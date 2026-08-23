"""
Shared trunk: a deliberately standard pre-norm transformer.

Every objective arm uses this identical trunk. The only thing that varies
between arms is the loss/head/data-order. That is the whole point: if an arm
wins, the win is attributable to the objective, not to an architecture change.

The trunk exposes `hidden(idx)` (returns the final normed hidden states) so
auxiliary heads can attach without duplicating the forward pass.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    vocab_size: int = 4096
    seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    causal: bool = True          # False => bidirectional (any-order / diffusion arm)
    tie_embeddings: bool = True


# ---------------------------------------------------------------- RoPE
def rope_tables(T: int, dh: int, device, base: float = 10000.0):
    i = torch.arange(0, dh, 2, device=device, dtype=torch.float32)
    inv = base ** (-i / dh)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    return cos[None, None], sin[None, None]


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos + rot * sin


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.H = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.causal = cfg.causal
        self.norm = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).split(d, dim=-1)
        q = q.view(B, T, self.H, self.dh).transpose(1, 2)
        k = k.view(B, T, self.H, self.dh).transpose(1, 2)
        v = v.view(B, T, self.H, self.dh).transpose(1, 2)
        cos, sin = rope_tables(T, self.dh, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.drop(self.o(out))


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.norm_ffn(x)))))
        return x


# ---------------------------------------------------------------- LM
class LM(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, idx):
        """Final normed hidden states, (B, T, d). Shared by every head."""
        x = self.emb_drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def logits_from_hidden(self, h):
        if self.cfg.tie_embeddings:
            return F.linear(h, self.tok_emb.weight)
        return self.lm_head(h)

    def forward(self, idx):
        return self.logits_from_hidden(self.hidden(idx))

    def n_params(self, trainable_only=True):
        ps = self.parameters()
        return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


# ---------------------------------------------------------------- checks
@torch.no_grad()
def verify_causality(model, device, vocab_size, seq_len=64, tol=1e-4):
    """Perturb the second half of the input; first-half logits must not move."""
    model.eval()
    B, T = 2, seq_len
    x = torch.randint(0, vocab_size, (B, T), device=device)
    l1 = model(x)
    cut = T // 2
    x2 = x.clone()
    x2[:, cut:] = torch.randint(0, vocab_size, (B, T - cut), device=device)
    l2 = model(x2)
    diff = (l1[:, :cut] - l2[:, :cut]).abs().max().item()
    nan = bool(torch.isnan(l1).any().item())
    model.train()
    ok = (diff < tol) and not nan
    return ok, diff, nan
