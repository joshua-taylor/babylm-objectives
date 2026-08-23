"""
Latent-target auxiliary objective (JEPA / BYOL lineage, adapted to LM).

The idea it tests: cross-entropy's gradient on non-observed tokens is
identity-blind -- when the truth is "happy", the update pushes down on "glad"
and "asparagus" by amounts that depend only on current probability, not on
semantic distance. A latent target gives a GRADED error signal, because
distance in a learned representation space is graded by construction.

Two constraints inherited from the project's own findings:

1. This is an AUXILIARY head. The token-level readout is never replaced. A
   pooled latent target is exactly the kind of pooled representation that the
   retrieval work found destroys precise recall, so it shapes the trunk and
   never becomes the thing the model reads from.
2. Collapse is the default outcome, not an edge case. Anti-collapse is
   structural: EMA target encoder + stop-gradient + VICReg variance/covariance
   terms. The effective rank of the TARGETS is a pre-registered kill criterion.

Ablations shipped alongside, because each is a live alternative explanation:
  target=ema      -- the real thing
  target=frozen   -- targets from a frozen random encoder (tests whether the
                     benefit is just "predict some smooth function of context")
  target=shuffle  -- targets shuffled across the batch (tests whether the
                     benefit is just the regularisation from the VICReg terms)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diagnostics import covariance_penalty, effective_rank, variance_penalty
from .token_losses import Loss, _flat_nll


class Predictor(nn.Module):
    def __init__(self, d, hidden_mult=2):
        super().__init__()
        h = d * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, d)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class LatentLoss(Loss):
    name = "latent"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        dev = next(model.parameters()).device
        d = model.cfg.d_model
        self.K = args.latent_horizon           # how far ahead to summarise
        self.lam = args.latent_weight
        self.momentum = args.latent_momentum
        self.var_w = args.latent_var_weight
        self.cov_w = args.latent_cov_weight
        self.target_mode = args.latent_target  # ema | frozen | shuffle

        self.predictor = Predictor(d).to(dev)
        self.target_encoder = copy.deepcopy(model).to(dev)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()
        self._stats = {}

    def extra_parameters(self):
        return list(self.predictor.parameters())

    @torch.no_grad()
    def on_optimizer_step(self, step):
        if self.target_mode != "ema":
            return
        m = self.momentum
        for pt, ps in zip(self.target_encoder.parameters(), self.model.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for bt, bs in zip(self.target_encoder.buffers(), self.model.buffers()):
            bt.copy_(bs)

    @torch.no_grad()
    def _targets(self, x):
        h = self.target_encoder.hidden(x)                    # (B, T, d)
        B, T, d = h.shape
        K = min(self.K, T - 1)
        # mean of the next K hidden states, per position
        cs = torch.cat([torch.zeros(B, 1, d, device=h.device, dtype=h.dtype),
                        h.cumsum(dim=1)], dim=1)             # (B, T+1, d)
        idx = torch.arange(T, device=h.device)
        hi = (idx + 1 + K).clamp(max=T)
        lo = (idx + 1).clamp(max=T)
        cnt = (hi - lo).clamp(min=1).float()[None, :, None]
        tgt = (cs[:, hi] - cs[:, lo]) / cnt
        valid = (idx + 1 + K) <= T                           # full window only
        if self.target_mode == "shuffle":
            perm = torch.randperm(B, device=h.device)
            tgt = tgt[perm]
        return tgt, valid

    def __call__(self, x, y, span_starts, step):
        h = self.model.hidden(x)
        logits = self.model.logits_from_hidden(h)
        nll = _flat_nll(logits, y)
        main = nll.mean()

        tgt, valid = self._targets(x)
        pred = self.predictor(h)

        p = F.normalize(pred[:, valid], dim=-1)
        t = F.normalize(tgt[:, valid], dim=-1).detach()
        if p.numel() == 0:
            aux = torch.zeros((), device=h.device)
            var = cov = torch.zeros((), device=h.device)
        else:
            aux = (1.0 - (p * t).sum(-1)).mean()
            pf = p.reshape(-1, p.shape[-1])
            var = variance_penalty(pf)
            cov = covariance_penalty(pf)

        loss = main + self.lam * (aux + self.var_w * var + self.cov_w * cov)

        if step % self.args.diag_every == 0:
            with torch.no_grad():
                tf = tgt[:, valid].reshape(-1, tgt.shape[-1])
                self._stats = {
                    "latent_cos_loss": float(aux.item()),
                    "latent_var_pen": float(var.item()),
                    "latent_cov_pen": float(cov.item()),
                    "latent_target_erank": effective_rank(tf) if tf.numel() else float("nan"),
                    "latent_target_erank_frac": (
                        effective_rank(tf) / tgt.shape[-1] if tf.numel() else float("nan")
                    ),
                    "latent_target_std": float(tf.std(0).mean().item()) if tf.numel() else 0.0,
                }
        return {
            "loss": loss,
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": 1.0,
        }

    def diagnostics(self):
        return dict(self._stats)
