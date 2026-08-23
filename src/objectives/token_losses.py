"""
Token-level objectives.

  ntp        -- standard next-token cross-entropy. The baseline everything is
                measured against.
  mtp        -- multi-token prediction (Gloeckle et al. 2024). Included as a
                KNOWN QUANTITY, not as a hopeful. The project dossier already
                records that an auxiliary t+2 objective slows next-token
                convergence at a fixed step budget, so this arm is judged on
                best-achievable val loss and on the train/val gap, never on
                fixed-step main perplexity.
  selective  -- train only on the top-p fraction of tokens by EXCESS surprisal
                over a trigram reference (RHO-1 in spirit, cheap reference).
                Targets the real defect in cross-entropy at a fixed budget:
                every token gets equal gradient weight, so most of the budget
                goes to tokens that were already nearly free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    """Interface. `__call__` returns a dict with at least `loss` and `nll`."""

    name = "base"
    requires_causal = True
    supports_ar_eval = True

    def __init__(self, model, corpus, args):
        self.model = model
        self.corpus = corpus
        self.args = args

    def extra_parameters(self):
        return []

    def on_optimizer_step(self, step):
        pass

    def diagnostics(self):
        return {}


def _flat_nll(logits, y):
    """Per-token NLL, shape (B, T)."""
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1), reduction="none"
    ).view(B, T)


# ------------------------------------------------------------------ ntp
class NTPLoss(Loss):
    name = "ntp"

    def __call__(self, x, y, span_starts, step):
        logits = self.model(x)
        nll = _flat_nll(logits, y)
        return {
            "loss": nll.mean(),
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": 1.0,
        }


# ------------------------------------------------------------------ mtp
class MTPLoss(Loss):
    name = "mtp"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        d = model.cfg.d_model
        self.horizon = args.mtp_horizon          # extra heads beyond t+1
        self.weight = args.mtp_weight
        self.heads = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(self.horizon)]
        ).to(next(model.parameters()).device)
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)

    def extra_parameters(self):
        return list(self.heads.parameters())

    def __call__(self, x, y, span_starts, step):
        h = self.model.hidden(x)
        logits = self.model.logits_from_hidden(h)
        nll = _flat_nll(logits, y)
        loss = nll.mean()
        aux_terms = []
        for k, head in enumerate(self.heads, start=1):
            if y.size(1) <= k:
                continue
            al = self.model.logits_from_hidden(head(h[:, :-k]))
            at = y[:, k:]
            aux_terms.append(_flat_nll(al, at).mean())
        if aux_terms:
            loss = loss + self.weight * torch.stack(aux_terms).mean()
        return {
            "loss": loss,
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": 1.0,
        }


# ------------------------------------------------------------------ selective
class SelectiveLoss(Loss):
    name = "selective"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        import os

        import numpy as np

        from ..ngram import reference_nll

        cache = os.path.join(args.cache_dir, f"refnll_{corpus.cfg.vocab_size}_{corpus.cfg.n_tokens}.npy")
        ids = corpus.train.detach().cpu().numpy()
        ref = reference_nll(ids, cache_path=cache, vocab_size=corpus.vocab_size)
        self.ref = torch.from_numpy(np.asarray(ref)).to(corpus.train.device)
        self.keep = args.selective_keep          # fraction of tokens kept
        self.mode = args.selective_mode          # 'excess' | 'refhigh' | 'random'
        self._last_frac = 1.0

    def _token_value(self, nll, y_positions):
        ref = self.ref[y_positions]              # reference surprisal of the TARGET token
        if self.mode == "excess":
            return nll.detach() - ref
        if self.mode == "refhigh":               # ablation: ignore the model, use ref only
            return ref
        if self.mode == "random":                # ablation: same sparsity, no signal
            return torch.rand_like(nll)
        raise ValueError(self.mode)

    def __call__(self, x, y, span_starts, step):
        logits = self.model(x)
        nll = _flat_nll(logits, y)

        B, T = y.shape
        # target token i of span starting at s sits at corpus index s + i + 1
        pos = span_starts[:, None] + torch.arange(T, device=y.device)[None, :] + 1
        pos = pos.clamp(max=self.ref.numel() - 1)

        value = self._token_value(nll, pos)
        k = max(1, int(self.keep * T))
        thresh = value.topk(k, dim=1).values[:, -1:]
        mask = (value >= thresh).float()
        self._last_frac = mask.mean().item()

        loss = (nll * mask).sum() / mask.sum().clamp(min=1.0)
        return {
            "loss": loss,
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": self._last_frac,
        }

    def diagnostics(self):
        return {"selective_frac": self._last_frac}
