"""
N-gram-anchored soft targets: graded partial credit from the corpus.

This is the direct answer to the objection that started this project. Plain
cross-entropy's gradient on non-observed tokens is identity-blind: when the
truth is "happy", "glad" and "asparagus" are pushed down by amounts that depend
only on their current probability, not on whether they fit. The complaint is not
that hedging is punished (a proper scoring rule rewards it), it is that the
update carries no information about which alternatives were reasonable.

The target becomes

    q = (1 - lam) * onehot(y) + lam * ngram_posterior(context)

where the posterior is the LEAVE-ONE-OUT top-m successor distribution from
src/ngram_table.py. Three properties matter:

* EXTERNAL. It comes from the corpus, not from the model's own beliefs, so it
  cannot degenerate into the entropy knob that kills self-referential objectives.
* CANNOT COLLAPSE. Targets are fixed before training starts. No EMA encoder, no
  stop-gradient, no anti-collapse machinery needed.
* NOT A DISTILLATION FLOOR. The n-gram model is much weaker than the network, so
  it is used only to shape which alternatives receive credit, never as a target
  the model is asked to match. Mass (1-lam) always stays on the observed token.

Controls, both of which produce the same amount of smoothing with less structure:
  uniform  -- standard label smoothing (spread lam uniformly)
  unigram  -- spread lam over corpus token frequencies, ignoring context

If `uniform` matches `trigram`, the benefit was generic smoothing and the
context-grounded structure bought nothing.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from .token_losses import Loss, _flat_nll


class NgramSoftLoss(Loss):
    name = "ngram_soft"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        from ..ngram_table import NgramTable

        self.lam = args.soft_lambda
        self.mode = args.soft_mode              # trigram | uniform | unigram
        self.m = args.soft_top_m
        dev = corpus.train.device

        if self.mode == "uniform":
            self.idx = self.prob = None         # handled analytically
        else:
            tag = f"{corpus.vocab_size}_{corpus.train.numel()}_m{self.m}"
            tbl = NgramTable(corpus.train_np, corpus.vocab_size, top_m=self.m,
                             min_count=args.soft_min_count)
            if self.mode == "unigram":
                ui, up = tbl.unigram_topm()
                self.idx = torch.from_numpy(ui.astype(np.int64)).to(dev)
                self.prob = torch.from_numpy(up).to(dev)
            else:
                cache = os.path.join(args.cache_dir, f"ngramtab_{tag}.npz")
                idx, prob = tbl.build(cache_path=cache)
                self.idx = torch.from_numpy(idx.astype(np.int64)).to(dev)
                self.prob = torch.from_numpy(prob).to(dev)
        self._stats = {}

    def _soft_term(self, logits, pos):
        """-sum_k q_k log p_k over the smoothing distribution only."""
        logp = F.log_softmax(logits.float(), dim=-1)
        if self.mode == "uniform":
            return -logp.mean(dim=-1)                          # uniform over vocab
        if self.mode == "unigram":
            B, T, _ = logits.shape
            lp = logp[..., self.idx]                           # (B,T,m)
            return -(self.prob.view(1, 1, -1) * lp).sum(-1)
        ix = self.idx[pos]                                     # (B,T,m)
        pr = self.prob[pos]
        lp = torch.gather(logp, -1, ix)
        return -(pr * lp).sum(-1)

    def __call__(self, x, y, span_starts, step):
        logits = self.model(x)
        nll = _flat_nll(logits, y)

        B, T = y.shape
        pos = (span_starts[:, None] + torch.arange(T, device=y.device)[None, :] + 1)
        pos = pos.clamp(max=self.idx.shape[0] - 1) if self.mode == "trigram" else pos

        soft = self._soft_term(logits, pos)
        loss = (1 - self.lam) * nll.mean() + self.lam * soft.mean()

        if step % self.args.diag_every == 0:
            with torch.no_grad():
                d = {"soft_term": float(soft.mean().item())}
                if self.mode == "trigram":
                    cov = (self.prob[pos].sum(-1) > 0).float().mean()
                    ent = -(self.prob[pos] * (self.prob[pos] + 1e-9).log()).sum(-1).mean()
                    # how often does the anchor already put mass on the true token?
                    hit = (self.idx[pos] == y.unsqueeze(-1)).any(-1).float().mean()
                    d.update(soft_coverage=float(cov.item()),
                             soft_target_entropy=float(ent.item()),
                             soft_anchor_hit_rate=float(hit.item()))
                self._stats = d
        return {
            "loss": loss,
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": 1.0,
        }

    def diagnostics(self):
        return dict(self._stats)
