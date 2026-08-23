"""
Privileged-information distillation from a non-parametric teacher.

WHAT THIS IS
------------
Not label smoothing with a fancy distribution. When a context occurs ten times
in the corpus with successors {w1:5, w2:3, w3:2}, the one-hot targets across
those ten positions already convey that distribution IN AGGREGATE -- in
expectation the gradients are identical. What a soft target changes is that
each step receives the CONDITIONAL EXPECTATION of the target instead of a
single Monte Carlo sample of it.

That is Rao-Blackwellisation of the training target. It cannot increase
variance and generally reduces it, and gradient variance is most costly exactly
where contexts are rare -- which is the whole small-data regime. It predicts
that the benefit shrinks as data grows and concentrates on moderate-count
contexts, both of which are testable.

Leave-one-out is what makes the teacher's information genuinely privileged
rather than a copy of the label: it answers "what followed this context
ELSEWHERE", which the student at this position cannot see.

TWO FORMS
---------
mixture  q = (1-lam)*onehot(y) + lam*teacher.  Symmetric: the teacher can push
         probability up on its support and, implicitly, down elsewhere.

hinge    penalise relu(log q_k - log p_k) on the teacher's support only. The
         teacher can raise a floor and never lower anything, and full mass
         stays on the observed token.

         The motivation is the same asymmetry that governs the dream judges: a
         sparse-count teacher is reliable when it says "this token is plausible
         here" (it observed it) and unreliable when it says "this token is
         implausible" (it merely lacked evidence). Measured on this corpus, the
         trigram teacher has no attestation for the true token about 40% of the
         time, and on those positions the mixture form moves mass off the truth.
         The hinge removes that failure mode and the weak-teacher ceiling with
         it: the student may freely become better than the teacher, but never
         worse on tokens the teacher has evidence for.

COUNT-ADAPTIVE LAMBDA
---------------------
    lam_j = lam_max * n_j / (n_j + kappa)

Bayesian shrinkage: trust the teacher in proportion to the evidence behind it.
Replaces the hard `min_count` cutoff, which threw away weak-but-real signal and
kept strong-but-noisy signal at the same weight.
"""

import torch
import torch.nn.functional as F

from ..teachers import build_teacher
from .token_losses import Loss, _flat_nll


class SoftTargetLoss(Loss):
    name = "ngram_soft"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        dev = corpus.train.device
        self.lam = args.soft_lambda
        self.form = args.soft_form                  # mixture | hinge
        self.spec = getattr(args, "teacher", None) or getattr(args, "soft_mode", "trigram")
        self.adaptive = bool(args.soft_adaptive_lambda)
        self.kappa = args.soft_kappa
        self.uniform_only = self.spec == "uniform"

        if self.uniform_only:
            self.idx = self.prob = self.cnt = None
            self.treport = {"teacher": "uniform", "lacks": "CONTROL: label smoothing"}
        else:
            T = build_teacher(self.spec, corpus.train_np, corpus.vocab_size,
                              m=args.soft_top_m, min_count=args.soft_min_count, args=args)
            idx, prob, cnt, mass = T.build(cache_dir=args.cache_dir)
            self.treport = T.report(idx, prob, cnt, mass)
            self.idx = torch.from_numpy(idx.astype("int64")).to(dev)
            self.prob = torch.from_numpy(prob).to(dev)
            self.cnt = torch.from_numpy(cnt).to(dev)
        self._stats = {}

    def teacher_report(self):
        return dict(self.treport)

    def _lam(self, pos):
        if not self.adaptive or self.cnt is None:
            return self.lam
        n = self.cnt[pos]
        return self.lam * (n / (n + self.kappa))

    def __call__(self, x, y, span_starts, step):
        logits = self.model(x)
        nll = _flat_nll(logits, y)
        logp = F.log_softmax(logits.float(), dim=-1)

        B, T = y.shape
        pos = (span_starts[:, None] + torch.arange(T, device=y.device)[None, :] + 1)

        if self.uniform_only:
            soft = -logp.mean(dim=-1)
            loss = (1 - self.lam) * nll.mean() + self.lam * soft.mean()
            self._stats = {"soft_term": float(soft.mean().item())}
            return self._pack(loss, nll)

        pos = pos.clamp(max=self.idx.shape[0] - 1)
        q = self.prob[pos]                                   # (B,T,m)
        ix = self.idx[pos]
        lam = self._lam(pos)
        lam = lam if torch.is_tensor(lam) else torch.full_like(nll, lam)
        live = (q.sum(-1) > 0).float()
        lam = lam * live                                     # no teacher -> plain CE

        lq = torch.gather(logp, -1, ix)                      # model logprob on support

        if self.form == "hinge":
            # one-sided: raise a floor, never push anything down
            live_k = (q > 0).float()
            gap = torch.relu(torch.log(q.clamp(min=1e-6)) - lq) * live_k
            # mean over the live support, not sum, so lambda means the same thing
            # in both forms and is comparable across teachers with different m
            pen = gap.sum(-1) / live_k.sum(-1).clamp(min=1.0)
            loss = nll.mean() + (lam * pen).mean()
        else:
            soft = -(q * lq).sum(-1)
            loss = ((1 - lam) * nll + lam * soft).mean()

        if step % self.args.diag_every == 0:
            with torch.no_grad():
                hit = (ix == y.unsqueeze(-1)) & (q > 0)
                ent = -(q * (q + 1e-9).log()).sum(-1)
                self._stats = {
                    "soft_lambda_eff": float(lam.mean().item()),
                    "soft_coverage": float(live.mean().item()),
                    "soft_anchor_hit_rate": float(hit.any(-1).float().mean().item()),
                    "soft_target_entropy": float((ent * live).sum().item()
                                                 / live.sum().clamp(min=1).item()),
                }
        return self._pack(loss, nll)

    def _pack(self, loss, nll):
        return {"loss": loss, "nll": nll.mean().detach(),
                "per_span_nll": nll.mean(dim=1).detach(), "frac_supervised": 1.0}

    def diagnostics(self):
        return dict(self._stats)


NgramSoftLoss = SoftTargetLoss
