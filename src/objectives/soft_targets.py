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

import copy

import torch
import torch.nn.functional as F

from ..teachers import build_teacher
from ..teachers.base import truncate_topm
from .token_losses import Loss, _flat_nll


SELF_SPECS = {"self", "self_probs"}


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
        self.dynamic = self.spec in SELF_SPECS
        self.self_flat = self.spec == "self"

        if self.dynamic:
            # THE CONTROL THAT COULD COLLAPSE THE WHOLE LADDER.
            #
            # Run 4's finding was that the SUPPORT carries the signal and the
            # teacher's probabilities carry nothing. If that is true, there is a
            # much cheaper source of a context-appropriate candidate set than any
            # corpus statistic: the model's own top-k, read off an EMA copy of
            # itself. No table, no precomputation, no external teacher, and it
            # scales to any model size for the price of one parameter copy.
            #
            # This is NOT the self-scoring trap. That failed because using a
            # model's own PROBABILITIES as a target reduces to an entropy knob --
            # E_{y~p}[log p(y)] is negative entropy. Using its own SUPPORT with
            # flat probabilities is a different object, and run 4 says support is
            # the part that matters.
            #
            # If this matches the n-gram teacher, the ladder is unnecessary and
            # the result is a one-line trick. If it does not, the external corpus
            # statistics are doing real work and the ladder is justified. Either
            # outcome is worth knowing; not knowing is not.
            self.idx = self.prob = self.cnt = None
            self.ema_model = copy.deepcopy(model).to(dev)
            for q in self.ema_model.parameters():
                q.requires_grad_(False)
            self.ema_model.eval()
            self.self_momentum = args.self_momentum
            self.self_warmup = args.self_warmup
            self.self_exclude_true = bool(args.self_exclude_true)
            self.treport = {"teacher": self.spec,
                            "lacks": "CONTROL: the model's own EMA top-k support"}
        elif self.uniform_only:
            self.idx = self.prob = self.cnt = None
            self.treport = {"teacher": "uniform", "lacks": "CONTROL: label smoothing"}
        else:
            # Build wide, truncate at use: makes the m sweep free for cached
            # teachers (no rebuild) now that m is the central variable.
            cache_m = max(args.teacher_cache_m, args.soft_top_m)
            T = build_teacher(self.spec, corpus.train_np, corpus.vocab_size,
                              m=cache_m, min_count=args.soft_min_count, args=args)
            idx, prob, cnt, mass = T.build(cache_dir=args.cache_dir)
            if args.soft_top_m < idx.shape[1]:
                idx, prob = truncate_topm(idx, prob, args.soft_top_m)
            if getattr(args, "soft_flatten", 0):
                # Support-only: keep WHICH tokens, discard HOW likely. This is the
                # entire content of the run-4 finding, applied to any teacher.
                import numpy as np
                live = (prob > 0).astype("float32")
                prob = live / np.maximum(live.sum(1, keepdims=True), 1.0)
            self.treport = T.report(idx, prob, cnt, mass)
            self.treport["m_used"] = int(idx.shape[1])
            self.idx = torch.from_numpy(idx.astype("int64")).to(dev)
            self.prob = torch.from_numpy(prob).to(dev)
            self.cnt = torch.from_numpy(cnt).to(dev)
        self._stats = {}

    def teacher_report(self):
        return dict(self.treport)

    @torch.no_grad()
    def on_optimizer_step(self, step):
        if not self.dynamic:
            return
        if step <= self.self_warmup:
            # Reset rather than accumulate during warmup: an EMA that averages in
            # the random initialisation produces a garbage candidate set, and the
            # damage persists for thousands of steps.
            for pt, ps in zip(self.ema_model.parameters(), self.model.parameters()):
                pt.copy_(ps.detach())
            return
        m = self.self_momentum
        for pt, ps in zip(self.ema_model.parameters(), self.model.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for bt, bs in zip(self.ema_model.buffers(), self.model.buffers()):
            bt.copy_(bs)

    @torch.no_grad()
    def _self_target(self, x, y, step):
        """Top-k support from the model's own EMA copy, computed on the fly."""
        logits = self.ema_model(x).float()
        if self.self_exclude_true:
            logits = logits.scatter(-1, y.unsqueeze(-1), float("-inf"))
        k = min(self.args.soft_top_m, logits.size(-1))
        tv, ti = torch.topk(torch.softmax(logits, dim=-1), k, dim=-1)
        if self.self_flat:
            q = torch.full_like(tv, 1.0 / k)
        else:
            q = tv / tv.sum(-1, keepdim=True).clamp(min=1e-9)
        return ti, q

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

        if self.dynamic:
            if step <= self.self_warmup:
                return self._pack(nll.mean(), nll)     # plain CE until the EMA is real
            ix, q = self._self_target(x, y, step)
            lam = torch.full_like(nll, self.lam)
            lq = torch.gather(logp, -1, ix)
            if self.form == "hinge":
                pen = (torch.relu(torch.log(q.clamp(min=1e-6)) - lq)).mean(-1)
                loss = nll.mean() + (lam * pen).mean()
            else:
                loss = ((1 - lam) * nll - lam * (q * lq).sum(-1)).mean()
            if step % self.args.diag_every == 0:
                with torch.no_grad():
                    hit = (ix == y.unsqueeze(-1))
                    self._stats = {
                        "self_hit_rate": float(hit.any(-1).float().mean().item()),
                        "self_top1_is_true": float((ix[..., 0] == y).float().mean().item()),
                        "self_target_entropy": float(
                            (-(q * (q + 1e-9).log()).sum(-1)).mean().item()),
                    }
            return self._pack(loss, nll)

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
