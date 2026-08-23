"""
Controls and combination.

CONTROLS. The previous round's control was entropy-mismatched: uniform label
smoothing spreads lambda over 2048 tokens (entropy 7.6 nats) while the trigram
anchor spreads it over ~4 (entropy 1.4). Comparing them tests "concentrated vs
diffuse", not "context vs no context". These fix that.

  shuffled      take a real teacher's row from a DIFFERENT random position.
                Identical shape, entropy, support size and count statistics.
                Wrong context. This is the control that isolates the claim, and
                it is the direct analogue of latent_shuffle.
  topm_uniform  a real teacher's support with FLAT probabilities. Separates
                "which tokens are plausible here" from "how plausible each is".
  unigram       context-free corpus frequency.
  uniform       classic label smoothing over the whole vocabulary.

MIXTURE. Different teachers hold different things, so the interesting question
is not which rung wins but whether they COMPOSE. The exact-match rungs are
precise where they have attestation and silent where they do not; the
abstraction rungs are never silent but always vague. If perplexity gains add,
that is labour division and it is the result worth writing up.
"""

import numpy as np

from .base import Teacher


class ShuffledTeacher(Teacher):
    name = "shuffled"
    lacks = "CONTROL: matched shape and entropy, wrong context"

    def __init__(self, *a, inner=None, seed=0, **kw):
        super().__init__(*a, **kw)
        self.inner = inner
        self.seed = seed

    def signature(self):
        return {"inner": self.inner.name, "seed": self.seed, **self.inner.signature()}

    def _compute(self):
        idx, prob, cnt, mass = self.inner._compute()
        perm = np.random.default_rng(self.seed).permutation(len(idx))
        return idx[perm], prob[perm], cnt[perm], mass[perm]


class TopMUniformTeacher(Teacher):
    name = "topm_uniform"
    lacks = "CONTROL: right support, no probability structure"

    def __init__(self, *a, inner=None, **kw):
        super().__init__(*a, **kw)
        self.inner = inner

    def signature(self):
        return {"inner": self.inner.name, **self.inner.signature()}

    def _compute(self):
        idx, prob, cnt, mass = self.inner._compute()
        live = (prob > 0).astype(np.float32)
        n = live.sum(1, keepdims=True)
        return idx, np.divide(live, np.maximum(n, 1)).astype(np.float32), cnt, mass


class UnigramTeacher(Teacher):
    name = "unigram"
    lacks = "CONTROL: context-free corpus frequency"

    def _compute(self):
        ids, m, N, V = self.ids, self.m, len(self.ids), self.V
        uni = np.bincount(ids, minlength=V).astype(np.float64)
        top = np.argsort(-uni)[:m]
        p = uni[top] / max(uni[top].sum(), 1.0)
        idx = np.tile(top.astype(np.int32), (N, 1))
        prob = np.tile(p.astype(np.float32), (N, 1))
        return idx, prob, np.full(N, float(N), np.float32), np.ones(N, np.float32)


class UniformTeacher(Teacher):
    """Marker: classic label smoothing, handled analytically in the loss."""

    name = "uniform"
    lacks = "CONTROL: unstructured label smoothing"

    def _compute(self):
        N = len(self.ids)
        return (np.zeros((N, self.m), np.int32), np.zeros((N, self.m), np.float32),
                np.zeros(N, np.float32), np.zeros(N, np.float32))


class MixtureTeacher(Teacher):
    name = "mixture"
    lacks = "labour division: different rungs cover different positions"

    def __init__(self, *a, parts=None, weights=None, **kw):
        super().__init__(*a, **kw)
        self.parts = parts or []
        self.weights = weights or [1.0 / max(len(self.parts), 1)] * len(self.parts)

    def signature(self):
        return {"parts": ",".join(p.name for p in self.parts),
                "w": ",".join(f"{x:.3f}" for x in self.weights)}

    def _compute(self):
        N, m = len(self.ids), self.m
        idxs, probs, cnts, masses = [], [], [], []
        for p, w in zip(self.parts, self.weights):
            i, pr, c, ms = p._compute()
            idxs.append(i)
            probs.append(pr * np.float32(w))
            cnts.append(c)
            masses.append(ms)

        idx = np.concatenate(idxs, 1).astype(np.int64)
        prob = np.concatenate(probs, 1).astype(np.float64)

        # merge duplicate token ids: sort by id, carry each run's sum to its last slot
        o = np.argsort(idx, axis=1, kind="stable")
        idx = np.take_along_axis(idx, o, 1)
        prob = np.take_along_axis(prob, o, 1)
        for j in range(1, idx.shape[1]):
            dup = idx[:, j] == idx[:, j - 1]
            prob[dup, j] += prob[dup, j - 1]
            prob[dup, j - 1] = 0.0

        k = min(m, idx.shape[1])
        sel = np.argpartition(-prob, k - 1, axis=1)[:, :k]
        pv = np.take_along_axis(prob, sel, 1)
        iv = np.take_along_axis(idx, sel, 1)
        ordr = np.argsort(-pv, axis=1)
        pv, iv = np.take_along_axis(pv, ordr, 1), np.take_along_axis(iv, ordr, 1)

        s = pv.sum(1, keepdims=True)
        prob_out = np.where(s > 0, pv / np.maximum(s, 1e-12), 0.0).astype(np.float32)
        return (iv.astype(np.int32), prob_out,
                np.max(np.stack(cnts), 0).astype(np.float32),
                np.mean(np.stack(masses), 0).astype(np.float32))
