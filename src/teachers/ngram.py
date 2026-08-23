"""
Exact-match teachers: cross-position pooling.

`trigram` answers "what followed this exact 2-token context elsewhere in the
corpus". `varorder` answers the same at the LONGEST context that still has
enough evidence, backing off order by order -- which is the infini-gram
construction, and equally is an induction head's computation moved out of the
architecture and into the loss. If varorder beats trigram, exact-match
retrieval helps whether you hard-wire it into the model or into the training
signal, which is a stronger claim than either result alone.

The order cap is a guard, not a limitation. At 1M tokens a very long exact
match may have only one other occurrence, usually a near-duplicate passage, so
an uncapped teacher degenerates into an instruction to memorise. `min_count`
requires several DISTINCT attestations before a long context is trusted.
"""

import numpy as np

from .base import Teacher, grouped_topm, loo_normalise, rolling_hash


class NgramTeacher(Teacher):
    name = "trigram"
    lacks = "cross-position pooling: what followed this exact context elsewhere"

    def __init__(self, *a, order=2, **kw):
        super().__init__(*a, **kw)
        self.order = order

    def signature(self):
        return {"order": self.order}

    def _compute(self):
        ids, m = self.ids, self.m
        M = m + 2
        key = rolling_hash(ids, self.order)
        uctx, tidx, tcnt, ttot = grouped_topm(key, ids, M)

        row = np.searchsorted(uctx, key)
        row = np.clip(row, 0, len(uctx) - 1)
        idx, prob, tot, mass = loo_normalise(tidx[row], tcnt[row], ttot[row], ids, m)

        dead = (np.arange(len(ids)) < self.order) | (tot < self.min_count)
        prob[dead] = 0.0
        return idx, prob, tot.astype(np.float32), np.where(dead, 0, mass).astype(np.float32)


class VarOrderTeacher(Teacher):
    """Adaptive-order backoff: use the longest context with >= min_count evidence."""

    name = "varorder"
    lacks = "adaptive-length exact-match retrieval (an induction head, in the loss)"

    def __init__(self, *a, max_order=6, min_order=1, **kw):
        super().__init__(*a, **kw)
        self.max_order = max_order
        self.min_order = min_order

    def signature(self):
        return {"max_order": self.max_order, "min_order": self.min_order}

    def _compute(self):
        ids, m, N = self.ids, self.m, len(self.ids)
        M = m + 2
        idx = np.zeros((N, m), dtype=np.int32)
        prob = np.zeros((N, m), dtype=np.float32)
        cnt = np.zeros(N, dtype=np.float32)
        mass = np.zeros(N, dtype=np.float32)
        chosen = np.zeros(N, dtype=np.int8)
        todo = np.ones(N, dtype=bool)          # positions still unassigned

        # longest order first; once a position is served, lower orders skip it
        for order in range(self.max_order, self.min_order - 1, -1):
            if not todo.any():
                break
            key = rolling_hash(ids, order)
            uctx, tidx, tcnt, ttot = grouped_topm(key, ids, M)
            row = np.clip(np.searchsorted(uctx, key), 0, len(uctx) - 1)
            i2, p2, t2, m2 = loo_normalise(tidx[row], tcnt[row], ttot[row], ids, m)

            ok = todo & (np.arange(N) >= order) & (t2 >= self.min_count) & (p2.sum(1) > 0)
            idx[ok], prob[ok], cnt[ok], mass[ok] = i2[ok], p2[ok], t2[ok], m2[ok]
            chosen[ok] = order
            todo &= ~ok

        self.order_hist = np.bincount(chosen, minlength=self.max_order + 1)
        return idx, prob, cnt, mass

    def report(self, *a):
        r = super().report(*a)
        h = getattr(self, "order_hist", None)
        if h is not None:
            tot = max(int(h.sum()), 1)
            r["order_mix"] = {f"o{k}": round(float(h[k]) / tot, 3)
                              for k in range(len(h)) if h[k] > 0}
        return r
