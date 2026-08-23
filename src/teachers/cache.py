"""
Recency cache: unbounded burstiness.

This is the teacher with the cleanest claim to privileged information. The
student has a 256-token window and, because spans are drawn independently at
training time, sees nothing of the surrounding corpus stream. The cache sees
thousands of tokens of causal history. Lexical burstiness -- a word that just
occurred is far more likely to occur again -- is one of the strongest and
cheapest regularities in language, and a fixed-window model cannot represent
the part of it that lives beyond its window.

Grave et al.'s neural cache keys by the model's own hidden states, which would
make the teacher circular. This one keys purely on token identity and decayed
recency, so it stays outside the weights.

    p_cache(w | j) proportional to sum over i<j of decay^(j-i) * [ids[i] = w]

Everything is strictly causal, so no leave-one-out correction is needed: the
target at j is never in its own window. If the target DID occur earlier that is
the burstiness signal, not leakage.

Implementation note: the exact top-m is computed blockwise. Within a block the
carried-in decayed counts are scaled by a single factor decay^t, which
preserves their ordering, so a token can only enter the top-m by receiving a
contribution inside the block. Candidates are therefore (top-m of the carry-in)
union (tokens occurring in the block) -- exact, and small.
"""

import numpy as np

from .base import Teacher


class CacheTeacher(Teacher):
    name = "cache"
    lacks = "unbounded recency/burstiness beyond the 256-token window"

    def __init__(self, *a, half_life=512.0, window=4096, block=256, **kw):
        super().__init__(*a, **kw)
        self.half_life = float(half_life)
        self.window = int(window)
        self.block = int(block)

    def signature(self):
        return {"hl": self.half_life, "win": self.window}

    def _compute(self):
        ids, m, N, V = self.ids, self.m, len(self.ids), self.V
        decay = float(np.exp(-np.log(2.0) / self.half_life))
        B = self.block

        idx = np.zeros((N, m), dtype=np.int32)
        prob = np.zeros((N, m), dtype=np.float32)
        cnt = np.zeros(N, dtype=np.float32)
        mass = np.ones(N, dtype=np.float32)

        carry = np.zeros(V, dtype=np.float64)
        pw = decay ** np.arange(B + 1, dtype=np.float64)

        for start in range(0, N, B):
            end = min(start + B, N)
            L = end - start
            blk = ids[start:end]

            # candidates: strongest carry-in tokens + everything seen in-block
            k = min(V, m + 4)
            cand = np.union1d(np.argpartition(-carry, k - 1)[:k], np.unique(blk))
            nc = len(cand)
            pos = -np.ones(V, dtype=np.int64)
            pos[cand] = np.arange(nc)

            # carry-in contribution, scaled by decay^t
            vals = carry[cand][None, :] * pw[1 : L + 1][:, None]        # (L, nc)

            # in-block contributions: token at start+s affects positions t > s
            rows = pos[blk]
            for s in range(L):
                c = rows[s]
                t = np.arange(s + 1, L)
                if t.size:
                    vals[t, c] += pw[t - s]

            k2 = min(m, nc)
            sel = np.argpartition(-vals, k2 - 1, axis=1)[:, :k2]
            sv = np.take_along_axis(vals, sel, 1)
            o = np.argsort(-sv, axis=1)
            sel, sv = np.take_along_axis(sel, o, 1), np.take_along_axis(sv, o, 1)

            tot = vals.sum(1)
            idx[start:end, :k2] = cand[sel].astype(np.int32)
            denom = np.maximum(sv.sum(1, keepdims=True), 1e-12)
            prob[start:end, :k2] = (sv / denom).astype(np.float32)
            cnt[start:end] = tot.astype(np.float32)
            mass[start:end] = (sv.sum(1) / np.maximum(tot, 1e-12)).astype(np.float32)

            # advance the carry to the start of the next block
            carry *= pw[L]
            np.add.at(carry, blk, pw[L - 1 - np.arange(L)])
            if self.window > 0:
                carry[carry < 1e-8] = 0.0

        dead = cnt < self.min_count
        prob[dead] = 0.0
        return idx, prob, cnt, np.where(dead, 0, mass).astype(np.float32)
