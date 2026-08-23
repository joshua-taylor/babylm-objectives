"""
Precomputed n-gram successor distributions -- the external anchor.

For every training position j this stores the top-m tokens that could follow the
context (ids[j-2], ids[j-1]) according to the REST of the corpus, with their
probabilities. Backoff: trigram -> bigram -> unigram, whichever context has
enough evidence.

THE CRITICAL DETAIL: LEAVE-ONE-OUT
----------------------------------
The occurrence being scored is subtracted from the counts. Without this, a
context seen once has exactly one successor -- the true next token -- so the
"soft target" collapses back to the one-hot target and the whole thing becomes a
memorisation amplifier. With leave-one-out, the distribution answers the question
we actually want:

    "given this context, what ELSE does the corpus say could come next?"

That is graded partial credit, grounded in real data, computed independently of
the model's own beliefs. It is the thing plain cross-entropy cannot supply: when
the truth is "happy", the gradient on "glad" and "asparagus" is identity-blind,
and this table knows the difference because the corpus does.

Contexts with fewer than `min_count` remaining observations back off, so rare
contexts contribute smoothing rather than memorised continuations.
"""

import os
from collections import defaultdict

import numpy as np


class NgramTable:
    def __init__(self, ids, vocab_size, top_m=8, min_count=3):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.V = int(vocab_size)
        self.m = top_m
        self.min_count = min_count
        self._build_counts()

    def _build_counts(self):
        ids = self.ids
        self.uni = np.bincount(ids, minlength=self.V).astype(np.int64)
        self.succ2 = defaultdict(lambda: defaultdict(int))   # (b,a) -> {w: count}
        self.succ1 = defaultdict(lambda: defaultdict(int))   # a     -> {w: count}
        self.bi_count = defaultdict(int)                     # (a,w) -> count
        for i in range(1, len(ids)):
            a, w = int(ids[i - 1]), int(ids[i])
            self.succ1[a][w] += 1
            self.bi_count[(a, w)] += 1
            if i >= 2:
                b = int(ids[i - 2])
                self.succ2[(b, a)][w] += 1

    # ----------------------------------------------------------- judges
    def unseen_bigram(self, a, w):
        """High-precision 'definitely off-distribution' detector.

        Used only as a NEGATIVE signal. The n-gram model is a bad judge of what
        is good and a reliable judge of what never occurs.
        """
        return self.bi_count.get((int(a), int(w)), 0) == 0

    # ----------------------------------------------------------- table
    def _dist_at(self, j):
        """Leave-one-out successor distribution for position j (target ids[j])."""
        w_true = int(self.ids[j])
        if j >= 2:
            key = (int(self.ids[j - 2]), int(self.ids[j - 1]))
            d = self.succ2.get(key)
            if d is not None:
                tot = sum(d.values()) - 1
                if tot >= self.min_count:
                    items = [(w, c - (1 if w == w_true else 0)) for w, c in d.items()]
                    items = [(w, c) for w, c in items if c > 0]
                    if items:
                        return items, tot
        if j >= 1:
            a = int(self.ids[j - 1])
            d = self.succ1.get(a)
            if d is not None:
                tot = sum(d.values()) - 1
                if tot >= self.min_count:
                    items = [(w, c - (1 if w == w_true else 0)) for w, c in d.items()]
                    items = [(w, c) for w, c in items if c > 0]
                    if items:
                        return items, tot
        cnt = self.uni.copy()
        cnt[w_true] -= 1
        top = np.argpartition(-cnt, self.m)[: self.m]
        return [(int(w), int(cnt[w])) for w in top if cnt[w] > 0], int(cnt.sum())

    def build(self, cache_path=None):
        """-> (idx int32 [N,m], prob float32 [N,m]). Rows sum to 1 (or 0 if empty)."""
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path)
            return z["idx"], z["prob"]

        N, m = len(self.ids), self.m
        idx = np.zeros((N, m), dtype=np.int32)
        prob = np.zeros((N, m), dtype=np.float32)
        for j in range(N):
            items, _ = self._dist_at(j)
            if not items:
                continue
            items.sort(key=lambda t: -t[1])
            items = items[:m]
            tot = float(sum(c for _, c in items))
            if tot <= 0:
                continue
            for k, (w, c) in enumerate(items):
                idx[j, k] = w
                prob[j, k] = c / tot

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(cache_path, idx=idx, prob=prob)
        return idx, prob

    def unigram_topm(self):
        """Context-free frequency distribution -- the control for the table above."""
        cnt = self.uni.astype(np.float64)
        top = np.argsort(-cnt)[: self.m]
        p = cnt[top] / max(cnt[top].sum(), 1.0)
        return top.astype(np.int32), p.astype(np.float32)
