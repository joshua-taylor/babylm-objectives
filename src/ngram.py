"""
A cheap n-gram model, used for two things.

1. TOKEN VALUE (selective arm): per-token "excess surprisal" -- surprisal the
   neural model has that a trigram model does not already explain. Scored with
   leave-one-out counts on the corpus the model was fit on.

2. ABSOLUTE ANCHOR (every arm): a trigram fit on train and scored on val gives
   a floor that any competent neural model must beat. This exists because of a
   bug that shipped in the first version of this repo: the train/val split was
   accidentally a domain split, every arm scored at or above the uniform-random
   ceiling, and nothing in the pipeline noticed. An absolute anchor makes that
   class of failure impossible to miss -- if the model cannot beat a trigram on
   val, the run is broken and no comparison between arms means anything.
"""

import os
from collections import defaultdict

import numpy as np

BACKOFF = 0.4


class Trigram:
    def __init__(self, ids, vocab_size=None):
        ids = np.asarray(ids, dtype=np.int64)
        self.V = int(vocab_size or (ids.max() + 1))
        self.N = len(ids)
        self.uni = defaultdict(int)
        self.bi = defaultdict(int)
        self.tri = defaultdict(int)
        self.ctx1 = defaultdict(int)
        self.ctx2 = defaultdict(int)
        prev2 = prev1 = None
        for w in ids:
            self.uni[w] += 1
            if prev1 is not None:
                self.bi[(prev1, w)] += 1
                self.ctx1[prev1] += 1
                if prev2 is not None:
                    self.tri[(prev2, prev1, w)] += 1
                    self.ctx2[(prev2, prev1)] += 1
            prev2, prev1 = prev1, w

    def score(self, ids, loo=False):
        """Per-token NLL in nats. `loo` subtracts the observed event (use only
        when scoring the same corpus the model was fit on)."""
        ids = np.asarray(ids, dtype=np.int64)
        d = 1 if loo else 0
        total = float(self.N - d)
        out = np.zeros(len(ids), dtype=np.float32)
        for i in range(len(ids)):
            w = ids[i]
            p = (self.uni[w] - d + 1.0) / (total + self.V)
            if i >= 1:
                a = ids[i - 1]
                c_bi, c_c1 = self.bi[(a, w)] - d, self.ctx1[a] - d
                p = (c_bi / c_c1) if (c_bi > 0 and c_c1 > 0) else BACKOFF * p
                if i >= 2:
                    b = ids[i - 2]
                    c_tri, c_c2 = self.tri[(b, a, w)] - d, self.ctx2[(b, a)] - d
                    p = (c_tri / c_c2) if (c_tri > 0 and c_c2 > 0) else BACKOFF * p
            out[i] = -np.log(max(p, 1e-12))
        return out


def reference_nll(ids, cache_path=None, vocab_size=None):
    """Leave-one-out self-scored surprisal (token-value signal for `selective`)."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    out = Trigram(ids, vocab_size).score(ids, loo=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, out)
    return out


def anchor_nll(train_ids, val_ids, cache_path=None, vocab_size=None):
    """Fit on train, score val. Returns mean nats/token -- the absolute floor."""
    if cache_path and os.path.exists(cache_path):
        return float(np.load(cache_path))
    m = Trigram(train_ids, vocab_size)
    val = float(m.score(val_ids, loo=False).mean())
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, np.array(val))
    return val
