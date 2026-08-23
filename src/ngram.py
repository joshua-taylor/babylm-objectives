"""
A cheap n-gram reference model, used only to answer one question per token:

    "was this token already predictable from local surface statistics?"

The selective-loss arm (RHO-1 in spirit) trains on tokens with high *excess*
surprisal -- surprisal the model has that a trigram model does not explain.
The intuition from the design discussion: at a fixed token budget, most gradient
is spent on tokens that were already nearly free (the `u` after `q`). This
reference makes "already nearly free" measurable without training a second
neural model.

Honesty notes
-------------
* The trigram model is fit on the same corpus it scores, which makes it
  optimistic. We use leave-one-out counts (subtract the observed event) to
  damp this. It is still not a held-out estimate.
* Stupid backoff, not Kneser-Ney. We only need a ranking, not a calibrated
  probability, and the arm's kill criterion does not depend on the reference
  being well-calibrated.
"""

import os
from collections import defaultdict

import numpy as np

BACKOFF = 0.4


def _counts(ids):
    uni = defaultdict(int)
    bi = defaultdict(int)
    tri = defaultdict(int)
    ctx1 = defaultdict(int)
    ctx2 = defaultdict(int)
    prev2 = prev1 = None
    for w in ids:
        uni[w] += 1
        if prev1 is not None:
            bi[(prev1, w)] += 1
            ctx1[prev1] += 1
            if prev2 is not None:
                tri[(prev2, prev1, w)] += 1
                ctx2[(prev2, prev1)] += 1
        prev2, prev1 = prev1, w
    return uni, bi, tri, ctx1, ctx2


def reference_nll(ids: np.ndarray, cache_path: str = None, vocab_size: int = None):
    """Per-token negative log-likelihood under a leave-one-out stupid-backoff trigram.

    Returns float32 array aligned with `ids` (position i = surprisal of ids[i]).
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    ids = ids.astype(np.int64)
    N = len(ids)
    V = int(vocab_size or (ids.max() + 1))
    uni, bi, tri, ctx1, ctx2 = _counts(ids)
    total = float(N)

    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        w = ids[i]
        # unigram (leave-one-out, add-1 smoothed)
        p = (uni[w] - 1 + 1.0) / (total - 1 + V)
        if i >= 1:
            a = ids[i - 1]
            c_bi, c_c1 = bi[(a, w)] - 1, ctx1[a] - 1
            if c_bi > 0 and c_c1 > 0:
                p = c_bi / c_c1
            else:
                p = BACKOFF * p
            if i >= 2:
                b = ids[i - 2]
                c_tri, c_c2 = tri[(b, a, w)] - 1, ctx2[(b, a)] - 1
                if c_tri > 0 and c_c2 > 0:
                    p = c_tri / c_c2
                else:
                    p = BACKOFF * p
        out[i] = -np.log(max(p, 1e-12))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, out)
    return out
