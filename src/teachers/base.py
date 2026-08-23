"""
Teachers: the privileged-information ladder.

THE ORGANISING PRINCIPLE
------------------------
Self-training cannot add information about the target distribution -- the
data-processing inequality settles that, and it is why scoring your own
rollouts reduces to an entropy knob. What self-training CAN do is convert
implicit knowledge into explicit knowledge, but only when the generating
process is stronger than the student. At 1M parameters there is no search and
no chain of thought, so the amplifier has to come from somewhere else:

    at training time we have information the causal student
    structurally cannot access at inference time.

Every teacher here is built from that asymmetry. A teacher is not chosen for
being a good language model -- a trigram is a terrible language model -- but
for holding something a 4-layer causal transformer with a 256-token window
does not hold. What matters is COMPLEMENTARITY, not quality.

THE LADDER, by what the student lacks
-------------------------------------
  trigram    cross-position pooling. What followed this exact context ELSEWHERE
             in the corpus. The student sees each position once per epoch and
             must discover the pooling by gradient descent.
  varorder   the same, at adaptive order (the longest context with enough
             evidence). This is exact-match retrieval -- an induction head's
             computation -- relocated from the architecture into the loss.
  cache      unbounded recency. Lexical burstiness over a window far wider than
             the student's 256 tokens. The student literally cannot see it.
  class      distributional abstraction. Pools successors by learned token
             CLASS, so a (context, token) pair never observed together still
             gets mass if their classes co-occur. This is the only teacher that
             can assign probability to a combination absent from the corpus --
             the "ten times table" mechanism.
  embed      soft context generalisation. Pools successors over SIMILAR
             contexts, attacking the ~40% of positions where exact-match
             teachers have no attestation for the true token.
  mixture    labour division. Different teachers cover different positions.

CONTROLS (in controls.py), each matched on a different axis:
  shuffled       identical shape, entropy and support statistics; wrong context.
  topm_uniform   right support, flat probabilities: is it WHICH tokens or HOW MUCH?
  unigram        context-free frequency.
  uniform        classic label smoothing.

INTERFACE
---------
Every teacher produces, for each training position j, the same object: the
top-m successors of the context at j with probabilities, plus the evidence
count behind them. Uniform interface, so the loss never knows which rung it is
on and rungs compose.

LEAVE-ONE-OUT is applied everywhere and is load-bearing. Without it, a context
seen once has exactly one successor -- the true token -- the soft target
collapses to one-hot, and the mechanism becomes a memorisation amplifier.
"""

import hashlib
import os

import numpy as np

FLOOR = 1e-9


# --------------------------------------------------------------------------
# vectorised segmented top-k: the workhorse behind most teachers
# --------------------------------------------------------------------------
def rolling_hash(ids: np.ndarray, order: int, seed: int = 0x9E3779B1) -> np.ndarray:
    """uint64 hash of the `order` tokens preceding each position.

    Positions < order get hash 0 and are treated as having no context.
    Collision probability at N=1e6 over 2^64 is negligible.
    """
    n = len(ids)
    h = np.zeros(n, dtype=np.uint64)
    P = np.uint64(1099511628211)
    base = np.uint64(seed + order * 7919)
    acc = np.full(n, base, dtype=np.uint64)
    for k in range(1, order + 1):
        shifted = np.zeros(n, dtype=np.uint64)
        shifted[k:] = ids[: n - k].astype(np.uint64) + np.uint64(1)
        acc = acc * P + shifted
    h[:] = acc
    h[:order] = np.uint64(0)
    return h


def grouped_topm(ctx_key, nxt, m, n_positions=None):
    """Top-m successors per context, fully vectorised.

    Returns
      row_of_ctx : (n_ctx,) sorted unique context keys
      top_idx    : (n_ctx, m) successor token ids (-1 = empty slot)
      top_cnt    : (n_ctx, m) counts
      ctx_total  : (n_ctx,) total successor count for that context
    """
    ctx_key = np.asarray(ctx_key)
    nxt = np.asarray(nxt, dtype=np.int64)

    # unique (ctx, tok) pairs with counts
    order = np.lexsort((nxt, ctx_key))
    ck, nx = ctx_key[order], nxt[order]
    newpair = np.empty(len(ck), dtype=bool)
    newpair[0] = True
    newpair[1:] = (ck[1:] != ck[:-1]) | (nx[1:] != nx[:-1])
    starts = np.flatnonzero(newpair)
    pair_ctx, pair_tok = ck[starts], nx[starts]
    pair_cnt = np.diff(np.append(starts, len(ck)))

    # per-context totals
    uctx, ctx_inv = np.unique(pair_ctx, return_inverse=True)
    ctx_total = np.zeros(len(uctx), dtype=np.int64)
    np.add.at(ctx_total, ctx_inv, pair_cnt)

    # sort pairs by (context, descending count) -> rank within context
    o2 = np.lexsort((-pair_cnt, ctx_inv))
    g, t, c = ctx_inv[o2], pair_tok[o2], pair_cnt[o2]
    gstart = np.zeros(len(g), dtype=np.int64)
    first = np.empty(len(g), dtype=bool)
    first[0] = True
    first[1:] = g[1:] != g[:-1]
    fidx = np.flatnonzero(first)
    gstart[fidx] = fidx
    np.maximum.accumulate(gstart, out=gstart)
    rank = np.arange(len(g)) - gstart

    keep = rank < m
    top_idx = np.full((len(uctx), m), -1, dtype=np.int64)
    top_cnt = np.zeros((len(uctx), m), dtype=np.int64)
    top_idx[g[keep], rank[keep]] = t[keep]
    top_cnt[g[keep], rank[keep]] = c[keep]
    return uctx, top_idx, top_cnt, ctx_total


def loo_normalise(idx, cnt, total, y, m):
    """Leave-one-out then renormalise to a proper distribution over top-m.

    `total` is the full successor count for the context, so removing the
    observed occurrence is exact; truncating to top-m afterwards is the only
    approximation (a count-1 true token can drop out and let another in, which
    is why callers gather m+2 and truncate here).
    """
    hit = (idx == y[:, None]) & (idx >= 0)
    cnt = cnt - hit.astype(np.int64)
    tot = np.maximum(total - 1, 0).astype(np.float64)

    valid = (idx >= 0) & (cnt > 0)
    cnt = np.where(valid, cnt, 0)

    ordr = np.argsort(-cnt, axis=1, kind="stable")[:, :m]
    idx = np.take_along_axis(idx, ordr, 1)
    cnt = np.take_along_axis(cnt, ordr, 1)
    valid = np.take_along_axis(valid, ordr, 1)
    idx = np.where(valid, idx, 0)
    cnt = np.where(valid, cnt, 0)

    covered = cnt.sum(1)
    prob = cnt.astype(np.float32) / np.maximum(covered, 1)[:, None].astype(np.float32)
    mass = (covered / np.maximum(tot, 1)).astype(np.float32)   # top-m share of the tail
    return idx.astype(np.int32), prob.astype(np.float32), tot.astype(np.float32), mass


# --------------------------------------------------------------------------
class Teacher:
    """Produces (idx, prob, count, mass) aligned with the training corpus.

    idx   (N, m) int32   candidate next tokens
    prob  (N, m) float32 probabilities, rows sum to 1 (or 0 where no evidence)
    count (N,)   float32 evidence behind the context -> drives count-adaptive lambda
    mass  (N,)   float32 share of the context's successor mass inside top-m
    """

    name = "base"
    lacks = "unspecified"          # what the causal student lacks that this supplies

    def __init__(self, ids, vocab_size, m=8, min_count=3, args=None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.V = int(vocab_size)
        self.m = int(m)
        self.min_count = int(min_count)
        self.args = args

    def _tag(self):
        parts = [self.name, str(self.V), str(len(self.ids)), str(self.m), str(self.min_count)]
        parts += [f"{k}={v}" for k, v in sorted(self.signature().items())]
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    def signature(self):
        return {}

    def _compute(self):
        raise NotImplementedError

    def build(self, cache_dir=None):
        path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"teacher_{self.name}_{self._tag()}.npz")
            if os.path.exists(path):
                z = np.load(path)
                return z["idx"], z["prob"], z["count"], z["mass"]
        idx, prob, count, mass = self._compute()
        if path:
            np.savez_compressed(path, idx=idx, prob=prob, count=count, mass=mass)
        return idx, prob, count, mass

    # ---------------------------------------------------------- diagnostics
    def report(self, idx, prob, count, mass):
        """What this teacher knows, measured. Run before spending GPU time."""
        y = self.ids
        hit = ((idx == y[:, None]) & (prob > 0))
        has = prob.sum(1) > 0
        hit_any = hit.any(1)
        p_true = np.where(hit, prob, 0).sum(1)
        ent = -(prob * np.log(prob + FLOOR)).sum(1)
        # NLL the teacher itself assigns the true token (floored, not a real LM)
        nll = -np.log(np.maximum(p_true, 1e-6))
        return {
            "teacher": self.name,
            "lacks": self.lacks,
            "coverage": float(has.mean()),
            "hit_rate": float(hit_any.mean()),
            "p_true_mean": float(p_true.mean()),
            "teacher_nll_on_hits": float(nll[hit_any].mean()) if hit_any.any() else float("nan"),
            "entropy_nats": float(ent[has].mean()) if has.any() else 0.0,
            "eff_support": float(np.exp(ent[has].mean())) if has.any() else 0.0,
            "topm_mass": float(mass[has].mean()) if has.any() else 0.0,
            "median_evidence": float(np.median(count[has])) if has.any() else 0.0,
        }
