"""
Distributional abstraction teachers.

`class` is the rung that answers the ten-times-table question directly.

Every other teacher on the ladder is an exact-match device: if the pair
(context, token) never co-occurs in the corpus, they assign it nothing. That
means they can pool information ACROSS positions but never propose a
combination the corpus does not contain. A model that has seen `jump`,
`jumped`, `walking` and `running` but never `jumping` gets no help from them.

This teacher clusters tokens by distributional similarity, builds the n-gram at
the level of CLASSES, and expands back to tokens:

    p(w | context) = p(class(w) | class(context)) * p(w | class(w))

A never-observed (context, token) pair receives mass whenever their classes
co-occur. That is a compositional prior, learned entirely from corpus
statistics, and it is the mechanism by which a training signal can point at
something absent from the training set.

`embed` is the soft version applied to the context side rather than the
successor side: pool successors over SIMILAR contexts instead of identical
ones. It targets the measured failure of the exact-match teachers -- on this
corpus the trigram anchor has no attestation for the true token about 40% of
the time, and on those positions it moves mass away from the truth.

Both share a fixed PPMI+SVD embedding computed from the training corpus. It is
external to the model, so no circularity: the teacher never consults the
student's beliefs.
"""

import numpy as np

from .base import Teacher, grouped_topm


def ppmi_svd(ids, V, dim=64, window=2, seed=0):
    """Co-occurrence -> PPMI -> truncated SVD. Fixed, external, cheap."""
    C = np.zeros((V, V), dtype=np.float64)
    for off in range(1, window + 1):
        np.add.at(C, (ids[:-off], ids[off:]), 1.0)
        np.add.at(C, (ids[off:], ids[:-off]), 1.0)
    tot = C.sum()
    if tot <= 0:
        return np.zeros((V, dim), dtype=np.float32)
    rows = C.sum(1, keepdims=True)
    cols = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * tot) / np.maximum(rows * cols, 1e-12))
    pmi = np.nan_to_num(pmi, neginf=0.0, posinf=0.0)
    np.maximum(pmi, 0.0, out=pmi)
    U, S, _ = np.linalg.svd(pmi, full_matrices=False)
    d = min(dim, U.shape[1])
    return (U[:, :d] * np.sqrt(S[:d])).astype(np.float32)


def kmeans(X, K, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    K = min(K, len(Xn))
    cent = Xn[rng.choice(len(Xn), K, replace=False)].copy()
    lab = np.zeros(len(Xn), dtype=np.int64)
    for _ in range(iters):
        lab = np.argmax(Xn @ cent.T, axis=1)
        for k in range(K):
            sel = lab == k
            if sel.any():
                c = Xn[sel].mean(0)
                cent[k] = c / (np.linalg.norm(c) + 1e-8)
            else:
                cent[k] = Xn[rng.integers(len(Xn))]
    return lab, cent


class ClassTeacher(Teacher):
    name = "class"
    lacks = "distributional abstraction: mass for combinations never observed"

    def __init__(self, *a, n_classes=128, emb_dim=64, top_classes=4, **kw):
        super().__init__(*a, **kw)
        self.K = n_classes
        self.dim = emb_dim
        self.top_c = top_classes

    def signature(self):
        return {"K": self.K, "dim": self.dim, "tc": self.top_c}

    def _compute(self):
        ids, m, N, V = self.ids, self.m, len(self.ids), self.V
        # Guard: as K approaches V, k-means gives every token its own class and
        # the teacher silently degenerates into the token-level trigram --
        # losing the entire point of the rung. Verified empirically: at K == V
        # the probability assigned to a held-out composition drops to exactly 0.
        if self.K > V // 2:
            self.K = max(4, V // 2)
        E = ppmi_svd(ids, V, self.dim)
        cls, _ = kmeans(E, self.K, seed=0)
        K = int(cls.max()) + 1
        self.cls = cls
        sizes = np.bincount(cls, minlength=K)
        self.singleton_frac = float((sizes == 1).mean())
        self.mean_class_size = float(sizes[sizes > 0].mean())

        # p(w | class): unigram share within the class
        uni = np.bincount(ids, minlength=V).astype(np.float64)
        cls_tot = np.zeros(K)
        np.add.at(cls_tot, cls, uni)
        p_w_c = uni / np.maximum(cls_tot[cls], 1e-9)

        # top tokens per class
        order = np.lexsort((-p_w_c, cls))
        tc_idx = np.full((K, m), -1, dtype=np.int64)
        tc_p = np.zeros((K, m), dtype=np.float64)
        g = cls[order]
        first = np.empty(len(g), bool); first[0] = True; first[1:] = g[1:] != g[:-1]
        gs = np.zeros(len(g), dtype=np.int64); fi = np.flatnonzero(first); gs[fi] = fi
        np.maximum.accumulate(gs, out=gs)
        rank = np.arange(len(g)) - gs
        keep = rank < m
        tc_idx[g[keep], rank[keep]] = order[keep]
        tc_p[g[keep], rank[keep]] = p_w_c[order[keep]]

        # class-level trigram: (class(j-2), class(j-1)) -> class(j)
        cid = cls[ids]
        ctx = np.full(N, -1, dtype=np.int64)
        ctx[2:] = cid[:-2] * K + cid[1:-1]
        uctx, ci, cc, ct = grouped_topm(ctx[2:], cid[2:], self.top_c)
        rowof = np.clip(np.searchsorted(uctx, ctx), 0, len(uctx) - 1)

        # expand each class-context into token candidates
        c_idx = ci[rowof]                                        # (N, top_c)
        c_cnt = cc[rowof].astype(np.float64)
        c_tot = ct[rowof].astype(np.float64)

        # leave-one-out at class level
        hit = c_idx == cid[:, None]
        c_cnt = np.maximum(c_cnt - hit, 0.0)
        c_tot = np.maximum(c_tot - 1.0, 0.0)
        c_p = c_cnt / np.maximum(c_cnt.sum(1, keepdims=True), 1e-9)

        safe = np.clip(c_idx, 0, K - 1)
        cand = tc_idx[safe]                                      # (N, top_c, m)
        cand_p = tc_p[safe] * c_p[:, :, None]
        cand = cand.reshape(N, -1)
        cand_p = cand_p.reshape(N, -1)
        cand_p = np.where(cand >= 0, cand_p, 0.0)

        k2 = min(m, cand.shape[1])
        sel = np.argpartition(-cand_p, k2 - 1, axis=1)[:, :k2]
        pv = np.take_along_axis(cand_p, sel, 1)
        iv = np.take_along_axis(cand, sel, 1)
        o = np.argsort(-pv, axis=1)
        pv, iv = np.take_along_axis(pv, o, 1), np.take_along_axis(iv, o, 1)

        prob = (pv / np.maximum(pv.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        idx = np.maximum(iv, 0).astype(np.int32)
        dead = (np.arange(N) < 2) | (c_tot < self.min_count) | (pv.sum(1) <= 0)
        prob[dead] = 0.0
        return idx, prob, c_tot.astype(np.float32), np.where(dead, 0, 1.0).astype(np.float32)

    def report(self, *a):
        r = super().report(*a)
        r["class_singleton_frac"] = round(getattr(self, "singleton_frac", float("nan")), 3)
        r["mean_class_size"] = round(getattr(self, "mean_class_size", float("nan")), 1)
        if r["class_singleton_frac"] > 0.5:
            r["WARNING"] = "most classes are singletons: no abstraction, lower --teacher-n-classes"
        return r


class EmbedTeacher(Teacher):
    name = "embed"
    lacks = "soft context generalisation: pool successors over SIMILAR contexts"

    def __init__(self, *a, emb_dim=64, n_neighbours=8, self_weight=1.0, **kw):
        super().__init__(*a, **kw)
        self.dim = emb_dim
        self.k = n_neighbours
        self.self_weight = self_weight

    def signature(self):
        return {"dim": self.dim, "k": self.k, "sw": self.self_weight}

    def _compute(self):
        ids, m, N, V = self.ids, self.m, len(self.ids), self.V
        E = ppmi_svd(ids, V, self.dim)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        sim = En @ En.T
        np.fill_diagonal(sim, -np.inf)
        k = min(self.k, V - 1)
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, nn, 1)
        w = np.maximum(w, 0.0)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)

        # dense bigram counts, then smooth across neighbouring contexts
        B = np.zeros((V, V), dtype=np.float64)
        np.add.at(B, (ids[:-1], ids[1:]), 1.0)
        Bn = B / np.maximum(B.sum(1, keepdims=True), 1e-9)
        S = self.self_weight * Bn + (w[:, :, None] * Bn[nn]).sum(1)

        M = min(m + 2, V)
        sel = np.argpartition(-S, M - 1, axis=1)[:, :M]
        sv = np.take_along_axis(S, sel, 1)

        prev = np.zeros(N, dtype=np.int64)
        prev[1:] = ids[:-1]
        idx = sel[prev]
        val = sv[prev].copy()

        # approximate leave-one-out: remove this occurrence's own contribution
        rowtot = B.sum(1)[prev]
        selfw = self.self_weight / np.maximum(rowtot, 1.0)
        hit = idx == ids[:, None]
        val = np.maximum(val - hit * selfw[:, None], 0.0)

        o = np.argsort(-val, axis=1)[:, :m]
        idx = np.take_along_axis(idx, o, 1).astype(np.int32)
        val = np.take_along_axis(val, o, 1)
        prob = (val / np.maximum(val.sum(1, keepdims=True), 1e-12)).astype(np.float32)

        cnt = np.maximum(rowtot - 1.0, 0.0).astype(np.float32)
        dead = (np.arange(N) < 1) | (cnt < self.min_count) | (val.sum(1) <= 0)
        prob[dead] = 0.0
        return idx, prob, cnt, np.where(dead, 0, 1.0).astype(np.float32)
