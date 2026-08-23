"""
Novel-combination evaluation.

Aggregate perplexity averages away exactly the capability in question. If you
learn your ten times table you know 10x10=100 without having seen it; the
question is whether a language model does the analogous thing, and a single
averaged number cannot answer it.

So slice the validation set by whether the (context, target) combination was
ever observed in training. Both components can be individually frequent while
the combination is absent -- that IS the ten-times-table case, and it needs no
corpus surgery because the split already contains thousands of instances.

    seen_bigram    (prev, target) occurs in train
    novel_bigram   (prev, target) never occurs, but BOTH tokens are frequent
    novel_trigram  (prev2, prev, target) never occurs

`novel_bigram_nll` is the compositional generalisation metric. Restricting to
frequent components is what separates "composition" from "rare word": a novel
combination of two common tokens is a slot-filling failure, whereas a novel
combination involving a token seen twice is just sparsity.

The prediction the teacher ladder makes is sharp and falsifiable: teachers that
pool over ABSTRACTIONS (class, embed) should help disproportionately here,
because pooling across contexts is precisely how a model discovers that an
affix is a slot rather than a string. Exact-match teachers (trigram, varorder)
should help mainly on the `seen` slice. If the advantage is flat across slices,
the mechanism is regularisation and the compositional story is wrong.
"""

import numpy as np
import torch
import torch.nn.functional as F


class NoveltySlices:
    def __init__(self, corpus, min_token_count=20):
        tr = corpus.train_np
        V = corpus.vocab_size
        self.V = V
        uni = np.bincount(tr, minlength=V)
        self.frequent = uni >= min_token_count

        # observed bigrams / trigrams, as hashed sets
        self.bi = set((int(a) << 20) | int(b) for a, b in zip(tr[:-1], tr[1:]))
        self.tri = set((int(a) << 40) | (int(b) << 20) | int(c)
                       for a, b, c in zip(tr[:-2], tr[1:-1], tr[2:]))

    def masks(self, x, y):
        """Boolean masks over (B,T) positions for each slice."""
        xb = x.detach().cpu().numpy()
        yb = y.detach().cpu().numpy()
        B, T = yb.shape
        seen_b = np.zeros((B, T), bool)
        seen_t = np.zeros((B, T), bool)
        freq = self.frequent[yb] & self.frequent[xb]
        for i in range(B):
            for j in range(T):
                a, c = int(xb[i, j]), int(yb[i, j])
                seen_b[i, j] = ((a << 20) | c) in self.bi
                if j >= 1:
                    p = int(xb[i, j - 1])
                    seen_t[i, j] = ((p << 40) | (a << 20) | c) in self.tri
        return {
            "seen_bigram": seen_b,
            "novel_bigram": (~seen_b) & freq,
            "novel_trigram": (~seen_t) & seen_b,     # context novel, pair attested
        }


@torch.no_grad()
def novelty_report(model, corpus, slices, batch_size=16, n_batches=12,
                   amp_dtype=None, device="cpu"):
    model.eval()
    tot = {k: 0.0 for k in ["seen_bigram", "novel_bigram", "novel_trigram"]}
    cnt = {k: 0 for k in tot}
    seen_batches = 0
    for x, y in corpus.val_stream(batch_size):
        if seen_batches >= n_batches:
            break
        seen_batches += 1
        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                              y.reshape(-1), reduction="none").view(y.shape)
        nll = nll.detach().cpu().numpy()
        for k, mk in slices.masks(x, y).items():
            if mk.any():
                tot[k] += float(nll[mk].sum())
                cnt[k] += int(mk.sum())
    model.train()
    out = {}
    for k in tot:
        # an EMPTY slice must not report 0.0 -- that reads as a perfect score
        out[f"{k}_nll"] = round(tot[k] / cnt[k], 4) if cnt[k] >= 50 else float("nan")
        out[f"{k}_n"] = cnt[k]
    return out


# --------------------------------------------------------------------------
DEFAULT_FIT_TARGETS = (3.20, 3.10, 3.00, 2.95, 2.92, 2.90)


def val_at_matched_train(history, targets=DEFAULT_FIT_TARGETS):
    """Val loss interpolated at fixed TRAIN loss levels.

    Comparing arms at their own minima confounds "learns better" with
    "regularises": each arm's minimum sits at a different point on the
    fit/generalisation frontier. Reading val at MATCHED train loss removes both
    that confound and the where-did-the-minimum-land variance that inflates the
    noise floor. It needs no extra runs -- the training trace already has it.
    """
    pts = [(h["train"], h["val"]) for h in history
           if h.get("train") == h.get("train") and h["train"] is not None]
    pts = [p for p in pts if not (np.isnan(p[0]) or np.isnan(p[1]))]
    if len(pts) < 2:
        return {}
    pts.sort(key=lambda p: -p[0])            # train loss decreases over training
    tr = np.array([p[0] for p in pts])
    vl = np.array([p[1] for p in pts])
    out = {}
    for t in targets:
        if t > tr[0] or t < tr[-1]:
            continue                          # arm never reached this fit level
        out[f"val@train{t:.2f}"] = round(float(np.interp(-t, -tr, vl)), 4)
    return out
