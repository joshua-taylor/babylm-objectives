"""
Diagnostics that decide arms, not just decorate them.

Two families:

1. COLLAPSE DETECTORS (for any arm with self-generated targets).
   If prediction targets are produced by the network itself, there is always a
   trivial solution: all representations constant, all error zero, loss
   minimised, nothing learned. Effective rank is the kill criterion. It is
   checked *during* training, and a falling rank ends the run regardless of
   what the loss curve is doing.

2. DEGENERATION METRICS (for any arm).
   Perplexity is structurally blind to "this continuation is fine but drifts
   into repetition after forty tokens". If an arm's benefit is fewer
   repetitions, perplexity will not show it. These will.
"""

import math

import torch


# ------------------------------------------------------------------ collapse
@torch.no_grad()
def effective_rank(z: torch.Tensor, eps: float = 1e-9) -> float:
    """Entropy-based effective rank of a (N, d) matrix of representations.

    erank = exp(H(p)) where p_i = s_i / sum(s). Ranges from 1 (total collapse,
    all mass on one direction) to d (isotropic).
    """
    z = z.reshape(-1, z.shape[-1]).float()
    z = z - z.mean(0, keepdim=True)
    if z.shape[0] < 2:
        return float("nan")
    try:
        s = torch.linalg.svdvals(z)
    except Exception:
        return float("nan")
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    h = -(p * (p + eps).log()).sum()
    return float(torch.exp(h).item())


@torch.no_grad()
def repr_stats(z: torch.Tensor) -> dict:
    z2 = z.reshape(-1, z.shape[-1]).float()
    return {
        "erank": effective_rank(z2),
        "erank_frac": effective_rank(z2) / z2.shape[-1],
        "feat_std": float(z2.std(0).mean().item()),
        "feat_std_min": float(z2.std(0).min().item()),
    }


# ------------------------------------------------------------------ VICReg
def variance_penalty(z, target_std=1.0, eps=1e-4):
    std = torch.sqrt(z.float().var(dim=0) + eps)
    return torch.relu(target_std - std).mean()


def covariance_penalty(z):
    z = z.float()
    n, d = z.shape
    if n < 2:
        return z.sum() * 0.0
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return (off.pow(2).sum() / d)


# ------------------------------------------------------------------ degeneration
def _ngrams(seq, n):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(sequences, n):
    """Fraction of generated n-grams that are unique, pooled over samples."""
    all_g, uniq = 0, set()
    for s in sequences:
        g = _ngrams(s, n)
        all_g += len(g)
        uniq.update(g)
    return len(uniq) / max(all_g, 1)


def rep_rate(sequences, n=4):
    """Fraction of n-grams in a sample that already appeared earlier in THAT sample.

    This is the degeneration signal cross-entropy cannot see.
    """
    rates = []
    for s in sequences:
        seen, rep, tot = set(), 0, 0
        for g in _ngrams(s, n):
            tot += 1
            if g in seen:
                rep += 1
            seen.add(g)
        rates.append(rep / max(tot, 1))
    return sum(rates) / max(len(rates), 1)


@torch.no_grad()
def generate(model, prompt_ids, max_new, seq_len, temperature=1.0, top_k=None, device="cpu"):
    model.eval()
    x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    for _ in range(max_new):
        xi = x[:, -seq_len:]
        logits = model(xi)[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
    model.train()
    return x[0].tolist()


@torch.no_grad()
def degeneration_report(model, corpus, n_samples=16, max_new=128, temperature=1.0,
                        top_k=None, device="cpu", seed=1234):
    """Sample continuations from val prompts and measure degeneration."""
    g = torch.Generator(device=corpus.val.device).manual_seed(seed)
    prompts, _ = corpus.val_batch(n_samples, generator=g)
    prompts = prompts[:, :32].tolist()
    outs = []
    for p in prompts:
        full = generate(model, p, max_new, corpus.cfg.seq_len, temperature, top_k, device)
        outs.append(full[len(p):])
    return {
        "rep4": rep_rate(outs, 4),
        "rep8": rep_rate(outs, 8),
        "distinct1": distinct_n(outs, 1),
        "distinct4": distinct_n(outs, 4),
    }


# ------------------------------------------------------------------ self-endorsement
@torch.no_grad()
def self_endorsement(model, corpus, n_samples=16, max_new=128, n=4,
                     temperature=1.0, device="cpu", seed=99):
    """Does the model assign HIGHER log-prob to tokens that continue a repeated
    n-gram than to non-repeated tokens?

    Holtzman et al. (2020): in real degeneration loops the model's confidence
    RISES with each repetition. If that holds here, self-scoring cannot detect
    the failure mode -- which is the one-hour diagnostic that kills the naive
    "let the model grade its own rollouts" idea before any training run.

    Returns mean logprob on repeated vs non-repeated positions. If
    `delta = repeated - nonrepeated` is POSITIVE, self-scoring is dead.
    """
    g = torch.Generator(device=corpus.val.device).manual_seed(seed)
    prompts, _ = corpus.val_batch(n_samples, generator=g)
    prompts = prompts[:, :32].tolist()

    rep_lp, non_lp = [], []
    model.eval()
    for p in prompts:
        full = generate(model, p, max_new, corpus.cfg.seq_len, temperature, None, device)
        x = torch.tensor([full], device=device, dtype=torch.long)[:, -corpus.cfg.seq_len:]
        logits = model(x[:, :-1])
        lp = torch.log_softmax(logits.float(), -1)
        tgt = x[0, 1:]
        tok_lp = lp[0, torch.arange(tgt.numel(), device=device), tgt]

        seq = x[0].tolist()
        seen = set()
        flags = [False] * (len(seq) - 1)
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i : i + n])
            if gram in seen and (i + n - 2) < len(flags):
                flags[i + n - 2] = True
            seen.add(gram)
        f = torch.tensor(flags, device=device)
        if f.any():
            rep_lp.append(tok_lp[f].mean().item())
        if (~f).any():
            non_lp.append(tok_lp[~f].mean().item())
    model.train()

    r = sum(rep_lp) / max(len(rep_lp), 1) if rep_lp else float("nan")
    nr = sum(non_lp) / max(len(non_lp), 1) if non_lp else float("nan")
    return {
        "endorse_repeated_lp": r,
        "endorse_nonrepeated_lp": nr,
        "endorse_delta": r - nr if not (math.isnan(r) or math.isnan(nr)) else float("nan"),
    }
