"""
Samplers decide WHICH spans the model sees and how often.

All samplers draw the same number of spans per step and run for the same number
of steps, so the token-visit budget is identical. A replay sampler makes an
EPOCH-ALLOCATION choice: it revisits some spans more than others at exactly the
same compute. That is what makes "replay helped" a claim rather than "we
trained longer".

ANTI-COLLAPSE (added after the first run)
-----------------------------------------
The first version had a runaway feedback loop: a span whose loss was falling
got sampled more, which made its loss fall further, which got it sampled more.
Within ~500 steps `replay_progress` had collapsed onto a handful of spans,
driven batch loss to 0.14 nats (ppl 1.15) while the held-out train-span eval
read ppl 272, and degraded trunk effective rank from 218 to 145. It then looked
like a WIN on the summary table. Prioritised replay needs anti-collapse
machinery for exactly the reason latent targets do.

Three guards, all on by default:
  * VISIT CAP     -- a span above `max_visit_ratio` x the uniform rate is
                     excluded until the rest catch up. Hard bound on Gini.
  * NOVELTY BONUS -- UCB-style sqrt(log t / (visits+1)) term, so unvisited and
                     stale spans stay in contention.
  * STALENESS     -- progress estimates decay toward zero when a span has not
                     been visited recently, so a stale "improving" reading
                     cannot keep winning forever.

`replay_visit_gini` is logged every run and a value above --replay-gini-warn
writes a confound into the register automatically.
"""

import math

import torch


class UniformSampler:
    name = "uniform"

    def __init__(self, corpus, args):
        self.corpus = corpus

    def sample(self, batch_size, step):
        return self.corpus.random_spans(batch_size)

    def update(self, span_ids, per_span_nll, step):
        pass

    def diagnostics(self):
        return {}


class _PrioritySampler:
    """Per-span EMA loss at two timescales, plus the three guards."""

    def __init__(self, corpus, args):
        self.corpus = corpus
        n = corpus.n_spans
        dev = corpus.train.device
        self.n = n
        self.fast = torch.zeros(n, device=dev)
        self.slow = torch.zeros(n, device=dev)
        self.visits = torch.zeros(n, device=dev)
        self.last_seen = torch.zeros(n, device=dev)
        self.beta_fast = args.replay_beta_fast
        self.beta_slow = args.replay_beta_slow
        self.temp = args.replay_temp
        self.eps = args.replay_eps
        self.warmup = args.replay_warmup
        self.novelty = args.replay_novelty
        self.max_ratio = args.replay_max_visit_ratio
        self.stale_halflife = args.replay_stale_halflife
        self.batch_size = args.batch_size

    def _raw_score(self):
        raise NotImplementedError

    def _staleness_decay(self, step):
        age = (step - self.last_seen).clamp(min=0)
        return torch.exp(-age * math.log(2.0) / max(self.stale_halflife, 1.0))

    def sample(self, batch_size, step):
        if step < self.warmup or (self.visits > 0).float().mean() < 0.9:
            return self.corpus.random_spans(batch_size)

        s = self._raw_score() * self._staleness_decay(step)
        s = (s - s.mean()) / (s.std() + 1e-6)

        # UCB-style novelty: keeps unvisited and stale spans in contention
        if self.novelty > 0:
            s = s + self.novelty * torch.sqrt(
                math.log(step + 1) / (self.visits + 1.0)
            )

        p = torch.softmax(s / self.temp, dim=0)

        # hard visit cap -- the guard that actually bounds collapse
        if self.max_ratio > 0:
            expected = step * self.batch_size / self.n
            over = self.visits > self.max_ratio * max(expected, 1.0)
            if over.any() and not over.all():
                p = p.masked_fill(over, 0.0)

        p = p / p.sum().clamp(min=1e-12)
        p = (1 - self.eps) * p + self.eps / self.n
        return torch.multinomial(p, batch_size, replacement=True)

    def update(self, span_ids, per_span_nll, step):
        uniq, inv = torch.unique(span_ids, return_inverse=True)
        acc = torch.zeros_like(uniq, dtype=per_span_nll.dtype)
        cnt = torch.zeros_like(acc)
        acc.index_add_(0, inv, per_span_nll)
        cnt.index_add_(0, inv, torch.ones_like(per_span_nll))
        mean_nll = acc / cnt.clamp(min=1)

        first = self.visits[uniq] == 0
        self.fast[uniq] = torch.where(
            first, mean_nll, self.beta_fast * self.fast[uniq] + (1 - self.beta_fast) * mean_nll)
        self.slow[uniq] = torch.where(
            first, mean_nll, self.beta_slow * self.slow[uniq] + (1 - self.beta_slow) * mean_nll)
        self.visits[uniq] += cnt
        self.last_seen[uniq] = float(step)

    def diagnostics(self):
        v = self.visits
        vs, _ = torch.sort(v)
        n = vs.numel()
        idx = torch.arange(1, n + 1, device=vs.device, dtype=vs.dtype)
        gini = ((2 * idx - n - 1) * vs).sum() / (n * vs.sum().clamp(min=1e-6))
        seen = v > 0
        return {
            "replay_coverage": (seen.float().mean().item()),
            "replay_visit_gini": gini.item(),
            "replay_visit_max": v.max().item(),
            "replay_visit_mean": v.mean().item(),
            "replay_progress_mean": (self.slow - self.fast)[seen].mean().item()
            if seen.any() else 0.0,
        }


class ProgressSampler(_PrioritySampler):
    name = "progress"

    def _raw_score(self):
        return self.slow - self.fast      # positive => loss falling => improving


class HardSampler(_PrioritySampler):
    name = "hard"

    def _raw_score(self):
        return self.fast                  # positive => high loss => difficult


SAMPLERS = {"uniform": UniformSampler, "progress": ProgressSampler, "hard": HardSampler}


def build_sampler(name, corpus, args):
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; choose from {list(SAMPLERS)}")
    return SAMPLERS[name](corpus, args)
