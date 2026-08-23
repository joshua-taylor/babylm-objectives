"""
Samplers decide WHICH spans the model sees and how often.

All samplers draw the same number of spans per step and run for the same number
of steps, so the token-visit budget is identical across them. A replay sampler
is therefore making an *epoch-allocation* choice: it revisits some spans more
than others while consuming exactly as much compute. This is the control that
makes "replay helped" a meaningful claim rather than "we trained longer".

Three samplers, deliberately:

  uniform   -- the control. i.i.d. spans, i.e. standard training.
  progress  -- learning-progress replay. Prioritise spans whose loss is
               FALLING fastest (Graves et al. 2017, automated curriculum
               learning; biologically, replay prioritisation by novelty).
  hard      -- the naive alternative a reviewer will ask for: prioritise spans
               with the HIGHEST loss. If `hard` matches `progress`, the
               learning-progress framing has bought nothing.

`progress` vs `hard` is the whole scientific content of this axis. If you only
run `progress` vs `uniform` you cannot tell which of the two stories is true.
"""

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
    """Shared machinery: per-span EMA loss at two timescales."""

    def __init__(self, corpus, args):
        self.corpus = corpus
        n = corpus.n_spans
        dev = corpus.train.device
        self.fast = torch.zeros(n, device=dev)
        self.slow = torch.zeros(n, device=dev)
        self.visits = torch.zeros(n, device=dev)
        self.beta_fast = args.replay_beta_fast
        self.beta_slow = args.replay_beta_slow
        self.temp = args.replay_temp
        self.eps = args.replay_eps            # uniform mixing, keeps coverage
        self.warmup = args.replay_warmup      # steps of uniform before prioritising

    def _score(self):
        raise NotImplementedError

    def sample(self, batch_size, step):
        if step < self.warmup or (self.visits > 0).float().mean() < 0.9:
            return self.corpus.random_spans(batch_size)
        s = self._score()
        # standardise so `temp` means the same thing regardless of loss scale
        s = (s - s.mean()) / (s.std() + 1e-6)
        p = torch.softmax(s / self.temp, dim=0)
        p = (1 - self.eps) * p + self.eps / p.numel()
        return torch.multinomial(p, batch_size, replacement=True)

    def update(self, span_ids, per_span_nll, step):
        # scatter-mean in case a span appears twice in one batch
        uniq, inv = torch.unique(span_ids, return_inverse=True)
        acc = torch.zeros_like(uniq, dtype=per_span_nll.dtype)
        cnt = torch.zeros_like(acc)
        acc.index_add_(0, inv, per_span_nll)
        cnt.index_add_(0, inv, torch.ones_like(per_span_nll))
        mean_nll = acc / cnt.clamp(min=1)

        first = self.visits[uniq] == 0
        self.fast[uniq] = torch.where(
            first, mean_nll, self.beta_fast * self.fast[uniq] + (1 - self.beta_fast) * mean_nll
        )
        self.slow[uniq] = torch.where(
            first, mean_nll, self.beta_slow * self.slow[uniq] + (1 - self.beta_slow) * mean_nll
        )
        self.visits[uniq] += cnt

    def diagnostics(self):
        v = self.visits
        seen = (v > 0).float().mean().item()
        # Gini of the visit distribution: 0 = uniform coverage, 1 = collapsed
        vs, _ = torch.sort(v)
        n = vs.numel()
        idx = torch.arange(1, n + 1, device=vs.device, dtype=vs.dtype)
        gini = ((2 * idx - n - 1) * vs).sum() / (n * vs.sum().clamp(min=1e-6))
        return {
            "replay_coverage": seen,
            "replay_visit_gini": gini.item(),
            "replay_visit_max": v.max().item(),
            "replay_progress_mean": (self.slow - self.fast)[v > 0].mean().item()
            if (v > 0).any()
            else 0.0,
        }


class ProgressSampler(_PrioritySampler):
    name = "progress"

    def _score(self):
        # positive when the fast EMA has dropped below the slow EMA => improving
        return self.slow - self.fast


class HardSampler(_PrioritySampler):
    name = "hard"

    def _score(self):
        return self.fast


SAMPLERS = {
    "uniform": UniformSampler,
    "progress": ProgressSampler,
    "hard": HardSampler,
}


def build_sampler(name, corpus, args):
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; choose from {list(SAMPLERS)}")
    return SAMPLERS[name](corpus, args)
