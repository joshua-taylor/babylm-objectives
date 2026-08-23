"""
Dreaming, anchored in external judges.

WHY THE EARLIER VERSION WAS DEAD, AND WHAT THE ANCHOR FIXES
-----------------------------------------------------------
Sampling a rollout and training on the model's own score of it optimises
E_{y~p}[log p(y)], which is exactly negative entropy: a purely self-referential
objective is an entropy knob, not a learning signal. Worse, a causal model
rescoring its own sample recomputes the identical conditional it used to
generate, so there is no second opinion. And models grow MORE confident inside
repetition loops (Holtzman et al. 2020), so self-scoring fails precisely on the
failure mode it was meant to catch.

An n-gram model fit on the real corpus breaks the circularity: it is external to
the weights. But it is also WEAKER than the network, so using it as a positive
target would teach the model to be more trigram-like, i.e. worse. The asymmetry
that makes it usable:

    a corpus n-gram model is an unreliable judge of what is GOOD
    and a reliable judge of what is DEFINITELY WRONG.

So the judges are used as NEGATIVE DETECTORS ONLY. Positives are always real
data. Nothing the model generates is ever used as a target.

Judges (all cheap, all external to the weights):
  ngram  -- the generated bigram occurs nowhere in the corpus
  rep    -- the token continues an n-gram already emitted in this sample

Flagged positions get an unlikelihood term (Welleck et al. 2019),
-log(1 - p(w_t | context)), added to the standard next-token loss on real data.

Honest expectation: roughly neutral on perplexity, plausible win on degeneration.
Perplexity is structurally blind to "drifts into repetition after forty tokens",
so `rep4_greedy` and `distinct4_greedy` are the metrics that can see this. If you
judge this arm on val loss alone you will conclude nothing happened.

Controls, both essential:
  dream_rep_only -- the hand-coded repetition penalty a reviewer will ask for.
                    If it matches, the n-gram judge bought nothing.
  dream_random   -- flag the same NUMBER of positions at random. Isolates
                    "unlikelihood on anything" from "unlikelihood on judged-bad".
"""

import torch
import torch.nn.functional as F

from .token_losses import Loss, _flat_nll


class DreamLoss(Loss):
    name = "dream"

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        from ..ngram_table import NgramTable

        self.every = args.dream_every
        self.n_seq = args.dream_batch
        self.gen_len = args.dream_len
        self.prompt_len = args.dream_prompt
        self.weight = args.dream_weight
        self.temp = args.dream_temp
        self.judges = set(args.dream_judges.split(","))
        self.rep_n = args.dream_rep_n
        self.random_rate = args.dream_random_rate
        self.tbl = None
        if "ngram" in self.judges:
            self.tbl = NgramTable(corpus.train_np, corpus.vocab_size,
                                  top_m=4, min_count=args.soft_min_count)
        self._stats = {"dream_flag_rate": 0.0, "dream_ul": 0.0,
                       "dream_flag_ngram": 0.0, "dream_flag_rep": 0.0}

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def _rollout(self, x):
        """Sample continuations from real prefixes. Model in eval mode (no dropout)."""
        was_training = self.model.training
        self.model.eval()
        seq = x[: self.n_seq, : self.prompt_len].clone()
        for _ in range(self.gen_len):
            logits = self.model(seq[:, -self.model.cfg.seq_len:])[:, -1, :].float()
            p = F.softmax(logits / max(self.temp, 1e-6), dim=-1)
            seq = torch.cat([seq, torch.multinomial(p, 1)], dim=1)
        if was_training:
            self.model.train()
        return seq

    # ------------------------------------------------------------ judges
    def _flags(self, seq):
        """Boolean mask over generated positions. True == judged DEFINITELY bad."""
        B, L = seq.shape
        s = seq.tolist()
        flags = torch.zeros(B, L, dtype=torch.bool, device=seq.device)
        n_ng = n_rep = 0

        for b in range(B):
            row = s[b]
            seen = set()
            for i in range(len(row)):
                if i < self.prompt_len:
                    continue
                bad = False
                if "ngram" in self.judges and self.tbl is not None:
                    if self.tbl.unseen_bigram(row[i - 1], row[i]):
                        bad = True
                        n_ng += 1
                if "rep" in self.judges and i >= self.rep_n - 1:
                    g = tuple(row[i - self.rep_n + 1 : i + 1])
                    if g in seen:
                        bad = True
                        n_rep += 1
                    seen.add(g)
                if bad:
                    flags[b, i] = True

        if "random" in self.judges:
            r = torch.rand(B, L, device=seq.device) < self.random_rate
            r[:, : self.prompt_len] = False
            flags = r
        return flags, n_ng, n_rep

    # ------------------------------------------------------------ loss
    def __call__(self, x, y, span_starts, step):
        logits = self.model(x)
        nll = _flat_nll(logits, y)
        loss = nll.mean()

        if self.weight > 0 and step % self.every == 0:
            seq = self._rollout(x)
            flags, n_ng, n_rep = self._flags(seq)
            tgt_flags = flags[:, 1:]
            if tgt_flags.any():
                inp = seq[:, :-1][:, -self.model.cfg.seq_len:]
                tgt = seq[:, 1:][:, -self.model.cfg.seq_len:]
                fl = tgt_flags[:, -self.model.cfg.seq_len:]
                lg = self.model(inp)
                p = F.softmax(lg.float(), dim=-1)
                pt = torch.gather(p, -1, tgt.unsqueeze(-1)).squeeze(-1)
                # unlikelihood: push down p on positions the external judges reject
                ul = -torch.log((1.0 - pt).clamp(min=1e-6))
                ul = (ul * fl.float()).sum() / fl.float().sum().clamp(min=1.0)
                loss = loss + self.weight * ul
                denom = max(flags.numel() - flags.shape[0] * self.prompt_len, 1)
                self._stats = {
                    "dream_flag_rate": float(flags.float().sum().item()) / denom,
                    "dream_ul": float(ul.item()),
                    "dream_flag_ngram": n_ng / denom,
                    "dream_flag_rep": n_rep / denom,
                }

        return {
            "loss": loss,
            "nll": nll.mean().detach(),
            "per_span_nll": nll.mean(dim=1).detach(),
            "frac_supervised": 1.0,
        }

    def diagnostics(self):
        return dict(self._stats)
