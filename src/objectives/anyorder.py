"""
Any-order / absorbing-state masked diffusion (MDLM-style).

This is the one arm that genuinely replaces next-token prediction rather than
reweighting it. The trunk runs BIDIRECTIONALLY; a random fraction t of positions
are replaced with <mask> and the model predicts them from both sides.

Objective (MDLM, linear schedule alpha_t = 1 - t):

    NELBO = E_{t ~ U(0,1)} [ (1/t) * sum_{i masked} -log p(x_i | x_t) ] / T

which is a valid upper bound on the negative log-likelihood in nats/token, and
is therefore comparable to the AR arms' loss -- as a BOUND, always looser than
the AR number. Reporting it as if it were an equal-footing perplexity would be
dishonest, so `eval_nelbo` returns it labelled as a bound and the registry
records `loss_unit='nats_bound'`.

The confound to state up front
------------------------------
At a matched token-VISIT budget, this arm receives supervision on only ~50% of
positions on average (E[t] = 0.5), where the AR arms get 100%. So a loss at
equal steps understates it. `--match-supervision` doubles this arm's steps so
that supervised-position count matches instead. Run BOTH accountings; they
answer different questions and the difference between them is itself a result.
"""

import torch
import torch.nn.functional as F

from .token_losses import Loss


class AnyOrderLoss(Loss):
    name = "anyorder"
    requires_causal = False
    supports_ar_eval = False

    def __init__(self, model, corpus, args):
        super().__init__(model, corpus, args)
        self.mask_id = corpus.mask_id
        self.eps = args.diffusion_eps
        self._last_t = 0.5

    def _masked_loss(self, x, t):
        """x: (B,T) clean tokens. t: (B,1) mask probability per sequence."""
        B, T = x.shape
        drop = torch.rand(B, T, device=x.device) < t
        # guarantee at least one masked position per sequence
        force = torch.randint(0, T, (B, 1), device=x.device)
        drop.scatter_(1, force, True)

        xin = torch.where(drop, torch.full_like(x, self.mask_id), x)
        logits = self.model(xin)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction="none"
        ).view(B, T)
        per_seq = (nll * drop.float()).sum(1) / t.squeeze(1).clamp(min=self.eps) / T
        return per_seq, drop.float().mean().item()

    def __call__(self, x, y, span_starts, step):
        # bidirectional: reconstruct the input itself, not a shifted target
        B = x.size(0)
        t = torch.rand(B, 1, device=x.device).clamp(min=self.eps, max=1.0)
        per_seq, frac = self._masked_loss(x, t)
        self._last_t = float(t.mean().item())
        loss = per_seq.mean()
        return {
            "loss": loss,
            "nll": loss.detach(),           # already nats/token (a bound)
            "per_span_nll": per_seq.detach(),
            "frac_supervised": frac,
        }

    @torch.no_grad()
    def eval_nelbo(self, corpus, n_batches=20, batch_size=16, mc=8, seed=99):
        """Monte-Carlo NELBO bound in nats/token over the val set."""
        self.model.eval()
        g = torch.Generator(device=corpus.val.device).manual_seed(seed)
        tot, n = 0.0, 0
        for _ in range(n_batches):
            x, _ = corpus.val_batch(batch_size, generator=g)
            for _ in range(mc):
                t = torch.rand(x.size(0), 1, device=x.device, generator=g).clamp(
                    min=self.eps, max=1.0
                )
                per_seq, _ = self._masked_loss(x, t)
                tot += float(per_seq.mean().item())
                n += 1
        self.model.train()
        return tot / max(n, 1)

    def diagnostics(self):
        return {"diffusion_mean_t": self._last_t}
