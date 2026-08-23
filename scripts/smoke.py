"""
CPU smoke test on synthetic data. No download, no GPU, a few seconds.

Standing rule in this project: never spend GPU time on a script that has not
passed a tiny CPU test for shapes, NaNs, gradients and causality.

  python -m scripts.smoke
"""

import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import build_parser  # noqa: E402
from src.arms import ARMS, apply_arm  # noqa: E402
from src.data import Corpus, DataCfg  # noqa: E402
from src.model import LM, ModelCfg, verify_causality  # noqa: E402
from src.objectives import build_loss, build_sampler  # noqa: E402


def synthetic_corpus(vocab=64, n=20_000, seq_len=32, device="cpu"):
    cfg = DataCfg(n_tokens=n, val_tokens=n // 5, vocab_size=vocab,
                  seq_len=seq_len, cache_dir="./cache_smoke")
    rng = np.random.default_rng(0)
    # mild markov structure so losses actually move
    tr = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        tr[i] = (tr[i - 1] * 7 + rng.integers(0, 5)) % vocab
    va = tr[: n // 5].copy()
    return Corpus(cfg, tr, va, None, {"vocab_size": vocab}, device)


def smoke(args=None):
    p = build_parser()
    args = p.parse_args([])
    from run import resolve_size
    args = resolve_size(args)
    args.seq_len = 32
    args.d_model = 32
    args.n_layers = 2
    args.n_heads = 2
    args.d_ff = 64
    args.batch_size = 4
    args.latent_horizon = 4
    args.diag_every = 1
    args.cache_dir = "./cache_smoke"
    args.ngram_anchor = 0
    args.soft_top_m = 4
    args.teacher_n_classes = 8
    args.teacher_max_order = 4
    args.cache_half_life = 32.0
    args.ema_decay = 0.99
    args.novelty_slices = 0

    corpus = synthetic_corpus(seq_len=args.seq_len)
    failures = []

    for arm in sorted(ARMS):
        a = apply_arm(copy.deepcopy(args), arm)
        mcfg = ModelCfg(vocab_size=corpus.vocab_size, seq_len=a.seq_len, d_model=a.d_model,
                        n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff,
                        causal=bool(a.causal))
        model = LM(mcfg)
        try:
            obj = build_loss(a.loss, model, corpus, a)
            sampler = build_sampler(a.sampler, corpus, a)

            span_ids = sampler.sample(a.batch_size, step=0)
            x, y, starts = corpus.spans_to_batch(span_ids)
            out = obj(x, y, starts, step=1)
            loss = out["loss"]
            loss.backward()

            gnan = any(p.grad is not None and torch.isnan(p.grad).any()
                       for p in list(model.parameters()) + obj.extra_parameters())
            lnan = bool(torch.isnan(loss).any().item())
            sampler.update(span_ids, out["per_span_nll"].float(), step=1)
            obj.on_optimizer_step(1)

            if mcfg.causal:
                cok, cdiff, _ = verify_causality(model, "cpu", corpus.vocab_size, seq_len=a.seq_len)
            else:
                cok, cdiff = True, float("nan")

            ok = (not gnan) and (not lnan) and cok
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {arm:<20} loss={loss.item():7.4f} "
                  f"grad_nan={gnan} causal_diff={cdiff:.1e} "
                  f"sup={out['frac_supervised']:.2f} extra_params="
                  f"{sum(q.numel() for q in obj.extra_parameters())}")
            d = obj.diagnostics()
            if d:
                print(f"          diag: { {k: round(v,4) if isinstance(v,float) else v for k,v in d.items()} }")
            if not ok:
                failures.append(arm)
        except Exception as e:
            print(f"  [FAIL] {arm:<20} {type(e).__name__}: {e}")
            failures.append(arm)

    # anyorder NELBO evaluator path
    try:
        a = apply_arm(copy.deepcopy(args), "anyorder")
        mcfg = ModelCfg(vocab_size=corpus.vocab_size, seq_len=a.seq_len, d_model=a.d_model,
                        n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff, causal=False)
        m = LM(mcfg)
        obj = build_loss("anyorder", m, corpus, a)
        b = obj.eval_nelbo(corpus, n_batches=2, batch_size=4, mc=2)
        print(f"  [OK ] anyorder eval_nelbo -> {b:.4f} nats/tok (bound)")
    except Exception as e:
        print(f"  [FAIL] anyorder eval_nelbo {type(e).__name__}: {e}")
        failures.append("anyorder_eval")

    # degeneration + self-endorsement diagnostics
    try:
        from src.diagnostics import degeneration_report, self_endorsement
        mcfg = ModelCfg(vocab_size=corpus.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
                        n_layers=2, n_heads=2, d_ff=64, causal=True)
        m = LM(mcfg)
        dr = degeneration_report(m, corpus, n_samples=2, max_new=16)
        se = self_endorsement(m, corpus, n_samples=2, max_new=16)
        print(f"  [OK ] diagnostics {  {k: round(v,3) for k,v in {**dr, **se}.items()} }")
    except Exception as e:
        print(f"  [FAIL] diagnostics {type(e).__name__}: {e}")
        failures.append("diagnostics")

    print("\nSMOKE " + ("OK" if not failures else f"FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(smoke())
