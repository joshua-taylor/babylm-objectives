"""
Train a teacher model on the same corpus and cache its top-k predictions.

  python -m scripts.train_teacher --teacher-size base --n-steps 9000

Produces `teacher_neural_<tag>.npz` in the cache dir, holding for every training
position the teacher's top-K next-token IDs and probabilities.

TWO DETAILS THAT MATTER
-----------------------
1. STRIDED WINDOWS. If you cache predictions by chopping the corpus into
   non-overlapping spans, the first tokens of every span are predicted with
   almost no left context, and the teacher looks far worse than it is. Here the
   corpus is swept with stride L/2 and only the second half of each window is
   kept, so every position has at least L/2 tokens of real context.

2. CACHE WIDE, TRUNCATE AT USE. The table is cached at K=64 and the loss
   truncates to --soft-top-m and renormalises at load time. That makes the m
   sweep free -- no retraining, no rebuilding -- which matters because m is now
   the central variable.

The teacher is trained on exactly the same corpus as the student. It sees no
extra data. Whatever it transfers is a reorganisation of information the student
already has access to, not new information -- which is the honest framing for
why distillation can beat direct training at all.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import SIZES, build_parser, resolve_size  # noqa: E402
from src.data import DataCfg, build_or_load  # noqa: E402
from src.train import eval_ar, run as train_run  # noqa: E402


@torch.no_grad()
def cache_topk(model, corpus, K, path, batch_windows=16, device="cpu",
               amp_dtype=None, val_nats=float("nan"), params_m=float("nan")):
    model.eval()
    L = corpus.span_len
    N = corpus.train.numel()
    half = L // 2

    idx = np.zeros((N, K), dtype=np.int32)
    prob = np.zeros((N, K), dtype=np.float32)
    filled = np.zeros(N, dtype=bool)

    starts = list(range(0, max(N - L, 1), half))
    for b0 in range(0, len(starts), batch_windows):
        ws = starts[b0 : b0 + batch_windows]
        off = torch.arange(L, device=corpus.train.device)
        st = torch.tensor(ws, device=corpus.train.device)
        x = corpus.train[st[:, None] + off[None, :]]

        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=-1)
        tv, ti = torch.topk(p, min(K, p.size(-1)), dim=-1)
        tv, ti = tv.cpu().numpy(), ti.cpu().numpy()

        for r, w in enumerate(ws):
            # logits[i] predicts corpus position w+i+1
            lo = 0 if w == 0 else half           # only keep well-contextualised half
            for i in range(lo, L):
                tgt = w + i + 1
                if tgt >= N or filled[tgt]:
                    continue
                idx[tgt, : ti.shape[-1]] = ti[r, i]
                prob[tgt, : tv.shape[-1]] = tv[r, i]
                filled[tgt] = True

    # position 0 and any stragglers: fall back to the teacher's unigram-ish first step
    if (~filled).any():
        miss = np.flatnonzero(~filled)
        idx[miss] = idx[filled][0]
        prob[miss] = prob[filled][0]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, idx=idx, prob=prob,
                        val_nats=np.array(val_nats), params_m=np.array(params_m))
    model.train()
    return path


def main():
    base = build_parser()
    p = argparse.ArgumentParser(parents=[base], conflict_handler="resolve")
    p.add_argument("--teacher-size", dest="teacher_size", default="base",
                   choices=sorted(SIZES), help="teacher trunk size")
    p.add_argument("--teacher-k", dest="teacher_k", type=int, default=64,
                   help="cache width; the loss truncates to --soft-top-m")
    p.add_argument("--teacher-tag", dest="teacher_tag", default=None)
    args = p.parse_args()

    args.size = args.teacher_size
    args.d_model = args.n_layers = args.n_heads = args.d_ff = None
    args = resolve_size(args)
    args.arm = f"TEACHER_{args.teacher_size}"
    args.sampler, args.loss, args.is_baseline = "uniform", "ntp", 0
    args.hypothesis = "teacher model for support-only distillation"
    args.novelty_slices = 0

    print(f"\n=== training teacher ({args.teacher_size}) on the SAME corpus ===")
    res, model, corpus = train_run(args)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    amp_dtype = torch.float16 if (device == "cuda" and args.amp) else None
    val = eval_ar(model, corpus, args.eval_batch, amp_dtype, device)

    tag = args.teacher_tag or f"{args.teacher_size}_v{corpus.vocab_size}_n{corpus.train.numel()}"
    path = os.path.join(args.cache_dir, f"teacher_neural_{tag}.npz")
    print(f"\ncaching top-{args.teacher_k} predictions for {corpus.train.numel():,} positions ...")
    cache_topk(model, corpus, args.teacher_k, path, device=device, amp_dtype=amp_dtype,
               val_nats=val, params_m=model.n_params() / 1e6)

    import math
    print(f"\nteacher val {val:.4f} nats (ppl {math.exp(min(val,20)):.2f}) | "
          f"{model.n_params()/1e6:.2f}M params")
    print(f"table -> {path}")
    print(f"\nnow run:  python run.py --arm teach_neural --teacher-table {path}")


if __name__ == "__main__":
    main()
