"""
Training loop.

Discipline baked in (inherited from the project's standing practice):
  * full-model causality check before any run that claims to be autoregressive
  * deterministic full-sweep validation, never a random subsample
  * train/val gap reported every eval, because the gap is the real ceiling here
  * pre-registered collapse kill criterion checked in-flight, not post-hoc
  * matched token-VISIT budget across arms, printed at startup so it is
    impossible to compare two runs that had different budgets by accident
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import DataCfg, build_or_load
from .diagnostics import degeneration_report, repr_stats, self_endorsement
from .model import LM, ModelCfg, verify_causality
from .objectives import build_loss, build_sampler


def set_seed(s):
    import random

    import numpy as np

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_ar(model, corpus, batch_size=32, amp_dtype=None, device="cpu"):
    """Deterministic full sweep of the val set. Returns mean nats/token."""
    model.eval()
    tot, n = 0.0, 0
    for x, y in corpus.val_stream(batch_size):
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += l.item() * y.numel()
        n += y.numel()
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def eval_train_subset(model, corpus, n_spans=64, batch_size=32, amp_dtype=None, device="cpu"):
    """Fixed, deterministic slice of train spans -> the gap denominator."""
    model.eval()
    ids = torch.arange(min(n_spans, corpus.n_spans), device=corpus.train.device)
    tot, n = 0.0, 0
    for i in range(0, ids.numel(), batch_size):
        x, y, _ = corpus.spans_to_batch(ids[i : i + batch_size])
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        l = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += l.item() * y.numel()
        n += y.numel()
    model.train()
    return tot / max(n, 1)


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(args.seed)

    dcfg = DataCfg(
        n_tokens=args.n_tokens,
        val_tokens=args.val_tokens,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        cache_dir=args.cache_dir,
    )
    corpus = build_or_load(dcfg, device=device, use_synthetic=bool(getattr(args,'synthetic',0)))

    mcfg = ModelCfg(
        vocab_size=corpus.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        causal=bool(args.causal),
    )
    model = LM(mcfg).to(device)
    objective = build_loss(args.loss, model, corpus, args)
    sampler = build_sampler(args.sampler, corpus, args)

    n_visits = args.n_steps * args.batch_size * args.seq_len
    print(f"\n{'='*72}\nARM: {args.arm}   (sampler={args.sampler}, loss={args.loss}, seed={args.seed})")
    print(f"{'='*72}")
    for line in corpus.report():
        print(line)
    anchor = float("nan")
    if not getattr(args, "synthetic", 0) and args.ngram_anchor:
        anchor = corpus.anchor(cache_dir=args.cache_dir)
        print(f"  trigram anchor (fit train, score val) {anchor:.4f} nats"
              f" = ppl {math.exp(min(anchor,20)):.1f}   <- the model MUST beat this")
    print(f"  model      {model.n_params()/1e6:.2f}M params"
          f" ({model.n_nonemb_params()/1e6:.2f}M non-embedding) | causal={mcfg.causal}"
          f" | dropout={mcfg.dropout}")
    print(f"  budget     {args.n_steps} steps x {args.batch_size} x {args.seq_len}"
          f" = {n_visits:,} token-visits ({n_visits/corpus.train.numel():.1f} epochs)")

    if mcfg.causal:
        ok, diff, nan = verify_causality(model, device, corpus.vocab_size)
        print(f"  causality  diff={diff:.2e} nan={nan} [{'OK' if ok else 'FAIL'}]")
        assert ok, "causality check failed"
    else:
        print("  causality  n/a (bidirectional arm)")

    params = [p for p in model.parameters() if p.requires_grad] + objective.extra_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    if args.lr_schedule == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_steps,
                                                         eta_min=args.lr * 0.05)
    else:
        sch = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)

    amp_dtype = torch.float16 if (device == "cuda" and args.amp) else None
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype is not None)

    best, best_step = float("inf"), 0
    hist, killed, kill_reason = [], False, ""
    low_rank_strikes = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for step in range(1, args.n_steps + 1):
        span_ids = sampler.sample(args.batch_size, step)
        x, y, starts = corpus.spans_to_batch(span_ids)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = objective(x, y, starts, step)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sch.step()
        objective.on_optimizer_step(step)
        sampler.update(span_ids, out["per_span_nll"].float(), step)

        # ---- pre-registered collapse kill criterion ----
        if step % args.diag_every == 0:
            d = objective.diagnostics()
            ef = d.get("latent_target_erank_frac")
            if ef is not None and not math.isnan(ef):
                if ef < args.collapse_erank_frac:
                    low_rank_strikes += 1
                    if low_rank_strikes >= 2:
                        killed = True
                        kill_reason = (f"representation collapse: target effective rank "
                                       f"{ef:.3f} of d < {args.collapse_erank_frac} twice")
                        print(f"  !! KILLED at step {step}: {kill_reason}")
                        break
                else:
                    low_rank_strikes = 0

        if step % args.log_every == 0:
            tps = step * args.batch_size * args.seq_len / (time.time() - t0)
            print(f"  step {step:5d} | loss {loss.item():6.3f} | nll {out['nll'].item():6.3f} "
                  f"| sup {out['frac_supervised']:.2f} | lr {sch.get_last_lr()[0]:.2e} "
                  f"| tok/s {tps:,.0f}")

        if step % args.eval_every == 0 or step == args.n_steps:
            if objective.supports_ar_eval:
                vl = eval_ar(model, corpus, args.eval_batch, amp_dtype, device)
                tl = eval_train_subset(model, corpus, args.gap_spans, args.eval_batch,
                                       amp_dtype, device)
                ceiling = math.log(corpus.vocab_size)
                flag = "  <<< AT/ABOVE UNIFORM-RANDOM, RUN IS BROKEN" if vl >= ceiling * 0.99 else ""
                tag = (f"val ppl {math.exp(min(vl,20)):7.2f} | train ppl "
                       f"{math.exp(min(tl,20)):7.2f} | gap {vl-tl:+.3f}{flag}")
            else:
                vl = objective.eval_nelbo(corpus, n_batches=args.eval_nelbo_batches,
                                          batch_size=args.eval_batch)
                tl = float("nan")
                tag = f"NELBO bound {vl:.4f} nats/tok"
            mark = " *" if vl < best else ""
            if vl < best:
                best, best_step = vl, step
            hist.append({"step": step, "val": vl, "train": tl})
            print(f"          {tag}{mark}")

    elapsed = time.time() - t0
    tok_s = (step * args.batch_size * args.seq_len) / max(elapsed, 1e-6)

    # ---------------------------------------------------------- final report
    diags = {}
    diags.update(sampler.diagnostics())
    diags.update(objective.diagnostics())
    if objective.supports_ar_eval and not killed:
        diags.update(degeneration_report(model, corpus, n_samples=args.gen_samples,
                                         max_new=args.gen_len, device=device))
        diags.update(self_endorsement(model, corpus, n_samples=args.gen_samples,
                                      max_new=args.gen_len, device=device))
    with torch.no_grad():
        x, y, _ = corpus.spans_to_batch(torch.arange(8, device=corpus.train.device))
        diags.update({f"trunk_{k}": v for k, v in repr_stats(model.hidden(x)).items()})

    confounds = [kill_reason] if kill_reason else []
    if hist and best_step == hist[0]["step"] and len(hist) > 1:
        confounds.append("best at FIRST eval: comparison is of undertrained models")
    if hist and best_step == hist[-1]["step"] and len(hist) > 1:
        # Run 2 shipped with this unflagged: every arm's best landed on the final
        # eval, so the whole table ranked convergence speed at a truncated budget
        # rather than generalisation. In that regime any objective that diverts
        # gradient from next-token prediction is a pure tax and will always lose.
        confounds.append("best at LAST eval: budget-limited, never reached a val "
                         "minimum; measures convergence speed not generalisation"
                         + (" (cosine LR guarantees this)"
                            if args.lr_schedule == "cosine" else ""))
    if objective.supports_ar_eval and best >= math.log(corpus.vocab_size) * 0.99:
        confounds.append("best val at/above uniform-random ceiling: run is broken, "
                         "do not compare arms")
    if not math.isnan(anchor) and objective.supports_ar_eval and best >= anchor:
        confounds.append(f"model failed to beat the trigram anchor ({anchor:.3f} nats)")
    g = diags.get("replay_visit_gini")
    if g is not None and g > args.replay_gini_warn:
        confounds.append(f"sampler concentration: visit Gini {g:.2f} > "
                         f"{args.replay_gini_warn}; possible replay collapse")

    final = hist[-1] if hist else {"val": float("nan"), "train": float("nan")}
    res = {
        "arm": args.arm,
        "seed": args.seed,
        "n_params": model.n_params(),
        "best_val_loss": round(best, 5),
        "best_step": best_step,
        "best_val_ppl": round(math.exp(min(best, 20)), 3) if objective.supports_ar_eval else "",
        "final_val_ppl": round(math.exp(min(final["val"], 20)), 3) if objective.supports_ar_eval else "",
        "final_train_ppl": round(math.exp(min(final["train"], 20)), 3)
        if objective.supports_ar_eval and not math.isnan(final["train"]) else "",
        "gap_nats": round(final["val"] - final["train"], 4)
        if objective.supports_ar_eval and not math.isnan(final["train"]) else "",
        "loss_unit": "nats_bound" if not objective.supports_ar_eval else "nats",
        "tok_s": round(tok_s),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else "",
        "history": hist,
        "diagnostics": diags,
        "outcome": "killed" if killed else "completed",
        "confound": " | ".join(confounds),
        "ngram_anchor_nats": None if math.isnan(anchor) else round(anchor, 5),
        "beats_anchor": (None if math.isnan(anchor) or not objective.supports_ar_eval
                         else bool(best < anchor)),
        "token_visits": n_visits,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n  RESULT  best={best:.4f} nats @ step {best_step} | {elapsed/60:.1f} min")
    if not math.isnan(anchor) and objective.supports_ar_eval:
        print(f"    vs trigram anchor {anchor:.4f}: "
              f"{'BEATS by ' + format(anchor-best,'.4f') if best < anchor else 'FAILS TO BEAT'}")
    for c in confounds:
        print(f"    !! CONFOUND: {c}")
    for k, v in diags.items():
        print(f"    {k:32s} {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.arm}_seed{args.seed}"
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
        json.dump(res, f, indent=2)

    return res, model, corpus
