#!/usr/bin/env python
"""
Entry point.

  python run.py --arm baseline --seeds 3          # stage 1: establish noise floor
  python run.py --stage screen                    # stage 2: one seed per arm
  python run.py --arm selective --seeds 3         # stage 3: confirm a survivor
  python run.py --summarise                       # read the register, apply the rule

Decision rule (pre-registered, do not change after seeing results)
------------------------------------------------------------------
  noise_floor = std of `baseline` best val loss across >= 3 seeds
  An arm is INTERESTING only if it beats baseline by more than 2 x noise_floor.
  An arm is REAL only if it also beats its own control by more than 2 x noise_floor.
  Anything else is logged as `refuted` and closed.
"""

import argparse
import copy
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.arms import ARMS, apply_arm
from src.registry import append_row, next_record_id, row_from_result
from src.train import run as train_run

# Size presets. At 1M tokens the first run memorised the training set (train ppl
# 17 after 16 epochs) with `base`. `small` is the new default; `tiny` is there
# for when even that memorises. Non-embedding params are what track capacity --
# the tied embedding table dominates the total at small d_model.
SIZES = {
    "tiny":  dict(d_model=96,  n_layers=3, n_heads=3, d_ff=384),   # ~0.14M non-emb
    "small": dict(d_model=128, n_layers=4, n_heads=4, d_ff=512),   # ~0.79M non-emb
    "base":  dict(d_model=256, n_layers=4, n_heads=4, d_ff=1024),  # ~3.15M non-emb
}

LADDER_ARMS = ["teach_unigram", "teach_trigram", "teach_varorder", "teach_cache",
               "teach_embed", "teach_class", "teach_mix"]

CONTROL_OF = {
    "selective": ["selective_random", "selective_ref"],
    "selective_soft": ["selective_random"],
    "ngram_soft": ["ngram_soft_uniform", "ngram_soft_unigram"],
    "teach_trigram": ["teach_shuffled", "teach_topm_uniform", "teach_uniform"],
    "teach_varorder": ["teach_shuffled", "teach_topm_uniform", "teach_uniform"],
    "teach_cache": ["teach_shuffled", "teach_uniform"],
    "teach_embed": ["teach_shuffled", "teach_uniform"],
    "teach_class": ["teach_shuffled", "teach_uniform"],
    "teach_mix": ["teach_shuffled", "teach_uniform"],
    "teach_best": ["teach_shuffled", "teach_uniform"],
    "dream": ["dream_rep_only", "dream_random"],
    "replay_progress": ["replay_hard"],
    "latent": ["latent_shuffle", "latent_frozen"],
    "anyorder": ["anyorder_matched"],
}

SCREEN_ORDER = [
    "baseline",
    "teach_self", "teach_self_probs",
    "teach_neural", "teach_neural_probs", "teach_neural_shuffled",
    # the ladder
    "teach_trigram", "teach_varorder", "teach_cache", "teach_embed", "teach_class",
    "teach_mix",
    # matched controls
    "teach_shuffled", "teach_topm_uniform", "teach_uniform",
    # loss form
    "teach_hinge", "teach_adaptive",
    # survivors from earlier rounds
    "selective_soft", "dream",
]


def build_parser():
    p = argparse.ArgumentParser()
    # what to run
    p.add_argument("--arm", default="baseline", choices=sorted(ARMS))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default=None,
                   choices=["screen", "noisefloor", "sweep"])
    p.add_argument("--arms", default=None,
                   help="comma-separated arm list, overrides SCREEN_ORDER")
    p.add_argument("--sweep-param", dest="sweep_param", default="soft-top-m")
    p.add_argument("--sweep-values", dest="sweep_values", default="2,4,8,16,32")
    p.add_argument("--summarise", action="store_true")
    p.add_argument("--metric", default="robust", choices=["robust", "best"])
    p.add_argument("--fit-at", dest="fit_at", default="3.10")
    p.add_argument("--smoke", action="store_true")

    # data
    p.add_argument("--n-tokens", dest="n_tokens", type=int, default=1_000_000)
    p.add_argument("--val-tokens", dest="val_tokens", type=int, default=100_000)
    p.add_argument("--vocab-size", dest="vocab_size", type=int, default=2048)
    p.add_argument("--block-chars", dest="block_chars", type=int, default=8192)
    p.add_argument("--split-seed", dest="split_seed", type=int, default=0)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=256)
    p.add_argument("--cache-dir", dest="cache_dir", default="./cache")

    # model
    p.add_argument("--size", default="small", choices=sorted(SIZES),
                   help="preset; explicit --d-model etc. override it")
    p.add_argument("--d-model", dest="d_model", type=int, default=None)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=None)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=None)
    p.add_argument("--d-ff", dest="d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--causal", type=int, default=1)

    # optimisation (identical across arms -- do not tune per arm)
    p.add_argument("--n-steps", dest="n_steps", type=int, default=6000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    # Run 2 lesson: cosine anneals to 5%, so the LAST step is almost always the
    # best and every arm's best@ landed on the final eval. That measures
    # convergence speed, not generalisation. Constant LR exposes a true val
    # minimum, which is the only regime where sample-efficiency claims mean
    # anything.
    p.add_argument("--lr-schedule", dest="lr_schedule", default="constant",
                   choices=["cosine", "constant"])
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--synthetic", type=int, default=0,
                   help="Markov synthetic corpus; validates the loop offline. Never a result.")

    # eval / logging
    p.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    p.add_argument("--log-every", dest="log_every", type=int, default=250)
    p.add_argument("--ngram-anchor", dest="ngram_anchor", type=int, default=1,
                   help="fit a trigram on train, score val: the absolute floor")
    p.add_argument("--diag-every", dest="diag_every", type=int, default=100)
    p.add_argument("--eval-batch", dest="eval_batch", type=int, default=32)
    p.add_argument("--eval-nelbo-batches", dest="eval_nelbo_batches", type=int, default=12)
    p.add_argument("--gap-spans", dest="gap_spans", type=int, default=64)
    p.add_argument("--gen-samples", dest="gen_samples", type=int, default=12)
    p.add_argument("--gen-len", dest="gen_len", type=int, default=128)
    p.add_argument("--ema-decay", dest="ema_decay", type=float, default=0.999,
                   help="Polyak weight averaging; 0 disables")
    p.add_argument("--ema-warmup", dest="ema_warmup", type=int, default=500)
    p.add_argument("--robust-k", dest="robust_k", type=int, default=3,
                   help="robust floor = mean of the k lowest evals")
    p.add_argument("--fit-targets", dest="fit_targets", default=None,
                   help="comma-separated TRAIN loss levels for the matched-fit metric")
    p.add_argument("--novelty-slices", dest="novelty_slices", type=int, default=1)
    p.add_argument("--novelty-batches", dest="novelty_batches", type=int, default=8)
    p.add_argument("--out-dir", dest="out_dir", default="./results")
    p.add_argument("--registry", default="./results/experiments_objectives.csv")
    p.add_argument("--chat-title", dest="chat_title", default="Bio-inspired objectives on BabyLM-1M")

    # replay
    p.add_argument("--replay-beta-fast", dest="replay_beta_fast", type=float, default=0.7)
    p.add_argument("--replay-beta-slow", dest="replay_beta_slow", type=float, default=0.95)
    p.add_argument("--replay-temp", dest="replay_temp", type=float, default=1.0)
    p.add_argument("--replay-eps", dest="replay_eps", type=float, default=0.15)
    p.add_argument("--replay-warmup", dest="replay_warmup", type=int, default=200)
    p.add_argument("--replay-novelty", dest="replay_novelty", type=float, default=0.5,
                   help="UCB bonus weight; 0 disables")
    p.add_argument("--replay-max-visit-ratio", dest="replay_max_visit_ratio",
                   type=float, default=4.0, help="hard visit cap vs uniform rate; 0 disables")
    p.add_argument("--replay-stale-halflife", dest="replay_stale_halflife",
                   type=float, default=300.0)
    p.add_argument("--replay-gini-warn", dest="replay_gini_warn", type=float, default=0.45)

    # selective
    p.add_argument("--selective-keep", dest="selective_keep", type=float, default=0.5)
    p.add_argument("--selective-mode", dest="selective_mode", default="excess",
                   choices=["excess", "refhigh", "random"])
    p.add_argument("--selective-weighting", dest="selective_weighting", default="hard",
                   choices=["hard", "soft"])
    p.add_argument("--selective-beta", dest="selective_beta", type=float, default=1.0)

    # n-gram-anchored soft targets
    p.add_argument("--soft-lambda", dest="soft_lambda", type=float, default=0.15)
    p.add_argument("--teacher", default=None,
                   help="trigram|varorder|cache|class|embed|unigram|uniform, "
                        "or mix:a+b, shuffled:X, topm_uniform:X")
    p.add_argument("--soft-mode", dest="soft_mode", default="trigram")
    p.add_argument("--soft-form", dest="soft_form", default="mixture",
                   choices=["mixture", "hinge"])
    p.add_argument("--soft-adaptive-lambda", dest="soft_adaptive_lambda", type=int, default=0)
    p.add_argument("--soft-kappa", dest="soft_kappa", type=float, default=10.0)
    p.add_argument("--teacher-order", dest="teacher_order", type=int, default=2)
    p.add_argument("--teacher-max-order", dest="teacher_max_order", type=int, default=6)
    p.add_argument("--teacher-n-classes", dest="teacher_n_classes", type=int, default=128)
    p.add_argument("--teacher-emb-dim", dest="teacher_emb_dim", type=int, default=64)
    p.add_argument("--teacher-top-classes", dest="teacher_top_classes", type=int, default=4)
    p.add_argument("--teacher-neighbours", dest="teacher_neighbours", type=int, default=8)
    p.add_argument("--cache-half-life", dest="cache_half_life", type=float, default=512.0)
    p.add_argument("--cache-window", dest="cache_window", type=int, default=4096)
    p.add_argument("--mixture-weights", dest="mixture_weights", default=None)
    p.add_argument("--teacher-table", dest="teacher_table", default=None,
                   help="path to a cached neural-teacher npz (scripts/train_teacher.py)")
    p.add_argument("--soft-flatten", dest="soft_flatten", type=int, default=0,
                   help="discard teacher probabilities, keep support only")
    p.add_argument("--teacher-cache-m", dest="teacher_cache_m", type=int, default=32,
                   help="cache teachers this wide; --soft-top-m truncates at use")
    p.add_argument("--self-momentum", dest="self_momentum", type=float, default=0.999)
    p.add_argument("--self-warmup", dest="self_warmup", type=int, default=1000)
    p.add_argument("--self-exclude-true", dest="self_exclude_true", type=int, default=0)
    p.add_argument("--teacher-report", dest="teacher_report", default=None,
                   help="build teachers and print their diagnostics, then exit")
    p.add_argument("--soft-top-m", dest="soft_top_m", type=int, default=8)
    p.add_argument("--soft-min-count", dest="soft_min_count", type=int, default=3)

    # dreaming with external judges
    p.add_argument("--dream-every", dest="dream_every", type=int, default=20)
    p.add_argument("--dream-batch", dest="dream_batch", type=int, default=8)
    p.add_argument("--dream-len", dest="dream_len", type=int, default=48)
    p.add_argument("--dream-prompt", dest="dream_prompt", type=int, default=32)
    p.add_argument("--dream-weight", dest="dream_weight", type=float, default=0.5)
    p.add_argument("--dream-temp", dest="dream_temp", type=float, default=1.0)
    p.add_argument("--dream-judges", dest="dream_judges", default="ngram,rep")
    p.add_argument("--dream-rep-n", dest="dream_rep_n", type=int, default=4)
    p.add_argument("--dream-random-rate", dest="dream_random_rate", type=float, default=0.05)

    # mtp
    p.add_argument("--mtp-horizon", dest="mtp_horizon", type=int, default=1)
    p.add_argument("--mtp-weight", dest="mtp_weight", type=float, default=0.3)

    # latent
    p.add_argument("--latent-horizon", dest="latent_horizon", type=int, default=16)
    p.add_argument("--latent-weight", dest="latent_weight", type=float, default=0.5)
    p.add_argument("--latent-momentum", dest="latent_momentum", type=float, default=0.996)
    p.add_argument("--latent-var-weight", dest="latent_var_weight", type=float, default=1.0)
    p.add_argument("--latent-cov-weight", dest="latent_cov_weight", type=float, default=0.04)
    p.add_argument("--latent-target", dest="latent_target", default="ema",
                   choices=["ema", "frozen", "shuffle"])
    p.add_argument("--collapse-erank-frac", dest="collapse_erank_frac", type=float, default=0.05)

    # diffusion
    p.add_argument("--diffusion-eps", dest="diffusion_eps", type=float, default=1e-3)
    p.add_argument("--match-supervision", dest="match_supervision", type=int, default=0)
    return p


def resolve_size(args):
    preset = SIZES[args.size]
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def one_run(args, arm, seed, tag=None):
    a = copy.deepcopy(args)
    a = resolve_size(a)
    a = apply_arm(a, arm)
    a.arm = arm
    a.seed = seed
    if getattr(a, "match_supervision", 0):
        a.n_steps = a.n_steps * 2
        print(f"  [match-supervision] doubling steps to {a.n_steps}")
    res, model, corpus = train_run(a)
    res["run_id"] = f"{arm}_{tag}_s{seed}" if tag else f"{arm}_s{seed}"
    if tag:
        res["arm"] = f"{arm}[{tag}]"
    res["record_id"] = next_record_id(a.registry)
    row = row_from_result(res, a, model, corpus)
    append_row(a.registry, row)
    return res


def teacher_report(args):
    """Build each teacher and print what it knows. Do this before spending GPU time."""
    import numpy as np

    from src.data import DataCfg, build_or_load
    from src.teachers import LADDER, build_teacher

    args = resolve_size(args)
    dcfg = DataCfg(n_tokens=args.n_tokens, val_tokens=args.val_tokens,
                   vocab_size=args.vocab_size, seq_len=args.seq_len,
                   cache_dir=args.cache_dir)
    corpus = build_or_load(dcfg, device="cpu")
    specs = LADDER if args.teacher_report == "all" else args.teacher_report.split(",")
    print(f"\ncorpus {corpus.train.numel():,} tokens | vocab {corpus.vocab_size}\n")
    rows = []
    for spec in specs:
        T = build_teacher(spec, corpus.train_np, corpus.vocab_size,
                          m=args.soft_top_m, min_count=args.soft_min_count, args=args)
        r = T.report(*T.build(cache_dir=args.cache_dir))
        rows.append(r)
        print(f"{spec:28s} cov {r['coverage']:.3f}  hit {r['hit_rate']:.3f}  "
              f"p_true {r['p_true_mean']:.3f}  eff_supp {r['eff_support']:5.2f}  "
              f"med_ev {r['median_evidence']:8.0f}")
        print(f"{'':28s} lacks: {r['lacks']}")
        for k in ("order_mix", "class_singleton_frac", "mean_class_size", "WARNING"):
            if k in r:
                print(f"{'':28s} {k}: {r[k]}")
    print("\nhit_rate = fraction of positions where the teacher independently attests")
    print("the true token (leave-one-out). Low hit_rate + high p_true = confident and")
    print("often wrong, which is what the one-sided hinge is for.")
    return rows


def _welch(a, b):
    """Welch's t on the DIFFERENCE between two arms.

    The old rule -- beat baseline by 2x the baseline's own std -- is the wrong
    test. It ignores the arm's own variance entirely, and it uses a threshold
    rather than a standard error on the quantity actually being estimated. This
    returns (mean difference, standard error of that difference, n_a, n_b).
    """
    import math

    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    se = math.sqrt(va / max(na, 1) + vb / max(nb, 1))
    return ma - mb, se, na, nb


def summarise(args):
    import csv
    import json as _json

    if not os.path.exists(args.registry):
        print("no registry yet")
        return
    rows = list(csv.DictReader(open(args.registry)))
    seen, by_arm, steps, flags, fits = set(), {}, {}, {}, {}
    for r in rows:
        rid = r.get("run_id", "")
        if rid in seen:
            continue
        seen.add(rid)
        try:
            v = float(r["metric_primary_value"])
        except (ValueError, KeyError):
            continue
        a = r["model_name"]
        by_arm.setdefault(a, []).append(v)
        try:
            steps.setdefault(a, []).append(int(r.get("best_step") or 0))
        except ValueError:
            pass
        if r.get("confound"):
            flags.setdefault(a, set()).add(r["confound"].split(" | ")[0][:44])
        try:
            ev = _json.loads(r.get("evidence") or "{}")
            key = f"val@train{args.fit_at}"
            if key in ev:
                fits.setdefault(a, []).append(float(ev[key]))
        except Exception:
            pass

    if "baseline" not in by_arm:
        print("run `--stage noisefloor` first to establish the noise floor")
        return
    base = by_arm["baseline"]
    nb = len(base)
    sd = statistics.stdev(base) if nb > 1 else float("nan")
    print(f"\nbaseline: {statistics.mean(base):.4f} nats over {nb} seed(s), sd {sd:.4f}")
    if nb < 3:
        print("!! fewer than 3 baseline seeds: no usable noise floor.")
    print(f"metric: {args.metric}   matched-fit column: val@train{args.fit_at}\n")

    hdr = (f"{'arm':<22} {'n':>2} {'mean':>8} {'delta':>8} {'SE':>7} "
           f"{'t':>6} {'fit':>8} {'verdict':>26}")
    print(hdr)
    print("-" * len(hdr))
    for arm, vs in sorted(by_arm.items(), key=lambda kv: statistics.mean(kv[1])):
        d, se, na, _ = _welch(vs, base)
        t = d / se if se > 0 else float("nan")
        f = statistics.mean(fits[arm]) if fits.get(arm) else float("nan")
        if arm == "baseline":
            verdict = "reference"
        elif arm.startswith("anyorder"):
            verdict = "NELBO bound, not comparable"
        elif na < 2 or nb < 2:
            verdict = "need >=2 seeds both sides"
        elif t < -2:
            ctrls = CONTROL_OF.get(arm, [])
            cv = [by_arm[c] for c in ctrls if c in by_arm and len(by_arm[c]) > 1]
            if not cv:
                verdict = "beats baseline; run control"
            else:
                worst = max(cv, key=statistics.mean)
                dc, sec, _, _ = _welch(vs, worst)
                tc = dc / sec if sec > 0 else float("nan")
                verdict = ("REAL (beats baseline+control)" if tc < -2
                           else "explained by control")
        elif t > 2:
            verdict = "worse than baseline"
        else:
            verdict = "within noise"
        print(f"{arm:<22} {na:>2} {statistics.mean(vs):>8.4f} {d:>+8.4f} {se:>7.4f} "
              f"{t:>6.2f} {f:>8.4f} {verdict:>26}")
        for fl in sorted(flags.get(arm, [])):
            print(f"{'':<22}   !! {fl}")

    print("\n`t` is the difference divided by its own standard error (Welch), not")
    print("a fixed multiple of the baseline's spread. |t| > 2 is the bar.")
    print("`fit` is val loss at MATCHED train loss -- it separates 'learns better'")
    print("from 'regularises', which comparing minima cannot.")
    bs_all = [x for v in steps.values() for x in v]
    if bs_all and len(set(bs_all)) == 1:
        print("\n" + "=" * 74)
        print("WARNING: every arm's best val loss is at the SAME step. No arm reached")
        print("a validation minimum, so this ranks convergence speed at a truncated")
        print("budget, not sample efficiency. Raise --n-steps until best@ is INTERIOR.")
        print("=" * 74)


def main():
    args = build_parser().parse_args()

    if args.teacher_report:
        return teacher_report(args)

    if args.summarise:
        return summarise(args)

    if args.smoke:
        from scripts.smoke import smoke
        return smoke(args)

    if args.stage == "noisefloor":
        for s in range(3):
            one_run(args, "baseline", s)
        return summarise(args)

    if args.stage == "screen":
        # BUG FIXED (run 4): this ignored --seeds and ran one seed per arm, which
        # is why an entire screen came back n=1 with "no noise floor yet" despite
        # --seeds 3 being passed. Nothing in that table was testable.
        arms = args.arms.split(",") if args.arms else SCREEN_ORDER
        print(f"screening {len(arms)} arms x {args.seeds} seed(s) = "
              f"{len(arms) * args.seeds} runs")
        for arm in arms:
            for i in range(args.seeds):
                try:
                    one_run(args, arm, args.seed + i)
                except Exception as e:  # one broken arm must not kill the sweep
                    print(f"  !! arm {arm} seed {args.seed+i} failed: "
                          f"{type(e).__name__}: {e}")
        return summarise(args)

    if args.stage == "sweep":
        vals = [float(v) for v in args.sweep_values.split(",")]
        print(f"sweeping --{args.sweep_param} over {vals} on arm {args.arm}")
        for v in vals:
            for i in range(args.seeds):
                a = copy.deepcopy(args)
                setattr(a, args.sweep_param.replace("-", "_"),
                        int(v) if float(v).is_integer() and args.sweep_param != "soft-lambda" else v)
                a.sweep_tag = f"{args.sweep_param}={v}"
                try:
                    one_run(a, args.arm, args.seed + i, tag=f"{args.sweep_param}{v}")
                except Exception as e:
                    print(f"  !! {args.sweep_param}={v} failed: {type(e).__name__}: {e}")
        return summarise(args)

    results = [one_run(args, args.arm, args.seed + i) for i in range(args.seeds)]
    if len(results) > 1:
        vals = [r["best_val_loss"] for r in results]
        print(f"\n{args.arm}: mean {statistics.mean(vals):.4f} "
              f"std {statistics.stdev(vals):.4f} over {len(vals)} seeds")
    print(json.dumps({r["run_id"]: r["best_val_loss"] for r in results}, indent=2))


if __name__ == "__main__":
    main()
