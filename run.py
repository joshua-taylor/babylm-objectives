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

CONTROL_OF = {
    "selective": ["selective_random", "selective_ref"],
    "selective_soft": ["selective_random"],
    "ngram_soft": ["ngram_soft_uniform", "ngram_soft_unigram"],
    "dream": ["dream_rep_only", "dream_random"],
    "replay_progress": ["replay_hard"],
    "latent": ["latent_shuffle", "latent_frozen"],
    "anyorder": ["anyorder_matched"],
}

SCREEN_ORDER = [
    "baseline",
    "ngram_soft", "ngram_soft_uniform", "ngram_soft_unigram",
    "selective_soft", "selective", "selective_random",
    "dream", "dream_rep_only", "dream_random",
    "replay_progress", "replay_hard",
    "latent", "latent_shuffle",
]


def build_parser():
    p = argparse.ArgumentParser()
    # what to run
    p.add_argument("--arm", default="baseline", choices=sorted(ARMS))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default=None, choices=["screen", "noisefloor"])
    p.add_argument("--summarise", action="store_true")
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
    p.add_argument("--soft-mode", dest="soft_mode", default="trigram",
                   choices=["trigram", "uniform", "unigram"])
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


def one_run(args, arm, seed):
    import copy

    a = copy.deepcopy(args)
    a = resolve_size(a)
    a = apply_arm(a, arm)
    a.arm = arm
    a.seed = seed
    if getattr(a, "match_supervision", 0):
        a.n_steps = a.n_steps * 2
        print(f"  [match-supervision] doubling steps to {a.n_steps}")
    res, model, corpus = train_run(a)
    res["run_id"] = f"{arm}_s{seed}"
    res["record_id"] = next_record_id(a.registry)
    row = row_from_result(res, a, model, corpus)
    append_row(a.registry, row)
    return res


def summarise(args):
    import csv

    if not os.path.exists(args.registry):
        print("no registry yet")
        return
    rows = list(csv.DictReader(open(args.registry)))
    seen, by_arm, steps, flags = set(), {}, {}, {}
    for r in rows:
        rid = r.get("run_id", "")
        if rid in seen:          # reruns of the same arm+seed must not inflate n
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
            flags.setdefault(a, set()).add(r["confound"].split(" | ")[0][:40])

    if "baseline" not in by_arm:
        print("run `--arm baseline --seeds 3` first to establish the noise floor")
        return
    base = by_arm["baseline"]
    nf = statistics.stdev(base) if len(base) > 1 else float("nan")
    print(f"\nbaseline: mean {statistics.mean(base):.4f} nats over {len(base)} seed(s)")
    print(f"noise floor (std): {nf:.4f}   threshold = 2x = {2*nf:.4f}\n")
    print(f"{'arm':<20} {'n':>3} {'mean':>9} {'delta':>9} {'best@':>7} {'verdict':>30}")
    print("-" * 84)
    for arm, vs in sorted(by_arm.items(), key=lambda kv: statistics.mean(kv[1])):
        m = statistics.mean(vs)
        d = m - statistics.mean(base)
        if arm == "baseline":
            verdict = "reference"
        elif arm.startswith("anyorder"):
            verdict = "NELBO bound, not comparable"
        elif nf != nf:
            verdict = "no noise floor yet"
        elif d > 2 * nf:
            verdict = "worse than baseline"
        elif d < -2 * nf:
            ctrls = CONTROL_OF.get(arm, [])
            cvals = [statistics.mean(by_arm[c]) for c in ctrls if c in by_arm]
            if not cvals:
                verdict = "beats baseline; run control"
            elif m < min(cvals) - 2 * nf:
                verdict = "REAL (beats baseline + control)"
            else:
                verdict = "explained by control"
        else:
            verdict = "refuted (within noise)"
        bs = statistics.mean(steps.get(arm, [0])) if steps.get(arm) else 0
        print(f"{arm:<20} {len(vs):>3} {m:>9.4f} {d:>+9.4f} {bs:>7.0f} {verdict:>30}")
        for f in sorted(flags.get(arm, [])):
            print(f"{'':<20}   !! {f}")
    print("\nNote: `anyorder` rows are a NELBO bound, not an equal-footing loss.")
    bs_all = [s for v in steps.values() for s in v]
    if bs_all and len(set(bs_all)) == 1:
        print()
        print("=" * 74)
        print("WARNING: every arm's best val loss is at the SAME step. No arm reached")
        print("a validation minimum, so this table ranks convergence speed at a")
        print("truncated budget, not sample efficiency. In that regime plain NTP is")
        print("optimal by construction and every alternative objective is a tax.")
        print("Raise --n-steps until best@ is INTERIOR, and use --lr-schedule constant.")
        print("=" * 74)


def main():
    args = build_parser().parse_args()

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
        for arm in SCREEN_ORDER:
            try:
                one_run(args, arm, args.seed)
            except Exception as e:  # one broken arm must not kill the sweep
                print(f"  !! arm {arm} failed: {type(e).__name__}: {e}")
        return summarise(args)

    results = [one_run(args, args.arm, args.seed + i) for i in range(args.seeds)]
    if len(results) > 1:
        vals = [r["best_val_loss"] for r in results]
        print(f"\n{args.arm}: mean {statistics.mean(vals):.4f} "
              f"std {statistics.stdev(vals):.4f} over {len(vals)} seeds")
    print(json.dumps({r["run_id"]: r["best_val_loss"] for r in results}, indent=2))


if __name__ == "__main__":
    main()
