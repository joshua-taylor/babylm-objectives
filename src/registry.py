"""
Append results to the project experiment register, using the existing
experiments_2.csv schema so these runs sit alongside the induction/PASM arcs
rather than in a parallel format.

Every completed run writes a row automatically. Nothing here should need to be
filled in by hand afterwards except `key_finding` and `outcome`, which are
judgements.
"""

import csv
import json
import os
from datetime import date

COLUMNS = [
    "record_id", "arc_id", "run_id", "chat_title", "date_logged", "model_name",
    "model_family", "mixer_type", "model_detail", "is_baseline",
    "has_softmax_attention", "is_subquadratic", "dataset", "tokenizer",
    "vocab_size", "seq_len", "d_model", "n_layers", "budget_type", "n_params_m",
    "n_steps", "best_step", "n_epochs", "n_seeds", "val_ppl", "best_val_ppl",
    "train_ppl", "test_ppl", "val_loss", "loss_unit", "gap_bits", "accuracy",
    "blimp", "tok_s_train", "tok_s_infer", "peak_mem_gb", "ppl_match",
    "ppl_nomatch", "pct_match", "hypothesis", "key_finding", "outcome",
    "evidence", "delta_vs_baseline", "noise_floor", "confound",
    "source_chat_uuid", "evidence_ordinal", "is_comparable_group",
    "metric_primary", "metric_primary_value",
]

ARC_ID = "O_objectives_babylm1m"


def append_row(csv_path, row: dict):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    full = {c: row.get(c, "") for c in COLUMNS}
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(full)
    return full


def next_record_id(csv_path, arc_id=ARC_ID, prefix="O"):
    n = 0
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("arc_id") == arc_id:
                    n += 1
    return f"{prefix}{n + 1}"


def row_from_result(res: dict, args, model, corpus) -> dict:
    n_visits = args.n_steps * args.batch_size * args.seq_len
    epochs = n_visits / max(corpus.train.numel(), 1)
    loss_unit = "nats_bound" if res.get("loss_unit") == "nats_bound" else "ppl"
    return {
        "record_id": res["record_id"],
        "arc_id": ARC_ID,
        "run_id": res["run_id"],
        "chat_title": args.chat_title,
        "date_logged": date.today().isoformat(),
        "model_name": args.arm,
        "model_family": "attention",
        "mixer_type": "softmax_full" if args.causal else "softmax_bidirectional",
        "model_detail": f"{args.n_layers}L d{args.d_model} | sampler={args.sampler} loss={args.loss}",
        "is_baseline": args.is_baseline,
        "has_softmax_attention": 1,
        "is_subquadratic": 0,
        "dataset": f"babylm_2026_strict_small_{corpus.train.numel()//1000}k",
        "tokenizer": "byte_level_bpe",
        "vocab_size": corpus.vocab_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "budget_type": "fixed_token_visits",
        "n_params_m": round(res["n_params"] / 1e6, 3),
        "n_steps": args.n_steps,
        "best_step": res.get("best_step", ""),
        "n_epochs": round(epochs, 2),
        "n_seeds": 1,
        "val_ppl": res.get("final_val_ppl", ""),
        "best_val_ppl": res.get("best_val_ppl", ""),
        "train_ppl": res.get("final_train_ppl", ""),
        "val_loss": res.get("best_val_loss", ""),
        "loss_unit": loss_unit,
        "gap_bits": res.get("gap_nats", ""),
        "tok_s_train": int(res.get("tok_s", 0) or 0),
        "peak_mem_gb": res.get("peak_mem_gb", ""),
        "hypothesis": args.hypothesis,
        "key_finding": res.get("key_finding", ""),
        "outcome": res.get("outcome", ""),
        "evidence": json.dumps(res.get("diagnostics", {}))[:900],
        "delta_vs_baseline": res.get("delta_vs_baseline", ""),
        "noise_floor": res.get("noise_floor", ""),
        "confound": res.get("confound", ""),
        "is_comparable_group": 1,
        "metric_primary": "best_val_loss_nats",
        "metric_primary_value": res.get("best_val_loss", ""),
    }
