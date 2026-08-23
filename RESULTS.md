# Results log

## Run 1 (2026-08-23) — VOID. Data pipeline bug.

**Status: all twelve arms discarded. No comparison from this run is usable.**

### What happened

`_raw_text` read rows until it had ~6.6M characters and stopped. BabyLM
concatenates six source files in order, so the slice was:

| file | size | fate |
|---|---|---|
| bnc_spoken | 4.06 MB | all of it, into TRAIN |
| childes | 15.2 MB | first 2.5 MB; the tail became the ENTIRE VAL SET |
| gutenberg | 14.0 MB | never read |
| open_subtitles | 12.2 MB | never read |
| simple_wiki | 8.86 MB | never read |
| switchboard | 0.12 MB | never read |

Train was adult conversation transcripts; val was pure child-directed speech.
Four of six domains were absent entirely.

### How it showed up

Vocabulary was 4096, so `log(4096) = 8.318` nats is what a model that has learned
nothing scores. Every arm's **best** val loss was at or above that line:

| arm | best val (nats) | vs uniform-random ceiling |
|---|---|---|
| replay_progress | 7.949 | 95.6% |
| latent_shuffle | 8.212 | 98.7% |
| baseline (3 seeds) | 8.32–8.66 | 100–104% |
| selective | 8.687 | 104.4% |
| mtp | 9.027 | 108.5% |
| selective_ref | 9.674 | 116.3% |

The baseline finished at **10.46 nats (ppl 34,947), 35% worse than uniform
random guessing.** Overfitting cannot produce that; only a distribution the
model has never seen can.

Because val loss rose from the first eval, `best_val_loss` landed at **step 250
for all twelve arms** — the decision rule ranked models that had seen 12% of
their budget. The 0.189-nat "noise floor" is the seed variance of a barely
trained model.

### The one finding that survives

`replay_progress` was flagged REAL by the decision rule and is a **false
positive** — but the diagnostics caught the mechanism:

* batch loss fell to **0.144 nats (ppl 1.15)** at step 750 while the held-out
  train-span eval simultaneously read **ppl 272**
* `replay_visit_max = 903` against a uniform expectation of ~16 (56x)
* `trunk_erank` degraded from 218 (baseline) to **145**
* train ppl *rose* across training (185 → 272 → 212 → 269): catastrophic forgetting

Cause: a positive feedback loop. A span with falling loss is sampled more, which
makes its loss fall further, which gets it sampled more. With `temp=0.5` over
~3,900 spans this ran away within a few hundred steps. Its apparent win was
being effectively undertrained on the corpus, and therefore less confidently
wrong about a foreign val domain.

**Prioritised replay needs anti-collapse machinery for the same reason latent
targets do.** This was built for `latent` and not for `replay`. It is now built
for both.

### Weak directional negatives (not evidence at 250 steps)

* `selective_ref` stalled at train ppl 160 vs 17 for baseline. Selecting purely
  on n-gram rarity starves the model of frequent structure. Rarity alone is a
  dead token-value signal.
* `latent_shuffle` (8.212) beat `latent` (8.520). If it held, the VICReg term
  would be doing whatever work exists and the prediction target none.
* `anyorder` trunk effective rank 0.19 of d, vs 0.85 for causal arms. Worth
  re-measuring on clean data.

### Machinery that worked

* the collapse kill criterion never fired spuriously (`latent_target_erank_frac`
  0.70 / 0.45)
* budget parity accounting was correct throughout
* `self_endorsement` was uninformative: it sampled at temperature 1.0 from
  near-random models, so its "repeated n-grams" were accidental collisions.
  Degeneration is a maximisation-decoding phenomenon; it now decodes greedily
  and immediately reports the expected positive `endorse_delta`.

### Fixes applied

1. **Block-interleaved split.** Whole corpus read, cut into 8192-char blocks,
   permuted with a fixed seed, train and val drawn from the same stream.
   Identical domain mixture by construction.
2. **Trigram anchor gate.** A trigram fit on train and scored on val prints at
   startup. Any model that fails to beat it writes a confound into the register.
3. **Uniform-ceiling gate.** Val loss at or above `log(vocab)` prints
   `<<< RUN IS BROKEN` inline and flags the row.
4. **Unigram KL(val||train)** printed at startup.
5. **Replay anti-collapse:** hard visit cap (4x uniform rate), UCB novelty bonus,
   staleness decay, higher temperature and epsilon. Visit Gini logged; above 0.45
   writes a confound.
6. **Smaller models.** `--size small` (0.79M non-embedding) is the new default;
   `tiny` is 0.34M. Dropout default 0.0 -> 0.1. Vocab 4096 -> 2048.
7. **`eval_every` 250 -> 100**, and a "best at FIRST eval" confound flag.
8. **Summariser** dedupes by run_id and prints `best@`.
