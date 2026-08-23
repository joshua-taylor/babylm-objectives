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


---

## Run 2 (2026-08-23) — split fixed, but WRONG REGIME. Ranking not usable.

**Status: the data bug is fixed and the model genuinely learns. The comparison
between arms is still void, for a different reason.**

### What worked

Baseline reached **3.6038 nats = ppl 36.7** against a uniform ceiling of
`log(2048) = 7.625`. The block-interleaved split and the sanity gates did their
job. Noise floor collapsed to **0.0027 nats** across 3 seeds, which is a very
sensitive instrument.

### Why the table is still void

`best@` was **1500 — the final eval — for every single arm.** No arm ever reached
a validation minimum. Two causes, both mine:

1. The budget was too short. 12 epochs over 1M tokens with 0.79M non-embedding
   params and dropout 0.1 was still firmly in the improving regime.
2. **The LR schedule was left on cosine**, annealing to 5% of peak. That
   structurally guarantees the last step is the best unless overfitting is strong
   enough to overcome the anneal. It manufactures `best@last`.

In a compute-limited regime, any objective that diverts gradient from next-token
prediction is a pure tax, and the table is exactly that — monotonically ordered
by how much gradient each arm diverts:

| arm | delta vs baseline | what it diverts |
|---|---|---|
| latent | +0.0029 | light auxiliary head |
| mtp | +0.0224 | full auxiliary vocabulary head |
| latent_frozen | +0.0408 | auxiliary head, useless target |
| replay_progress | +0.0475 | non-uniform data order |
| selective | +0.0891 | 50% of tokens discarded |

This is the *same confound* previously flagged for `mtp` alone, and it was
allowed to contaminate all twelve arms. Run 1 was massively overfit on broken
data; run 2 was underfit on good data.

### The one real signal

Isolating the selective pair, which differ only in whether the selection carries
information:

```
selective        +0.0891   (50% sparsity + excess-surprisal signal)
selective_random +0.1048   (50% sparsity, no signal)
                 -------
signal recovers   0.0157 nats = 5.8x the noise floor
```

**The excess-surprisal token-value signal is real** — comfortably above threshold
— but it only recovers ~15% of what discarding the tokens costs. That is a design
error, not a refutation: the signal should REWEIGHT, not DISCARD. Fixed in
`selective_soft`, which keeps every token at mean weight 1.0.

### Fixes applied

1. **`--lr-schedule constant` is now the default.** A true val minimum is the
   only regime where a sample-efficiency claim means anything.
2. **`--n-steps` default 1500 -> 6000.** Train until val turns up.
3. **"best at LAST eval" confound flag**, symmetric with the existing first-eval
   flag. `--summarise` now refuses to rank the table when every arm shares the
   same best step, and says why.
4. **`selective_soft`**: sigmoid reweighting instead of hard masking.
5. New axes: `ngram_soft` (external graded partial credit) and `dream`
   (anchored, negative-only rollout critique). See README.


---

## Run 3 (9k steps, constant LR) — the split and regime are right; the measurement was not

Baseline 3.4022 nats (ppl 30.0) vs a trigram anchor of 3.745 and a uniform
ceiling of 7.625. Data and regime are finally sound.

### The summary table was misleading; the traces were not

Comparing arms at their own minima confounds "learns better" with
"regularises". Reading val loss at MATCHED train loss, from the same traces:

| arm | val ppl at train ppl ~18.5 |
|---|---|
| **ngram_soft** | **27.9** |
| dream | 29.5 |
| latent_shuffle | 29.65 |
| baseline | ~29.9 |
| ngram_soft_uniform | never reaches this fit |

At final step `ngram_soft` reached train ppl 18.22 vs baseline 18.02 -- an
IDENTICAL fit -- with val 27.52 vs 29.22. It generalises 6% better without
fitting less. Uniform label smoothing gets its (smaller) gain the classic way,
by fitting worse: train 21.72. Different mechanisms, and the endpoint
comparison conflated them. The verdict "explained by control" was an artefact
of an entropy-mismatched control: uniform spreads lambda over 2048 tokens
(entropy 7.6 nats), trigram over ~4 (1.43 nats).

`selective_soft` was similarly underrated: (20.08, 28.71) vs baseline 30.15 at
train ppl 20.2 -- 4.8% better at matched fit, from an arm the table called
"within noise".

### Diagnostics

* `soft_anchor_hit_rate = 0.596`. The true token is independently attested
  elsewhere 60% of the time. It also means trigram applies ~18% LESS effective
  smoothing to the true token than uniform at the same lambda, and still wins --
  which disfavours "it is just more regularisation". The other 40% is where the
  mixture form moves mass off the truth, and is what the one-sided hinge fixes.
* `soft_target_entropy = 1.429` nats -> effective support 4.2 of top-8. The
  m=8 truncation is NOT binding; a bigger m will not help, a different teacher will.
* `endorse_delta = +1.6 to +1.9 nats` on EVERY arm. The model assigns ~1.7 nats
  HIGHER log-probability to tokens continuing a repeated 4-gram. This is the
  Holtzman effect confirmed in our own setup, and it closes naive self-scoring
  with data rather than argument.

### Dream: real gain, wrong mechanism

At matched fit dream beats both controls (29.43 vs 30.21 rep-only, 30.54
random) and fits MORE than baseline while generalising better, which is not
regulariser behaviour. But the degeneration metrics went the wrong way:
`rep4_greedy` 0.751 vs baseline 0.721, while `dream_rep_only` improved it to
0.677. The n-gram judge did nothing for repetition; the repetition judge did.
So dream's perplexity gain is not a degeneration effect.

Two confounds: flag rates differ 9x between dream (11.7%) and its control
(1.3%), and across arms better perplexity CORRELATES with worse greedy
repetition -- better models are more confident and loop more readily under
argmax, so degeneration metrics cannot be read standalone.

### Closed

* **Replay, definitively.** Visit Gini 0.129 and coverage 1.0 -- the
  anti-collapse guards worked perfectly -- and still +0.063 worse, with an
  unstable val curve. Not collapse: gradient bias from non-i.i.d. sampling. At
  74 epochs there is no allocation problem to solve. Three runs, three failure
  modes, consistently negative.
* **Latent targets.** `latent_cos_loss = 0.163` means the predictor SOLVES the
  latent task, and it is the earliest-overfitting arm (best@4250). The task is
  solved and solving it hurts. A pooled next-K target pulls the trunk toward
  smooth low-frequency content -- the same pooling the retrieval line found
  destroys precision. "Pooled values destroy retrieval" appears to extend to
  pooled TARGETS. `latent_shuffle` beat it for the third time, so the VICReg
  term is the only active ingredient.

### Changes

The teacher ladder (see README), matched controls (`shuffled`, `topm_uniform`),
the one-sided hinge, count-adaptive lambda, matched-fit metric, EMA weights,
robust floor, and novelty slices.
