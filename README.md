# Learning objectives for small-data language modelling

A testbed for the question: **at a fixed and very small token budget, can something
other than plain next-token cross-entropy extract more from the data?**

Corpus: a fixed 1M-token slice of `BabyLM-community/BabyLM-2026-Strict-Small`.
Architecture: a deliberately standard pre-norm transformer with RoPE, identical
across every arm. Size presets: `tiny` (0.34M params), `small` (1.05M, default),
`base` (3.68M). At 1M tokens `base` memorises — train ppl 17 after 16 epochs — so
`small` with dropout 0.1 is the default working point. **Nothing about the architecture varies.** If an arm
wins, the win is attributable to the objective.

---

## The three motivating problems

Cross-entropy is a proper scoring rule, so the common complaint — "you get punished
for predicting a relevant-but-different token" — is not quite right: you are only
penalised for failing to put mass on what actually occurred, and hedging across
plausible continuations is rewarded. But three real defects hide behind that
intuition, and each gets its own axis here.

| Defect | Axis | Arms |
|---|---|---|
| Every token gets equal gradient weight, so most of a small budget is spent on tokens that were already nearly free | which *tokens* get gradient | `selective` |
| Epochs are allocated uniformly across the corpus regardless of where learning is happening | which *spans* get revisited | `replay_progress` |
| The gradient on non-observed tokens is identity-blind: no partial credit in the direction of the update | what the *error signal* is made of | `latent` |
| The objective itself is single-horizon and left-to-right | replace next-token prediction outright | `anyorder` |
| No graded partial credit: the update knows nothing about which alternatives fit | make the error signal graded, from an EXTERNAL anchor | `ngram_soft` |
| The model is never trained on its own trajectories | dream, judged by external detectors | `dream` |

## The teacher ladder

The organising principle, arrived at after three rounds of results:

> Self-training cannot add information about the target distribution. What it
> CAN do is convert implicit knowledge into explicit knowledge -- but only when
> the generating process is stronger than the student. At 1M parameters there
> is no search and no chain of thought, so the amplifier has to be:
> **at training time we have information the causal student structurally cannot
> access at inference time.**

A teacher is therefore chosen for what it HOLDS, not for being a good language
model. A trigram is a terrible language model. That is not the point; the point
is complementarity.

| rung | what the student lacks | arm |
|---|---|---|
| trigram | cross-position pooling: what followed this exact context elsewhere | `teach_trigram` |
| varorder | adaptive-length exact match — an induction head, in the loss | `teach_varorder` |
| cache | unbounded recency/burstiness beyond the 256-token window | `teach_cache` |
| embed | soft context generalisation: pool over SIMILAR contexts | `teach_embed` |
| class | distributional abstraction: mass for combinations never observed | `teach_class` |
| mixture | labour division across rungs | `teach_mix` |

`class` is the only rung that can propose a (context, token) pair absent from
the corpus. Verified on a synthetic corpus with a deliberately held-out
composition: exact-match teachers assign it exactly 0.0000, while `class` and
`embed` assign it real mass. That is the ten-times-table mechanism, made
measurable.

Diagnose before training: `python run.py --teacher-report all` builds each
teacher and prints coverage, hit rate, effective support and evidence, with no
GPU time spent.

### Loss forms

`--soft-form hinge` penalises `relu(log q - log p)` on the teacher's support:
the teacher raises a floor and never pushes anything down. Motivated by the
same asymmetry as the dream judges — a sparse teacher is reliable when it says
"plausible here" and unreliable when it says "implausible". The measured
trigram hit rate is 0.596, so the mixture form moves mass off the true token
about 40% of the time.

`--soft-adaptive-lambda 1` sets `lam = lam_max * n/(n+kappa)`: Bayesian
shrinkage on the evidence behind each context, replacing the hard `min_count`
cutoff.

## Support-only distillation (the headline test)

Run 4 found that `topm_uniform` — the teacher's candidate set with the
probabilities **thrown away and flattened** — matched or beat the full teacher.
The *identity* of the plausible tokens carried the signal; the probability
structure carried nothing measurable.

That was a trigram teacher. Whether it holds for a **neural** teacher is both
untested and consequential. Classical distillation transfers the full soft
distribution, which is what makes it expensive: you either keep the teacher
resident, or cache a full distribution per token — 128k floats per position at
a modern vocabulary. **If only the support matters, you cache 8 int16 per
token.** Four orders of magnitude, and distillation stops being an
infrastructure project.

```bash
# 1. train a larger teacher on the SAME corpus and cache its top-64
python -m scripts.train_teacher --teacher-size base --n-steps 9000     --cache-dir ./cache

# 2. the three-way comparison
python run.py --stage screen --seeds 5     --arms baseline,teach_neural,teach_neural_probs,teach_neural_shuffled     --teacher-table ./cache/teacher_neural_base_v2048_n1000000.npz
```

| arm | what it does |
|---|---|
| `teach_neural` | teacher's top-k IDs, **flattened** — the novel claim |
| `teach_neural_probs` | the same top-k **with probabilities** — classical distillation |
| `teach_neural_shuffled` | matched shape, wrong context |

The table is cached at K=64 and `--soft-top-m` truncates at load, so the m sweep
is free — no retraining, no rebuilding.

## The cheap control that could collapse the whole ladder

`teach_self` takes the model's **own EMA copy's top-k**, flattens it, and uses
that as the smoothing distribution. No table, no corpus statistics, no external
teacher, and it scales to any model size for one parameter copy.

**This is not the self-scoring trap.** That failed because using a model's own
*probabilities* as a target reduces to an entropy knob — `E_{y~p}[log p(y)]` is
negative entropy. Using its own *support* with flat probabilities is a different
object, and run 4 says support is the part that matters. `teach_self_probs` runs
the probability version alongside, so the two can be separated.

If `teach_self` matches the n-gram teacher, **the ladder is unnecessary** and the
result is a one-line trick. If it does not, the external corpus statistics are
doing real work. Either outcome is worth knowing. Diagnostics `self_hit_rate` and
`self_top1_is_true` report how often the EMA's support contains the true token —
at high epoch counts the EMA has partly memorised the corpus, and that needs
watching.

## Measurement (rebuilt after run 3)

Run 3's noise floor was 0.024 nats, nine times run 2's, and every arm's best
landed at the final eval. Four fixes:

* **matched-fit metric.** Comparing arms at their own minima confounds "learns
  better" with "regularises". `val@train3.00` reads val loss at MATCHED train
  loss. It needs no extra runs — the trace already contains it. On run 3 this
  changed three verdicts.
* **EMA weights** (`--ema-decay`). Under constant LR the model bounces at
  temperature and that noise lands in the metric.
* **robust floor** (`--robust-k`). The minimum over ~36 noisy evals is
  downward-biased in proportion to trajectory noise, and that bias differs
  across arms.
* **novelty slices.** `novel_bigram_nll` is val loss restricted to (prev,
  target) pairs never seen in training where both tokens are individually
  frequent. **Caveat found in run 4: this metric is confounded by global
  confidence.** Plain label smoothing "wins" it by a mile simply by flattening
  the distribution. Read the *gap* (novel minus seen), not the raw number.
* **Welch statistics on the difference.** The old rule — beat baseline by 2x the
  baseline's own standard deviation — ignored the arm's own variance and tested
  a threshold rather than a standard error on the quantity being estimated.
  `--summarise` now reports the mean difference, its standard error, and `t`.

## Bugs fixed after run 4

1. **`--stage screen` ignored `--seeds`** and ran one seed per arm. An entire
   14-arm screen came back `n=1` with "no noise floor yet" despite `--seeds 3`
   being passed. Nothing in that table was testable.
2. **EMA warm-up spike.** The average accumulated from step 0 but evaluation
   switched to it at `--ema-warmup`, so val ppl spiked to 132 at step 750 and
   took ~2000 steps to wash out. The EMA is now reset to the live weights at
   warm-up instead of averaging in the random initialisation.
3. **Cached teachers lost their build-time diagnostics** (`class_singleton_frac`
   came back `nan`), which silently hid the exact degeneracy the diagnostic
   exists to catch. Now saved to a JSON sidecar.
4. **Primary metric is now the robust floor** (mean of the k lowest evals) rather
   than the single minimum, which is downward-biased in proportion to trajectory
   noise — and that bias differs across arms.

## The two earlier axes

**`ngram_soft` — graded partial credit that cannot collapse.** The target becomes
`q = (1-lam)*onehot(y) + lam*ngram_posterior(context)`, where the posterior is a
**leave-one-out** top-m successor distribution from the corpus. Leave-one-out is
the load-bearing detail: without it a context seen once has exactly one successor
(the true token), the soft target collapses back to one-hot, and the mechanism
becomes a memorisation amplifier. With it, the target answers "what ELSE does the
corpus say could follow this context" — which is exactly the graded signal plain
cross-entropy cannot supply. The anchor is external to the weights, so no EMA
encoder, no stop-gradient, no anti-collapse machinery is needed. Controls:
`ngram_soft_uniform` (label smoothing) and `ngram_soft_unigram` (frequency, no
context).

**`dream` — anchored, negative-only.** Naive self-scoring is dead:
`E_{y~p}[log p(y)]` is negative entropy, and a causal model rescoring its own
sample recomputes the identical conditional. An n-gram model breaks the
circularity because it is external. But it is also *weaker* than the network, so
using it as a positive target would teach the model to be more trigram-like. The
usable asymmetry: **a corpus n-gram model is an unreliable judge of what is good
and a reliable judge of what is definitely wrong.** Judges (unseen corpus bigram,
self-repetition) are therefore negative detectors only, applied via unlikelihood.
Positives are always real data; nothing generated is ever a target. Judge this arm
on `rep4_greedy` and `distinct4_greedy`, never on perplexity. Controls:
`dream_rep_only` (the hand-coded penalty) and `dream_random` (same flag rate,
no signal).

---

## Arms, and the control that sits next to each one

The question a reviewer asks is never "did it beat nothing" but "did it beat the
cheap analogue". Every hopeful ships with its most likely alternative explanation.

| Arm | What it tests | Its control | What the control rules out |
|---|---|---|---|
| `selective` | train only on the top-50% of tokens by *excess* surprisal over a trigram reference (RHO-1 in spirit) | `selective_random` | that the gain is just sparsity/regularisation |
| | | `selective_ref` | that the gain is just "train on rare tokens" |
| `replay_progress` | replay spans whose loss is falling fastest (Graves et al. 2017; replay prioritisation) | `replay_hard` | that the gain is just "train on hard spans" |
| `latent` | auxiliary graded target: predict an EMA encoder's summary of the next K tokens (JEPA/BYOL lineage) | `latent_shuffle` | that the gain is the VICReg regularisation, not the prediction task |
| | | `latent_frozen` | that predicting *any* smooth function of future context suffices |
| `anyorder` | absorbing-state masked diffusion (MDLM), bidirectional trunk | `anyorder_matched` | that the difference is the supervision rate, not the objective |
| `mtp` | multi-token prediction | — | known quantity, included for continuity (see confound below) |
| `combo_sel_replay` | do the token and span axes compose? | — | run only if both parents survive |

---

## Pre-registered decision rule

Fix this before looking at any result.

```
noise_floor  = std of `baseline` best val loss across >= 3 seeds
INTERESTING  = arm beats baseline by more than 2 x noise_floor
REAL         = INTERESTING and also beats its own control by more than 2 x noise_floor
otherwise    = refuted, logged, closed
```

`python run.py --summarise` applies this rule mechanically against the register.
It is deliberately not a judgement call.

## Sanity gates (added after run 1 shipped a broken split — see RESULTS.md)

Three things print at startup and are checked automatically. They exist because
the first version of this repo silently made the validation set a *different
domain* from training, every arm scored at or above the uniform-random ceiling,
and the harness cheerfully ranked twelve arms against the resulting noise.

| gate | what it catches |
|---|---|
| unigram `KL(val \|\| train)` | distribution mismatch between the splits |
| trigram anchor (fit train, score val) | a model that has learned nothing transferable — it must beat this |
| uniform-random ceiling `log(vocab)` | val loss at or above it means the run is broken, printed inline |

Any of these failing writes a `confound` into the register row automatically, and
`--summarise` surfaces it under the arm. **A run that fails a gate produces no
usable comparison, however clean the loss curve looks.**

## Pre-registered kill criteria

0. **Sampler collapse (`replay_*`).** Prioritised replay has a positive feedback
   loop: a span whose loss is falling gets sampled more, which makes it fall
   further. Run 1 collapsed onto ~1% of spans and looked like a win. Guards:
   a hard visit cap at `--replay-max-visit-ratio` x the uniform rate, a UCB
   novelty bonus, and staleness decay on progress estimates. `replay_visit_gini`
   is logged and flags a confound above `--replay-gini-warn`.
1. **Representation collapse (`latent`).** Any objective whose targets come from
   the network itself has a trivial solution: constant representations, zero
   error, nothing learned. The effective rank of the *targets* is checked every
   `--diag-every` steps. Two consecutive readings below `--collapse-erank-frac`
   (default 0.05 of `d_model`) abort the run and log `outcome=killed`, regardless
   of what the loss curve is doing.
2. **Budget parity.** Every arm gets identical `steps x batch x seq_len` token
   visits, printed at startup. A replay arm reallocates epochs; it never gets more
   data. `anyorder_matched` is the one deliberate exception and is labelled as such.
3. **Degeneration, not perplexity.** If an arm's benefit is fewer repetitions,
   perplexity will not show it. `rep4` / `rep8` / `distinct-n` are logged for
   every autoregressive arm.

## Confounds stated up front

- **`mtp` measures convergence speed, not generalisation** at a fixed step budget.
  This project has already been burned by exactly this. Judge it on best-achievable
  val loss and on the train/val gap, never on fixed-step perplexity.
- **`anyorder`'s loss is a NELBO bound**, always looser than an AR likelihood. It is
  logged with `loss_unit=nats_bound` and the summariser refuses to rank it against
  the AR arms. Compare diffusion arms to each other.
- **`anyorder` supervises ~50% of positions per step** (E[t]=0.5) where AR arms
  supervise 100%. Run both `anyorder` and `anyorder_matched`; the gap between them
  is itself the result.
- **The trigram reference for `selective` is fit on the corpus it scores.**
  Leave-one-out counts damp this but it is not a held-out estimate.
- **The train/val gap is the real ceiling** on data this small. Expect every arm
  to overfit; watch `gap_bits` as closely as `best_val_ppl`.

---

## Running it

```bash
pip install -r requirements.txt

# 0. never spend GPU time on unsmoked code (CPU, ~10s, no download)
python -m scripts.smoke

# 1. noise floor first. nothing is interpretable without it.
python run.py --stage noisefloor

# 2. screen every arm at one seed
python run.py --stage screen

# 3. apply the decision rule
python run.py --summarise

# 4. confirm survivors at 3 seeds
python run.py --arm selective --seeds 3
```

Useful flags: `--n-steps` (budget), `--n-tokens` (corpus size), `--lr-schedule constant`
(when looking for a true floor rather than a fixed-budget number), `--synthetic 1`
(validate the loop offline; never a result).

Results land in `./results/*.json` and append to
`./results/experiments_objectives.csv`, which uses the same 51-column schema as
the project's `experiments_2.csv` so these rows can be concatenated onto the
existing register. Arc id: `O_objectives_babylm1m`.

---

## A note on what is *not* here

An earlier design considered a "dreaming" arm: sample rollouts, have the model
score them, and train on its own assessment. It was dropped after working through
two problems. First, `E_{y~p} log p(y)` is exactly negative entropy — a purely
self-referential objective is an entropy knob, not a learning signal, unless
real data anchors one side of it. Second, a causal model rescoring its own
sample recomputes the identical conditional it used when generating; there is no
second opinion. And empirically, models grow *more* confident inside repetition
loops (Holtzman et al. 2020), so self-scoring fails precisely on the failure
mode it was meant to catch.

The diagnostic that would have decided it is still shipped, in
`src/diagnostics.py::self_endorsement`: it measures whether tokens continuing a
repeated n-gram receive higher log-probability than non-repeated ones. A positive
`endorse_delta` confirms self-scoring is dead. It runs automatically on every AR
arm and costs nothing.
