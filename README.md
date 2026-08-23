# Learning objectives for small-data language modelling

A testbed for the question: **at a fixed and very small token budget, can something
other than plain next-token cross-entropy extract more from the data?**

Corpus: a fixed 1M-token slice of `BabyLM-community/BabyLM-2026-Strict-Small`.
Architecture: a deliberately standard 4-layer pre-norm transformer with RoPE,
identical across every arm. **Nothing about the architecture varies.** If an arm
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

## Pre-registered kill criteria

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
