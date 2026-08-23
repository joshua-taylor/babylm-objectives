"""
Arm presets.

An "arm" is a (sampler, loss, overrides) triple. Every hopeful arm ships with
the ablation that is its most likely alternative explanation, because the
question a reviewer asks is never "did it beat nothing" but "did it beat the
cheap non-biological analogue".

  hopeful              its control
  -------              -----------
  replay_progress      replay_hard      (is it progress, or just difficulty?)
  selective            selective_random (is it the signal, or just sparsity?)
                       selective_ref    (is it excess, or just rarity?)
  latent               latent_shuffle   (is it the target, or just VICReg?)
                       latent_frozen    (is it learned targets, or any smooth fn?)
  anyorder             anyorder_matched (is it the objective, or the supervision rate?)
  mtp                  --               (known quantity; convergence-speed confound)
"""

ARMS = {
    # ---- baseline -------------------------------------------------------
    "baseline": dict(sampler="uniform", loss="ntp", is_baseline=1,
                     hypothesis="Reference next-token cross-entropy. Defines the noise floor."),

    # ---- axis 1: which tokens get gradient ------------------------------
    "selective": dict(sampler="uniform", loss="selective", selective_mode="excess",
                      hypothesis="Training only on tokens with high excess surprisal over a "
                                 "trigram reference concentrates a fixed gradient budget on "
                                 "contextual prediction and improves val loss per token."),
    "selective_random": dict(sampler="uniform", loss="selective", selective_mode="random",
                             hypothesis="CONTROL: same token sparsity, no signal. Isolates the "
                                        "effect of dropping tokens from the effect of choosing them."),
    "selective_ref": dict(sampler="uniform", loss="selective", selective_mode="refhigh",
                          hypothesis="CONTROL: select by raw n-gram surprisal only (rarity), "
                                     "ignoring the model. Isolates 'excess' from 'rare'."),

    # ---- axis 2: which spans get revisited ------------------------------
    "replay_progress": dict(sampler="progress", loss="ntp",
                            hypothesis="Replaying spans whose loss is falling fastest allocates "
                                       "epochs where learning is actually happening and beats "
                                       "uniform revisiting at a matched token-visit budget."),
    "replay_hard": dict(sampler="hard", loss="ntp",
                        hypothesis="CONTROL: replay by highest loss instead of fastest progress. "
                                   "If this matches replay_progress, the learning-progress "
                                   "framing has bought nothing."),

    # ---- axis 3: what the error signal is made of -----------------------
    "latent": dict(sampler="uniform", loss="latent", latent_target="ema",
                   hypothesis="An auxiliary graded latent target (EMA encoder, next-K summary) "
                              "supplies semantic partial credit that token cross-entropy cannot, "
                              "improving val loss without collapsing representations."),
    "latent_shuffle": dict(sampler="uniform", loss="latent", latent_target="shuffle",
                           hypothesis="CONTROL: targets shuffled across the batch. Any gain here "
                                      "is the VICReg regularisation, not the prediction task."),
    "latent_frozen": dict(sampler="uniform", loss="latent", latent_target="frozen",
                          hypothesis="CONTROL: targets from a frozen random encoder. Tests whether "
                                     "predicting ANY smooth function of future context suffices."),

    # ---- axis 4: replace next-token prediction outright -----------------
    "anyorder": dict(sampler="uniform", loss="anyorder", causal=0,
                     hypothesis="Any-order masked diffusion trained at a matched token-visit "
                                "budget reaches a competitive NELBO bound and better infilling "
                                "than autoregressive cross-entropy."),
    "anyorder_matched": dict(sampler="uniform", loss="anyorder", causal=0,
                             match_supervision=1,
                             hypothesis="CONTROL: same arm at matched SUPERVISED-position count "
                                        "(2x steps). Separates the objective from the fact that "
                                        "diffusion supervises ~50%% of positions per step."),

    # ---- THE TEACHER LADDER ---------------------------------------------
    # Ordered by what the causal student structurally LACKS, not by teacher
    # quality. A trigram is a terrible language model; that is not the point.
    "teach_trigram": dict(sampler="uniform", loss="soft", teacher="trigram",
        hypothesis="Cross-position pooling: what followed this exact context elsewhere. "
                   "Rao-Blackwellises the training target, reducing gradient variance "
                   "where contexts are rare."),
    "teach_varorder": dict(sampler="uniform", loss="soft", teacher="varorder",
        hypothesis="Adaptive-order exact match (infini-gram). This is an induction "
                   "head's computation moved from the architecture into the loss; if it "
                   "beats trigram, exact-match retrieval helps in either location."),
    "teach_cache": dict(sampler="uniform", loss="soft", teacher="cache",
        hypothesis="Unbounded recency. Burstiness over a window far wider than the "
                   "student's 256 tokens, which it structurally cannot see."),
    "teach_embed": dict(sampler="uniform", loss="soft", teacher="embed",
        hypothesis="Soft context generalisation. Pools successors over SIMILAR contexts, "
                   "attacking the ~40%% of positions where exact match has no attestation "
                   "for the true token."),
    "teach_class": dict(sampler="uniform", loss="soft", teacher="class",
        hypothesis="Distributional abstraction. The only rung that can assign mass to a "
                   "(context, token) combination absent from the corpus. Judged on "
                   "novel_bigram_nll, not aggregate perplexity."),
    "teach_mix": dict(sampler="uniform", loss="soft", teacher="mix:varorder+cache+class",
        hypothesis="Labour division: exact match is precise but silent where it lacks "
                   "attestation; abstraction is never silent but always vague. Do the "
                   "gains ADD?"),

    # ---- matched controls (the entropy-mismatched uniform control was wrong) --
    "teach_shuffled": dict(sampler="uniform", loss="soft", teacher="shuffled:varorder",
        hypothesis="CONTROL: identical shape, entropy, support size and count statistics; "
                   "wrong context. This is the control that isolates the claim."),
    "teach_topm_uniform": dict(sampler="uniform", loss="soft", teacher="topm_uniform:varorder",
        hypothesis="CONTROL: right support, flat probabilities. Separates WHICH tokens are "
                   "plausible from HOW plausible each is."),
    "teach_unigram": dict(sampler="uniform", loss="soft", teacher="unigram",
        hypothesis="CONTROL: context-free corpus frequency."),
    "teach_uniform": dict(sampler="uniform", loss="soft", teacher="uniform",
        hypothesis="CONTROL: classic label smoothing."),

    # ---- loss-form variants on the best rung ------------------------------
    "teach_hinge": dict(sampler="uniform", loss="soft", teacher="varorder", soft_form="hinge",
        hypothesis="One-sided: the teacher raises a floor and never pushes anything down. "
                   "Removes the weak-teacher ceiling and the 40%%-miss failure mode."),
    "teach_adaptive": dict(sampler="uniform", loss="soft", teacher="varorder",
        soft_adaptive_lambda=1,
        hypothesis="Count-adaptive lambda: trust the teacher in proportion to the evidence "
                   "behind it, replacing the hard min_count cutoff."),
    "teach_best": dict(sampler="uniform", loss="soft", teacher="mix:varorder+class",
        soft_form="hinge", soft_adaptive_lambda=1,
        hypothesis="Everything that survived: mixture teacher, one-sided hinge, "
                   "count-adaptive lambda."),

    # ---- back-compat aliases ---------------------------------------------
    "ngram_soft": dict(sampler="uniform", loss="soft", teacher="trigram",
                       hypothesis="alias of teach_trigram (run-3 configuration)"),
    "ngram_soft_uniform": dict(sampler="uniform", loss="soft", teacher="uniform",
                               hypothesis="alias of teach_uniform"),
    "ngram_soft_unigram": dict(sampler="uniform", loss="soft", teacher="unigram",
                               hypothesis="alias of teach_unigram"),

    # ---- axis 6: dreaming, anchored, negative-only -----------------------
    "dream": dict(sampler="uniform", loss="dream", dream_judges="ngram,rep",
                  hypothesis="Rollouts judged by EXTERNAL cheap detectors (unseen corpus "
                             "bigram, self-repetition) and penalised via unlikelihood "
                             "reduce degeneration without costing val loss. Judged on "
                             "rep4_greedy/distinct4_greedy, NOT on perplexity."),
    "dream_rep_only": dict(sampler="uniform", loss="dream", dream_judges="rep",
                           hypothesis="CONTROL: the hand-coded repetition penalty a "
                                      "reviewer will ask for. If it matches `dream`, the "
                                      "n-gram judge contributed nothing."),
    "dream_random": dict(sampler="uniform", loss="dream", dream_judges="random",
                         hypothesis="CONTROL: flag the same number of positions at random. "
                                    "Isolates 'unlikelihood on anything' from "
                                    "'unlikelihood on judged-bad'."),

    # ---- fix for the run-2 selective failure -----------------------------
    "selective_soft": dict(sampler="uniform", loss="selective", selective_mode="excess",
                           selective_weighting="soft",
                           hypothesis="Run 2: hard-masking 50%% of tokens cost 0.105 nats "
                                      "while the excess-surprisal signal recovered only "
                                      "0.016. Reweight instead of discard: keep every "
                                      "token, mean weight 1.0, so the signal is free."),

    # ---- known quantity --------------------------------------------------
    "mtp": dict(sampler="uniform", loss="mtp",
                hypothesis="Multi-token prediction as auxiliary loss. Prior work in this project "
                           "shows it slows fixed-step convergence; judged on best-achievable val "
                           "loss and on the train/val gap, not fixed-step perplexity."),

    # ---- composition (only if both parents survive stage 2) --------------
    "combo_sel_replay": dict(sampler="progress", loss="selective", selective_mode="excess",
                             hypothesis="Do the token-selection and span-selection axes compose, "
                                        "or do they target the same variance?"),
}


def apply_arm(args, arm_name):
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm {arm_name!r}; choose from {sorted(ARMS)}")
    spec = dict(ARMS[arm_name])
    args.sampler = spec.pop("sampler")
    args.loss = spec.pop("loss")
    args.is_baseline = spec.pop("is_baseline", 0)
    args.hypothesis = spec.pop("hypothesis", "")
    for k, v in spec.items():
        setattr(args, k, v)
    return args
