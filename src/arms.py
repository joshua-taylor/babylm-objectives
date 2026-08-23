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
