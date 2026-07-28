# Thread 03 — "Teaching the model the rulebook"

**Status**: SCAFFOLD — copy structure final, numbers are `{{PLACEHOLDER}}` until
the constraint-aware fine-tuning runs land. DO NOT POST until every
placeholder is replaced with a measured value and figures are regenerated
from the real results file.

**Depends on**: the fine-tuning phase (SOGB-style Modal scale-up: CPU smoke →
L4 @ ~1.5B → A100 @ 7–8B). Experiment spec: `Percepta_Transformer_VM/`
`experiment_results.md` § "Next experiments" #2 (constraint-aware
fine-tuning) — train the mis-calibrated model on trajectories generated
under per-step enforcement; track disagreement (KL/step), fluency (logP),
and enforcement work (forced steps) across training.

**Figures**: `generate_figures.py` reads `results/constraint_ft_results.json`
(schema in the script docstring). `--placeholder` mode renders
layout previews watermarked "PLACEHOLDER — ILLUSTRATIVE SHAPE ONLY".

**Pairs with blog post**: new post (working title: "Teaching the Model the
Rulebook") — to be drafted with the results.

---

## Post 1 — hook  📎 fig1_disagreement_falls.png

Previously: we can force an AI to follow a domain's rules — 100% valid
output, guaranteed, with a receipt.

The catch: when the model disagrees with the rulebook, forcing costs
fluency.

New result: you can train that cost away — and keep the guarantee the whole
time. {{HEADLINE_STAT}} 🧵

> **Alt text**: Line chart titled "Training on rule-checked outputs teaches
> the model the rules." Model–rulebook disagreement (KL per step) falls from
> {{KL_START}} to {{KL_END}} over {{N_STEPS}} fine-tuning steps.

## Post 2 — recap for new readers

Quick recap (full thread linked at the end):

A rulebook (ontology) lists the only legal steps in a domain. Checking every
generated step against it guarantees valid output — but when the model's
habits clash with the rules, enforcement overrides it constantly, and
fluency pays: ~5.5 nats/sequence in our worst synthetic case.

## Post 3 — the idea

The fix is almost embarrassingly simple:

1. Run the model WITH enforcement on. Every output is valid.
2. Fine-tune the model on its own enforced outputs.
3. The model internalizes the rulebook — so enforcement has less and less to
   override.

The guardrail generates its own training data.

## Post 4 — disagreement falls  📎 fig1_disagreement_falls.png

We fine-tuned {{MODEL_NAME}} ({{MODEL_SIZE}}) this way on {{DOMAIN_DESC}}.

Disagreement with the rulebook (KL per step — how hard enforcement must
reshape the model's choices) fell {{KL_DROP_PCT}} over {{N_STEPS}} steps:
{{KL_START}} → {{KL_END}}.

The model is learning the rules, not just being fenced in by them.

## Post 5 — fluency recovers  📎 fig2_fluency_recovers.png

And the fluency price of enforcement? {{LOGP_RECOVERY_SUMMARY}}

Before training: {{LOGP_BEFORE}} logprob/sequence under enforcement.
After: {{LOGP_AFTER}} — {{LOGP_RECOVERY_PCT}} of the gap to the
well-calibrated frontier closed.

Soundness during ALL of this: 100%. The guarantee never lapsed.

## Post 6 — the interesting probe  📎 fig3_enforcer_off.png

The probe we find most interesting: turn the enforcer OFF and test the
trained model bare.

Before training: {{SOUNDNESS_OFF_BEFORE}} of outputs were fully valid.
After: {{SOUNDNESS_OFF_AFTER}}.

The model absorbed the rulebook into its weights. (We still ship with the
enforcer on — that's what makes the guarantee a guarantee.)

## Post 7 — why this matters for small models

Why this matters: the rulebook + enforcement carry the reliability, so the
model doesn't have to be huge.

A {{MODEL_SIZE}} model with a rulebook gives you something no frontier model
gives you bare: provably valid domain behavior, with a verifiable receipt
per output — {{SMALL_VS_BASELINE_COMPARISON}}.

## Post 8 — why this matters for frontier models

And in the other direction: enforcement lives at decoding time. No
from-scratch retraining, no architecture surgery.

That means auditable action & reasoning traces can be layered onto models
that already exist — the certificate says exactly which rule licensed each
step of a conclusion.

## Post 9 — the honest caveats

Caveats, because they matter:

• "Valid" = valid under the rulebook. The guarantee is exactly as good as
the ontology you wrote. That's a feature (it's inspectable) and a
responsibility.
• Results are {{MODEL_SIZE}}, {{N_DOMAINS}} domain(s), seed(s) {{SEEDS}}.
{{SIGNIFICANCE_CAVEAT}}

## Post 10 — links

Everything is open and reproducible:

Repo: github.com/MikeHLee/structure_of_clear_thinking
Training + eval code: {{TRAINING_SCRIPT_PATH}}
Thread 1 (why per-step checking beats attention masking): {{THREAD1_LINK}}
Paper: {{ARXIV_LINK}}

---

## Placeholder inventory (all must be filled before posting)

| Token | Source |
|-------|--------|
| `{{HEADLINE_STAT}}` | strongest single measured number, chosen after runs |
| `{{KL_START}} / {{KL_END}} / {{KL_DROP_PCT}} / {{N_STEPS}}` | training curve, `constraint_ft_results.json` |
| `{{MODEL_NAME}} / {{MODEL_SIZE}}` | e.g. Qwen2.5-1.5B on L4 first; 7–8B on A100 later |
| `{{DOMAIN_DESC}}` | e.g. "a 7-type e-commerce ontology" (+ any added Ologs) |
| `{{LOGP_BEFORE}} / {{LOGP_AFTER}} / {{LOGP_RECOVERY_PCT}} / {{LOGP_RECOVERY_SUMMARY}}` | fluency eval |
| `{{SOUNDNESS_OFF_BEFORE}} / {{SOUNDNESS_OFF_AFTER}}` | enforcement-off eval |
| `{{SMALL_VS_BASELINE_COMPARISON}}` | only if a larger-model baseline is actually run; otherwise cut Post 7's final clause |
| `{{N_DOMAINS}} / {{SEEDS}} / {{SIGNIFICANCE_CAVEAT}}` | honesty block — state n, seeds, and "single seed, directional" if that's the truth |
| `{{TRAINING_SCRIPT_PATH}} / {{THREAD1_LINK}} / {{ARXIV_LINK}}` | filled at posting time |

## Posting notes

- Keep the SOGB honesty discipline: if a delta is within noise, say so in
  Post 9 rather than rounding up the claim.
- If the enforcement-off soundness (Post 6) doesn't improve, that is itself
  a publishable finding — reframe Post 6 as "the rules don't transfer into
  weights; enforcement remains load-bearing" and keep the thread.
- Alt text for figs 2–3 follows the fig 1 pattern: title + the two numbers.
