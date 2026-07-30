# Thread 03 — "Teaching the model the rulebook"

**Status**: DRAFT — numbers filled 2026-07-28 from the completed L4 run
(`results/constraint_ft_results.json`, Qwen2.5-1.5B + LoRA, Modal).
Figures regenerated from the real results file (no placeholders).
**Depends on**: nothing — run complete. A dense-checkpoint re-run (~$1.30,
see posting notes) would improve figs 1–2 before posting.
**Pairs with blog post**: `blog/06_teaching_the_rulebook.md`

Each post is ≤280 characters. Attach the figure named under the post; alt
text goes in X's image description field.

---

## Post 1 — hook  📎 fig3_enforcer_off.png

Previously: we can force an AI to follow a domain's rules — 100% valid
output, guaranteed, with a receipt.

The catch: forcing costs fluency when the model disagrees with the rules.

New result: a 1.5B model trained the cost away entirely — 0% → 100%
rule-following even with the enforcer OFF. 🧵

> **Alt text**: Bar chart titled "The rulebook transfers into the weights."
> Percent of outputs fully valid with the enforcer switched off, for
> Qwen2.5-1.5B on a 7-type e-commerce ontology: 0.0% before fine-tuning,
> 100.0% after. Caption notes the shipped configuration keeps the enforcer
> on.

## Post 2 — recap for new readers

Quick recap (thread 1 linked at the end):

A rulebook (ontology) lists the only legal steps in a domain. Checking every
generated step against it guarantees valid output — but when the model's
habits clash with the rules, the enforcer overrides it constantly, and
fluency pays the price.

## Post 3 — the idea

The fix is almost embarrassingly simple:

1. Run the model WITH enforcement on. Every output is valid.
2. Fine-tune the model on its own enforced outputs.
3. The model internalizes the rulebook — so enforcement has less and less to
   override.

The guardrail generates its own training data.

## Post 4 — disagreement falls  📎 fig1_disagreement_falls.png

We fine-tuned Qwen2.5-1.5B (LoRA) this way on a 7-type shop ontology.

Disagreement with the rulebook — how hard enforcement must reshape the
model's choices, measured as KL per step — fell 0.95 → 0.00.

And it didn't take long: converged before the first checkpoint at step 250.

> **Alt text**: Line chart titled "Training on rule-checked outputs teaches
> the model the rules." Model–rulebook disagreement (KL per generation step)
> for Qwen2.5-1.5B falls from 0.95 at step 0 to 0.00 by the first checkpoint
> at step 250, and stays at zero through step 2000.

## Post 5 — the fluency cost vanishes  📎 fig2_fluency_recovers.png

The fluency price of enforcement went from −11.4 log-prob per sequence to
−0.0: 100% of the gap to the aligned frontier, closed.

Output validity during ALL of this: 100%. The guarantee never lapsed — the
enforcer just ran out of things to correct.

> **Alt text**: Line chart titled "The fluency cost of enforcement trains
> away." Log-probability per enforced sequence rises from −11.4 before
> training to 0.0 by step 250, meeting the dashed "well-calibrated frontier"
> reference line, and stays there through step 2000.

## Post 6 — what "learned the rules" means here

Full honesty about what happened: the model didn't just improve — it
SATURATED this small rulebook. It now walks legal paths deterministically.

Great for rule-following. But it means this domain is too easy to test
fluency claims. (That's what the next run — bigger, harder ontologies — is
for.)

## Post 7 — why this matters for small models

The pattern that matters: rulebook + enforcement carry the reliability, so
the model doesn't have to be huge.

A 1.5B model — small enough to run on a laptop — now gives provably valid
domain behavior with a verifiable receipt per output (thread 2).

Reliability from structure, not scale.

## Post 8 — why this matters for frontier models

And in the other direction: enforcement lives at decoding time. No
from-scratch retraining, no architecture surgery.

Auditable action & reasoning traces can be layered onto models that already
exist — and this result says the fluency tax of doing so is trainable to
zero.

## Post 9 — the honest caveats

The caveats, because they're load-bearing:

• Single seed, one small domain (7 types), 1.5B model. Mechanism
demonstrated; generality not yet.
• The domain saturates — a 135M model also mastered it. Harder ontologies
next.
• "Valid" = valid under the rulebook. The guarantee is as good as the rules
you wrote.

## Post 10 — links

Everything is open and reproducible:

Repo: github.com/MikeHLee/structure_of_clear_thinking
Training + eval code: Percepta_Transformer_VM/experiment_constraint_ft.py
Results: results/constraint_ft_results.json
Thread 1 (why per-step checking beats attention masking): [LINK]
Paper: [ARXIV LINK TBD]

---

## Numbers provenance (filled 2026-07-28)

All from `results/constraint_ft_results.json` (L4 run, wall time 3h53m):
KL/step 0.954 → 0.000 · logP −11.444 → −0.000 (frontier 0.000) ·
enforcer-off soundness 0.0% → 100.0% · seed 42 · 2000 LoRA steps
(converged ≤250) · eval N=100/checkpoint, soundness N=300.

## Posting notes

- **Recommended before posting**: a dense-checkpoint re-run (eval every 10
  steps for the first 150, smaller eval N) so figs 1–2 show the actual
  learning curve instead of a cliff between two checkpoints. ~$1.30 on L4.
  The current figures are honest but coarse.
- Post 6 is the determinism/mode-collapse disclosure — do not soften it; it
  pre-empts the sharpest technical objection to the fluency claim.
- Post after threads 01 and 02; add both links.
- **Generalization nuance (added after the merged-graph split)**: "the
  rulebook transfers into the weights" (Posts 1/6, fig 3) is true only
  region-locally — on a merged 19-ontology graph, trained regions hit 92.5%
  enforcer-off soundness while held-out regions stay near baseline (15.8%).
  Fold this into Post 6/9 or make it the bridge to thread 04. Do not post
  the "transfers into weights" framing without this qualifier.

## Citations & positioning (verified 2026-07-30; post as REPLIES)

The COST is known; the CURE is the contribution. Tam et al.
(arXiv:2408.02442) showed format constraints degrade reasoning — cite it
before showing the KL curve. CRANE (arXiv:2502.09061) mitigates by
*relaxing* the constraint; we train *through* it with the guarantee never
lapsing. Process-reward contrast: Lightman et al., "Let's Verify Step by
Step" (arXiv:2305.20050) *scores* steps statistically; enforcement
*forbids* invalid steps by construction. Memorization-not-abstraction has
published backbone: Dziri et al., "Faith and Fate" (arXiv:2305.18654) —
frame our region split as a clean quantified instance with a soundness
oracle, not a discovery of the phenomenon.

**Reply A (cost lineage)** — after Post 5:
"The constraint-costs-fluency effect is documented (arXiv:2408.02442), and
prior mitigations relax the constraint to recover it (arXiv:2502.09061).
The alternative shown here: keep the constraint, train on its outputs —
the cost measurably goes to ~zero and the guarantee never lapses."

No verified author handles for this thread's citations — cite papers only.
