# Thread 04 — "The map is not in the attention"

**Status**: DRAFT — ready for review (post after thread 03)
**Figures**: `figures/` (generated from `results/qk_causal.json`,
`results/qk_region.json`, `results/constraint_ft_generalization.json`)
**Pairs with blog post**: `blog/07_the_map_is_not_the_mechanism.md`

Each post ≤280 characters. Alt text goes in X's image description field.

---

## Post 1 — hook  📎 fig1_soft_graph_exists.png

We went looking for the knowledge graph inside a language model's attention
heads.

We found it: specific heads separate real ontology relations from fake ones
at up to 0.90 AUC — on scrambled text, before ANY training.

Then we deleted those heads. Nothing happened. 🧵

> **Alt text**: Histogram titled "Some attention heads already know the
> ontology — before any training." Distribution of rule-detection AUC across
> Qwen2.5-1.5B's 336 attention heads on scrambled sequences: most heads near
> chance (0.5), a tail of heads up to 0.76 labeled (L10.H11), and one head
> reaching 0.90 on politics/sports relations, annotated as popular
> pretraining topics.

## Post 2 — how we looked

Method in one breath: feed sequences of typed concepts, and check whether
the attention weight between two concepts predicts whether a REAL rule
connects them in the domain's rulebook.

Crucially: we also use scrambled sequences, so word position can't fake the
signal. Only type identity is left.

## Post 3 — credit where due

That attention heads encode relational knowledge isn't new — "knowledge
hub" heads extracting subject–attribute relations are documented
(arXiv:2304.14767, Geva et al.).

What we add: graph-level AUC against a full typed ontology — and then the
part nobody had done: causal deletion + training dynamics.

## Post 4 — the deletion test  📎 fig2_ablation_inert.png

We fine-tuned the model to 100% rule-following, then zeroed out the top
"graph heads."

Rule-following after deleting the 8 best: 99%.
After deleting 8 random heads instead: 99–100%.

The heads that best ENCODE the rulebook contribute nothing measurable to
FOLLOWING it.

> **Alt text**: Bar chart titled "Deleting the 'graph heads' changes
> nothing." Rule-following with the enforcer off: no ablation 100%, delete
> top-8 graph heads 99%, delete top-4 100%, two random-8-head controls 99%
> and 100%.

## Post 5 — training doesn't touch the map

Stranger still: fine-tuning that takes the model from 13% → 100%
rule-following moves the attention map by nothing.

Attention patterns: Δ < 0.02 AUC. Pre-softmax q·k geometry: Δ < 0.01.
Same top heads before and after.

The learning went AROUND the map, not into it.

## Post 6 — no geometric shadow  📎 fig3_no_geometric_shadow.png

The cleanest version: train on some regions of a 127-type merged ontology,
hold others out.

Behavior: trained regions jump 8% → 92%. Held-out: 4% → 16%.
Attention geometry's knowledge of those SAME rules: flat everywhere
(0.50→0.51 trained, 0.47→0.49 held-out).

An 84-point behavioral change with no geometric shadow.

> **Alt text**: Two-panel chart titled "Behavior transformed. The attention
> map didn't move." Left: rule-valid output before/after training — trained
> regions 8% to 92%, never-seen regions 4% to 16%. Right: mean
> rule-detection AUC in attention for the same rules — trained regions 0.50
> to 0.51, never-seen 0.47 to 0.49, both at chance level.

## Post 7 — so where does the skill live?

Where did the learning go? Downstream: MLP and value pathways — the same
place the editing literature localizes factual associations
(ROME, arXiv:2202.05262), and consistent with transformers solving
structured tasks by pattern-matching rather than rules ("Faith and Fate,"
arXiv:2305.18654).

The map is in attention. The habit is elsewhere.

## Post 8 — the dissociation that stings

The sharpest single fact: that 0.90-AUC head knows politician/sports
relations *better than almost anything else in the network* — and the model
generates valid sequences in exactly that domain 4% of the time.

Knowing and doing are different circuits. The read-out is the gap.

## Post 9 — why your KG projects failed

This may explain a common production experience: LLMs reasoning natively
over knowledge graphs underperform (vs SQL, vs schemas).

The internal graph is real but diffuse, generation doesn't consult it, and
fine-tuning doesn't strengthen it.

Structure pays off when enforced OUTSIDE the weights. (Threads 1–2.)

## Post 10 — caveats + links

Caveats: one model (Qwen2.5-1.5B, LoRA), one ontology family (DBpedia),
attention-level probes — value/MLP-side probing is the obvious next
experiment. Deletion + training-dynamics evidence, not a full circuit
analysis.

All code, data, probes: github.com/MikeHLee/structure_of_clear_thinking

---

## Citations & positioning (verified 2026-07-30; post as REPLIES)

**Lead with the inertness, never the existence** — attention heads encoding
relational maps is published (Geva et al., arXiv:2304.14767 "Dissecting
Recall of Factual Associations"; also Geva et al. arXiv:2012.14913 for
MLPs-as-key-value-memories, Hernandez et al. arXiv:2308.09124 for linear
relation decoding). Our novel conjunction: the graph exists (up to 0.90
AUC), is causally inert for trained rule-following, is untouched by
fine-tuning, and the behavioral generalization split has no geometric
counterpart.

**Reply A (mech-interp lineage)** — after Post 7:
"Lineage for the 'where knowledge lives' picture: MLPs as key-value
memories (arXiv:2012.14913), factual associations editable in MLPs (ROME,
arXiv:2202.05262), attention 'knowledge hubs' (arXiv:2304.14767,
@megamor2), relations as linear maps (arXiv:2308.09124). Our contribution
is the causal-inertness + training-dynamics side."

Verified tags: @megamor2 (most on-topic, Post 3 or Reply A), @NeelNanda5
(audience multiplier for the mech-interp framing — tag on Post 1 or 4 only
if the thread stands on its own; don't over-tag).

## Posting notes

- Post 8's dissociation is the thread's most quotable card — consider it as
  the hook if Post 1 tests weak.
- If asked "did you check values/MLP side?": answer honestly — no, that's
  the stated next probe; deletion evidence localizes where the skill ISN'T.
- Post 9 links this thread back to threads 1–2; keep those links handy.
