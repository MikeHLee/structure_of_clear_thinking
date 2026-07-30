# Thread 05 — "The result that died in replication (and what survived)"

**Status**: DRAFT — ready for review (post after thread 04)
**Figures**: `figures/` (generated from `results/routing_ft_results*.json`,
3 seeds)
**Pairs with blog post**: new post to draft (working title: "Kill Your
Favorite Result")

Each post ≤280 characters. Alt text goes in X's image description field.

---

## Post 1 — hook  📎 fig1_effect_dissolves.png

Last week, one experiment told a beautiful story: restrict fine-tuning to
ATTENTION weights and the model generalizes rules ~2× better than tuning
MLPs.

Great mechanism. Great narrative. p≈0.07.

We pre-registered a replication before posting it. The story died. Posting
the corpse anyway. 🧵

> **Alt text**: Chart titled "One seed looked exciting. Three seeds say: no
> effect." Four conditions (attention routing q/k, attention read-out v/o,
> MLPs only, all weights) with three seed dots each and pooled bars at
> 19.2%, 20.0%, 15.0%, 16.7% rule-following on never-trained regions. The
> within-condition spread visibly exceeds the between-condition differences;
> chi-squared p = 0.46.

## Post 2 — the setup

The question: when a model learns a rulebook, does it matter WHERE the
gradient is allowed to write?

4 conditions — attention routing (q/k), attention read-out (v/o), MLPs
only, everything — trained on the IDENTICAL corpus in identical batch
order, scored on rule-following in regions of the ontology the model never
saw.

## Post 3 — seed 42, the seducer

First seed: read-out-only tuning hit 96% on trained regions AND doubled
transfer to unseen regions vs MLP tuning, with 4× fewer parameters. Clean
mechanistic story to go with it (thread 4's inert-knowledge head).

We wrote "directional, single seed, replication required" — and meant it.

## Post 4 — seeds 43 and 44

The replication: same everything, new seeds.

Seed 43: the read-out condition fell to LAST place.
Seed 44: the all-weights condition — dead last in seed 42 — came SECOND.

Pooled (n=240/condition): 19.2 / 20.0 / 15.0 / 16.7%. χ² p=0.46.

The ordering was noise wearing a narrative.

## Post 5 — the verdict

At this scale, there is NO reliable evidence that LoRA placement affects
out-of-region rule generalization.

The seed-42 numbers should never be cited as a finding. (This thread exists
partly so they can't be quietly resurrected later — including by us.)

## Post 6 — what replicated, 12 for 12  📎 fig2_generalization_wall.png

What held in every seed × every condition: the generalization wall.

Trained regions: 85–92% rule-following. Never-seen regions: 15–20%.
No placement, no rank, no parameter budget punched through it.

Fine-tuning buys regional mastery. It does not buy the abstraction.

> **Alt text**: Grouped bar chart titled "What replicates in every run: the
> generalization wall." For all four conditions, trained-region
> rule-following is 85–92% while never-seen-region rule-following is
> 15–20%. Twelve runs total across three seeds.

## Post 7 — the literature agrees, from the other side

Fair notice: the LoRA-placement literature (arXiv:2405.09673 "LoRA Learns
Less and Forgets Less"; Thinking Machines' "LoRA Without Regret") already
found attention-only tuning is NOT special for in-domain learning.

We measured a different axis — out-of-region transfer — and got the same
null. The field's prior survives our test.

## Post 8 — the meta-lesson

The trap, spelled out: a single-seed result with p≈0.07, a plausible
mechanism, and a story you WANT to be true is precisely the result most
likely to fool you.

The replication cost $4 of GPU time and one day. Cheapest insurance in ML.

## Post 9 — what it means for the program

The conclusion this strengthens: you cannot PLACE your way to generalizable
rule-following. The wall stands wherever the gradient writes.

Which is why the guarantee in this series never lived in the weights: it
lives in per-step enforcement + a verifiable receipt (threads 1–2).

## Post 10 — links

Full data — all three seeds, all four conditions, the pre-registration
trail in the commit history:

github.com/MikeHLee/structure_of_clear_thinking

Threads 1–4 linked below. The wall is the finding. Enforcement is the
answer.

---

## Citations & positioning (verified 2026-07-30)

- The null is CONSISTENT with Biderman et al. (arXiv:2405.09673) and
  "LoRA Without Regret" (thinkingmachines.ai/blog/lora/) — cite both in
  Post 7; frame as "the field's prior survives a transfer-axis test," not
  as a contradiction.
- The wall (Post 6) has published backbone: Dziri et al. "Faith and Fate"
  (arXiv:2305.18654) — memorization/pattern-matching over systematic rules.
  Add as a reply if pressed: "the wall is a clean quantified instance with
  a soundness oracle, not a new phenomenon."
- No verified author handles for this thread's citations — cite papers, do
  not guess handles.

## Posting notes

- This thread's credibility IS the product: do not soften Posts 4–5. The
  explicit "including by us" self-binding in Post 5 is deliberate.
- Post 3 must match what we actually wrote at the time (it does — the
  seed-42 commit says "directional, needs seeds"). The commit history is
  the receipt; that's on-brand for the series.
- If engagement asks "so does placement matter at 7B/70B?": honest answer —
  unknown; effects could exist at other scales; this is 1.5B, n=240,
  one ontology family.
