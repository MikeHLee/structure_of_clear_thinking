# Kill Your Favorite Result

*Part 8 of the Structure of Clear Thinking series. One experiment told a beautiful story with a plausible mechanism and marginal statistics. We pre-registered a replication before publicizing it. The story died in two seeds. This post is the corpse, the autopsy, and the finding that survived — because the survival is the point.*

---

## The seducer

[Part 7](07_the_map_is_not_the_mechanism.md) ended on a dissociation: the model's attention contains relational knowledge (one head at 0.90 AUC) that its generation never uses. That suggested an intervention with real stakes: if fine-tuning normally bypasses the attention structure and memorizes downstream, what happens if we *force* the gradient to write to attention instead?

So we ran the generalization split from [part 6](06_teaching_the_rulebook.md) four ways, identical in every respect — same base-model-generated corpus, same batch order, same evaluation streams — except for which weights LoRA was allowed to touch: attention routing (q/k projections), attention read-out (v/o projections), MLPs only, or everything.

Seed one delivered a dream. Read-out-only tuning posted the best trained-region score (96%) *and* nearly double the held-out transfer of MLP tuning (20.0% vs 13.8%), using a quarter of the parameters. Routing-only had the best transfer ratio. All-weights — the default everyone uses — transferred *worst*, as if the gradient, given freedom, preferentially takes the memorization shortcut. It fit part 7's mechanism perfectly. It suggested a mitigation for the KG-underperformance story. It was quotable: *"the knowledge was always there; we tuned the map-reading."*

It also had p ≈ 0.07 on its best contrast, at one seed. We wrote "directional, single seed, replication required" in the commit message, budgeted four dollars, and ran two more seeds before saying anything in public.

## The execution

Seed two: the read-out condition fell to last place. Seed three: all-weights — dead last in seed one — came second. Pooled across three seeds (n=240 per condition), held-out transfer landed at 19.2% / 20.0% / 15.0% / 16.7% across the four conditions. Four-way χ² p = 0.46. The within-condition spread across seeds (the read-out condition alone ranged 12.5%–27.5%) simply swamps the between-condition differences.

Verdict, stated as bluntly as we'd have stated the positive: **at this scale there is no reliable evidence that LoRA placement affects out-of-region rule generalization.** The seed-one ordering was noise wearing a narrative. The numbers from that seed should not be cited as a finding — by anyone, including us. This post exists partly to make quiet resurrection impossible.

Two things soften nothing but deserve saying. The null is *consistent* with the LoRA-placement literature, which already found attention-only tuning unremarkable for in-domain learning (Biderman et al., [arXiv:2405.09673](https://arxiv.org/abs/2405.09673); Thinking Machines' ["LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/)) — we tested a different axis, transfer, and the field's prior survived. And the mechanism from part 7 remains intact; what died is only the claim that weight placement converts it into an intervention.

## What replicated, twelve times out of twelve

Here is what held in every seed and every condition, without exception: the **generalization wall**. Trained regions, 85–92% rule-following. Never-seen regions, 15–20%. No placement, rank, or parameter budget punched through it. Fine-tuning buys regional mastery; it does not buy the abstraction — this is Dziri et al.'s pattern-matching-not-rules result ([arXiv:2305.18654](https://arxiv.org/abs/2305.18654)) with a soundness oracle attached.

Which means the replication *strengthened* the series' core claim while killing its newest one. You cannot place your way to generalizable rule-following. The guarantee has to come from somewhere other than the weights — per-step enforcement and a verifiable receipt ([parts 4](04_building_auditable_ai.md)–[5](05_compiling_programs_into_attention.md)) — and the wall is the standing reason why.

## The autopsy: why this trap catches people

The seed-one result had every property of a maximally dangerous finding: a marginal p-value; a mechanism narrative assembled *before* the statistics, waiting to explain whatever appeared; an outcome we wanted (it validated a mitigation we'd proposed); and small n per cell, where 5-vs-9 successes out of 80 reads as "nearly double." None of these are exotic failures. They are the default conditions of fast-moving ML work.

The countermeasures we used are boring and cheap, which is the argument for them. State the caveat *in the artifact* at the moment of the finding, not in your memory of it — our seed-one commit message is the pre-registration receipt, and this post's claims can be checked against it in the repository history. Decide the replication before the tweet. And budget for it: two seeds cost four dollars and a day, roughly the price of the coffee consumed while over-interpreting seed one. We have been burned before — an earlier project of ours publicly retracted a headline claim after a 50-seed re-run reversed it — and the lesson generalizes: the prettier the story, the earlier the replication.

## Caveats — including on the null

Symmetry demands skepticism of the null too. Three seeds at n=240 per condition can rule out the large effect seed one suggested; they cannot rule out small placement effects, other scales (this is one 1.5B model), other ontology families, or full fine-tuning rather than LoRA. "No reliable evidence at this scale" is the claim. It is enough to kill the headline, and not one word more.

---

*All three seeds, all four conditions, and the pre-registration trail in commit history: [github.com/MikeHLee/structure_of_clear_thinking](https://github.com/MikeHLee/structure_of_clear_thinking). The wall is the finding. Enforcement is the answer.*
