# Teaching the Model the Rulebook

*Part 6 of the Structure of Clear Thinking series. We fine-tuned a language model on its own rule-enforced outputs. The fluency cost of enforcement trained away to zero, the guarantee never lapsed — and then a bigger experiment showed us exactly what the model had and hadn't learned.*

---

## Where we left off

Earlier in this series we established two things. First, checking every generated step against a domain rulebook — an ontology of types and the legal relations between them — gives you output that is valid *by construction*: 100% of sequences follow the rules, because illegal steps cannot be emitted ([part 5](05_compiling_programs_into_attention.md); the attention-masking shortcut fails, and we measured how). Second, every output can carry a receipt: a small JSON certificate that anyone can re-verify with the rulebook alone ([part 4](04_building_auditable_ai.md)).

One objection hangs over both results: *doesn't forcing a model to follow rules it didn't choose make it worse?* The constrained-generation literature says yes — format restrictions measurably degrade reasoning performance (Tam et al., [arXiv:2408.02442](https://arxiv.org/abs/2408.02442)). Prior mitigations relax the constraint to recover fluency (CRANE, [arXiv:2502.09061](https://arxiv.org/abs/2502.09061)). We wanted a different answer: keep the constraint at full strength, and train the disagreement away.

## The loop: the guardrail generates its own training data

The scheme is almost embarrassingly simple:

1. Run the model with per-step enforcement ON. Every output is valid.
2. Fine-tune the model (LoRA) on its own enforced outputs.
3. The model internalizes the rulebook — so enforcement has less and less to override.

We can watch this happen quantitatively, because enforcement gives us a natural gauge: the KL divergence per step between the model's raw next-step distribution and the rule-masked one. High KL means the rules are doing violence to the model's preferences; zero KL means the model already agrees with the rulebook.

On a small e-commerce ontology, fine-tuning Qwen2.5-1.5B this way took KL/step from 0.95 to 0.00, took the fluency cost of enforcement from −11.4 log-probability per sequence to zero, and took *enforcer-off* rule-following — sample freely, no mask, and count fully-valid sequences — from 0% to 100%. All of it converged within the first 250 training steps, on about three dollars of cloud GPU.

Two honest footnotes. The domain was small enough that a 135-million-parameter model also mastered it (in thirty steps), and the trained model became deterministic on it — it memorized the legal paths rather than learning a distribution over them. Toy domains saturate. So we went looking for domains that don't.

## Real rulebooks, and a surprise about their shape

We rebuilt the experiment on real ontologies from Text2KGBench — DBpedia domain schemas with real-world types and up to 68 typed relations. Two findings.

First, the intended one: the training loop works on real vocabulary too. On the WrittenWork and MeanOfTransportation ontologies, disagreement fell from ~2.0 to ~0.005 by step 50, enforcer-off soundness went from ~12% to 100%, and — unlike the toy — the model kept a genuine distribution over legal paths rather than collapsing to one.

Second, the unplanned one: **most "real ontologies" have no sequential structure to enforce.** Thirteen of the nineteen DBpedia ontologies are star graphs — one hub type owns every relation, and every path is one hop long. On a star, reachability and legality coincide, enforcement is provably inert, and rule-following is trivially learnable. Difficulty lives in *chain structure* — types whose targets are themselves sources — not in rule count. Whether enforcement can even matter is a measurable property of the rulebook's shape. If you are evaluating an LLM system "on a knowledge graph," it is worth checking which kind of graph you actually have.

## The generalization wall

Then the experiment that mattered most. We merged all nineteen ontologies into one graph — 127 types, 598 rules, fused wherever they share types like *Person* and *City* — and held out three entire regions (Politician, Astronaut, SportsTeam: 89 rules that exist nowhere else). The model was fine-tuned only on trajectories from the remaining regions; a mechanical check confirmed that not one held-out rule ever appeared in its training text. Then we asked, with the enforcer off: how rule-abiding is it on territory it trained on, versus territory it has never seen?

Trained regions: 8% → **92.5%**.
Held-out regions: 4% → **15.8%**.

That is the result of the whole phase, and it survived a three-seed replication in every configuration we tried (more on that in [part 8](08_kill_your_favorite_result.md)). What fine-tuning installs is not the *skill* of consulting a rulebook. It is a regional map — memorized legal continuations for the territory it saw. Off the map, the model reverts to nearly its untrained self. This matches the broader evidence that transformers solve structured tasks by pattern-matching rather than systematic rules (Dziri et al., "Faith and Fate," [arXiv:2305.18654](https://arxiv.org/abs/2305.18654)); our version simply has a soundness oracle attached, so the failure is exact and quantified.

## Why this is good news for the program

It sounds like a negative result. For the architecture this series proposes, it is the load-bearing positive one.

If internalization had generalized, enforcement would be scaffolding — train hard enough and you could remove the mask, trust the weights, and lose the receipt. It doesn't generalize. The guarantee genuinely cannot live in the weights, at least at this scale — and the places it fails (inputs the model hasn't seen) are exactly the places guarantees exist for. So the division of labor stands: *training* reduces the cost of enforcement to zero where the model has experience; *enforcement* carries the guarantee everywhere, including where it doesn't; the *certificate* proves which one you got.

Note what this does and doesn't claim. A process reward model (Lightman et al., [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) *scores* reasoning steps statistically; enforcement *forbids* invalid ones outright. And as always in this series: valid means valid under the rulebook in force. The certificate makes that rulebook public. Writing good rulebooks remains the real frontier.

## Caveats

Single model family (Qwen2.5-1.5B, LoRA), one benchmark's ontologies, enforcer-off sampling restricted to the rulebook's label vocabulary (generous to the model — open-vocabulary generation could only score worse). The small-domain runs saturate and are demonstrations of mechanism, not capability claims.

---

*Code, data, and every result file in this post: [github.com/MikeHLee/structure_of_clear_thinking](https://github.com/MikeHLee/structure_of_clear_thinking). Next: [The Map Is Not the Mechanism](07_the_map_is_not_the_mechanism.md) — we go looking for the rulebook inside the model's attention heads, find it, and then discover it isn't doing anything.*
