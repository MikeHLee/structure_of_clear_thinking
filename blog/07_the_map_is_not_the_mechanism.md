# The Map Is Not the Mechanism

*Part 7 of the Structure of Clear Thinking series. We went looking for the knowledge graph inside a language model's attention heads. We found it — one head recognizes real-world relations at 0.90 AUC before any training. Then we deleted it, and nothing happened. What that dissociation explains about LLMs and knowledge graphs.*

---

## The question

[Part 6](06_teaching_the_rulebook.md) ended with a behavioral puzzle: fine-tuning teaches a model to follow the rulebook regions it trained on (92.5% valid, enforcer off) while leaving it nearly helpless on regions it didn't (15.8%). A regional map, not a mapmaking skill. The obvious next question is mechanistic: *where in the network does that map live?* And underneath it, an older intuition many of us share: don't transformers already implement something like a soft knowledge graph in their attention — token-to-token routing that mirrors which concepts relate to which?

That intuition has published support. Attention heads that act as "knowledge hubs," extracting subject–attribute relations, are documented (Geva et al., [arXiv:2304.14767](https://arxiv.org/abs/2304.14767)); many relations decode as linear maps in activation space (Hernandez et al., [arXiv:2308.09124](https://arxiv.org/abs/2308.09124)). What hasn't been done — because it needs an exact ground-truth rulebook, before/after checkpoints of the same model on that rulebook, and a behavioral soundness metric, all of which the previous experiments happened to leave on our workbench — is to test the intuition *causally*.

## Finding the map

Method, briefly. Feed the model sequences of typed concepts. For every pair of concept positions, record each attention head's weight between them, and label the pair by the ontology: directly connected by a real rule, merely reachable, or unrelated. A head "knows the graph" to the degree its attention ranks rule-pairs above non-pairs — an AUC per head. Two guards keep the measurement honest: adjacent-in-sequence pairs are excluded (in valid text, adjacency and relatedness coincide, so a boring "attend to the previous phrase" head would fake a high score), and half the stimuli are *scrambled* — random types in the same surface format — so position cannot carry the signal at all.

The result, on pretrained Qwen2.5-1.5B with no fine-tuning: most of the 336 heads sit at chance. A specific tail does not. The best general head separates real DBpedia rules from non-rules at 0.76 on scrambled input; and one head reaches **0.90** on relations from politics and sports — topics the pretraining corpus is soaked in. The soft knowledge graph is real. It is diffuse, it is concentrated in identifiable heads, and it was there before we touched anything.

## Three ways the map turned out not to matter

Then the campaign of disillusionment, in ascending order of surprise.

**Deleting the map changes nothing.** We took the fine-tuned, 100%-rule-following model and zeroed out the top graph heads entirely. Rule-following after deleting the best eight: 99%. After deleting eight *random* heads instead: 99–100%. The heads that best encode the rulebook contribute nothing measurable to following it.

**Training never wrote to the map.** Fine-tuning that takes the model from 13% to 100% rule-following moves the attention patterns by less than 0.02 AUC and the pre-softmax query–key geometry by less than 0.01. Same top heads before and after. Whatever the gradient built, it built it around the attention structure, not into it — consistent with where the editing literature localizes factual associations: the MLPs (ROME, [arXiv:2202.05262](https://arxiv.org/abs/2202.05262); Geva et al., [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)).

**The behavioral wall has no geometric shadow.** On the merged graph from part 6, behavior split 92.5% vs 15.8% between trained and held-out regions. The attention geometry's knowledge of those same rules: 0.50 → 0.51 on trained regions, 0.47 → 0.49 on held-out. Flat, at chance, everywhere. An 84-point behavioral transformation with no detectable reorganization of the "graph" in attention.

## The dissociation that stings

Put the sharpest two facts side by side. The single most knowledgeable structure we found in the entire network — the 0.90 head — covers politician and sports relations. The model's ability to *generate* valid sequences in exactly that territory, enforcer off: 4%.

Knowing and doing are different circuits. The model contains a decent map it does not consult when it speaks. Whatever bottleneck separates represented relational knowledge from emitted behavior — call it the read-out path — it, and not the map's quality, is where the failure lives. (We tested one version of this directly: restricting fine-tuning to the value/output projections, the read-out side of attention. One seed looked spectacular. [Part 8](08_kill_your_favorite_result.md) is about what happened when we tried to replicate it.)

## Why your knowledge-graph project disappointed you

This picture retrodicts a production experience that seems near-universal, ours included: LLMs are poor at *natively reasoning over* knowledge graphs — multi-hop traversal, graph-query generation against KG stores underperforming plain SQL — while the same graphs work beautifully as *design-time* structure: schemas, type systems, application ontologies.

The mechanism now looks concrete. Native traversal asks the model to be the graph engine: compose edge lookups internally, hop after hop, carrying path state. Its internal edge signal is 0.55-diffuse with a thin 0.7-to-0.9 tail — and per-hop unreliability compounds exponentially. Prompting doesn't upgrade the routing; fine-tuning, we now know, doesn't either — it writes memorized paths downstream and hits the generalization wall on any subgraph it didn't see. An arbitrary user-supplied KG *is* a held-out region. Meanwhile SQL rides on schema-enforced structure where the database does the composing — structure enforced outside the weights, which is this series' thesis wearing someone else's clothes.

## Caveats

One model at one scale, one ontology family, and attention-level evidence: deletion plus training dynamics localize where the skill *isn't*, which is not a full circuit analysis of where it is. Probing the value/MLP side is the stated next experiment. The 0.90 head's domain (politics/sports) confounds pretraining frequency with head specialization — we note it, not over-read it.

---

*Code, probes, and per-head results: [github.com/MikeHLee/structure_of_clear_thinking](https://github.com/MikeHLee/structure_of_clear_thinking). Next: [Kill Your Favorite Result](08_kill_your_favorite_result.md) — the replication that executed our best story, and why we published the corpse.*
