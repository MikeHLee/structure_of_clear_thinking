# Structure of Clear Thinking: where the project stands

*A summary of the work so far, and the four result threads we start posting this weekend.*

We started this series with a bee. In 1946, Karl von Frisch decoded the honeybee's waggle dance. A bee that has found nectar comes back and dances a figure eight: the angle of the dance gives the direction and its vigor gives the distance. The other bees fly straight to a source they have never seen, and any of them can check the message by flying it.

Our working thesis is that many hallucinations are invalid combinations of relations, and that a model with no type system for its domain has no way to rule them out. Since January we have been building that type system.

## Timeline

- January 19: the first design handoff for the project (then called Topos-Bridge).
- April 25: the repository went public, with the core embedding and attention layers tested and a proof engine that checks claims in three modes (strict, compositional, and reachability).
- May 5: we wrote up TLTS-Compilation, the framework that ties together the four lines of work below.
- June: we finished a 9-page paper on TLTS-Compilation for NeSy 2026. The OpenReview upload failed and the organizers did not respond before the cycle closed, so we have released it as a self-published preprint.
- July 30: we finished drafting and fact-checking the result threads, with citations checked against the original papers.
- This weekend: we start posting them.

The posts over the next few days will look like a burst of activity. They are a backlog of dated work that has been in a public repository for months, and we are releasing it one piece per day.

## What we built

All four lines of work constrain a language model with a categorical ontology, called an olog. The objects in an olog are types, and the morphisms are the relations the domain allows.

### Typed attention

Standard attention lets any token attend to any other token, whether or not the domain has a relation between them. Our masked variant only allows attention between types that can reach each other. On a 23-type ontology drawn from several domains (2,536 examples, 300 epochs), standard attention put 29.5% of its weight on invalid token pairs, and the typed version put 0%. Test accuracy was similar (47.4% for standard, 45.7% for typed). The zero comes from the mask, so it holds whatever the training does.

### A structural contradiction detector

We compute sheaf cohomology, a tool from algebraic topology, over knowledge graphs to find internal inconsistency. When we injected 76 conflicts into a clean benchmark subgraph, the first cohomology dimension (H¹) rose from 5 to 58. H¹ is computed directly from the graph, with no trained classifier. The same pipeline reaches a mean reciprocal rank of 0.346 for link prediction on FB15K-237, in the range reported for ConvE and RotatE.

### Soundness by construction (TLTS-Compilation)

The obvious fix is to mask attention so the model cannot see parts of the ontology it cannot reach. It helps less than we expected. In a synthetic test with 1,000 generated sequences per condition, reachability masking raised the share of fully valid sequences from 48% to 62% for a model that already knew the domain, and from 4.2% to 4.3% for one that did not. When the ontology has loops, as real domains do, most of that gain disappears. Checking every step against the rulebook during generation gave 100% valid sequences in both conditions. Each output also comes with a JSON audit certificate that anyone can re-check without access to the model. This is the subject of the preprint.

### The epistemic-status bridge

We give each claim a model makes one of four statuses: sourced, falsifiable but unsourced, unfalsifiable, or unknowable. The status comes from the tokenizer's modality and provenance fields, and it decides whether the model states the claim, hedges it, or declines to answer. The code and its tests are in the repository. We have not yet tested it on a trained model.

## The trade

We can make a model's output follow a rulebook every time. The price is that the model sometimes has to say it does not know.

A statistical guarantee says a model is usually right. A structural guarantee says it cannot emit a step the rulebook does not allow. We think unattended use in medicine, law, and finance needs the second kind. That guarantee is only as good as the rulebook, and writing good rulebooks is the hard part of the program.

## What's next

Starting this weekend, we will post four result threads on X, one per day, each with figures you can regenerate from the repository:

1. Reachability masking is not enough: the result above in full, including the test with loops.
2. Receipts for AI: how the audit certificates work, and how anyone can check one without our model weights.
3. The map is not in the attention: a knowledge-graph structure we found in a model's attention heads, at up to 0.90 AUC. When we deleted those heads, the model's rule-following did not change.
4. The result that died in replication: an effect we liked that did not hold up in a replication we had committed to in advance, and the result that held in every run.

The experiments in the first two threads run on a laptop. The training and probe runs in the last two used one cloud GPU and cost a few dollars each. Code, data, and figures are at github.com/MikeHLee/structure_of_clear_thinking.

*Structure of Clear Thinking is a research project on category theory, algebraic topology, and language models.*
