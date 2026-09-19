# Thread 1 (day 1): Reachability masking is not enough

Source draft: `threads/01_reachability_is_not_enough/thread.md`
Figures: `threads/01_reachability_is_not_enough/figures/`

## Post 1  [attach: fig2_soundness_by_method.png]

We tested the usual way to stop an AI from taking illegal actions: block what the rulebook marks unreachable.

It helped a little. When the model had bad habits, it did nothing (4.2% valid without it, 4.3% with it).

Checking every step against the rulebook gave 100%. 🧵

Alt text: Bar chart titled "Only checking every step guarantees rule-following output." For a well-calibrated model: no enforcement 48.1%, reachability masking 61.5%, per-step rule check 100%. For a mis-calibrated model: 4.2%, 4.3%, and 100%. The per-step check is labeled "guaranteed by construction."

## Post 2

The setup: the AI gets a rulebook (an ontology) listing the kinds of things in a domain and the only legal steps between them.

In a shop: a Customer fills a Cart, the Cart goes to Checkout, Checkout collects Payment.

Language models often generate steps that break these rules.

## Post 3  [attach: fig1_reachable_vs_legal.png]

The usual fix: mask attention so the model cannot look at unreachable things.

Reachability says whether you can get somewhere eventually. It does not say whether this step is legal. A rook can reach almost any square in a few moves, but most are illegal next moves.

Alt text: Diagram titled "A reachable destination is not a legal move." Boxes for Customer, Cart, Checkout, Payment, Order, and Item are joined by solid arrows, each one a legal step. A dashed red arrow jumps from Customer straight to Payment. The label says Payment is reachable through Cart and Checkout, so a reachability mask allows the jump, but no rule permits it.

## Post 4

We generated 1,000 action sequences per condition and counted the ones with zero illegal steps.

No enforcement: 48% (model knows the domain) / 4% (bad habits)
Reachability masking: 61.5% / 4.3%
Checking each step against the rulebook: 100% / 100%

## Post 5  [attach: fig3_cyclic_collapse.png]

Real rulebooks have loops: a Delivery leads back to a Customer. With loops, nearly every type can reach every other, so the mask blocks almost nothing. Its advantage over no enforcement drops from 17 points to 1.

The step check works the same whatever the shape of the rulebook.

Alt text: Slope chart titled "Add loops to the rulebook and reachability masking stops working." From a rulebook without loops (34.7% of pairs reachable) to one with loops (100% reachable), reachability masking falls from 65% to 8% and no enforcement falls from 48% to 7%. The per-step rule check stays at 100%.

## Post 6  [attach: fig4_attention_mass.png]

We also audited our own typed-attention layer. Two thirds of its attention mass lands on pairs that are reachable but have no rule between them. None lands on unreachable pairs.

So the mask works as designed. It just does not make the output valid.

Alt text: A single stacked bar titled "Under a reachability mask, two thirds of attention flows through non-rules." Segments: token attending to itself 16.9%, legal step 16.4%, reachable but not a rule 66.7%. A note says the mask blocks all unreachable pairs but allows 4 times more pairs than the rulebook contains.

## Post 7

Does forcing the rules hurt fluency? Only when the model disagrees with the rulebook, and you can measure that. Our worst synthetic case cost about 5.5 nats per sequence. When the model knew the domain, the cost was near zero.

The same measure works as a drift alarm.

## Post 8

Because every step is checked, each output can ship with a receipt: a JSON certificate listing each step and the rule that allowed it.

Anyone with the rulebook can re-check it, with no model weights or API. In our tests the check caught a tampered receipt and a swapped rulebook.

## Post 9

Where this is going:

1. Auditable action and reasoning traces for frontier models. Enforcement runs at decoding time, so nobody has to retrain from scratch.
2. Small models that are provably reliable in a domain, because the rulebook carries the guarantee.

## Post 10

The caveat that matters most: "valid" means valid under the rulebook. The guarantee is only as good as the ontology someone wrote, and a wrong rule gives certified wrong behavior. Writing good rulebooks is the hard part.

Code: github.com/MikeHLee/structure_of_clear_thinking

## Reply A (post as a reply to Post 4)

Prior work: certified per-step constrained decoding is @GabrielPoesia and @noahdgoodman's LogicGuide (arXiv:2306.04031). The decoding machinery goes back to @remilouf's Outlines (arXiv:2307.09702), which guarantees format. We add a typed-rulebook guarantee over whole sequences.

## Reply B (post as a reply to Post 2)

Why check the structure instead of reading the model's explanation? Chain-of-thought explanations often misstate what drove the answer: @milesaturpin @sleepinyourhat et al. (arXiv:2305.04388), and Lanham et al. (arXiv:2307.13702).
