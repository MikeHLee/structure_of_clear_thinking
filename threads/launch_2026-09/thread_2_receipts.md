# Thread 2 (day 2): Receipts for AI

Source draft: `threads/02_receipts_for_ai/thread.md`
Figures: `threads/02_receipts_for_ai/figures/`

## Post 1  [attach: fig2_verification_flow.png]

Every AI output should come with a receipt.

Ours is a JSON file of about 1 KB that shows every step followed the domain's rules. Anyone can check it without the model, a GPU, an API, or trust in the vendor.

Yesterday covered how we enforce the rules. Today: the proof. 🧵

Alt text: Flow diagram titled "Anyone can check the receipt without the model." An AI system generates output with per-step rule checking on and emits the output plus a JSON receipt. An independent verifier that holds only the rulebook and the receipt re-checks every step. Three results: an honest output passes; one edited field ("Cart" changed to "Mars") is caught because Customer-has-Mars is not a rule; a substituted rulebook is caught because its fingerprint does not match.

## Post 2

Today, checking whether an AI followed the rules mostly means trusting it. Reruns are nondeterministic, the vendor writes the logs, and benchmarks measure a different question.

The output itself carries no evidence, and the receipt is our way of adding some.

## Post 3  [attach: fig1_certificate_anatomy.png]

For every step of the output, the receipt records where the system was, the move it made, where it landed, whether that move is in the rulebook, and whether the enforcer overrode the model.

It also holds a fingerprint of the exact rulebook in force.

Alt text: An annotated JSON certificate titled "The receipt: what ships with every AI output." The tlts_fingerprint field identifies the rulebook in force, and any edit to the rulebook changes it. Each step record has state_in, label, state_out, and an in_delta flag that says whether the move is a rule. The forced and masked_kl fields show where and how hard the enforcer overrode the model. A summary gives the soundness verdict.

## Post 4

Checking a receipt means re-checking each step against the rulebook. That takes a few dozen lines of code and a few milliseconds on any machine.

An auditor needs two text files and no GPU. Getting a behavioral guarantee out of a model's weights is a much harder audit.

## Post 5

We attacked our own receipts in real runs. We changed one field ("Cart" to "Mars") and the verifier flagged step 0, because that move is not in the rulebook. We swapped in a different rulebook and the fingerprint check failed.

A forger would have to produce a fully valid trace.

## Post 6  [attach: fig3_trust_signal.png]

The receipt also shows how often the enforcer overrode the model: 0% of steps when the model knows the domain, 71% when it has bad habits.

Both outputs are valid. At 71%, the rules are doing most of the work, which tells you to retrain the model or look again at the rulebook.

Alt text: Bar chart titled "The receipt doubles as a health gauge." The share of generation steps where the enforcer overrode the model's first choice is 0% for a well-calibrated model and 71% for a mis-calibrated one. A note says both outputs are valid, but at 71% the rules did most of the work.

## Post 7

The receipt costs nothing extra. The enforcement that guarantees validity already computes every fact the receipt records, so the certificate comes out of generation itself and needs no separate audit pass.

## Post 8

A proposal: papers that claim "our model follows structure X" should ship certificates with their outputs, so reviewers can re-check the claim without weights or compute.

A sample certificate and the verifier are in our repo. Please try to break them.

## Post 9

With certificates, "the AI concluded X" becomes "the AI concluded X through these steps, each allowed by this rule, and anyone can check." We want that audit trail for AI actions and conclusions, added at decoding time with no retraining from scratch.

## Post 10

The caveat still applies: a receipt proves the output complied with the rulebook, and says nothing about whether the rulebook is good. A bad rulebook gives certified bad behavior, but the receipt makes the rulebook public.

Code: github.com/MikeHLee/structure_of_clear_thinking

## Reply A (post as a reply to Post 9)

The multi-lab position paper on chain-of-thought monitorability (@tomekkorbak et al., arXiv:2507.11473) calls the reasoning trace a fragile safety opportunity. Our proposal: enforce the trace at decoding time and ship it as a certificate anyone can check.

## Reply B (post as a reply to Post 8)

Related work checks reasoning after the fact, for example typed post-hoc CoT checking (arXiv:2510.01069) and trace compilation (arXiv:2606.24124). Our receipt is written during decoding, so checking it never needs the model.
