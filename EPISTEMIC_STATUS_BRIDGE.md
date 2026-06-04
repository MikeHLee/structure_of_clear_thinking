# The Epistemic-Status Bridge: from formal guards to sourced claims

*Design note — also the seed for a blog post in the "Structure of Clear Thinking" series.*

## The observation

Modern frontier models already do something that looks a lot like the discipline
we are trying to formalize. Shown a messy macro thesis, a current model will
spontaneously **partition a claim into a sourced half and an unfalsifiable half**
and refuse to assert the latter as fact. A real example (an investment-reasoning
memo, 2026-06):

> The supply wall is real and historically unprecedented … combined >$200B vs a
> $45B total 2025 US IPO market **[SOURCED]** … That allocators are coordinating
> a "hold the line" pattern is **[UNKNOWABLE]** — unfalsifiable, and not something
> I can source. Reframe (b) from "conspiracy" to "structural incentive":
> issuers/underwriters/PE sponsors are individually incentivized to keep the tape
> orderly into their exits, which can look coordinated without being so.

That is exactly the premise→conclusion path-validation our proof engine does for
Olog morphisms — applied to conceptual claims instead of code.

## What it is, and what it isn't

The behaviour is **elicitable but soft**. There is (almost certainly) no fact
graph under the hood, nothing is actually retrieved or checked, and the model can
be confidently wrong about what it tags `[SOURCED]`. It has the *shape* of
sourcing without the *substance*. The value to us is not "the model has the
mechanism" — it is that **the behaviour is a learnable, expressible target**,
which is the precondition for enforcing it as a *hard* guarantee in a smaller
model via constrained decoding.

## The bridge

`hierarchical_tokenizer` already carries the two slots we need:

- `modality_code` — ASSERTION / HYPOTHESIS / QUESTION / CITATION
- `provenance_code` — a witness id, or `NULL_PROVENANCE`

`epistemic_status.py` adds a **derived axis** over `(modality, grounded?, falsifiable?)`:

| EpistemicStatus | Meaning | Gate |
|---|---|---|
| `SOURCED` | proof leaf resolves to a witness | **EMIT** |
| `FALSIFIABLE_UNSOURCED` | checkable in principle, no witness yet | DOWNGRADE (hedge / seek source) |
| `UNFALSIFIABLE` | no possible evidence settles it | DOWNGRADE (reframe, don't assert) |
| `UNKNOWABLE` | answer exists but is inaccessible | **ABSTAIN** → `CANNOT_ANSWER` |
| `UNVERIFIED` | default until assessed | DOWNGRADE |

`DOWNGRADE` is the formal analogue of the memo's "reframe conspiracy → structural
incentive": the claim may still be uttered, but never as a sourced assertion.
`ABSTAIN` routes to the tokenizer's existing `CANNOT_ANSWER` primitive.

Crucially this is **not a 5th bound GHRR slot** — it is an annotation the gate
layer (`ConstrainedDecoder` / `ProofEngine`) consults, so the 4-slot binding is
untouched. This keeps the change cheap and non-destabilizing.

## Why this matters for the small-model thesis

This is the conceptual extension of our "guards generalize from code to epistemic
reasoning" claim (TLTS post §f/g): the *same* admissibility relation δ that gates
Olog morphisms gates **claim assertability**. The functional-δ case is code; the
relational-δ case now includes "may this claim be asserted as sourced?"

## Honest gap

Soft elicitation in a frontier model ≠ a hard guarantee in a 0.5–7B model. This
module gives the *vocabulary and the gate*; it does **not** yet learn to assign
`EpistemicStatus` from raw text. That is the job of:

- **SCT-008** (constraint-aware fine-tuning): train a small LM so its claims,
  decoded under this gate, recover a GOOD epistemic prior.
- **SCT-011** (large→small distillation): distill a frontier model's soft
  partitioning into a small model under the gate.

## Eval

`eval/epistemic_tagging_memo.json` encodes the memo above as the first target
case (claims with gold `SOURCED`/`UNFALSIFIABLE`/… labels). `tests/test_epistemic_status.py`
asserts the guard reproduces the memo's own split — the supply half is
assertable, the coordination claim is gated out of assertion. This is the first
entry in what should become an **epistemic-tagging benchmark** for conceptual,
non-code generation guards.

## Files

- `epistemic_status.py` — the axis, the gate, `derive_status`, `map_legacy_tag`, `Claim`.
- `eval/epistemic_tagging_memo.json` — first gold-labeled eval case.
- `tests/test_epistemic_status.py` — unit + memo-partition tests.
