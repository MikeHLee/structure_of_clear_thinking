"""
Epistemic-status bridge — extends the SCT gated-generation guard from formal
languages (code / Olog morphisms) to *conceptual* claims.

Motivation
----------
Frontier models already, informally, partition a claim into a well-evidenced
half and an unfalsifiable half and hedge the latter. (Observed 2026-06 in an
investment-reasoning memo that tagged a market thesis [SOURCED] vs [UNKNOWABLE]
and reframed an unfalsifiable "coordination" claim as a "structural incentive".)
That behaviour is *elicitable but soft*: nothing is actually checked and the tag
can be wrong. This module makes the same partition a *first-class, gateable*
property so a constrained decoder / proof engine can enforce it rather than hope
for it.

Where it sits
-------------
`hierarchical_tokenizer` already gives every token two relevant slots:
    - modality_code   (Modality: ASSERTION / HYPOTHESIS / QUESTION / CITATION)
    - provenance_code (witness id; NULL_PROVENANCE if ungrounded)
EpistemicStatus is a *derived* axis over (modality, grounded?, falsifiable?).
It is intentionally NOT a 5th bound GHRR slot — it is an annotation the gate
layer (ConstrainedDecoder / ProofEngine) consults, so the existing 4-slot
binding is unchanged. Coupling to Modality is by value (the str-enum value,
e.g. "assertion") so this module stays import-light and unit-testable alone.

The gate
--------
    SOURCED               -> EMIT      (proof leaf resolves to a witness)
    FALSIFIABLE_UNSOURCED -> DOWNGRADE (checkable in principle; hedge / seek source)
    UNFALSIFIABLE         -> DOWNGRADE (reframe; never assert as fact)
    UNKNOWABLE            -> ABSTAIN   (emit CANNOT_ANSWER)
    UNVERIFIED            -> DOWNGRADE (default until assessed)

DOWNGRADE is the formal analogue of the memo's "reframe (b) from conspiracy to
structural incentive": the claim may still be uttered, but not as a sourced
assertion. ABSTAIN routes to the tokenizer's existing CANNOT_ANSWER primitive.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Modality values mirror hierarchical_tokenizer.Modality (str-enum) by value.
MODALITY_ASSERTION = "assertion"
MODALITY_HYPOTHESIS = "hypothesis"
MODALITY_QUESTION = "question"
MODALITY_CITATION = "citation"
MODALITY_UNTYPED = "untyped"


class EpistemicStatus(str, Enum):
    SOURCED = "sourced"
    FALSIFIABLE_UNSOURCED = "falsifiable_unsourced"
    UNFALSIFIABLE = "unfalsifiable"
    UNKNOWABLE = "unknowable"
    UNVERIFIED = "unverified"


class GateAction(str, Enum):
    EMIT = "emit"
    DOWNGRADE = "downgrade"
    ABSTAIN = "abstain"


# Inline free-text tags a frontier model emits, mapped onto the axis.
_LEGACY_TAG_MAP = {
    "sourced": EpistemicStatus.SOURCED,
    "falsifiable_unsourced": EpistemicStatus.FALSIFIABLE_UNSOURCED,
    "unfalsifiable": EpistemicStatus.UNFALSIFIABLE,
    "unknowable": EpistemicStatus.UNKNOWABLE,
    "unverified": EpistemicStatus.UNVERIFIED,
}

_GATE = {
    EpistemicStatus.SOURCED: GateAction.EMIT,
    EpistemicStatus.FALSIFIABLE_UNSOURCED: GateAction.DOWNGRADE,
    EpistemicStatus.UNFALSIFIABLE: GateAction.DOWNGRADE,
    EpistemicStatus.UNKNOWABLE: GateAction.ABSTAIN,
    EpistemicStatus.UNVERIFIED: GateAction.DOWNGRADE,
}


def map_legacy_tag(tag: str) -> EpistemicStatus:
    """Map an inline tag like '[SOURCED]' / 'UNKNOWABLE' to EpistemicStatus."""
    key = tag.strip().strip("[]").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _LEGACY_TAG_MAP:
        raise ValueError(f"unknown epistemic tag: {tag!r}")
    return _LEGACY_TAG_MAP[key]


def derive_status(
    modality: str,
    grounded: bool,
    falsifiable: Optional[bool] = None,
) -> EpistemicStatus:
    """
    Derive epistemic status from tokenizer slots already available.

    modality    : the Modality .value (e.g. "assertion").
    grounded    : TokenSlots.is_grounded()  (provenance != NULL_PROVENANCE).
    falsifiable : optional hint — could evidence even in principle settle it?
                  None means "unknown" and is treated conservatively.
    """
    m = modality.strip().lower()
    if m == MODALITY_CITATION or (m == MODALITY_ASSERTION and grounded):
        return EpistemicStatus.SOURCED
    if falsifiable is False:
        return EpistemicStatus.UNFALSIFIABLE
    if m == MODALITY_HYPOTHESIS:
        return EpistemicStatus.FALSIFIABLE_UNSOURCED
    if m == MODALITY_ASSERTION and not grounded:
        # asserted-but-ungrounded, in-principle-checkable -> needs a source
        return (EpistemicStatus.FALSIFIABLE_UNSOURCED
                if falsifiable else EpistemicStatus.UNVERIFIED)
    return EpistemicStatus.UNVERIFIED


def gate(status: EpistemicStatus) -> GateAction:
    """The generation guard's verdict for a claim of the given status."""
    return _GATE[status]


@dataclass
class Claim:
    """A conceptual claim carrying its epistemic annotation."""
    text: str
    status: EpistemicStatus = EpistemicStatus.UNVERIFIED
    axis: str = ""           # optional grouping (e.g. "supply" vs "intent")
    note: str = ""

    @property
    def action(self) -> GateAction:
        return gate(self.status)

    @property
    def assertable(self) -> bool:
        """True iff the guard would let this be emitted as a sourced fact."""
        return self.action is GateAction.EMIT


if __name__ == "__main__":
    assert map_legacy_tag("[SOURCED]") is EpistemicStatus.SOURCED
    assert gate(EpistemicStatus.UNKNOWABLE) is GateAction.ABSTAIN
    assert derive_status("assertion", grounded=True) is EpistemicStatus.SOURCED
    assert derive_status("assertion", grounded=False,
                         falsifiable=False) is EpistemicStatus.UNFALSIFIABLE
    print("epistemic_status smoke test OK")
