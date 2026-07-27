"""Tests for the epistemic-status bridge, including the market-memo eval case."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from epistemic_status import (  # noqa: E402
    EpistemicStatus, GateAction, Claim,
    map_legacy_tag, derive_status, gate,
)

EVAL = Path(__file__).resolve().parent.parent / "eval" / "epistemic_tagging_memo.json"


def test_legacy_tag_mapping():
    assert map_legacy_tag("[SOURCED]") is EpistemicStatus.SOURCED
    assert map_legacy_tag("UNKNOWABLE") is EpistemicStatus.UNKNOWABLE
    assert map_legacy_tag(" unfalsifiable ") is EpistemicStatus.UNFALSIFIABLE


def test_gate_policy():
    assert gate(EpistemicStatus.SOURCED) is GateAction.EMIT
    assert gate(EpistemicStatus.UNKNOWABLE) is GateAction.ABSTAIN
    # an unfalsifiable claim must never pass as a sourced assertion
    assert gate(EpistemicStatus.UNFALSIFIABLE) is not GateAction.EMIT


def test_derive_from_slots():
    assert derive_status("citation", grounded=True) is EpistemicStatus.SOURCED
    assert derive_status("assertion", grounded=True) is EpistemicStatus.SOURCED
    assert derive_status("assertion", grounded=False,
                         falsifiable=False) is EpistemicStatus.UNFALSIFIABLE
    assert derive_status("hypothesis", grounded=False) is EpistemicStatus.FALSIFIABLE_UNSOURCED


def test_memo_partition_reproduced():
    """The guard must reproduce the memo's own split: the supply half is
    assertable; the coordination/intent claim is gated out of assertion."""
    data = json.loads(EVAL.read_text())
    claims = [Claim(text=c["text"], status=map_legacy_tag(c["gold"]),
                    axis=c.get("axis", ""), note=c.get("note", ""))
              for c in data["claims"]]

    # every SOURCED fact is assertable
    assert all(c.assertable for c in claims if c.status is EpistemicStatus.SOURCED)
    # the unfalsifiable coordination claim is NOT assertable
    coord = [c for c in claims if c.status is EpistemicStatus.UNFALSIFIABLE]
    assert coord, "expected at least one unfalsifiable claim in the memo"
    assert all(not c.assertable for c in coord)


if __name__ == "__main__":
    test_legacy_tag_mapping()
    test_gate_policy()
    test_derive_from_slots()
    test_memo_partition_reproduced()
    print("all epistemic_status tests passed")
