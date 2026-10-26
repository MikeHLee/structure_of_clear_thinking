"""Tests for agentic_ontology.py — role/precondition/quantity-gated TLTS.

Two jobs:
  1. Parity: dropping role/precondition/quantity recovers the exact
     type-only admissibility of experiment_constraint_ft.TLTS (the
     already-validated thread-1 enforcer). This is the non-negotiable
     regression check before any new experiment runs.
  2. New behavior: role denial, precondition denial (via the live Bayesian
     gate), and rate limiting each fire independently and are each
     distinguishable in the receipt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ontology import AgenticTLTS, Rule, ActorState, check_step  # noqa: E402
from bayesian_gate import ClaimStore, RiskTier  # noqa: E402


def _shop_agentic_tlts() -> AgenticTLTS:
    """Same 7-type e-commerce domain as experiment_loci_comparison's
    relational_ecommerce_tlts, re-expressed as Rules with no role/
    precondition/quantity restriction — must be exactly type-equivalent."""
    return AgenticTLTS(
        types=["Customer", "Cart", "Item", "Checkout", "Payment", "Order", "Delivery"],
        labels=["has", "contains", "proceeds_to", "requires", "creates", "triggers"],
        rules=[
            Rule("Customer", "has", "Cart"),
            Rule("Cart", "contains", "Item"),
            Rule("Cart", "proceeds_to", "Checkout"),
            Rule("Checkout", "requires", "Payment"),
            Rule("Payment", "creates", "Order"),
            Rule("Order", "triggers", "Delivery"),
        ],
    )


def test_plain_tlts_equivalence():
    """admissible_from with no role/precondition/quantity restriction must
    match the plain TLTS triple set exactly (thread-1 parity)."""
    tlts = _shop_agentic_tlts()
    expected = {
        "Customer": {"has": "Cart"},
        "Cart": {"contains": "Item", "proceeds_to": "Checkout"},
        "Checkout": {"requires": "Payment"},
        "Payment": {"creates": "Order"},
        "Order": {"triggers": "Delivery"},
        "Item": {},
        "Delivery": {},
    }
    for t in tlts.types:
        assert tlts.admissible_from(t) == expected[t], t


def test_admissible_for_matches_admissible_from_when_unrestricted():
    """With an actor holding no roles and an empty claim store, admissible_for
    must return exactly admissible_from's targets — no rule in this fixture
    carries role/precondition/quantity, so nothing should be filtered."""
    tlts = _shop_agentic_tlts()
    actor = ActorState(actor_id="a1")
    claims = ClaimStore()
    for t in tlts.types:
        got = {label: target for label, (target, _r) in
               tlts.admissible_for(t, actor, claims).items()}
        assert got == tlts.admissible_from(t), t


def test_role_denial():
    tlts = AgenticTLTS(
        types=["Session", "ToolResult"], labels=["delete_file"],
        rules=[Rule("Session", "delete_file", "ToolResult",
                    roles=("admin",), risk_tier=RiskTier.HIGH)],
    )
    actor_plain = ActorState(actor_id="a1", roles=("viewer",))
    actor_admin = ActorState(actor_id="a2", roles=("admin",))
    claims = ClaimStore()
    # admin's claim store still needs the risk tier satisfied, so seed it
    # with strong confirming evidence to isolate the role check.
    for _ in range(30):
        claims.observe("dummy", confirming=True)

    r = check_step(tlts, "Session", "delete_file", actor_plain, claims)
    assert not r.role_checked
    assert r.denial_reason.startswith("role_denied")

    r2 = check_step(tlts, "Session", "delete_file", actor_admin, claims)
    assert r2.role_checked  # role clears; precondition list is empty so it's fully OK
    assert r2.fully_admissible


def test_precondition_denial_via_bayesian_gate():
    """A HIGH-risk action gated on a claim with only one weak observation
    must be denied; after enough confirming evidence it must clear."""
    tlts = AgenticTLTS(
        types=["Session", "ToolResult"], labels=["wire_transfer"],
        rules=[Rule("Session", "wire_transfer", "ToolResult",
                    preconditions=("recipient_verified",),
                    risk_tier=RiskTier.CRITICAL)],
    )
    actor = ActorState(actor_id="a1")
    claims = ClaimStore()

    r = check_step(tlts, "Session", "wire_transfer", actor, claims)
    assert not r.preconditions_checked
    assert r.denial_reason.startswith("preconditions_unmet")

    for _ in range(40):
        claims.observe("recipient_verified", confirming=True)
    r2 = check_step(tlts, "Session", "wire_transfer", actor, claims)
    assert r2.preconditions_checked
    assert r2.fully_admissible


def test_rate_limiting():
    tlts = AgenticTLTS(
        types=["Session", "ToolResult"], labels=["send_email"],
        rules=[Rule("Session", "send_email", "ToolResult",
                    max_per_window=2, window_seconds=60.0)],
    )
    actor = ActorState(actor_id="a1")
    claims = ClaimStore()
    now = 1000.0

    r1 = check_step(tlts, "Session", "send_email", actor, claims, now=now)
    assert r1.fully_admissible
    actor.record_call("send_email", now=now)

    r2 = check_step(tlts, "Session", "send_email", actor, claims, now=now + 1)
    assert r2.fully_admissible
    actor.record_call("send_email", now=now + 1)

    r3 = check_step(tlts, "Session", "send_email", actor, claims, now=now + 2)
    assert not r3.within_quantity
    assert r3.denial_reason.startswith("rate_limited")

    # outside the window, the limit resets
    r4 = check_step(tlts, "Session", "send_email", actor, claims, now=now + 120)
    assert r4.fully_admissible


def test_forced_flag_records_enforcer_override():
    tlts = _shop_agentic_tlts()
    actor = ActorState(actor_id="a1")
    claims = ClaimStore()
    r = check_step(tlts, "Customer", "has", actor, claims, model_argmax="bogus_label")
    assert r.forced
    r2 = check_step(tlts, "Customer", "has", actor, claims, model_argmax="has")
    assert not r2.forced


if __name__ == "__main__":
    test_plain_tlts_equivalence()
    test_admissible_for_matches_admissible_from_when_unrestricted()
    test_role_denial()
    test_precondition_denial_via_bayesian_gate()
    test_rate_limiting()
    test_forced_flag_records_enforcer_override()
    print("all agentic_ontology tests passed")
