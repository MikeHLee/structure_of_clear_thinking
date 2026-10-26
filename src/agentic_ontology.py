"""
agentic_ontology.py — role- and claim-gated TLTS for agentic tool-call
enforcement, extending experiment_constraint_ft.py's TLTS with:

  1. Dependency-typed schema (Aristotle said-of/present-in grid, i.e.
     independent-continuant / dependent-continuant, i.e. dependent types):
       - independent sorts:  Actor, Tool, Resource
       - dependent sorts:    Role      (indexed by an Actor instance)
                             Permission (indexed by an Actor x Resource pair;
                                         a durative STATE, not a punctual action)
       - occurrent, doubly-dependent: Action (indexed by Role x satisfied Claims)
  2. Quantity and Time as first-class measure/temporal fields on rules
     (rate limits, credential TTL) — the two forced Aristotelian categories
     missing from the plain TLTS.
  3. A live Bayesian claim store (bayesian_gate.ClaimStore) instead of a
     boolean precondition list: an action's preconditions are claim ids +
     the RiskTier the action carries, and admissibility asks the gate,
     not a lookup table.

This is a superset of experiment_constraint_ft.TLTS: dropping role,
preconditions, quantity, and time recovers the exact type-only enforcer
already validated in threads 1-4 (see `to_plain_tlts` and
`test_agentic_ontology.py::test_plain_tlts_equivalence`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from bayesian_gate import ClaimStore, RiskTier, GateAction, gate_action


# ---------------------------------------------------------------------------
# Rule: the extended transition — a superset of the plain TLTS's
# (domain, label, range) triple.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    domain: str                 # source type (e.g. "Session")
    label: str                  # action verb (e.g. "call_tool")
    range: str                  # target type (e.g. "ToolResult")
    roles: Tuple[str, ...] = ()          # allowed Role values; () = unrestricted
    preconditions: Tuple[str, ...] = ()  # claim ids that must gate EMIT
    risk_tier: RiskTier = RiskTier.LOW   # stakes this action carries
    max_per_window: Optional[int] = None  # Quantity: rate limit, None = unbounded
    window_seconds: Optional[float] = None  # Time: window the limit applies over

    def as_plain_triple(self) -> Tuple[str, str, str]:
        return (self.domain, self.label, self.range)


# ---------------------------------------------------------------------------
# Actor state — tracks which Role(s) an actor currently holds (a durative
# STATE/"having", distinct from the punctual Action that granted it) plus
# the rolling call counts that make rate limits (Quantity x Time) checkable.
# ---------------------------------------------------------------------------

@dataclass
class ActorState:
    actor_id: str
    roles: Tuple[str, ...] = ()
    _call_log: Dict[str, List[float]] = field(default_factory=dict)  # label -> timestamps

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def record_call(self, label: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self._call_log.setdefault(label, []).append(now)

    def calls_in_window(self, label: str, window_seconds: float,
                        now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        cutoff = now - window_seconds
        return sum(1 for t in self._call_log.get(label, []) if t >= cutoff)


# ---------------------------------------------------------------------------
# AgenticTLTS — the enforcer
# ---------------------------------------------------------------------------

@dataclass
class AgenticTLTS:
    types: List[str]
    labels: List[str]
    rules: List[Rule]

    def admissible_from(self, t: str) -> Dict[str, str]:
        """Type-only admissibility — matches plain TLTS.admissible_from
        exactly (drop role/precondition/quantity checks). Kept for direct
        parity testing against experiment_constraint_ft.TLTS."""
        return {r.label: r.range for r in self.rules if r.domain == t}

    def rules_from(self, t: str) -> List[Rule]:
        return [r for r in self.rules if r.domain == t]

    def to_plain_triples(self) -> List[Tuple[str, str, str]]:
        return [r.as_plain_triple() for r in self.rules]

    # -- the extended check: type AND role AND preconditions AND quantity --

    def admissible_for(
        self,
        t: str,
        actor: ActorState,
        claims: ClaimStore,
        now: Optional[float] = None,
    ) -> Dict[str, Tuple[str, Rule]]:
        """{label: (successor_type, rule)} for transitions out of t that
        actor is currently permitted to take, given the live claim store.

        A rule survives all four gates:
          (1) type-legal:      domain == t                          (thread 1's guarantee)
          (2) role-legal:      rule.roles is empty, or actor holds one of them
          (3) claims satisfied: every precondition claim gates EMIT
                                at the rule's risk_tier               (this extension)
          (4) within quantity:  max_per_window not exceeded in window_seconds
                                                                     (Quantity x Time)
        """
        out: Dict[str, Tuple[str, Rule]] = {}
        for r in self.rules_from(t):
            if r.roles and not any(actor.has_role(role) for role in r.roles):
                continue
            if not claims.all_satisfied(r.preconditions, r.risk_tier):
                continue
            if r.max_per_window is not None:
                n = actor.calls_in_window(r.label, r.window_seconds or 0.0, now)
                if n >= r.max_per_window:
                    continue
            out[r.label] = (r.range, r)
        return out

    def denial_reason(
        self, t: str, label: str, actor: ActorState, claims: ClaimStore,
        now: Optional[float] = None,
    ) -> str:
        """Diagnostic for the receipt: WHY was this label not admissible.
        Checked in the same order as admissible_for, first failure wins —
        matches the receipt's existing 'forced'/'masked_kl' style fields
        from thread 2, extended with the new failure modes."""
        candidates = [r for r in self.rules_from(t) if r.label == label]
        if not candidates:
            return "not_type_legal"
        r = candidates[0]
        if r.roles and not any(actor.has_role(role) for role in r.roles):
            return f"role_denied (needs one of {r.roles}, actor has {actor.roles})"
        unmet = [c for c in r.preconditions
                 if gate_action(claims.get(c), r.risk_tier) is not GateAction.EMIT]
        if unmet:
            details = ", ".join(
                f"{c}: mean={claims.get(c).mean:.2f} width={claims.get(c).ci_width():.2f} "
                f"(tier={r.risk_tier.value})" for c in unmet)
            return f"preconditions_unmet ({details})"
        if r.max_per_window is not None:
            n = actor.calls_in_window(r.label, r.window_seconds or 0.0, now)
            if n >= r.max_per_window:
                return f"rate_limited ({n}/{r.max_per_window} in {r.window_seconds}s)"
        return "admissible"


# ---------------------------------------------------------------------------
# Enforced-step receipt record — extends thread 2's certificate schema with
# the new failure modes this layer can produce (role_denied,
# preconditions_unmet, rate_limited) alongside the original in_delta flag.
# ---------------------------------------------------------------------------

@dataclass
class StepReceipt:
    state_in: str
    label: str
    state_out: Optional[str]
    in_delta: bool                 # type-legal, thread-1/2's original flag
    role_checked: bool
    preconditions_checked: bool
    within_quantity: bool
    forced: bool                   # enforcer overrode the model's argmax
    denial_reason: Optional[str] = None

    @property
    def fully_admissible(self) -> bool:
        return (self.in_delta and self.role_checked
                and self.preconditions_checked and self.within_quantity)


def check_step(
    tlts: AgenticTLTS, state_in: str, label: str, actor: ActorState,
    claims: ClaimStore, model_argmax: Optional[str] = None,
    now: Optional[float] = None,
) -> StepReceipt:
    """Build one receipt record for a proposed (state_in, label) step —
    the unit the enforcer checks and the unit the receipt ships."""
    admissible = tlts.admissible_for(state_in, actor, claims, now)
    plain_adm = tlts.admissible_from(state_in)
    in_delta = label in plain_adm
    fully_ok = label in admissible
    reason = None if fully_ok else tlts.denial_reason(state_in, label, actor, claims, now)
    role_ok = in_delta and (reason is None or not reason.startswith("role_denied"))
    precond_ok = in_delta and (reason is None or not reason.startswith("preconditions_unmet"))
    quantity_ok = in_delta and (reason is None or not reason.startswith("rate_limited"))
    return StepReceipt(
        state_in=state_in, label=label,
        state_out=admissible[label][0] if fully_ok else None,
        in_delta=in_delta, role_checked=role_ok,
        preconditions_checked=precond_ok, within_quantity=quantity_ok,
        forced=(model_argmax is not None and model_argmax != label and fully_ok),
        denial_reason=reason,
    )
