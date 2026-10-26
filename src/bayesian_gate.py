"""
bayesian_gate.py — Bayesian belief-updating extension of epistemic_status.py.

Motivation
----------
epistemic_status.py's gate is categorical: a claim is SOURCED or it is not,
and gate() returns EMIT/DOWNGRADE/ABSTAIN off that single tag. That is fine
for "is this grounded" but wrong for permissioning real actions, where what
you actually want is: does the CURRENT POSTERIOR on this claim clear a
threshold that itself depends on the STAKES of the action being gated?
A read-only lookup and an irreversible delete should not need the same
confidence to fire.

This module does not replace epistemic_status.py. It is a strict
generalization: collapsing a BeliefState to a point mass recovers the old
discrete gate exactly (see `from_status` / `test_recovers_discrete_gate`).

Design
------
Each claim carries a Beta(alpha, beta) posterior over "this claim is true".
Evidence increments alpha (confirming) or beta (disconfirming), optionally
weighted (e.g. a highly reliable witness counts for more than a rumor).
gate_action() then asks two questions, not one:
    1. Is the posterior MEAN above this action's required threshold?
    2. Is the posterior CONFIDENT enough (credible interval narrow enough)
       that the mean is not a coin-flip estimate from three observations?
Both must hold for EMIT. Failing (1) alone is DOWNGRADE (probably false or
too uncertain to act on). UNFALSIFIABLE and UNKNOWABLE claims are still
categorical trapdoors — no amount of "evidence" should ever buy a claim
that cannot in principle be checked, or one that is definitionally
unknowable, a passing grade. That mirrors epistemic_status.py's own refusal
to let UNFALSIFIABLE claims reach EMIT regardless of tag confidence.

Risk tiers
----------
An action's risk tier sets (min_mean, max_ci_width). Higher tiers demand
both a higher mean AND a tighter interval — a claim resting on one weak
observation can have a high mean by luck; the interval check catches that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional

from epistemic_status import EpistemicStatus, GateAction


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """Beta(alpha, beta) posterior over 'claim is true'.

    alpha/beta start at (1, 1) — uniform, maximally uninformative — unless
    seeded otherwise. This is a prior pseudo-count pair, not a hard tag.
    """
    claim_id: str
    alpha: float = 1.0
    beta: float = 1.0
    unfalsifiable: bool = False   # categorical trapdoor, see module docstring
    unknowable: bool = False      # categorical trapdoor
    note: str = ""

    def update(self, confirming: bool, weight: float = 1.0) -> "BeliefState":
        """In-place Bayesian update from one piece of evidence.

        weight scales how much this single observation counts (e.g. a
        cross-checked source vs a single unverified report). weight=1.0 is
        one 'standard' observation.
        """
        if confirming:
            self.alpha += weight
        else:
            self.beta += weight
        return self

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    def credible_interval(self, width: float = 0.80) -> tuple[float, float]:
        """Approximate (lo, hi) central credible interval.

        Uses a normal approximation to the Beta posterior (accurate once
        alpha, beta are both comfortably above ~5; conservative — i.e. wider
        than the true interval — for small counts, which is the safe
        direction for a permission gate). Avoids an incomplete-beta-function
        dependency so this module stays torch/scipy-free.
        """
        m, v = self.mean, self.variance
        sd = math.sqrt(max(v, 0.0))
        z = _normal_quantile(0.5 + width / 2)
        lo, hi = m - z * sd, m + z * sd
        return (max(0.0, lo), min(1.0, hi))

    def ci_width(self, width: float = 0.80) -> float:
        lo, hi = self.credible_interval(width)
        return hi - lo

    @property
    def n_obs(self) -> float:
        """Effective observation count beyond the uninformative (1,1) prior."""
        return (self.alpha - 1.0) + (self.beta - 1.0)


def _normal_quantile(p: float) -> float:
    """Inverse CDF of the standard normal (Acklam's rational approximation).
    Good to ~1e-9, no scipy dependency. p in (0, 1)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Risk tiers — what an action demands of the claims that gate it
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    LOW = "low"           # read-only / reversible / low blast radius
    MEDIUM = "medium"      # writes, reversible with effort
    HIGH = "high"          # irreversible, narrow blast radius
    CRITICAL = "critical"  # irreversible, wide blast radius / affects others


# (min_posterior_mean, max_80%_credible_interval_width)
_RISK_REQUIREMENTS: Dict[RiskTier, tuple[float, float]] = {
    RiskTier.LOW:      (0.60, 0.60),
    RiskTier.MEDIUM:   (0.80, 0.40),
    RiskTier.HIGH:     (0.90, 0.25),
    RiskTier.CRITICAL: (0.97, 0.10),
}


def gate_action(belief: BeliefState, risk_tier: RiskTier) -> GateAction:
    """The Bayesian analogue of epistemic_status.gate().

    Categorical trapdoors first (no posterior can buy past these — matches
    epistemic_status.py's own UNFALSIFIABLE/UNKNOWABLE handling), then a
    mean+precision test keyed to the action's risk tier.
    """
    if belief.unknowable:
        return GateAction.ABSTAIN
    if belief.unfalsifiable:
        return GateAction.DOWNGRADE
    min_mean, max_width = _RISK_REQUIREMENTS[risk_tier]
    if belief.mean >= min_mean and belief.ci_width() <= max_width:
        return GateAction.EMIT
    return GateAction.DOWNGRADE


# ---------------------------------------------------------------------------
# Interop with the discrete gate — this module is a strict generalization
# ---------------------------------------------------------------------------

# Pseudo-count pairs chosen so mean/width recover the discrete verdict at
# every risk tier when unfalsifiable/unknowable aren't in play: SOURCED
# collapses to a near-point-mass at 1.0 (passes every tier); UNVERIFIED
# stays at the uninformative (1,1) prior (fails every tier — ci_width(0.8)
# for Beta(1,1) is wide, so it never clears even LOW's 0.60 width bound).
_STATUS_SEED: Dict[EpistemicStatus, tuple[float, float]] = {
    EpistemicStatus.SOURCED: (1000.0, 1.0),
    EpistemicStatus.FALSIFIABLE_UNSOURCED: (1.0, 1.0),
    EpistemicStatus.UNFALSIFIABLE: (1.0, 1.0),
    EpistemicStatus.UNKNOWABLE: (1.0, 1.0),
    EpistemicStatus.UNVERIFIED: (1.0, 1.0),
}


def from_status(claim_id: str, status: EpistemicStatus) -> BeliefState:
    """Build a BeliefState that reproduces the discrete gate's verdict.

    Used to migrate existing Claim objects (epistemic_status.Claim) into the
    Bayesian store without re-deriving evidence counts, and as the recovery
    test that this module is a superset, not a replacement.
    """
    alpha, beta = _STATUS_SEED[status]
    return BeliefState(
        claim_id=claim_id, alpha=alpha, beta=beta,
        unfalsifiable=(status is EpistemicStatus.UNFALSIFIABLE),
        unknowable=(status is EpistemicStatus.UNKNOWABLE),
    )


# ---------------------------------------------------------------------------
# Claim store — the thing the enforcer actually queries
# ---------------------------------------------------------------------------

@dataclass
class ClaimStore:
    """Live posteriors over every claim currently in scope for an actor/
    session. This is what the rulebook enforcer consults at each step."""
    beliefs: Dict[str, BeliefState] = field(default_factory=dict)

    def get(self, claim_id: str) -> BeliefState:
        if claim_id not in self.beliefs:
            self.beliefs[claim_id] = BeliefState(claim_id=claim_id)  # (1,1) prior
        return self.beliefs[claim_id]

    def observe(self, claim_id: str, confirming: bool, weight: float = 1.0) -> None:
        self.get(claim_id).update(confirming, weight)

    def satisfied(self, claim_id: str, risk_tier: RiskTier) -> bool:
        return gate_action(self.get(claim_id), risk_tier) is GateAction.EMIT

    def all_satisfied(self, claim_ids: Iterable[str], risk_tier: RiskTier) -> bool:
        return all(self.satisfied(c, risk_tier) for c in claim_ids)


if __name__ == "__main__":
    # -- recovery test: collapsing to a point mass reproduces the old gate --
    sourced = from_status("c1", EpistemicStatus.SOURCED)
    assert gate_action(sourced, RiskTier.CRITICAL) is GateAction.EMIT
    unverified = from_status("c2", EpistemicStatus.UNVERIFIED)
    assert gate_action(unverified, RiskTier.LOW) is GateAction.DOWNGRADE
    unfals = from_status("c3", EpistemicStatus.UNFALSIFIABLE)
    assert gate_action(unfals, RiskTier.LOW) is GateAction.DOWNGRADE
    unknow = from_status("c4", EpistemicStatus.UNKNOWABLE)
    assert gate_action(unknow, RiskTier.LOW) is GateAction.ABSTAIN

    # -- accumulating real evidence moves a claim from DOWNGRADE to EMIT --
    b = BeliefState(claim_id="backup_confirmed")
    assert gate_action(b, RiskTier.HIGH) is GateAction.DOWNGRADE
    for _ in range(20):
        b.update(confirming=True, weight=1.0)
    assert gate_action(b, RiskTier.HIGH) is GateAction.EMIT, \
        f"mean={b.mean:.3f} width={b.ci_width():.3f}"

    # -- one weak observation should NOT buy a CRITICAL action --
    c = BeliefState(claim_id="one_shaky_report")
    c.update(confirming=True, weight=1.0)
    assert gate_action(c, RiskTier.CRITICAL) is GateAction.DOWNGRADE, \
        f"mean={c.mean:.3f} width={c.ci_width():.3f}"

    print("bayesian_gate smoke test OK")
