"""
experiment_agentic_ft.py — thread-1 protocol (enforcer on/off valid-sequence
rate) applied to the agentic tool-call rulebook, via a real LM prior.

Correction to an earlier claim in this session: this file's real-model path
needs torch/transformers, which are NOT available in this container (no
`modal` package, no local torch — see session status). So the "CPU smoke"
step is not actually Modal-free; it still runs through Modal's `smoke_agentic`
function (tiny model, CPU tier) — same as every other CPU-smoke step in this
project. What IS Modal-free and already done: the enforcer logic itself
(agentic_ontology.py, tested in tests/test_agentic_ontology.py, all passing,
torch-free) and this file's --fake path below (FakePrior, no torch), which
validates the wiring between the rulebook JSON and the measurement loop
before any GPU/CPU-tier Modal spend.

Protocol (mirrors experiment_constraint_ft.py / thread 1 exactly):
  - state is fixed at "Session" throughout a trajectory (tool calls don't
    change type the way shop-ontology steps do — an agent session issues
    many calls from the same state). Trajectory length is a call budget,
    not a chain depth.
  - actor holds ONE role for the whole trajectory (persona fixed per run).
  - claims: for run 1 (this file), every precondition claim id in the
    rulebook is pre-seeded into the ClaimStore with strong confirming
    evidence, so the Bayesian gate clears uniformly — i.e. claims are
    collapsed to boolean-SOURCED per the sequencing plan (measure the base
    role+quantity effect before turning on live claim uncertainty). Live
    claims are a follow-up run, not this one — see run(seed_claims=False).
  - enforcer OFF: sample from the raw LM label distribution over ALL
    rulebook labels; a trajectory is sound iff every sampled label is
    admissible_for (type + role + preconditions + quantity) at the state
    it was sampled from.
  - enforcer ON (D): mask to admissible_for labels, renormalize, sample —
    100% sound by construction, exactly thread 1's guarantee.

Run (local, no torch, validates wiring only):
    python experiment_agentic_ft.py --fake
Run (real model, via Modal — see scripts/modal_constraint_ft.py::smoke_agentic
     / train_agentic once access is unblocked):
    modal run scripts/modal_constraint_ft.py::smoke_agentic
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SRC = os.path.join(HERE, "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiment_constraint_ft import LMPrior, FakePrior  # noqa: E402
from agentic_ontology import AgenticTLTS, Rule, ActorState, check_step  # noqa: E402
from bayesian_gate import ClaimStore, RiskTier  # noqa: E402


# ---------------------------------------------------------------------------
# Rulebook loading
# ---------------------------------------------------------------------------

def load_rulebook(path: str) -> AgenticTLTS:
    with open(path) as f:
        d = json.load(f)
    tier_map = {t.value: t for t in RiskTier}
    rules = [
        Rule(
            domain=r["domain"], label=r["label"], range=r["range"],
            roles=tuple(r.get("roles", ())),
            preconditions=tuple(r.get("preconditions", ())),
            risk_tier=tier_map[r.get("risk_tier", "low")],
            max_per_window=r.get("max_per_window"),
            window_seconds=r.get("window_seconds"),
        )
        for r in d["rules"]
    ]
    return AgenticTLTS(types=d["types"], labels=d["labels"], rules=rules)


def seeded_claim_store(tlts: AgenticTLTS, confirms: int = 50) -> ClaimStore:
    """Pre-satisfy every precondition claim in the rulebook — the 'claims
    collapsed to boolean SOURCED' stance for this first run."""
    store = ClaimStore()
    claim_ids = {c for r in tlts.rules for c in r.preconditions}
    for cid in claim_ids:
        for _ in range(confirms):
            store.observe(cid, confirming=True)
    return store


# ---------------------------------------------------------------------------
# Serialization — same convention as experiment_constraint_ft.serialize
# ---------------------------------------------------------------------------

def serialize(calls: List[str]) -> str:
    if not calls:
        return "Session"
    return "Session " + " ".join(f"-{c}-> ToolResult ->" for c in calls) + " Session"


# ---------------------------------------------------------------------------
# Measurement — thread-1 protocol
# ---------------------------------------------------------------------------

@dataclass
class AgenticEvalResult:
    soundness_enforcer_off: float
    soundness_enforcer_on: float  # 100.0 by construction, kept for parity with thread-1 tables
    denial_reasons: Dict[str, int]  # tally of WHY off-trajectories failed


def _admissible_labels(tlts: AgenticTLTS, state: str, actor: ActorState,
                       claims: ClaimStore, now: Optional[float] = None) -> Dict[str, str]:
    return {l: t for l, (t, _r) in tlts.admissible_for(state, actor, claims, now).items()}


def soundness_off(prior, tlts: AgenticTLTS, actor: ActorState, claims: ClaimStore,
                  n_traj: int, rng, max_calls: int = 8) -> tuple[float, Dict[str, int]]:
    """Enforcer OFF: sample from the raw label distribution over the FULL
    rulebook label vocabulary; a trajectory is sound iff every sampled
    label clears admissible_for at the state it was sampled from."""
    ok = 0
    reasons: Dict[str, int] = {}
    for _ in range(n_traj):
        calls: List[str] = []
        sound = True
        local_actor = ActorState(actor_id=actor.actor_id, roles=actor.roles)
        for step in range(max_calls):
            dist = prior.label_dist(serialize(calls), tlts.labels)
            label = rng.choices(list(dist), weights=list(dist.values()))[0]
            adm = _admissible_labels(tlts, "Session", local_actor, claims)
            if label not in adm:
                sound = False
                reason = tlts.denial_reason("Session", label, local_actor, claims)
                reasons[reason] = reasons.get(reason, 0) + 1
                break
            local_actor.record_call(label)
            calls.append(label)
        ok += sound
    return 100.0 * ok / n_traj, reasons


def gen_enforced(prior, tlts: AgenticTLTS, actor: ActorState, claims: ClaimStore,
                 rng, max_calls: int = 8) -> List[str]:
    """(D) pre-decoder enforcement: mask to admissible_for labels, renormalize.
    100% sound by construction — the thread-1 guarantee, extended to the
    role+precondition+quantity gates."""
    local_actor = ActorState(actor_id=actor.actor_id, roles=actor.roles)
    calls: List[str] = []
    for _ in range(max_calls):
        adm = _admissible_labels(tlts, "Session", local_actor, claims)
        if not adm:
            break
        dist = prior.label_dist(serialize(calls), tlts.labels)
        masked = {l: dist[l] for l in adm}
        z = sum(masked.values())
        if z == 0:
            masked = {l: 1.0 / len(adm) for l in adm}
        else:
            masked = {l: p / z for l, p in masked.items()}
        label = rng.choices(list(masked), weights=list(masked.values()))[0]
        local_actor.record_call(label)
        calls.append(label)
    return calls


def run(rulebook_path: str, out_path: str, model_name: str = "Qwen/Qwen2.5-1.5B",
        role: str = "editor", n_soundness_traj: int = 300, max_calls: int = 8,
        seed: int = 42, fake: bool = False) -> dict:
    import random
    rng = random.Random(seed)
    tlts = load_rulebook(rulebook_path)
    actor = ActorState(actor_id="agent_1", roles=(role,))
    claims = seeded_claim_store(tlts)  # boolean-SOURCED stance for run 1

    t0 = time.time()
    prior = FakePrior(_wrap_tlts_for_fakeprior(tlts), seed) if fake else LMPrior(model_name)
    print(f"[{time.time()-t0:.0f}s] model loaded: {model_name if not fake else 'FAKE'}",
          flush=True)

    s_off, reasons = soundness_off(prior, tlts, actor, claims, n_soundness_traj,
                                   rng, max_calls=max_calls)
    print(f"[{time.time()-t0:.0f}s] enforcer OFF: {s_off:.1f}% sound; "
          f"denial reasons: {reasons}", flush=True)

    sample_enforced = [gen_enforced(prior, tlts, actor, claims, rng, max_calls=max_calls)
                       for _ in range(20)]
    print(f"[{time.time()-t0:.0f}s] enforcer ON sample trajectories (first 3): "
          f"{sample_enforced[:3]}", flush=True)

    results = {
        "model_name": model_name if not fake else "FAKE (logic test)",
        "rulebook": os.path.basename(rulebook_path),
        "role": role,
        "n_soundness_trajectories": n_soundness_traj,
        "max_calls": max_calls,
        "soundness_enforcer_off": s_off,
        "soundness_enforcer_on": 100.0,  # by construction
        "denial_reasons_off": reasons,
        "claims_stance": "boolean_sourced_seeded",  # not live Bayesian yet — see docstring
        "config": {"seed": seed, "wall_seconds": round(time.time() - t0, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return results


def _wrap_tlts_for_fakeprior(tlts: AgenticTLTS):
    """FakePrior (experiment_constraint_ft.py) expects a plain TLTS with
    .types/.labels for building its per-type favorite-label map. AgenticTLTS
    already exposes both, so this is just a thin shim satisfying that
    duck-typed contract without importing the plain TLTS class."""
    class _Shim:
        types = tlts.types
        labels = tlts.labels
    return _Shim()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rulebook", default=os.path.join(
        HERE, "..", "rulebooks", "agentic_tool_call_v1.json"))
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "results", "agentic_ft_results.json"))
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--role", default="editor")
    p.add_argument("--fake", action="store_true",
                   help="no-torch logic test with a synthetic prior")
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()
    run(args.rulebook, args.out, model_name=args.model, role=args.role,
        n_soundness_traj=args.n, fake=args.fake)


if __name__ == "__main__":
    main()
