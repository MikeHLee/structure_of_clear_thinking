"""
experiment_routing_ft.py — WHERE should rulebook learning go?

Tests the "strengthen the core attention ontology" mitigation: rerun the
merged-graph generalization split with LoRA restricted to different weight
families, and compare held-out-region transfer.

Conditions (fresh adapter per condition, same base model, SAME corpus —
generated once by the base model, so differences are purely in where the
gradient is allowed to write):

  qk_r64   — q_proj/k_proj only (the routing store; r=64 to narrow the
             trainable-parameter gap vs the MLP condition)
  vo_r32   — v_proj/o_proj only (the read-out path; motivated by the
             0.90-AUC held-out head whose knowledge never reached behavior)
  mlp_r16  — gate/up/down projections only (the memorization store)
  all_r16  — all-linear (replicates the v3 baseline)

Predictions under the hard-wired-routing/downstream-overfit theory:
mlp_r16 ≈ all_r16 (regional memorization, poor transfer); qk_r64 trains
slower/worse on train-region but transfers relatively better IF routing
learning generalizes; vo_r32 probes whether coupling existing routing
knowledge to the output is the binding constraint.

Metrics per condition: enforcer-off soundness on train-region and
held-out-region starts (full-rulebook grading), final KL/step + masked
entropy on train region. Shared baseline measured once on the base model.

Capacity caveat (reported in results): trainable params are not exactly
matched across conditions; r is chosen to narrow but not close the gap.
Conclusions should lean on transfer RATIOS, not absolute train-region wins.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from experiment_constraint_ft import (  # noqa: E402
    LMPrior, tlts_merged_text2kg, start_states, gen_enforced, evaluate,
    soundness_off)

CONDITIONS = [
    ("qk_r64", ["q_proj", "k_proj"], 64),
    ("vo_r32", ["v_proj", "o_proj"], 32),
    ("mlp_r16", ["gate_proj", "up_proj", "down_proj"], 16),
    ("all_r16", "all-linear", 16),
]


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_routing(model_name: str, out_path: str, ontology_dir: str,
                holdout_files: List[str], train_steps: int = 500,
                batch_size: int = 8, n_train_traj: int = 256,
                n_sound: int = 80, n_eval: int = 30, seed: int = 42,
                conditions: Optional[List[str]] = None) -> dict:
    import random
    full, train_tlts, info = tlts_merged_text2kg(ontology_dir, holdout_files)
    train_starts = [t for t in start_states(train_tlts)
                    if t not in set(info["holdout_start_states"])]
    holdout_starts = [t for t in info["holdout_start_states"]
                      if full.admissible_from(t)]
    print(f"merged: {json.dumps(info)}", flush=True)

    todo = [c for c in CONDITIONS
            if conditions is None or c[0] in conditions]

    t0 = time.time()
    # scoring-only prior: baseline snapshots + the SHARED corpus
    base = LMPrior(model_name, lora=False)
    rng = random.Random(seed)

    def snapshot(prior, tag):
        srng = random.Random(seed + 999)  # identical eval stream everywhere
        s_train = soundness_off(prior, full, n_sound, srng,
                                starts=train_starts)
        s_hold = soundness_off(prior, full, n_sound, srng,
                               starts=holdout_starts)
        ev = evaluate(prior, full, n_eval, srng, starts=train_starts)
        print(f"[{time.time()-t0:.0f}s] {tag}: train {s_train:.1f}%, "
              f"HELD-OUT {s_hold:.1f}%, KL {ev.mean_kl_per_step:.3f}, "
              f"ent {ev.mean_masked_entropy:.3f}", flush=True)
        return {"train_region": s_train, "holdout_region": s_hold,
                "kl_per_step": ev.mean_kl_per_step,
                "masked_entropy": ev.mean_masked_entropy}

    baseline = snapshot(base, "baseline (base model)")
    corpus = [gen_enforced(base, train_tlts, rng, starts=train_starts)[0]
              for _ in range(n_train_traj)]
    print(f"[{time.time()-t0:.0f}s] shared corpus: {len(corpus)} trajectories",
          flush=True)
    del base
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    results_by_cond: Dict[str, dict] = {}
    for name, targets, r in todo:
        crng = random.Random(seed)  # identical batch order per condition
        prior = LMPrior(model_name, target_modules=targets, lora_r=r)
        n_params = count_trainable(prior.model)
        print(f"[{time.time()-t0:.0f}s] condition {name}: "
              f"targets={targets}, r={r}, trainable={n_params:,}", flush=True)
        for step in range(1, train_steps + 1):
            loss = prior.train_step(
                crng.sample(corpus, min(batch_size, len(corpus))))
            if step % 100 == 0:
                print(f"[{time.time()-t0:.0f}s] {name} step {step}: "
                      f"loss {loss:.4f}", flush=True)
        results_by_cond[name] = snapshot(prior, f"{name} final")
        results_by_cond[name]["trainable_params"] = n_params
        del prior
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    results = {
        "experiment": "routing_targeted_ft_generalization",
        "model_name": model_name, "merged_info": info,
        "baseline": baseline, "conditions": results_by_cond,
        "config": {"train_steps": train_steps, "batch_size": batch_size,
                   "n_train_traj": n_train_traj, "n_sound": n_sound,
                   "holdout_files": holdout_files, "seed": seed,
                   "shared_corpus": True,
                   "capacity_note": "trainable params NOT exactly matched; "
                                    "lean on transfer ratios",
                   "wall_seconds": round(time.time() - t0, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--ontology-dir", required=True)
    p.add_argument("--holdout", default="6_politician_ontology.json,9_astronaut_ontology.json,15_sportsteam_ontology.json")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    holdouts = [h.strip() for h in args.holdout.split(",")]
    if args.smoke:
        run_routing("HuggingFaceTB/SmolLM2-135M", args.out,
                    args.ontology_dir, holdouts, train_steps=4,
                    batch_size=4, n_train_traj=6, n_sound=6, n_eval=3,
                    conditions=["qk_r64", "mlp_r16"])
    else:
        run_routing(args.model, args.out, args.ontology_dir, holdouts)


if __name__ == "__main__":
    main()
