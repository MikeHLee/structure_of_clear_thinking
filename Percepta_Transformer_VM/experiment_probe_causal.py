"""
experiment_probe_causal.py — three follow-ups to the attention-pattern probe.

1. HEAD ABLATION (causal): zero the output of the top soft-KG heads in the
   fine-tuned model and measure enforcer-off soundness + KL. Matched
   random-head ablations are the control. If rule-following lives in those
   heads, soundness drops; if it lives downstream (the working theory),
   ablating them is survivable.

2. QK GEOMETRY: pre-softmax, pre-RoPE q·k alignment between type-token
   positions (content-only by construction — RoPE hasn't injected position
   yet). Patterns can wash structure out through softmax + causal masking;
   geometry may hold it. Also the before/after contrast: does fine-tuning
   move QK geometry? (Working theory: barely.)

3. REGION PROBE (run_region_probe): fine-tune on the merged graph's train
   region, then compare edge-signal (patterns + QK) for train-region rules
   vs held-out-only rules. If geometry mirrors the behavioral 92.5%-vs-15.8%
   generalization split, the "regional map, not mapmaking" account gets
   mechanistic support.

GQA note: Qwen2.5 uses grouped KV heads; q-head h reads kv-head
h // (n_q_heads // n_kv_heads). Handled via model config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from experiment_constraint_ft import (  # noqa: E402
    LMPrior, tlts_from_text2kg, tlts_merged_text2kg, start_states,
    gen_enforced, evaluate, soundness_off)
from experiment_attention_probe import (  # noqa: E402
    reachability, pair_class, auc, random_sequences, type_positions,
    summarize)


# ---------------------------------------------------------------------------
# QK geometry probe
# ---------------------------------------------------------------------------

def qk_probe(prior: LMPrior, tlts, stimuli: Dict[str, List[str]]) -> dict:
    """Per-(layer, q-head) AUC of pre-RoPE q·k scores predicting rule-edge
    membership between type positions. Captures q_proj/k_proj outputs via
    hooks, so LoRA deltas are included after fine-tuning."""
    torch = prior.torch
    cfg = prior.model.config
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    group = n_q // n_kv
    head_dim = cfg.hidden_size // n_q

    captured: Dict[str, "object"] = {}
    hooks = []
    for name, mod in prior.model.named_modules():
        if name.endswith("self_attn.q_proj") or name.endswith("self_attn.k_proj"):
            def make(nm):
                def hook(module, args, output):
                    captured[nm] = output.detach()
                return hook
            hooks.append(mod.register_forward_hook(make(name)))

    def layer_of(name: str) -> int:
        parts = name.split(".")
        return int(parts[parts.index("layers") + 1])

    edges = {(a, b) for a, _, b in tlts.delta}
    reach = reachability(tlts)
    results = {}
    try:
        for set_name, texts in stimuli.items():
            buckets: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
            for text in texts:
                captured.clear()
                enc = prior.tokenizer(text, return_tensors="pt").to(prior.device)
                with torch.no_grad():
                    prior.model(**enc)
                qs = {layer_of(n): t for n, t in captured.items()
                      if n.endswith("q_proj")}
                ks = {layer_of(n): t for n, t in captured.items()
                      if n.endswith("k_proj")}
                tpos = type_positions(prior.tokenizer, text, tlts.types)
                pairs = []
                for ki in range(len(tpos)):
                    for qi in range(ki + 1, len(tpos)):
                        if qi - ki == 1:
                            continue
                        kpos, kt = tpos[ki]
                        qpos, qt = tpos[qi]
                        pairs.append((kpos, qpos,
                                      pair_class(kt, qt, edges, reach)))
                if not pairs:
                    continue
                for L in qs:
                    q = qs[L][0].float()  # [S, n_q*d]
                    k = ks[L][0].float()  # [S, n_kv*d]
                    for h in range(n_q):
                        qh = q[:, h * head_dim:(h + 1) * head_dim]
                        kv = h // group
                        kh = k[:, kv * head_dim:(kv + 1) * head_dim]
                        b = buckets.setdefault((L, h), {"edge": [],
                                                        "reachable": [],
                                                        "unreachable": []})
                        for kpos, qpos, cls in pairs:
                            s = float(qh[qpos] @ kh[kpos]) / head_dim ** 0.5
                            b[cls].append(s)
            heads = {}
            for (L, H), b in buckets.items():
                heads[f"L{L}.H{H}"] = {
                    "auc_edge_vs_unreachable": auc(b["edge"], b["unreachable"]),
                    "auc_edge_vs_reachable": auc(b["edge"], b["reachable"]),
                    "n": {k: len(v) for k, v in b.items()},
                }
            results[set_name] = heads
    finally:
        for h in hooks:
            h.remove()
    return results


# ---------------------------------------------------------------------------
# Head ablation
# ---------------------------------------------------------------------------

class HeadAblation:
    """Zero the o_proj input slice of chosen (layer, head) pairs — removes
    those heads' contribution to the residual stream."""

    def __init__(self, prior: LMPrior, heads: List[Tuple[int, int]]):
        self.prior = prior
        self.heads = heads
        self.hooks = []

    def __enter__(self):
        cfg = self.prior.model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        by_layer: Dict[int, List[int]] = {}
        for L, H in self.heads:
            by_layer.setdefault(L, []).append(H)
        for name, mod in self.prior.model.named_modules():
            if not name.endswith("self_attn.o_proj"):
                continue
            parts = name.split(".")
            L = int(parts[parts.index("layers") + 1])
            if L not in by_layer:
                continue

            def make(hs):
                def hook(module, args):
                    x = args[0].clone()
                    for H in hs:
                        x[..., H * head_dim:(H + 1) * head_dim] = 0
                    return (x,) + tuple(args[1:])
                return hook
            self.hooks.append(mod.register_forward_pre_hook(make(by_layer[L])))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def parse_head(name: str) -> Tuple[int, int]:
    L, H = name.split(".")
    return int(L[1:]), int(H[1:])


# ---------------------------------------------------------------------------
# Experiment 1+2: QK before/after + ablation battery on one ontology
# ---------------------------------------------------------------------------

def run_causal(model_name: str, out_path: str, ontology_path: str,
               n_stimuli: int = 30, train_steps: int = 200,
               batch_size: int = 8, n_train_traj: int = 128,
               n_sound: int = 100, n_eval: int = 30, seed: int = 42,
               topk: int = 8) -> dict:
    import random
    rng = random.Random(seed)
    tlts, info = tlts_from_text2kg(ontology_path)
    starts = start_states(tlts)
    print(f"ontology: {json.dumps(info)}", flush=True)

    t0 = time.time()
    prior = LMPrior(model_name, attn_implementation="eager")
    print(f"[{time.time()-t0:.0f}s] model loaded", flush=True)

    stimuli = {
        "valid_trajectories": [gen_enforced(prior, tlts, rng, starts=starts)[0]
                               for _ in range(n_stimuli)],
        "random_sequences": random_sequences(tlts, n_stimuli, rng),
    }
    qk_before = qk_probe(prior, tlts, stimuli)
    print(f"[{time.time()-t0:.0f}s] QK before: "
          f"{json.dumps(summarize(qk_before['random_sequences']))}", flush=True)

    corpus = [gen_enforced(prior, tlts, rng, starts=starts)[0]
              for _ in range(n_train_traj)]
    for step in range(1, train_steps + 1):
        loss = prior.train_step(rng.sample(corpus, min(batch_size, len(corpus))))
        if step % 100 == 0:
            print(f"[{time.time()-t0:.0f}s] step {step}: loss {loss:.4f}",
                  flush=True)

    qk_after = qk_probe(prior, tlts, stimuli)
    print(f"[{time.time()-t0:.0f}s] QK after: "
          f"{json.dumps(summarize(qk_after['random_sequences']))}", flush=True)

    # ---- ablation battery on the fine-tuned model ----
    ranked = sorted(
        ((h, d["auc_edge_vs_unreachable"])
         for h, d in qk_after["random_sequences"].items()
         if d["auc_edge_vs_unreachable"] is not None),
        key=lambda x: -x[1])
    top_heads = [parse_head(h) for h, _ in ranked[:topk]]
    all_heads = [parse_head(h) for h, _ in ranked]

    def battery(tag, heads):
        srng = random.Random(seed + 1000)  # same eval seed per condition
        if heads:
            with HeadAblation(prior, heads):
                s = soundness_off(prior, tlts, n_sound, srng, starts=starts)
                ev = evaluate(prior, tlts, n_eval, srng, starts=starts)
        else:
            s = soundness_off(prior, tlts, n_sound, srng, starts=starts)
            ev = evaluate(prior, tlts, n_eval, srng, starts=starts)
        print(f"[{time.time()-t0:.0f}s] ablate {tag}: off-soundness {s:.1f}%, "
              f"KL {ev.mean_kl_per_step:.3f}", flush=True)
        return {"heads": [f"L{L}.H{H}" for L, H in heads],
                "off_soundness": s, "kl_per_step": ev.mean_kl_per_step}

    conditions = {"none": battery("none", [])}
    conditions[f"top{topk}_softkg"] = battery(f"top{topk}", top_heads)
    conditions[f"top{topk//2}_softkg"] = battery(
        f"top{topk//2}", top_heads[:topk // 2])
    for i in range(2):
        rrng = random.Random(seed + i)
        rand = rrng.sample(all_heads, topk)
        conditions[f"random{topk}_{i}"] = battery(f"random{topk}_{i}", rand)

    results = {
        "experiment": "qk_geometry_and_head_ablation",
        "model_name": model_name, "ontology_info": info,
        "qk_before": qk_before, "qk_after": qk_after,
        "qk_summary": {
            "before_random": summarize(qk_before["random_sequences"]),
            "after_random": summarize(qk_after["random_sequences"]),
            "before_valid_edge_vs_reach": summarize(
                qk_before["valid_trajectories"], "auc_edge_vs_reachable"),
            "after_valid_edge_vs_reach": summarize(
                qk_after["valid_trajectories"], "auc_edge_vs_reachable"),
        },
        "ablation": conditions,
        "config": {"n_stimuli": n_stimuli, "train_steps": train_steps,
                   "n_sound": n_sound, "topk": topk, "seed": seed,
                   "wall_seconds": round(time.time() - t0, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Experiment 3: region probe on the merged graph
# ---------------------------------------------------------------------------

def run_region_probe(model_name: str, out_path: str, ontology_dir: str,
                     holdout_files: List[str], n_stimuli: int = 40,
                     train_steps: int = 300, batch_size: int = 8,
                     n_train_traj: int = 128, seed: int = 42) -> dict:
    """Fine-tune on the merged graph's train region, then compare the QK
    edge-signal for train-region rules vs held-out-only rules, using random
    sequences over the FULL type set (position-free, both regions present)."""
    import random
    rng = random.Random(seed)
    full, train_tlts, info = tlts_merged_text2kg(ontology_dir, holdout_files)
    train_starts = [t for t in start_states(train_tlts)
                    if t not in set(info["holdout_start_states"])]
    print(f"merged: {json.dumps(info)}", flush=True)

    t0 = time.time()
    prior = LMPrior(model_name, attn_implementation="eager")
    print(f"[{time.time()-t0:.0f}s] model loaded", flush=True)

    stimuli = {"random_sequences": random_sequences(full, n_stimuli, rng)}

    def region_auc(qk):
        """Split edge pairs into train-region vs holdout-only rules and
        score each against the same unreachable negatives."""
        heads = qk["random_sequences"]
        return heads  # per-head split happens below via re-bucketing

    # per-head, per-region AUC needs raw re-bucketing — do it inline
    train_edges = {(a, b) for a, _, b in train_tlts.delta}
    holdout_edges = {(a, b) for a, _, b in full.delta} - train_edges

    def qk_region(prior):
        torch = prior.torch
        cfg = prior.model.config
        n_q = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_q)
        group, head_dim = n_q // n_kv, cfg.hidden_size // n_q
        captured, hooks = {}, []
        for name, mod in prior.model.named_modules():
            if name.endswith("self_attn.q_proj") or name.endswith("self_attn.k_proj"):
                def make(nm):
                    def hook(m, a, o):
                        captured[nm] = o.detach()
                    return hook
                hooks.append(mod.register_forward_hook(make(name)))
        reach = reachability(full)
        buckets: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        try:
            for text in stimuli["random_sequences"]:
                captured.clear()
                enc = prior.tokenizer(text, return_tensors="pt").to(prior.device)
                with torch.no_grad():
                    prior.model(**enc)
                def layer_of(n):
                    p = n.split(".")
                    return int(p[p.index("layers") + 1])
                qs = {layer_of(n): t for n, t in captured.items() if n.endswith("q_proj")}
                ks = {layer_of(n): t for n, t in captured.items() if n.endswith("k_proj")}
                tpos = type_positions(prior.tokenizer, text, full.types)
                pairs = []
                for ki in range(len(tpos)):
                    for qi in range(ki + 1, len(tpos)):
                        if qi - ki == 1:
                            continue
                        kpos, kt = tpos[ki]
                        qpos, qt = tpos[qi]
                        if (kt, qt) in train_edges:
                            cls = "train_edge"
                        elif (kt, qt) in holdout_edges:
                            cls = "holdout_edge"
                        elif qt in reach.get(kt, set()):
                            cls = "reachable"
                        else:
                            cls = "unreachable"
                        pairs.append((kpos, qpos, cls))
                for L in qs:
                    q, k = qs[L][0].float(), ks[L][0].float()
                    for h in range(n_q):
                        qh = q[:, h * head_dim:(h + 1) * head_dim]
                        kh = k[:, (h // group) * head_dim:
                               (h // group + 1) * head_dim]
                        b = buckets.setdefault((L, h), {
                            "train_edge": [], "holdout_edge": [],
                            "reachable": [], "unreachable": []})
                        for kpos, qpos, cls in pairs:
                            b[cls].append(float(qh[qpos] @ kh[kpos])
                                          / head_dim ** 0.5)
        finally:
            for h in hooks:
                h.remove()
        heads = {}
        for (L, H), b in buckets.items():
            heads[f"L{L}.H{H}"] = {
                "auc_train_edge_vs_unreachable": auc(b["train_edge"],
                                                     b["unreachable"]),
                "auc_holdout_edge_vs_unreachable": auc(b["holdout_edge"],
                                                       b["unreachable"]),
                "n": {k: len(v) for k, v in b.items()},
            }
        return heads

    before = qk_region(prior)
    print(f"[{time.time()-t0:.0f}s] region QK before done", flush=True)

    corpus = [gen_enforced(prior, train_tlts, rng, starts=train_starts)[0]
              for _ in range(n_train_traj)]
    print(f"[{time.time()-t0:.0f}s] corpus built", flush=True)
    for step in range(1, train_steps + 1):
        loss = prior.train_step(rng.sample(corpus, min(batch_size, len(corpus))))
        if step % 100 == 0:
            print(f"[{time.time()-t0:.0f}s] step {step}: loss {loss:.4f}",
                  flush=True)

    after = qk_region(prior)
    print(f"[{time.time()-t0:.0f}s] region QK after done", flush=True)

    def summ(heads, key):
        vals = [v[key] for v in heads.values() if v[key] is not None]
        vals.sort(reverse=True)
        return {"mean": sum(vals) / max(len(vals), 1), "top5": vals[:5]}

    results = {
        "experiment": "qk_region_probe_merged_graph",
        "model_name": model_name, "merged_info": info,
        "before": before, "after": after,
        "summary": {
            phase: {"train_edge": summ(h, "auc_train_edge_vs_unreachable"),
                    "holdout_edge": summ(h, "auc_holdout_edge_vs_unreachable")}
            for phase, h in (("before", before), ("after", after))
        },
        "config": {"n_stimuli": n_stimuli, "train_steps": train_steps,
                   "holdout_files": holdout_files, "seed": seed,
                   "wall_seconds": round(time.time() - t0, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["causal", "region"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--ontology", default=None)
    p.add_argument("--ontology-dir", default=None)
    p.add_argument("--holdout", default="6_politician_ontology.json,9_astronaut_ontology.json,15_sportsteam_ontology.json")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.mode == "causal":
        kw = dict(n_stimuli=4, train_steps=6, n_train_traj=8, n_sound=10,
                  n_eval=4, topk=4) if args.smoke else {}
        run_causal(args.model if not args.smoke else "HuggingFaceTB/SmolLM2-135M",
                   args.out, args.ontology, **kw)
    else:
        holdouts = [h.strip() for h in args.holdout.split(",")]
        kw = dict(n_stimuli=4, train_steps=4, n_train_traj=6) if args.smoke else {}
        run_region_probe(args.model if not args.smoke else "HuggingFaceTB/SmolLM2-135M",
                         args.out, args.ontology_dir, holdouts, **kw)


if __name__ == "__main__":
    main()
