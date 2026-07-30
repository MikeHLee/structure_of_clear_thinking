"""
experiment_attention_probe.py — do attention heads reproduce the ontology?

Motivating claim (to support or refute): frontier LMs implicitly implement
soft knowledge graphs — and fine-tuning on a rulebook installs that graph
into identifiable attention structure.

Design. For a fixed ontology (TLTS), feed the model two stimulus sets:
  (a) valid enforced trajectories ("A -l-> B -m-> C"), and
  (b) random type/label sequences with the same surface format (invalid) —
      these decouple graph structure from sequence position.
For every ordered pair of type-token positions (key at i, query at j, i<j)
we record the attention weight per (layer, head), and label the pair by the
ontology relation between the KEY type and the QUERY type:
  'edge'        — a direct rule key_type -> query_type exists in delta
  'reachable'   — reachable in the transitive closure but no direct rule
  'unreachable' — neither
(the same three-way taxonomy the loci study used at the mask level).
Adjacent-in-sequence pairs are excluded from scoring: in valid trajectories
adjacency and edge-ness coincide, so they'd smuggle position into the AUC.

Per (layer, head) we report AUC(edge vs unreachable) and
AUC(edge vs reachable-only). Heads whose attention ranks direct rules above
non-rules are candidate "graph heads". We probe the SAME model before and
after constraint-aware fine-tuning (train loop reused from
experiment_constraint_ft) — the before/after contrast shows whether
fine-tuning installs the graph or it was already softly present.

Output JSON: per-(layer,head) AUCs for both stimulus sets, before + after,
plus summary (top heads, mean AUC). Figures are generated separately.

Run via scripts/modal_constraint_ft.py::probe_attention (L4) after the
CPU smoke (::smoke_probe).
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

from experiment_loci_comparison import TLTS  # noqa: E402
from experiment_constraint_ft import (  # noqa: E402
    LMPrior, tlts_from_text2kg, start_states, gen_enforced, serialize)


# ---------------------------------------------------------------------------
# Pair taxonomy
# ---------------------------------------------------------------------------

def reachability(tlts: TLTS) -> Dict[str, set]:
    reach: Dict[str, set] = {t: set() for t in tlts.types}
    for t in tlts.types:
        frontier = {t2 for _, _, t2 in [(a, l, b) for a, l, b in tlts.delta
                                        if a == t]}
        seen = set(frontier)
        while frontier:
            nxt = set()
            for u in frontier:
                for a, _, b in tlts.delta:
                    if a == u and b not in seen:
                        nxt.add(b)
            seen |= nxt
            frontier = nxt
        reach[t] = seen
    return reach


def pair_class(key_t: str, query_t: str, edges: set,
               reach: Dict[str, set]) -> str:
    if (key_t, query_t) in edges:
        return "edge"
    if query_t in reach.get(key_t, set()):
        return "reachable"
    return "unreachable"


def auc(pos: List[float], neg: List[float]) -> Optional[float]:
    """Rank-based AUC without sklearn. None if either class is empty."""
    if not pos or not neg:
        return None
    combined = sorted((v, 1) for v in pos)
    combined += sorted((v, 0) for v in neg)
    combined.sort(key=lambda x: x[0])
    rank_sum, rank = 0.0, 1
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if combined[k][1] == 1:
                rank_sum += avg_rank
        rank += j - i
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------

def random_sequences(tlts: TLTS, n: int, rng, length: int = 7) -> List[str]:
    """Same surface format, uniformly random types + labels — invalid text
    that decouples ontology structure from sequence position."""
    out = []
    for _ in range(n):
        t = rng.choice(tlts.types)
        steps = [(rng.choice(tlts.labels), rng.choice(tlts.types))
                 for _ in range(length)]
        out.append(serialize(t, steps))
    return out


def type_positions(tokenizer, text: str, types: List[str]
                   ) -> List[Tuple[int, str]]:
    """(last-token-position, type_name) for each type occurrence, via
    character offsets. Longest-match-first prevents substring collisions."""
    enc = tokenizer(text, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    hits: List[Tuple[int, int, str]] = []  # (char_start, char_end, type)
    taken: List[Tuple[int, int]] = []
    for t in sorted(types, key=len, reverse=True):
        start = 0
        while True:
            idx = text.find(t, start)
            if idx < 0:
                break
            span = (idx, idx + len(t))
            if not any(a < span[1] and span[0] < b for a, b in taken):
                hits.append((span[0], span[1], t))
                taken.append(span)
            start = idx + 1
    out = []
    for cs, ce, t in sorted(hits):
        last = None
        for pos, (a, b) in enumerate(offsets):
            if a < ce and cs < b:
                last = pos
        if last is not None:
            out.append((last, t))
    return out


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(prior: LMPrior, tlts: TLTS, stimuli: Dict[str, List[str]]
          ) -> dict:
    """Collect per-(layer,head) attention AUCs for each stimulus set."""
    torch = prior.torch
    edges = {(a, b) for a, _, b in tlts.delta}
    reach = reachability(tlts)
    results = {}
    for set_name, texts in stimuli.items():
        # per (layer, head): {class: [attn values]}
        buckets: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        for text in texts:
            enc = prior.tokenizer(text, return_tensors="pt").to(prior.device)
            with torch.no_grad():
                out = prior.model(**enc, output_attentions=True)
            if out.attentions is None or out.attentions[0] is None:
                raise RuntimeError("no attentions — load model with "
                                   "attn_implementation='eager'")
            tpos = type_positions(prior.tokenizer, text, tlts.types)
            pairs = []
            for ki in range(len(tpos)):
                for qi in range(ki + 1, len(tpos)):
                    if qi - ki == 1:
                        continue  # exclude sequence-adjacent pairs
                    kpos, kt = tpos[ki]
                    qpos, qt = tpos[qi]
                    pairs.append((kpos, qpos, pair_class(kt, qt, edges, reach)))
            if not pairs:
                continue
            for L, att in enumerate(out.attentions):
                a = att[0].float()  # [H, S, S]
                for H in range(a.shape[0]):
                    b = buckets.setdefault((L, H), {"edge": [], "reachable": [],
                                                    "unreachable": []})
                    for kpos, qpos, cls in pairs:
                        b[cls].append(a[H, qpos, kpos].item())
        heads = {}
        for (L, H), b in buckets.items():
            heads[f"L{L}.H{H}"] = {
                "auc_edge_vs_unreachable": auc(b["edge"], b["unreachable"]),
                "auc_edge_vs_reachable": auc(b["edge"], b["reachable"]),
                "n": {k: len(v) for k, v in b.items()},
            }
        results[set_name] = heads
    return results


def summarize(heads: dict, key: str = "auc_edge_vs_unreachable") -> dict:
    vals = [(h, d[key]) for h, d in heads.items() if d[key] is not None]
    vals.sort(key=lambda x: -x[1])
    return {"top5": vals[:5],
            "mean_auc": sum(v for _, v in vals) / max(len(vals), 1),
            "n_heads_above_0.8": sum(1 for _, v in vals if v > 0.8)}


# ---------------------------------------------------------------------------
# Experiment: probe -> fine-tune -> probe
# ---------------------------------------------------------------------------

def run_probe(model_name: str, out_path: str, ontology_path: str,
              n_traj_stimuli: int = 30, n_random_stimuli: int = 30,
              train_steps: int = 200, batch_size: int = 8,
              n_train_traj: int = 128, seed: int = 42) -> dict:
    import random
    rng = random.Random(seed)
    tlts, info = tlts_from_text2kg(ontology_path)
    starts = start_states(tlts)
    print(f"ontology: {json.dumps(info)}", flush=True)

    t0 = time.time()
    # eager attention so output_attentions returns real weights
    prior = LMPrior(model_name, attn_implementation="eager")
    print(f"[{time.time()-t0:.0f}s] model loaded", flush=True)

    stimuli = {
        "valid_trajectories": [gen_enforced(prior, tlts, rng, starts=starts)[0]
                               for _ in range(n_traj_stimuli)],
        "random_sequences": random_sequences(tlts, n_random_stimuli, rng),
    }
    print(f"[{time.time()-t0:.0f}s] stimuli built", flush=True)

    before = probe(prior, tlts, stimuli)
    print(f"[{time.time()-t0:.0f}s] BEFORE probe done — valid-traj summary: "
          f"{json.dumps(summarize(before['valid_trajectories']))}", flush=True)

    corpus = [gen_enforced(prior, tlts, rng, starts=starts)[0]
              for _ in range(n_train_traj)]
    for step in range(1, train_steps + 1):
        loss = prior.train_step(rng.sample(corpus, min(batch_size, len(corpus))))
        if step % 50 == 0:
            print(f"[{time.time()-t0:.0f}s] step {step}: loss {loss:.4f}",
                  flush=True)

    after = probe(prior, tlts, stimuli)
    print(f"[{time.time()-t0:.0f}s] AFTER probe done — valid-traj summary: "
          f"{json.dumps(summarize(after['valid_trajectories']))}", flush=True)

    results = {
        "experiment": "attention_graph_probe",
        "model_name": model_name,
        "ontology_info": info,
        "before": before, "after": after,
        "summary": {
            # valid trajectories have NO unreachable pairs by construction
            # (you walked there), so their contrast is edge-vs-reachable;
            # random sequences support both contrasts.
            phase: {
                "valid_trajectories": summarize(
                    phases["valid_trajectories"], "auc_edge_vs_reachable"),
                "random_sequences": {
                    "edge_vs_unreachable": summarize(
                        phases["random_sequences"], "auc_edge_vs_unreachable"),
                    "edge_vs_reachable": summarize(
                        phases["random_sequences"], "auc_edge_vs_reachable"),
                },
            }
            for phase, phases in (("before", before), ("after", after))
        },
        "config": {"n_traj_stimuli": n_traj_stimuli,
                   "n_random_stimuli": n_random_stimuli,
                   "train_steps": train_steps, "seed": seed,
                   "adjacent_pairs_excluded": True,
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
    p.add_argument("--ontology", required=True)
    p.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                 "attention_probe.json"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        run_probe("HuggingFaceTB/SmolLM2-135M", args.out, args.ontology,
                  n_traj_stimuli=4, n_random_stimuli=4, train_steps=6,
                  n_train_traj=8)
    else:
        run_probe(args.model, args.out, args.ontology)


if __name__ == "__main__":
    main()
