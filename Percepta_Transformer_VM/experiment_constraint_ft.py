"""
experiment_constraint_ft.py — constraint-aware fine-tuning with a real LM.

The trained-model version of the synthetic-prior loci study, and the
experiment behind thread 03 / the paper's "future work" gap:

  1. A real causal LM plays the role of the prior: its distribution over
     the TLTS label set at each state is obtained by scoring each label
     as a text continuation of the serialized trajectory so far.
  2. Trajectories are generated under (D) pre-decoder enforcement
     (mask to admissible labels, renormalize) — 100% sound by
     construction.
  3. The model is fine-tuned (LoRA) on its own enforced trajectories.
  4. At checkpoints we measure:
       - kl_per_step:  KL(masked prior ‖ raw prior) — how hard
         enforcement reshapes the model's choices (drift gauge)
       - logp_enforced: mean log-prob of enforced trajectories under the
         RAW prior — the fluency cost of enforcement
       - logp_masked:   same trajectories under the masked prior — the
         value logp_enforced attains if the model fully internalizes
         the rulebook (KL → 0); its final value is the
         "well-calibrated frontier" reference line
       - soundness_enforcer_off: fraction of trajectories with zero
         illegal steps when sampling from the RAW prior (enforcer OFF).
         Sampling is restricted to the TLTS label vocabulary, which is
         generous to the model; open-vocabulary generation could only
         be worse. Measured before training and at the final checkpoint.

Output: a JSON file matching the schema consumed by
threads/03_teaching_the_rulebook/generate_figures.py.

Run (CPU smoke):   python experiment_constraint_ft.py --smoke
Run (full, GPU):   via scripts/modal_constraint_ft.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from experiment_loci_comparison import TLTS, cyclic_ecommerce_tlts  # noqa: E402


# ---------------------------------------------------------------------------
# Real ontologies (Text2KGBench) as TLTSs
# ---------------------------------------------------------------------------

def tlts_from_text2kg(json_path: str) -> Tuple[TLTS, dict]:
    """Build a TLTS from a Text2KGBench ontology JSON.

    Each relation entry (pid, domain, range) becomes a rule
    (domain, pid, range). TLTS is a deterministic labeled transition
    system — one successor per (type, label) — so duplicate
    (domain, pid) pairs keep their first occurrence; the count of
    dropped duplicates is reported in the info dict.
    """
    with open(json_path) as f:
        d = json.load(f)
    delta: List[Tuple[str, str, str]] = []
    seen = set()
    dropped = 0
    for r in d.get("relations", []):
        dom, pid, rng = r.get("domain"), r.get("pid"), r.get("range")
        if not (dom and pid and rng):
            continue
        if (dom, pid) in seen:
            dropped += 1
            continue
        seen.add((dom, pid))
        delta.append((dom, pid, rng))
    types = sorted({c["qid"] for c in d.get("concepts", [])}
                   | {t for t, _, _ in delta} | {t for _, _, t in delta})
    labels = sorted({l for _, l, _ in delta})
    tlts = TLTS(types=types, labels=labels, delta=delta)
    info = {"ontology_id": d.get("id", os.path.basename(json_path)),
            "title": d.get("title", ""), "n_types": len(types),
            "n_labels": len(labels), "n_rules": len(delta),
            "dropped_duplicate_domain_pid": dropped}
    return tlts, info


def start_states(tlts: TLTS) -> List[str]:
    """Types with at least one outgoing rule — trajectory start candidates."""
    return [t for t in tlts.types if tlts.admissible_from(t)]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize(start: str, steps: List[Tuple[str, str]]) -> str:
    """'Customer -has-> Cart -contains-> Item' from start + [(label, type), …]."""
    out = start
    for label, t2 in steps:
        out += f" -{label}-> {t2}"
    return out


# ---------------------------------------------------------------------------
# Real-LM prior over the TLTS label set
# ---------------------------------------------------------------------------

class LMPrior:
    """Scores each TLTS label as a continuation; softmax over the label set.

    torch/transformers are imported lazily so the module can be tested
    without them (see FakePrior).
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 lora: bool = True, lr: float = 1e-4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(self.device)
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules="all-linear",
                             task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, cfg)
        self.model.train()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

    # -- scoring ------------------------------------------------------------
    def label_dist(self, prefix: str, labels: List[str]) -> Dict[str, float]:
        """p(label | prefix) over the candidate set, from summed token logprobs
        of the continuation ' -{label}->'. All candidates are scored in ONE
        padded batch forward — with real ontologies (40–70 labels) the
        sequential version dominated wall time."""
        torch = self.torch
        prefix_ids = self.tokenizer(prefix, return_tensors="pt").input_ids[0]
        P = prefix_ids.shape[0]
        cont_ids = [self.tokenizer(f" -{l}->", add_special_tokens=False,
                                   return_tensors="pt").input_ids[0]
                    for l in labels]
        lens = [c.shape[0] for c in cont_ids]
        Lmax = P + max(lens)
        pad = self.tokenizer.pad_token_id
        batch = torch.full((len(labels), Lmax), pad, dtype=torch.long)
        mask = torch.zeros((len(labels), Lmax), dtype=torch.long)
        for i, c in enumerate(cont_ids):
            L = P + c.shape[0]
            batch[i, :P] = prefix_ids
            batch[i, P:L] = c
            mask[i, :L] = 1
        batch, mask = batch.to(self.device), mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=batch, attention_mask=mask).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
        scores = []
        for i, c in enumerate(cont_ids):
            s = 0.0
            for j in range(c.shape[0]):
                s += logprobs[i, P + j - 1, batch[i, P + j]].item()
            scores.append(s)
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        return {l: e / z for l, e in zip(labels, exps)}

    # -- training -----------------------------------------------------------
    def train_step(self, texts: List[str]) -> float:
        torch = self.torch
        enc = self.tokenizer([t + self.tokenizer.eos_token for t in texts],
                             return_tensors="pt", padding=True)
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = self.model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return float(out.loss.item())


class FakePrior:
    """Deterministic stand-in for logic tests without torch: a fixed random
    'favorite' label per state gets 70% mass (mirrors the bad_prior of the
    synthetic harness)."""

    def __init__(self, tlts: TLTS, seed: int = 0):
        import random
        rng = random.Random(seed)
        self.fav = {t: rng.choice(tlts.labels) for t in tlts.types}

    def label_dist(self, prefix: str, labels: List[str]) -> Dict[str, float]:
        state = prefix.split()[-1] if "->" in prefix else prefix
        fav = self.fav.get(state, labels[0])
        rest = (1.0 - 0.7) / max(len(labels) - 1, 1)
        return {l: (0.7 if l == fav else rest) for l in labels}

    def train_step(self, texts: List[str]) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Trajectory generation + evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    mean_kl_per_step: float
    mean_logp_enforced: float
    mean_logp_masked: float


def gen_enforced(prior, tlts: TLTS, rng, max_len: int = 12,
                 starts: Optional[List[str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """(D) pre-decoder enforcement: mask to admissible labels, renormalize."""
    start = rng.choice(starts) if starts else "Customer"
    t = start
    steps: List[Tuple[str, str]] = []
    for _ in range(max_len):
        adm = tlts.admissible_from(t)
        if not adm:
            break
        dist = prior.label_dist(serialize(start, steps), tlts.labels)
        masked = {l: dist[l] for l in adm}
        z = sum(masked.values())
        masked = {l: p / z for l, p in masked.items()}
        label = rng.choices(list(masked), weights=list(masked.values()))[0]
        t = adm[label]
        steps.append((label, t))
    return serialize(start, steps), steps


def evaluate(prior, tlts: TLTS, n: int, rng, max_len: int = 12,
             starts: Optional[List[str]] = None) -> EvalResult:
    kls, logps_raw, logps_masked = [], [], []
    for _ in range(n):
        start = rng.choice(starts) if starts else "Customer"
        t = start
        steps: List[Tuple[str, str]] = []
        lp_raw = lp_masked = 0.0
        while len(steps) < max_len:
            adm = tlts.admissible_from(t)
            if not adm:
                break
            dist = prior.label_dist(serialize(start, steps), tlts.labels)
            masked = {l: dist[l] for l in adm}
            z = sum(masked.values())
            masked = {l: p / z for l, p in masked.items()}
            kls.append(sum(p * math.log(p / max(dist[l], 1e-12))
                           for l, p in masked.items() if p > 0))
            label = rng.choices(list(masked), weights=list(masked.values()))[0]
            lp_raw += math.log(max(dist[label], 1e-12))
            lp_masked += math.log(max(masked[label], 1e-12))
            t = adm[label]
            steps.append((label, t))
        logps_raw.append(lp_raw)
        logps_masked.append(lp_masked)
    return EvalResult(
        mean_kl_per_step=sum(kls) / max(len(kls), 1),
        mean_logp_enforced=sum(logps_raw) / max(len(logps_raw), 1),
        mean_logp_masked=sum(logps_masked) / max(len(logps_masked), 1),
    )


def soundness_off(prior, tlts: TLTS, n: int, rng, max_len: int = 12,
                  starts: Optional[List[str]] = None) -> float:
    """Enforcer OFF: sample from the raw label distribution; a trajectory is
    sound iff every sampled label is admissible from its state."""
    ok = 0
    for _ in range(n):
        start = rng.choice(starts) if starts else "Customer"
        t = start
        steps: List[Tuple[str, str]] = []
        sound = True
        while len(steps) < max_len:
            adm = tlts.admissible_from(t)
            if not adm:
                break
            dist = prior.label_dist(serialize(start, steps), tlts.labels)
            label = rng.choices(list(dist), weights=list(dist.values()))[0]
            if label not in adm:
                sound = False
                break
            t = adm[label]
            steps.append((label, t))
        ok += sound
    return 100.0 * ok / n


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run(model_name: str, out_path: str, train_steps: int = 2000,
        eval_every: int = 250, batch_size: int = 8, n_train_traj: int = 512,
        n_eval_traj: int = 100, n_soundness_traj: int = 300,
        seed: int = 42, fake: bool = False,
        ontology_path: Optional[str] = None) -> dict:
    import random
    rng = random.Random(seed)
    if ontology_path:
        tlts, ont_info = tlts_from_text2kg(ontology_path)
        starts = start_states(tlts)
        domain = (f"{ont_info['title'] or ont_info['ontology_id']} "
                  f"({ont_info['n_types']} types, {ont_info['n_rules']} rules)")
        print(f"ontology: {json.dumps(ont_info)}", flush=True)
    else:
        tlts, ont_info = cyclic_ecommerce_tlts(), None
        starts = None
        domain = "7-type cyclic e-commerce ontology"

    t0 = time.time()
    prior = FakePrior(tlts, seed) if fake else LMPrior(model_name)
    print(f"[{time.time()-t0:.0f}s] model loaded: {model_name}", flush=True)

    # baseline eval + enforcer-off soundness
    ev0 = evaluate(prior, tlts, n_eval_traj, rng, starts=starts)
    s_off_before = soundness_off(prior, tlts, n_soundness_traj, rng, starts=starts)
    print(f"[{time.time()-t0:.0f}s] baseline: KL/step {ev0.mean_kl_per_step:.3f}, "
          f"logp {ev0.mean_logp_enforced:.3f}, off-soundness {s_off_before:.1f}%",
          flush=True)

    # training corpus: the model's own enforced outputs
    corpus = [gen_enforced(prior, tlts, rng, starts=starts)[0]
              for _ in range(n_train_traj)]
    print(f"[{time.time()-t0:.0f}s] corpus: {n_train_traj} enforced trajectories",
          flush=True)

    steps_axis = [0]
    kl_series = [ev0.mean_kl_per_step]
    logp_series = [ev0.mean_logp_enforced]
    logp_masked_series = [ev0.mean_logp_masked]

    for step in range(1, train_steps + 1):
        batch = rng.sample(corpus, min(batch_size, len(corpus)))
        loss = prior.train_step(batch)
        if step % eval_every == 0 or step == train_steps:
            ev = evaluate(prior, tlts, n_eval_traj, rng, starts=starts)
            steps_axis.append(step)
            kl_series.append(ev.mean_kl_per_step)
            logp_series.append(ev.mean_logp_enforced)
            logp_masked_series.append(ev.mean_logp_masked)
            print(f"[{time.time()-t0:.0f}s] step {step}: loss {loss:.4f}, "
                  f"KL/step {ev.mean_kl_per_step:.3f}, "
                  f"logp {ev.mean_logp_enforced:.3f}", flush=True)

    s_off_after = soundness_off(prior, tlts, n_soundness_traj, rng, starts=starts)
    print(f"[{time.time()-t0:.0f}s] final off-soundness {s_off_after:.1f}%",
          flush=True)

    results = {
        "model_name": model_name if not fake else "FAKE (logic test)",
        "domain": domain,
        "ontology_info": ont_info,
        "steps": steps_axis,
        "kl_per_step": kl_series,
        "logp_enforced": logp_series,
        "logp_masked": logp_masked_series,
        "logp_reference_good": logp_masked_series[-1],
        "soundness_enforcer_off": {"before": s_off_before,
                                   "after": s_off_after},
        "soundness_enforced": 100.0,  # by construction; every gen is (D)-masked
        "n_eval_trajectories": n_soundness_traj,
        "seeds": [seed],
        "config": {"train_steps": train_steps, "eval_every": eval_every,
                   "batch_size": batch_size, "n_train_traj": n_train_traj,
                   "n_eval_traj": n_eval_traj, "lora": True,
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
    p.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                 "constraint_ft_results.json"))
    p.add_argument("--smoke", action="store_true",
                   help="tiny model + tiny counts (CPU-friendly)")
    p.add_argument("--fake", action="store_true",
                   help="no-torch logic test with a synthetic prior")
    p.add_argument("--ontology", default=None,
                   help="path to a Text2KGBench ontology JSON (real domain)")
    args = p.parse_args()

    if args.fake:
        run("FAKE", args.out, train_steps=4, eval_every=2, batch_size=4,
            n_train_traj=16, n_eval_traj=50, n_soundness_traj=200, fake=True,
            ontology_path=args.ontology)
    elif args.smoke:
        run("HuggingFaceTB/SmolLM2-135M", args.out, train_steps=30,
            eval_every=15, batch_size=4, n_train_traj=48, n_eval_traj=12,
            n_soundness_traj=25, ontology_path=args.ontology)
    else:
        run(args.model, args.out, ontology_path=args.ontology)


if __name__ == "__main__":
    main()
