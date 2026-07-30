#!/usr/bin/env python3
"""Modal runner for the constraint-aware fine-tuning experiment.

SOGB pathway: CPU smoke FIRST (it exercises the identical code path and
has repeatedly caught bugs a GPU run would have hidden), then L4.

    modal run scripts/modal_constraint_ft.py::smoke
    modal run --detach scripts/modal_constraint_ft.py::train_l4
    modal run scripts/modal_constraint_ft.py::download

Results land in the 'sct-constraint-ft' Volume; `download` copies the
JSON to results/constraint_ft_results.json for the thread-03 figures.
"""

import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PTVM = os.path.join(HERE, "..", "Percepta_Transformer_VM")

ONTOLOGIES = os.path.join(HERE, "..", "training_data", "Text2KGBench",
                          "data", "dbpedia_webnlg", "ontologies")

app = modal.App("sct-constraint-ft")
vol = modal.Volume.from_name("sct-constraint-ft", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate", "numpy")
    .add_local_dir(PTVM, remote_path="/root/ptvm")
    .add_local_dir(ONTOLOGIES, remote_path="/root/ontologies")
)

RESULTS_DIR = "/results"


def _run(model_name, out_name, **kwargs):
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_constraint_ft import run
    out = os.path.join(RESULTS_DIR, out_name)
    res = run(model_name, out, **kwargs)
    vol.commit()
    return {k: res[k] for k in ("model_name", "kl_per_step", "logp_enforced",
                                "soundness_enforcer_off", "config")}


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke():
    """Tiny model, tiny counts — proves the full code path end to end."""
    return _run("HuggingFaceTB/SmolLM2-135M", "smoke_results.json",
                train_steps=30, eval_every=15, batch_size=4,
                n_train_traj=48, n_eval_traj=12, n_soundness_traj=25)


@app.function(image=image, gpu="L4", timeout=4 * 3600,
              volumes={RESULTS_DIR: vol})
def train_l4():
    """Qwen2.5-1.5B, full run. ~L4-sized; detach and check back."""
    return _run("Qwen/Qwen2.5-1.5B", "constraint_ft_results.json",
                train_steps=2000, eval_every=250, batch_size=8,
                n_train_traj=512, n_eval_traj=100, n_soundness_traj=300)


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke_ontology(ontology_file: str = "1_university_ontology.json"):
    """Tiny model on a REAL ontology — proves the batched-scoring +
    random-start path before paying for GPU."""
    return _run("HuggingFaceTB/SmolLM2-135M",
                f"smoke_ont_{ontology_file.split('_')[1]}.json",
                train_steps=20, eval_every=10, batch_size=4,
                n_train_traj=24, n_eval_traj=8, n_soundness_traj=15,
                ontology_path=f"/root/ontologies/{ontology_file}")


@app.function(image=image, gpu="L4", timeout=2 * 3600,
              volumes={RESULTS_DIR: vol})
def train_ontology(ontology_file: str = "1_university_ontology.json"):
    """Qwen2.5-1.5B on a real Text2KGBench ontology. Cost-tuned config:
    convergence lessons from the e-commerce run (done by step 250) plus
    batched label scoring keep this ~$0.25-0.50 per ontology."""
    slug = os.path.splitext(ontology_file)[0]
    return _run("Qwen/Qwen2.5-1.5B", f"constraint_ft_{slug}.json",
                train_steps=500, eval_every=50, batch_size=8,
                n_train_traj=256, n_eval_traj=50, n_soundness_traj=150,
                ontology_path=f"/root/ontologies/{ontology_file}")


HOLDOUTS = "6_politician_ontology.json,9_astronaut_ontology.json,15_sportsteam_ontology.json"


def _run_gen(model_name, out_name, holdouts=HOLDOUTS, **kwargs):
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_constraint_ft import run_generalization
    out = os.path.join(RESULTS_DIR, out_name)
    res = run_generalization(model_name, out, "/root/ontologies",
                             [h.strip() for h in holdouts.split(",")], **kwargs)
    vol.commit()
    return {k: res[k] for k in ("model_name", "soundness_enforcer_off",
                                "corpus_holdout_rule_leaks", "config")}


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke_generalization():
    """Tiny model, tiny counts, full merged graph — proves the split logic
    (region separation, contamination check, chunked scoring) before GPU."""
    # 357-label scoring is ~6 chunk-forwards per step on CPU — counts must
    # stay tiny or the smoke hits its own timeout (learned 2026-07-29).
    return _run_gen("HuggingFaceTB/SmolLM2-135M", "smoke_gen_results.json",
                    train_steps=4, eval_every=2, batch_size=4,
                    n_train_traj=6, n_eval_traj=3, n_soundness_traj=6)


@app.function(image=image, gpu="L4", timeout=3 * 3600,
              volumes={RESULTS_DIR: vol})
def train_generalization():
    """Qwen2.5-1.5B on the merged 19-ontology graph, writtenwork+scientist
    held out. The question: does enforcer-off soundness rise on regions the
    model never trained on?"""
    return _run_gen("Qwen/Qwen2.5-1.5B", "constraint_ft_generalization.json")


PROBE_ONT = "14_writtenwork_ontology.json"


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke_probe():
    """Attention-probe path on CPU with a tiny model — verifies eager
    attention extraction, offset mapping, and the AUC bookkeeping."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_attention_probe import run_probe
    res = run_probe("HuggingFaceTB/SmolLM2-135M",
                    os.path.join(RESULTS_DIR, "smoke_probe.json"),
                    f"/root/ontologies/{PROBE_ONT}",
                    n_traj_stimuli=4, n_random_stimuli=4, train_steps=6,
                    n_train_traj=8)
    vol.commit()
    return res["summary"]


@app.function(image=image, gpu="L4", timeout=2 * 3600,
              volumes={RESULTS_DIR: vol})
def probe_attention():
    """Qwen2.5-1.5B: per-head ontology-edge AUC before vs after
    constraint-aware fine-tuning on WrittenWork (depth-5 chains)."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_attention_probe import run_probe
    res = run_probe("Qwen/Qwen2.5-1.5B",
                    os.path.join(RESULTS_DIR, "attention_probe.json"),
                    f"/root/ontologies/{PROBE_ONT}")
    vol.commit()
    return res["summary"]


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke_causal():
    """QK hooks + GQA mapping + head-ablation machinery on CPU/SmolLM
    (SmolLM2 is also GQA, so the kv-head mapping gets exercised)."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_probe_causal import run_causal
    res = run_causal("HuggingFaceTB/SmolLM2-135M",
                     os.path.join(RESULTS_DIR, "smoke_causal.json"),
                     f"/root/ontologies/{PROBE_ONT}",
                     n_stimuli=4, train_steps=6, n_train_traj=8,
                     n_sound=10, n_eval=4, topk=4)
    vol.commit()
    return res["qk_summary"]


@app.function(image=image, gpu="L4", timeout=2 * 3600,
              volumes={RESULTS_DIR: vol})
def probe_causal():
    """Qwen2.5-1.5B on WrittenWork: QK geometry before/after FT + head
    ablation battery with matched random controls."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_probe_causal import run_causal
    res = run_causal("Qwen/Qwen2.5-1.5B",
                     os.path.join(RESULTS_DIR, "qk_causal.json"),
                     f"/root/ontologies/{PROBE_ONT}")
    vol.commit()
    return {"qk_summary": res["qk_summary"], "ablation": res["ablation"]}


@app.function(image=image, gpu="L4", timeout=2 * 3600,
              volumes={RESULTS_DIR: vol})
def probe_region():
    """Merged-graph QK region probe: train-region vs held-out-only rule
    signal, before/after fine-tuning on the train region."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_probe_causal import run_region_probe
    res = run_region_probe("Qwen/Qwen2.5-1.5B",
                           os.path.join(RESULTS_DIR, "qk_region.json"),
                           "/root/ontologies",
                           [h.strip() for h in HOLDOUTS.split(",")])
    vol.commit()
    return res["summary"]


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={RESULTS_DIR: vol})
def smoke_routing():
    """Per-condition adapter creation + shared-corpus flow on CPU/SmolLM,
    2 conditions, tiny counts."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_routing_ft import run_routing
    res = run_routing("HuggingFaceTB/SmolLM2-135M",
                      os.path.join(RESULTS_DIR, "smoke_routing.json"),
                      "/root/ontologies",
                      [h.strip() for h in HOLDOUTS.split(",")],
                      train_steps=4, batch_size=4, n_train_traj=6,
                      n_sound=6, n_eval=3, conditions=["qk_r64", "mlp_r16"])
    vol.commit()
    return res["conditions"]


@app.function(image=image, gpu="L4", timeout=5 * 3600,
              volumes={RESULTS_DIR: vol})
def train_routing(seed: int = 42):
    """Qwen2.5-1.5B, merged-graph generalization split, 4 LoRA-target
    conditions (qk / vo / mlp / all) on one shared base-model corpus.
    Seed varies corpus sampling, batch order, and eval streams."""
    import sys
    sys.path.insert(0, "/root/ptvm")
    from experiment_routing_ft import run_routing
    name = ("routing_ft_results.json" if seed == 42
            else f"routing_ft_results_seed{seed}.json")
    res = run_routing("Qwen/Qwen2.5-1.5B",
                      os.path.join(RESULTS_DIR, name),
                      "/root/ontologies",
                      [h.strip() for h in HOLDOUTS.split(",")], seed=seed)
    vol.commit()
    return {"baseline": res["baseline"], "conditions": res["conditions"]}


@app.function(image=image, volumes={RESULTS_DIR: vol})
def _read(name: str) -> str:
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return f.read()


@app.local_entrypoint()
def download(name: str = "constraint_ft_results.json"):
    data = _read.remote(name)
    dest = os.path.join(HERE, "..", "results", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(data)
    print(f"wrote {dest}")
