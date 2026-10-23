"""
modal_hdc_filtered_eval.py  (Kolmogorov, 2026-10-23)
====================================================
Re-check of the Olog paper's headline "MRR 0.346 on FB15K-237, competitive with ConvE/RotatE".

Problems with the published number (modal_hdc_training.py):
  - test ranks only tails, and does NOT filter other true answers (raw protocol);
    ConvE 0.325 / RotatE 0.338 are FILTERED, head+tail averaged;
  - the summary quotes 40,943 entities (= WN18RR), so which dataset gave 0.346 is unclear;
  - no result file exists.

This script trains the SAME model with the SAME recipe (HolE-style circular-convolution
embeddings, margin 0.5, one random tail negative, Adam, batch 512, dim 4096, seed 42, 50 epochs)
and reports every protocol side by side, for both datasets:
  raw_tail (reproduces the published protocol), filtered_tail, filtered_head, filtered_both.

PRE-REGISTRATION (written 2026-10-23 02:3x UTC, before any run):
  - Claim under test: "competitive with ConvE (0.325)" on FB15K-237.
  - Metric: filtered_both MRR on the FB15K-237 test set, seed 42.
  - Pass mark: >= 0.31. Below that, the claim is withdrawn.
  - Expectation: about 10% chance of passing; best guess 0.15-0.25.
  - Also recorded: raw_tail on each dataset. If raw_tail ~= 0.346 on WN18RR but not on
    FB15K-237, the published number came from the wrong dataset.

Run (GPU, via Yes Man):
    modal run topics/structure_of_clear_thinking/scripts/modal_hdc_filtered_eval.py
Expected cost: 1x A100, ~40-60 min total (both datasets), about $3-4. Max runtime 2 h.
Reads data from the existing `hdc-sheaf-data` volume; writes JSON to `hdc-sheaf-results`.
"""

import json
from pathlib import Path

import modal

app = modal.App("hdc-sheaf-filtered-eval")
data_volume = modal.Volume.from_name("hdc-sheaf-data", create_if_missing=False)
results_volume = modal.Volume.from_name("hdc-sheaf-results", create_if_missing=True)
DATA_DIR, RESULTS_DIR = "/data", "/results"

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.2.2", "numpy>=1.24.0")


@app.function(image=image, gpu="A100", volumes={DATA_DIR: data_volume, RESULTS_DIR: results_volume},
              timeout=7200)
def run(dataset: str = "fb15k237", epochs: int = 50, batch_size: int = 512, dim: int = 4096,
        lr: float = 0.001, seed: int = 42) -> dict:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda")

    def load(split):
        out = []
        for line in open(Path(DATA_DIR) / dataset / f"{split}.txt"):
            p = line.strip().split("\t")
            if len(p) == 3:
                out.append((p[0], p[2], p[1]))  # (h, t, r), same as modal_hdc_training.py
        return out

    tr, va, te = load("train"), load("valid"), load("test")
    ents = sorted({x for h, t, _ in tr + va + te for x in (h, t)})
    rels = sorted({r for _, _, r in tr + va + te})
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(rels)}
    N, R = len(ents), len(rels)

    def ten(tr_):
        return (torch.tensor([e2i[h] for h, _, _ in tr_]), torch.tensor([r2i[r] for _, _, r in tr_]),
                torch.tensor([e2i[t] for _, t, _ in tr_]))

    E = torch.nn.Embedding(N, dim).to(dev)
    Rm = torch.nn.Embedding(R, dim).to(dev)
    torch.nn.init.normal_(E.weight, std=1.0 / np.sqrt(dim))
    torch.nn.init.normal_(Rm.weight, std=1.0 / np.sqrt(dim))
    opt = torch.optim.Adam(list(E.parameters()) + list(Rm.parameters()), lr=lr)

    def bind(x, y):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y)).real

    def score(h, r, t):
        return torch.nn.functional.cosine_similarity(bind(E(h), Rm(r)), E(t), dim=-1)

    th, trr, tt = ten(tr)
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            h, r, t = th[idx].to(dev), trr[idx].to(dev), tt[idx].to(dev)
            neg = torch.randint(0, N, (len(idx),), device=dev)
            loss = torch.relu(0.5 - score(h, r, t) + score(h, r, neg)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"{dataset} epoch {ep + 1} loss {tot / ((n + batch_size - 1) // batch_size):.4f}", flush=True)

    # filter sets over train+valid+test, the standard protocol
    tails, heads = {}, {}
    for h, t, r in tr + va + te:
        tails.setdefault((e2i[h], r2i[r]), set()).add(e2i[t])
        heads.setdefault((e2i[t], r2i[r]), set()).add(e2i[h])

    acc = {k: [] for k in ("raw_tail", "filtered_tail", "filtered_head")}
    with torch.no_grad():
        En = torch.nn.functional.normalize(E.weight, dim=-1)
        Ef = torch.fft.fft(E.weight)
        for h, t, r in te:
            hi, ti, ri = e2i[h], e2i[t], r2i[r]
            rv = Rm.weight[ri]
            # tail query (h, r, ?)
            q = torch.nn.functional.normalize(bind(E.weight[hi], rv), dim=-1)
            s = En @ q
            true = s[ti].item()
            acc["raw_tail"].append(int((s > true).sum().item()) + 1)
            m = s.clone()
            others = [x for x in tails[(hi, ri)] if x != ti]
            if others:
                m[others] = -float("inf")
            acc["filtered_tail"].append(int((m > true).sum().item()) + 1)
            # head query (?, r, t): score every h' by cos(bind(h', r), t)
            cand = torch.nn.functional.normalize(torch.fft.ifft(Ef * torch.fft.fft(rv)).real, dim=-1)
            s2 = cand @ En[ti]
            true2 = s2[hi].item()
            others2 = [x for x in heads[(ti, ri)] if x != hi]
            if others2:
                s2[others2] = -float("inf")
            acc["filtered_head"].append(int((s2 > true2).sum().item()) + 1)

    def summ(ranks):
        a = np.asarray(ranks, float)
        return {"mrr": float(np.mean(1 / a)), "hits1": float(np.mean(a <= 1)),
                "hits3": float(np.mean(a <= 3)), "hits10": float(np.mean(a <= 10))}

    res = {k: summ(v) for k, v in acc.items()}
    res["filtered_both"] = summ(acc["filtered_tail"] + acc["filtered_head"])
    res["_meta"] = {"dataset": dataset, "num_entities": N, "num_relations": R, "num_train": len(tr),
                    "num_test": len(te), "epochs": epochs, "dim": dim, "lr": lr, "seed": seed,
                    "batch_size": batch_size}
    Path(RESULTS_DIR, f"{dataset}_filtered_eval_seed{seed}.json").write_text(json.dumps(res, indent=2))
    results_volume.commit()
    print(json.dumps(res, indent=2), flush=True)
    return res


@app.local_entrypoint()
def main(epochs: int = 50, seed: int = 42):
    out = {d: run.remote(dataset=d, epochs=epochs, seed=seed) for d in ("fb15k237", "wn18rr")}
    print(json.dumps(out, indent=2))
    fb = out["fb15k237"]["filtered_both"]["mrr"]
    print(f"PRE-REG: FB15K-237 filtered_both MRR = {fb:.4f}; pass mark 0.31 -> {'PASS' if fb >= 0.31 else 'FAIL'}")
