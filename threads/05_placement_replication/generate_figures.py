#!/usr/bin/env python3
"""Figures for X/Twitter thread #5: the pre-registered placement replication.

Reads all three seed result files:
  results/routing_ft_results.json           (seed 42)
  results/routing_ft_results_seed43.json
  results/routing_ft_results_seed44.json

Usage: python threads/05_placement_replication/generate_figures.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASE, "xtick.color": INK2, "ytick.color": MUTED,
})

SEEDS = {
    42: "results/routing_ft_results.json",
    43: "results/routing_ft_results_seed43.json",
    44: "results/routing_ft_results_seed44.json",
}
DATA = {s: json.load(open(os.path.join(ROOT, p))) for s, p in SEEDS.items()}
CONDS = [
    ("qk_r64", "Attention\nrouting (q/k)"),
    ("vo_r32", "Attention\nread-out (v/o)"),
    ("mlp_r16", "MLPs only"),
    ("all_r16", "All weights"),
]


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


# ---------------------------------------------------------------- figure 1
def fig1_dissolves():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.07, right=0.96)
    for i, (key, label) in enumerate(CONDS):
        vals = [DATA[s]["conditions"][key]["holdout_region"] for s in DATA]
        pooled = sum(vals) / len(vals)
        ax.bar(i, pooled, width=0.55, color=BLUE, alpha=0.35, zorder=2)
        ax.text(i, pooled + 9.5, f"pooled\n{pooled:.1f}%", ha="center",
                fontsize=11.5, fontweight="bold", color=INK)
        for s, v in zip(DATA, vals):
            ax.scatter(i, v, s=90, color=ORANGE, edgecolor=SURFACE,
                       linewidth=1.5, zorder=4)
            ax.text(i + 0.3, v, f"s{s}", fontsize=9, color=MUTED,
                    va="center")
    headline(fig,
             "One seed looked exciting. Three seeds say: no effect.",
             "Rule-following on never-trained regions by which weights fine-tuning could write to. Dots = individual seeds;\nbars = 3-seed pooled mean. The spread WITHIN each condition swamps the differences BETWEEN them (χ² p=0.46).")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([l for _, l in CONDS], fontsize=11.5, color=INK)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%"], fontsize=10.5)
    ax.set_ylabel("% rule-valid on held-out regions, enforcer OFF",
                  fontsize=11.5, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    footer(fig, "3 seeds × n=80/region · results/routing_ft_results*.json · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig1_effect_dissolves.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2_wall():
    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=160)
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.07, right=0.96)
    w = 0.32
    for i, (key, label) in enumerate(CONDS):
        tr = sum(DATA[s]["conditions"][key]["train_region"]
                 for s in DATA) / len(DATA)
        ho = sum(DATA[s]["conditions"][key]["holdout_region"]
                 for s in DATA) / len(DATA)
        b1 = ax.bar(i - w / 2 - 0.01, tr, width=w, color=AQUA, zorder=3,
                    label="trained regions" if i == 0 else None)
        b2 = ax.bar(i + w / 2 + 0.01, ho, width=w, color=ORANGE, zorder=3,
                    label="never-seen regions" if i == 0 else None)
        ax.text(i - w / 2 - 0.01, tr + 2, f"{tr:.0f}%", ha="center",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(i + w / 2 + 0.01, ho + 2, f"{ho:.0f}%", ha="center",
                fontsize=12, fontweight="bold", color=INK)
    headline(fig,
             "What replicates in every run: the generalization wall",
             "3-seed means. No matter WHERE fine-tuning is allowed to write — attention, MLPs, everything — the model\nmasters trained regions and fails never-seen ones. The wall is the finding; enforcement is the answer to it.")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([l for _, l in CONDS], fontsize=11.5, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.set_ylabel("% rule-valid output, enforcer OFF", fontsize=11.5,
                  color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    footer(fig, "12 runs (3 seeds × 4 conditions) · results/routing_ft_results*.json · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig2_generalization_wall.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_dissolves()
    fig2_wall()
    print(f"wrote 2 figures to {OUT}")
