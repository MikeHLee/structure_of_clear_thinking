#!/usr/bin/env python3
"""Figures for X/Twitter thread #4: "The map is not in the attention".

All numbers read from committed result files:
  results/qk_causal.json                 (QK probe before/after + ablation)
  results/qk_region.json                 (merged-graph region probe)
  results/constraint_ft_generalization.json  (behavioral split)

Usage: python threads/04_map_not_in_attention/generate_figures.py
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


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


causal = json.load(open(os.path.join(ROOT, "results", "qk_causal.json")))
region = json.load(open(os.path.join(ROOT, "results", "qk_region.json")))
gen = json.load(open(os.path.join(ROOT, "results",
                                  "constraint_ft_generalization.json")))


# ---------------------------------------------------------------- figure 1
def fig1_soft_graph():
    heads = causal["qk_before"]["random_sequences"]
    aucs = sorted(v["auc_edge_vs_unreachable"] for v in heads.values()
                  if v["auc_edge_vs_unreachable"] is not None)
    top = sorted(((h, v["auc_edge_vs_unreachable"]) for h, v in heads.items()
                  if v["auc_edge_vs_unreachable"] is not None),
                 key=lambda x: -x[1])[:3]
    holdout_top = region["summary"]["before"]["holdout_edge"]["top5"][0]

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.96)
    ax.hist(aucs, bins=36, color=BLUE, zorder=3)
    ax.set_xlim(0.32, 0.97)  # keep the 0.90 head annotation on-canvas
    ax.axvline(0.5, color=MUTED, linewidth=1.6, linestyle=(0, (5, 4)))
    ax.text(0.5, ax.get_ylim()[1] * 0.97, " chance (0.5)", color=INK2,
            fontsize=11, va="top")
    for i, (h, v) in enumerate(top):
        ax.annotate(f"{h}: {v:.2f}", xy=(v, 1.5),
                    xytext=(v - 0.06, 13 + 5 * i),
                    fontsize=11.5, color=INK, fontweight="bold", ha="right",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate(f"one head reaches {holdout_top:.2f} on\npolitics/sports relations\n(popular pretraining topics)",
                xy=(holdout_top, 1.0), xytext=(holdout_top - 0.015, 30),
                fontsize=11.5, color=ORANGE, fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    headline(fig,
             "Some attention heads already know the ontology — before any training",
             "How well each of Qwen2.5-1.5B's 336 attention heads separates real ontology rules from non-rules (AUC),\nmeasured on scrambled sequences so position cannot help. Most heads ≈ chance; a specific few clearly encode the graph.")
    ax.set_xlabel("per-head AUC: attention ranks real rule pairs above non-rule pairs",
                  fontsize=11.5, color=INK2)
    ax.set_ylabel("number of heads", fontsize=11.5, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    footer(fig, "results/qk_causal.json + qk_region.json · pretrained model, no fine-tuning · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig1_soft_graph_exists.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2_ablation():
    ab = causal["ablation"]
    order = [("none", "No ablation"),
             ("top8_softkg", "Delete top-8\n'graph heads'"),
             ("top4_softkg", "Delete top-4\n'graph heads'"),
             ("random8_0", "Delete 8 random\nheads (control A)"),
             ("random8_1", "Delete 8 random\nheads (control B)")]
    vals = [ab[k]["off_soundness"] for k, _ in order]
    colors = [MUTED, ORANGE, ORANGE, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=160)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.96)
    bars = ax.bar(range(len(vals)), vals, width=0.55, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=13.5, fontweight="bold", color=INK)
    headline(fig,
             "Deleting the 'graph heads' changes nothing",
             "Rule-following of the fine-tuned model (enforcer off) after zeroing out attention heads — the heads that best\nencode the ontology (orange) hurt exactly as little as random heads (blue). The learned skill does not live there.")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l for _, l in order], fontsize=11.5, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.set_ylabel("% outputs fully rule-valid, enforcer OFF", fontsize=11.5,
                  color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    footer(fig, "results/qk_causal.json · N=100 trajectories per condition, identical eval stream · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig2_ablation_inert.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3_no_shadow():
    beh_b = gen["soundness_enforcer_off"]["before"]
    beh_a = gen["soundness_enforcer_off"]["after"]
    geo_b = region["summary"]["before"]
    geo_a = region["summary"]["after"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), dpi=160)
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.07, right=0.97,
                        wspace=0.3)
    headline(fig,
             "Behavior transformed. The attention map didn’t move.",
             "Fine-tuning on the merged 19-ontology graph: rule-following jumps 84 points on trained regions (left) —\nwhile the attention geometry’s knowledge of those same rules stays flat (right).")

    labels = ["Trained\nregions", "Never-seen\nregions"]
    for ax, before, after, title, ylim, fmt in (
        (axes[0], [beh_b["train_region"], beh_b["holdout_region"]],
         [beh_a["train_region"], beh_a["holdout_region"]],
         "BEHAVIOR: % rule-valid output (enforcer off)", (0, 112), "{:.0f}%"),
        (axes[1], [geo_b["train_edge"]["mean"], geo_b["holdout_edge"]["mean"]],
         [geo_a["train_edge"]["mean"], geo_a["holdout_edge"]["mean"]],
         "GEOMETRY: mean rule-detection AUC in attention", (0, 1.12), "{:.2f}"),
    ):
        x = [0, 1]
        w = 0.32
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], before, width=w,
                    color=MUTED, label="before training", zorder=3)
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], after, width=w,
                    color=AQUA, label="after training", zorder=3)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + ylim[1] * 0.015,
                        fmt.format(b.get_height()), ha="center",
                        fontsize=11.5, fontweight="bold", color=INK)
        if ax is axes[1]:
            ax.axhline(0.5, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)))
            ax.text(-0.42, 0.515, "chance", fontsize=10, color=MUTED,
                    va="bottom", ha="left")
        ax.set_title(title, fontsize=12, color=INK2, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11.5, color=INK)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper right")
    footer(fig, "results/constraint_ft_generalization.json + qk_region.json · Qwen2.5-1.5B · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig3_no_geometric_shadow.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_soft_graph()
    fig2_ablation()
    fig3_no_shadow()
    print(f"wrote 3 figures to {OUT}")
