#!/usr/bin/env python3
"""Figures for X/Twitter thread #1: "Reachability masking is not enough".

All numbers come from Percepta_Transformer_VM/experiment_results.md
(loci comparison, cyclic stress test, real-attention mask audit).
Figures are designed to be self-interpretable: the takeaway is in the
title, every mark is directly labeled, and no figure depends on the
thread text to be understood.

Usage: python threads/01_reachability_is_not_enough/generate_figures.py
Outputs PNGs into threads/01_reachability_is_not_enough/figures/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # method: unconstrained sampling
ORANGE = "#eb6834"  # method: reachability masking
AQUA = "#1baf7a"    # method: per-step rule check
CRIT = "#d03b3b"    # illegal move (status, paired with x label)
GOOD = "#0ca30c"    # legal move   (status, paired with check label)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "xtick.color": INK2,
    "ytick.color": MUTED,
})

METHODS = [
    ("No rule enforcement", BLUE),
    ("Reachability masking", ORANGE),
    ("Per-step rule check", AQUA),
]


def new_fig():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    return fig, ax


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


# ---------------------------------------------------------------- figure 1
# Concept: a reachable destination is not a legal move.
def fig1_concept():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    headline(fig,
             "A reachable destination is not a legal move",
             "The rulebook (ontology) for a shop: boxes are types of things, arrows are the only steps allowed between them.")

    nodes = {  # name -> (x, y)
        "Customer": (1.2, 3.2),
        "Cart":     (3.6, 3.2),
        "Checkout": (6.0, 3.2),
        "Payment":  (8.4, 3.2),
        "Order":    (10.8, 3.2),
        "Item":     (3.6, 1.1),
    }
    for name, (x, y) in nodes.items():
        box = FancyBboxPatch((x - 0.85, y - 0.42), 1.7, 0.84,
                             boxstyle="round,pad=0.06,rounding_size=0.14",
                             facecolor="#ffffff", edgecolor=INK2, linewidth=1.4)
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=13,
                color=INK, fontweight="bold")

    def edge(a, b):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        arr = FancyArrowPatch((x1 + 0.95, y1) if y1 == y2 else (x1, y1 - 0.55),
                              (x2 - 0.95, y2) if y1 == y2 else (x2, y2 + 0.55),
                              arrowstyle="-|>", mutation_scale=16,
                              linewidth=2.0, color=INK2)
        ax.add_patch(arr)

    edge("Customer", "Cart")
    edge("Cart", "Checkout")
    edge("Checkout", "Payment")
    edge("Payment", "Order")
    edge("Cart", "Item")

    # Legal single step, marked good
    ax.text(2.4, 3.95, "✓ each solid arrow = a legal step",
            ha="left", va="bottom", fontsize=11.5, color=GOOD,
            fontweight="bold")

    # The illegal-but-reachable jump
    arr = FancyArrowPatch((1.2, 3.85), (8.4, 3.85),
                          arrowstyle="-|>", mutation_scale=18,
                          linewidth=2.4, color=CRIT, linestyle=(0, (5, 4)),
                          connectionstyle="arc3,rad=-0.32")
    ax.add_patch(arr)
    ax.text(4.8, 5.75,
            "✗ Customer → Payment:  Payment IS reachable (via Cart → Checkout),\n"
            "so a reachability mask ALLOWS this step — but no rule permits it directly.",
            ha="center", va="top", fontsize=12.5, color=CRIT, fontweight="bold")

    ax.text(6.0, 0.25,
            "Chess analogy: your rook can eventually reach almost any square — "
            "that doesn’t make every square a legal next move.",
            ha="center", va="bottom", fontsize=11.5, color=INK2)

    footer(fig, "Structure of Clear Thinking · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig1_reachable_vs_legal.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Headline soundness comparison.
def fig2_soundness():
    fig, ax = new_fig()
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.07, right=0.97)

    scenarios = ["Model already knows the domain well\n(well-calibrated)",
                 "Model has wrong habits\n(mis-calibrated)"]
    data = {  # method -> (well, mis)
        "No rule enforcement": (48.1, 4.2),
        "Reachability masking": (61.5, 4.3),
        "Per-step rule check": (100.0, 100.0),
    }

    x0 = [0, 1]
    width = 0.24
    offsets = [-width - 0.01, 0, width + 0.01]  # ~2px surface gap
    for (name, color), off in zip(METHODS, offsets):
        vals = data[name]
        xs = [x + off for x in x0]
        bars = ax.bar(xs, vals, width=width, color=color, label=name,
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2.5,
                    f"{v:.0f}%" if v == 100 else f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=12.5,
                    color=INK, fontweight="bold")

    ax.annotate("guaranteed\nby construction",
                xy=(1 + offsets[2], 100), xytext=(1.42, 78),
                fontsize=11, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    headline(fig,
             "Only checking every step guarantees rule-following output",
             "% of 1,000 generated sequences that contain zero illegal steps (“soundness”), on a 7-type shop ontology.")
    ax.set_xticks(x0)
    ax.set_xticklabels(scenarios, fontsize=12, color=INK)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False,
              fontsize=11.5, ncol=1)
    footer(fig, "N=1,000 trajectories per condition, seed 42 · experiment_loci_comparison.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig2_soundness_by_method.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
# Cyclic collapse slope chart.
def fig3_cycles():
    fig, ax = new_fig()
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.09, right=0.80)

    pts = {  # method -> (acyclic, cyclic)
        "No rule enforcement": (48.1, 6.9),
        "Reachability masking": (65.0, 8.0),
        "Per-step rule check": (100.0, 100.0),
    }
    label_y = {  # dodge the overlapping 7%/8% end labels
        "No rule enforcement": 0.5,
        "Reachability masking": 14.0,
        "Per-step rule check": 100.0,
    }
    for name, color in METHODS:
        a, c = pts[name]
        ax.plot([0, 1], [a, c], color=color, linewidth=2.4, zorder=3,
                marker="o", markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.text(-0.04, a, f"{a:.0f}%", ha="right", va="center",
                fontsize=12, color=INK, fontweight="bold")
        ax.text(1.04, label_y[name], f"{c:.0f}%  {name}", ha="left", va="center",
                fontsize=12, color=INK, fontweight="bold" if "Per-step" in name else "normal")

    headline(fig,
             "Add loops to the rulebook and reachability masking stops working",
             "With loops, almost every type becomes “reachable” — so the mask blocks almost nothing. Per-step checks don’t care.")

    ax.set_xlim(-0.35, 1.9)
    ax.set_ylim(-4, 112)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rulebook without loops\n(34.7% of pairs reachable)",
                        "Rulebook with loops\n(100% of pairs reachable)"],
                       fontsize=12, color=INK)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(0.5, -0.185, "% of generated sequences with zero illegal steps (well-calibrated model)",
            transform=ax.transAxes, ha="center", fontsize=10.5, color=MUTED)
    footer(fig, "experiment_cyclic_stress.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig3_cyclic_collapse.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
# Attention-mass distribution under the production reachability mask.
def fig4_attention_mass():
    fig, ax = plt.subplots(figsize=(12, 3.9), dpi=160)
    fig.subplots_adjust(top=0.66, bottom=0.16, left=0.05, right=0.95)

    segs = [
        ("Token attending to itself", 16.9, MUTED),
        ("Legal step (a real rule)", 16.4, BLUE),
        ("Reachable — but NOT a rule", 66.7, ORANGE),
    ]
    left = 0.0
    for name, v, color in segs:
        ax.barh(0, v, left=left, color=color, height=0.55,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center",
                fontsize=14 if v > 30 else 12, color="#ffffff",
                fontweight="bold")
        ax.text(left + v / 2, -0.48, name, ha="center", va="top",
                fontsize=11.5, color=INK)
        left += v

    headline(fig,
             "Under a reachability mask, ⅔ of attention flows through non-rules",
             "Where attention mass actually lands in the typed-attention layer (production code, 6-token window, 36 query–key pairs).")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, 0.62)
    ax.axis("off")
    ax.text(50, 0.52,
            "The mask correctly blocks all UNreachable pairs (0% leakage) — "
            "but “reachable” admits 4× more pairs than the rulebook actually contains.",
            ha="center", va="bottom", fontsize=11.5, color=INK2)
    footer(fig, "experiment_real_attention_b.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig4_attention_mass.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_concept()
    fig2_soundness()
    fig3_cycles()
    fig4_attention_mass()
    print(f"wrote 4 figures to {OUT}")
