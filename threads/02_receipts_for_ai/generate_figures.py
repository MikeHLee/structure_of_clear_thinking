#!/usr/bin/env python3
"""Figures for X/Twitter thread #2: "Receipts for AI" (audit certificates).

All values are real outputs of Percepta_Transformer_VM/verification_certificate.py
(fingerprints 10f492de… / 9fae0d63…, the "Mars" tamper demo) and the sample
certificate in papers/nesy_submission/supplementary/sample_audit_certificate.json.

Usage: python threads/02_receipts_for_ai/generate_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
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
GOOD = "#0ca30c"
CRIT = "#d03b3b"
MONO = ["Menlo", "Monaco", "DejaVu Sans Mono"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
})


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


def rounded(ax, x, y, w, h, fc, ec, lw=1.4):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.10",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)


# ---------------------------------------------------------------- figure 1
# Anatomy of the certificate.
def fig1_anatomy():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    headline(fig,
             "The receipt: what ships with every AI output",
             "A real audit certificate (JSON, ~1 KB). Left: the raw fields. Right: what each one proves.")

    # left: JSON-ish card
    rounded(ax, 0.4, 0.5, 5.6, 5.3, "#ffffff", INK2)
    lines = [
        ('"tlts_fingerprint": "10f492de…"', BLUE),
        ('"steps": [', INK),
        ('  { "state_in":  "Customer",', INK),
        ('    "label":     "has",', INK),
        ('    "state_out": "Cart",', INK),
        ('    "in_delta":  true,', GOOD),
        ('    "forced":    false,', ORANGE),
        ('    "masked_kl": 0.223 }, …]', ORANGE),
        ('"soundness_passed": true,', GOOD),
        ('"summary": { "forced_step_count": 0,', INK),
        ('             "log_p_under_prior": -1.81 }', INK),
    ]
    y = 5.35
    for text, color in lines:
        ax.text(0.7, y, text, fontsize=11.5, color=color,
                family=MONO, va="top")
        y -= 0.45

    # right: annotations
    notes = [
        (5.05, "Which rulebook was in force — a hash of the exact\n"
               "types + rules. Any edit to the rulebook changes it.", BLUE),
        (3.60, "Every single step: where it was, what move it made,\n"
               "where it landed — and whether that move is a rule.", INK2),
        (2.60, "Where the enforcer had to override the model,\n"
               "and how hard the rules reshaped its choices.", ORANGE),
        (1.45, "The verdict + totals an auditor checks first.", GOOD),
    ]
    for y, text, color in notes:
        ax.annotate(text, xy=(6.0, y), xytext=(6.7, y),
                    fontsize=12, color=color, va="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

    footer(fig, "sample_audit_certificate.json (real output) · verification_certificate.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig1_certificate_anatomy.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Verification flow: no model needed.
def fig2_flow():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    headline(fig,
             "Anyone can check the receipt — without the model",
             "The verifier needs two small text files. No weights, no GPU, no API access, no trust in the vendor.")

    # Producer side
    rounded(ax, 0.4, 3.6, 3.4, 1.9, "#ffffff", INK2)
    ax.text(2.1, 5.05, "AI system", ha="center", fontsize=13.5,
            fontweight="bold", color=INK)
    ax.text(2.1, 4.55, "generates output with\nper-step rule checking ON",
            ha="center", va="center", fontsize=11, color=INK2)

    rounded(ax, 0.4, 0.9, 3.4, 1.9, "#ffffff", BLUE)
    ax.text(2.1, 2.35, "the output\n+ its receipt (JSON)", ha="center",
            va="center", fontsize=12.5, fontweight="bold", color=BLUE)
    ax.text(2.1, 1.45, "emitted as a byproduct —\nzero extra model work",
            ha="center", va="center", fontsize=10.5, color=INK2)

    arr = FancyArrowPatch((2.1, 3.55), (2.1, 2.95), arrowstyle="-|>",
                          mutation_scale=18, linewidth=2, color=INK2)
    ax.add_patch(arr)

    # Verifier side
    rounded(ax, 4.9, 2.2, 3.0, 2.4, "#ffffff", AQUA, lw=2)
    ax.text(6.4, 4.15, "Independent verifier", ha="center", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(6.4, 3.35, "holds only:\n• the rulebook\n• the receipt",
            ha="center", va="center", fontsize=11.5, color=INK2)
    ax.text(6.4, 2.55, "re-checks every step", ha="center",
            fontsize=10.5, color=MUTED, style="italic")

    arr = FancyArrowPatch((3.85, 1.85), (5.3, 2.15), arrowstyle="-|>",
                          mutation_scale=18, linewidth=2, color=BLUE)
    ax.add_patch(arr)

    # verdicts
    verdicts = [
        (5.35, "Honest output", "PASS", GOOD,
            "every step re-verifies against the rulebook"),
        (3.95, "One field edited (\"Cart\"→\"Mars\")", "CAUGHT", CRIT,
            "step 0: (Customer, has, Mars) is not a rule"),
        (2.55, "Different rulebook substituted", "CAUGHT", CRIT,
            "fingerprint 10f492de… ≠ 9fae0d63…"),
    ]
    for y, case, verdict, color, why in verdicts:
        arr = FancyArrowPatch((8.0, 3.4), (8.75, y - 0.15), arrowstyle="-|>",
                              mutation_scale=14, linewidth=1.6, color=MUTED)
        ax.add_patch(arr)
        ax.text(8.9, y, case, fontsize=11.5, color=INK, va="bottom")
        ax.text(8.9, y - 0.42, f"→ {verdict}", fontsize=12.5, color=color,
                fontweight="bold", va="bottom")
        ax.text(8.9, y - 0.78, why, fontsize=9.5, color=MUTED, va="bottom")

    footer(fig, "All three verdicts are real runs of verification_certificate.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig2_verification_flow.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
# The disagreement gauge: forced steps as a trust signal.
def fig3_trust_signal():
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=160)
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.09, right=0.96)

    cases = ["Model knows the domain\n(well-calibrated)",
             "Model has wrong habits\n(mis-calibrated)"]
    forced = [0.0, 71.0]  # % of steps where enforcer overrode the model
    bars = ax.bar([0, 1], forced, width=0.5, color=[AQUA, ORANGE], zorder=3)
    for b, v in zip(bars, forced):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=INK)
    ax.text(0, 14, "output valid ✓\nmodel did it itself", ha="center",
            fontsize=11, color=INK2)
    ax.annotate("output STILL valid ✓\nbut the receipt shows the rules\ndid most of the work — retrain or\nfix the rulebook",
                xy=(1, 71), xytext=(1.42, 45), fontsize=11, color=INK2,
                ha="left", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    headline(fig,
             "The receipt doubles as a health gauge",
             "% of generation steps where the enforcer had to override the model’s first choice (“forced steps”), from certificate summaries.")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cases, fontsize=12.5, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    ax.set_xlim(-0.5, 2.3)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    footer(fig, "N=1,000 trajectories per condition · experiment_loci_comparison.py · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig3_trust_signal.png"),
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_anatomy()
    fig2_flow()
    fig3_trust_signal()
    print(f"wrote 3 figures to {OUT}")
