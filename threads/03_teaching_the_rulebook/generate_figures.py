#!/usr/bin/env python3
"""Figures for X/Twitter thread #3: "Teaching the model the rulebook".

Reads the constraint-aware fine-tuning results file and renders the three
thread figures. Until the training runs land, use --placeholder to render
layout previews from synthetic curves — every preview is watermarked
"PLACEHOLDER — ILLUSTRATIVE SHAPE ONLY" and must never be posted.

Usage:
    python threads/03_teaching_the_rulebook/generate_figures.py \
        [--results results/constraint_ft_results.json] [--placeholder]

Expected results JSON schema (produced by the fine-tuning eval harness):
{
  "model_name": "Qwen2.5-1.5B",           # str, shown in subtitles
  "domain": "7-type e-commerce ontology",  # str, shown in subtitles
  "steps": [0, 100, ...],                  # training-step axis
  "kl_per_step": [1.92, ...],              # model-rulebook disagreement
  "logp_enforced": [-7.4, ...],            # fluency under enforcement
  "logp_reference_good": -1.47,            # well-calibrated frontier line
  "soundness_enforcer_off": {              # bare-model eval, % fully valid
      "before": 4.2, "after": null         # filled by eval harness
  },
  "n_eval_trajectories": 1000,
  "seeds": [42]
}
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (light mode) — same as thread 01
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
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "xtick.color": INK2,
    "ytick.color": MUTED,
})


def headline(fig, title, subtitle):
    fig.text(0.05, 0.955, title, fontsize=19, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=INK2,
             ha="left", va="top")


def footer(fig, text):
    fig.text(0.05, 0.02, text, fontsize=9.5, color=MUTED, ha="left")


def watermark(fig):
    fig.text(0.5, 0.45, "PLACEHOLDER — ILLUSTRATIVE SHAPE ONLY\nDO NOT POST",
             fontsize=34, fontweight="bold", color="#d03b3b", alpha=0.35,
             ha="center", va="center", rotation=18, zorder=10)


def placeholder_results():
    steps = list(range(0, 2001, 100))
    return {
        "model_name": "{{MODEL_NAME}}",
        "domain": "{{DOMAIN_DESC}}",
        "steps": steps,
        "kl_per_step": [1.92 * math.exp(-s / 700) + 0.15 for s in steps],
        "logp_enforced": [-7.4 + 5.2 * (1 - math.exp(-s / 800)) for s in steps],
        "logp_reference_good": -1.47,
        "soundness_enforcer_off": {"before": 4.2, "after": 78.0},
        "n_eval_trajectories": 1000,
        "seeds": [42],
    }


def base_axes(fig, ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def fig1_disagreement(res, ph):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.96)
    ax.plot(res["steps"], res["kl_per_step"], color=BLUE, linewidth=2.4,
            zorder=3)
    s0, s1 = res["kl_per_step"][0], res["kl_per_step"][-1]
    ax.annotate(f"start: {s0:.2f}", xy=(res["steps"][0], s0),
                xytext=(res["steps"][-1] * 0.06, s0), fontsize=12,
                fontweight="bold", color=INK, va="bottom")
    ax.annotate(f"after training: {s1:.2f}", xy=(res["steps"][-1], s1),
                xytext=(res["steps"][-1] * 0.68, s1 + (s0 - s1) * 0.12),
                fontsize=12, fontweight="bold", color=INK)
    conv = next((s for s, k in zip(res["steps"][1:], res["kl_per_step"][1:])
                 if k < 0.05), None)
    if conv is not None and conv <= res["steps"][-1] * 0.2:
        ax.annotate(f"≈0 by step {conv}\n({conv * 8} training examples)",
                    xy=(conv, res["kl_per_step"][res["steps"].index(conv)]),
                    xytext=(res["steps"][-1] * 0.12, s0 * 0.45),
                    fontsize=12, fontweight="bold", color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    headline(fig,
             "Training on rule-checked outputs teaches the model the rules",
             f"Model–rulebook disagreement (KL per generation step) while fine-tuning {res['model_name']} on its own enforced outputs · {res['domain']}.")
    ax.set_xlabel("fine-tuning steps", fontsize=11.5, color=INK2)
    ax.set_ylabel("disagreement (KL/step) — lower = model agrees with rulebook",
                  fontsize=11.5, color=INK2)
    ax.set_ylim(0, None)
    base_axes(fig, ax)
    if ph:
        watermark(fig)
    footer(fig, f"N={res['n_eval_trajectories']} eval trajectories, seeds {res['seeds']} · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig1_disagreement_falls.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig2_fluency(res, ph):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.96)
    ax.plot(res["steps"], res["logp_enforced"], color=AQUA, linewidth=2.4,
            zorder=3, label="fluency under enforcement (soundness stays 100%)")
    ref = res["logp_reference_good"]
    ax.axhline(ref, color=MUTED, linewidth=1.6, linestyle=(0, (5, 4)))
    ax.text(res["steps"][-1], ref, "  well-calibrated frontier",
            fontsize=11, color=INK2, va="bottom", ha="right")
    l0, l1 = res["logp_enforced"][0], res["logp_enforced"][-1]
    rec = 100 * (l1 - l0) / (ref - l0) if ref != l0 else 0
    ax.annotate(f"start: {l0:.2f}", xy=(res["steps"][0], l0),
                xytext=(res["steps"][-1] * 0.06, l0), fontsize=12,
                fontweight="bold", color=INK, va="bottom")
    ax.annotate(f"after: {l1:.2f}  ({rec:.0f}% of gap closed)",
                xy=(res["steps"][-1], l1),
                xytext=(res["steps"][-1] * 0.55, l1 - abs(ref - l0) * 0.10),
                fontsize=12, fontweight="bold", color=INK)
    headline(fig,
             "The fluency cost of enforcement trains away",
             f"Log-probability per generated sequence under per-step enforcement, {res['model_name']} · {res['domain']} · output validity is 100% throughout.")
    ax.set_xlabel("fine-tuning steps", fontsize=11.5, color=INK2)
    ax.set_ylabel("log-prob / sequence — higher = more fluent",
                  fontsize=11.5, color=INK2)
    base_axes(fig, ax)
    if ph:
        watermark(fig)
    footer(fig, f"N={res['n_eval_trajectories']} eval trajectories, seeds {res['seeds']} · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig2_fluency_recovers.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig3_enforcer_off(res, ph):
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=160)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.08, right=0.96)
    off = res["soundness_enforcer_off"]
    vals = [off["before"], off["after"]]
    labels = ["Before fine-tuning", "After fine-tuning"]
    bars = ax.bar([0, 1], vals, width=0.5, color=[ORANGE, AQUA], zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=INK)
    headline(fig,
             "The rulebook transfers into the weights",
             f"% of outputs fully valid with the enforcer switched OFF (bare model), {res['model_name']} · {res['domain']}. Shipped configuration keeps the enforcer on.")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12.5, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10.5)
    base_axes(fig, ax)
    if ph:
        watermark(fig)
    footer(fig, f"N={res['n_eval_trajectories']} eval trajectories, seeds {res['seeds']} · github.com/MikeHLee/structure_of_clear_thinking")
    fig.savefig(os.path.join(OUT, "fig3_enforcer_off.png"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results",
                   default=os.path.join(HERE, "..", "..", "results",
                                        "constraint_ft_results.json"))
    p.add_argument("--placeholder", action="store_true",
                   help="render watermarked layout previews from synthetic curves")
    args = p.parse_args()

    if args.placeholder:
        res = placeholder_results()
    else:
        with open(args.results) as f:
            res = json.load(f)
        if res["soundness_enforcer_off"]["after"] is None:
            raise SystemExit("results file incomplete: soundness_enforcer_off.after is null")

    fig1_disagreement(res, args.placeholder)
    fig2_fluency(res, args.placeholder)
    fig3_enforcer_off(res, args.placeholder)
    tag = " (PLACEHOLDER previews)" if args.placeholder else ""
    print(f"wrote 3 figures to {OUT}{tag}")


if __name__ == "__main__":
    main()
