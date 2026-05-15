"""scripts/plot_aug_ranking.py — visualise augmentation op difficulty ranking.

Reads results/aug_op_ranking.json (produced by experiments/rank_aug_ops.py) and
generates two figures for the thesis:

  1. aug_op_delta_loss.png   — horizontal bar chart of Δloss per op.
       Bar colour = manual tier.  Dashed lines = loss-based tier boundaries.
       Shows whether loss-based ordering matches the manual design.

  2. aug_op_rank_agreement.png — scatter: manual rank vs loss rank per op.
       Spearman ρ printed in the title.  Points on the diagonal = perfect agreement.

Usage:
    python -m scripts.plot_aug_ranking
    python -m scripts.plot_aug_ranking --ranking results/aug_op_ranking.json \\
                                        --output_dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

TIER_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
TIER_NAMES = {
    1: "Tier 1 — Geometric",
    2: "Tier 2 — Photometric",
    3: "Tier 3 — Destructive",
}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_delta_loss_bar(ranking: dict, out_path: str) -> None:
    """Horizontal bar chart sorted by Δloss; bars coloured by manual tier."""
    ops = sorted(ranking["ops"], key=lambda x: x["delta_loss"])
    names = [r["name"].replace("_", " ") for r in ops]
    deltas = [r["delta_loss"] for r in ops]
    loss_tiers = [r["loss_tier"] for r in ops]
    manual_tiers = [r["manual_tier"] for r in ops]

    fig, ax = plt.subplots(figsize=(11, 7))

    colors = [TIER_COLORS[t] for t in manual_tiers]
    ax.barh(names, deltas, color=colors, edgecolor="white", linewidth=0.6, zorder=2)

    # Loss-based tier boundary lines
    t1_boundary = sum(1 for t in loss_tiers if t == 1) - 0.5
    t2_boundary = sum(1 for t in loss_tiers if t <= 2) - 0.5
    for y, label in [(t1_boundary, "Loss T1/T2"), (t2_boundary, "Loss T2/T3")]:
        ax.axhline(
            y, color="black", linestyle="--", linewidth=1.3, alpha=0.75, zorder=3
        )
        ax.text(
            max(deltas) * 0.98,
            y + 0.15,
            label,
            fontsize=8,
            ha="right",
            va="bottom",
            color="black",
            alpha=0.8,
        )

    ax.axvline(0, color="#888888", linewidth=0.9, zorder=1)
    ax.set_xlabel(
        "Val loss delta  (op at strength=1.0  vs.  no-aug baseline)", fontsize=11
    )
    ax.set_title(
        f"Augmentation op difficulty — {ranking['metadata']['dataset'].upper()}\n"
        f"Bar colour = manual tier assignment  ·  Dashed lines = loss-based tier boundaries",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)

    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t]) for t in [1, 2, 3]
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_rank_agreement(ranking: dict, out_path: str) -> None:
    """Scatter: manual rank vs loss rank.  Spearman ρ in title."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy not available — skipping rank-agreement plot")
        return

    ops_data = ranking["ops"]

    # Manual rank: sort by (manual_tier, then registry insertion order)
    manual_order = sorted(ops_data, key=lambda x: (x["manual_tier"], x["name"]))
    manual_rank = {r["name"]: i for i, r in enumerate(manual_order)}

    # Loss rank: already sorted by delta_loss in the JSON
    loss_rank = {r["name"]: i for i, r in enumerate(ops_data)}

    x = np.array([manual_rank[r["name"]] for r in ops_data])
    y = np.array([loss_rank[r["name"]] for r in ops_data])
    colors = [TIER_COLORS[r["manual_tier"]] for r in ops_data]

    rho, pval = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=colors, s=90, zorder=3, edgecolors="white", linewidths=0.5)

    for r in ops_data:
        ax.annotate(
            r["name"].replace("_", "\n"),
            (manual_rank[r["name"]], loss_rank[r["name"]]),
            fontsize=6,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            color="#333333",
        )

    lim = len(ops_data) - 1
    ax.plot(
        [0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.4, label="Perfect agreement"
    )

    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t]) for t in [1, 2, 3]
    ]
    legend_patches.append(
        mpatches.Patch(color="none", label=f"Spearman ρ = {rho:.2f}  (p={pval:.3f})")
    )
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.9)

    ax.set_xlabel("Manual rank  (by tier)", fontsize=10)
    ax.set_ylabel("Loss-based rank  (by Δloss)", fontsize=10)
    ax.set_title(
        f"Manual vs loss-based op ordering — {ranking['metadata']['dataset'].upper()}\n"
        f"Spearman ρ = {rho:.2f}  (p = {pval:.3f})",
        fontsize=11,
    )
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_strength_curves(ranking: dict, out_path: str) -> None:
    """Line plot: Δloss vs strength for every op, coloured by manual tier.

    Also marks the recommended_strength for each op with a vertical tick,
    and draws horizontal dashed lines at the per-tier calibration targets.
    Skipped if the JSON predates the multi-strength sweep (no curve data).
    """
    if not ranking["ops"][0].get("strengths_tested"):
        print("  Skipping strength-curve plot — no per-strength data in JSON")
        return

    targets = ranking["metadata"].get("tier_delta_targets", {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    tier_titles = {
        1: "Tier 1 — Geometric",
        2: "Tier 2 — Photometric",
        3: "Tier 3 — Destructive",
    }

    for ax, tier in zip(axes, [1, 2, 3]):
        ops_in_tier = [r for r in ranking["ops"] if r["manual_tier"] == tier]
        color = TIER_COLORS[tier]

        for r in ops_in_tier:
            xs = r["strengths_tested"]
            ys = r["delta_losses_by_strength"]
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.75, label=r["name"])
            # Mark recommended strength
            rec = r.get("recommended_strength")
            if rec is not None and rec <= 1.0:
                rec_delta = np.interp(rec, xs, ys)
                ax.plot(rec, rec_delta, "o", color=color, markersize=6, zorder=4)

        # Tier calibration target line
        target = targets.get(str(tier)) or targets.get(tier)
        if target is not None:
            ax.axhline(target, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(
                0.02,
                target,
                f"target={target}",
                fontsize=7,
                va="bottom",
                color="black",
                alpha=0.7,
            )

        ax.set_title(tier_titles[tier], fontsize=10)
        ax.set_xlabel("Strength", fontsize=9)
        ax.set_ylabel("Δloss" if tier == 1 else "", fontsize=9)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.axhline(0, color="#888888", linewidth=0.7)

    fig.suptitle(
        f"Δloss vs strength per op — {ranking['metadata']['dataset'].upper()}\n"
        f"Dots = recommended_strength  ·  Dashed = tier calibration target",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def print_summary_table(ranking: dict) -> None:
    ops = ranking["ops"]  # already sorted by delta_loss
    print(f"\n  {'Op':<22} {'Δloss':>8}  loss_tier  manual_tier  agreement")
    print(f"  {'─' * 62}")
    matches = 0
    for r in ops:
        match = "✓" if r["loss_tier"] == r["manual_tier"] else "✗"
        if r["loss_tier"] == r["manual_tier"]:
            matches += 1
        print(
            f"  {r['name']:<22} {r['delta_loss']:>+8.4f}"
            f"      {r['loss_tier']}          {r['manual_tier']}         {match}"
        )
    pct = 100 * matches / len(ops)
    print(f"\n  Tier agreement: {matches}/{len(ops)} ops ({pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", default="results/aug_op_ranking.json")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    ranking = _load(args.ranking)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = ranking["metadata"]
    print(
        f"\nPlotting aug ranking  |  dataset={meta['dataset']}"
        f"  model={meta['model']}  date={meta['date']}"
    )

    plot_delta_loss_bar(ranking, str(out / "aug_op_delta_loss.png"))
    plot_rank_agreement(ranking, str(out / "aug_op_rank_agreement.png"))
    plot_strength_curves(ranking, str(out / "aug_op_strength_curves.png"))
    print_summary_table(ranking)


if __name__ == "__main__":
    main()
