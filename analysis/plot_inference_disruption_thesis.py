"""analysis/plot_inference_disruption_thesis.py

Publication-ready postprocessing of the inference-time disruption diagnostic
(experiments/run_augmentation_inference_disruption.py). This script does NOT
re-run that experiment and does NOT touch its output directory
(`results/augmentation_inference_disruption/`) — it only READS the CSVs and
statistics.txt that experiment already produced, and writes new, separate,
polished artifacts elsewhere. That separation matters: re-running the raw
diagnostic regenerates its own plain `tier_scatter.png` from scratch, and if
this script wrote into the same directory, that re-run would silently clobber
the thesis-ready version produced here.

Inputs (must already exist — run the diagnostic first if missing):
    results/augmentation_inference_disruption/summary_by_operation.csv
    results/augmentation_inference_disruption/statistics.txt

Outputs:
    results/figs/inference_disruption/tier_scatter.{png,pdf}
    results/tables/inference_disruption_summary.tex
        (requires \\usepackage{booktabs} in the thesis preamble)

Usage:
    python analysis/plot_inference_disruption_thesis.py
    python analysis/plot_inference_disruption_thesis.py --top_n_violations 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_thesis_convergence import _tex, configure_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "results" / "augmentation_inference_disruption"
FIG_OUT_DIR = PROJECT_ROOT / "results" / "figs" / "inference_disruption"
TABLE_OUT_DIR = PROJECT_ROOT / "results" / "tables"

# Same tier palette as run_augmentation_inference_disruption.py /
# run_augmentation_difficulty.py, for visual consistency across this
# diagnostic's whole figure family.
TIER_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
TIER_NAMES = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
MIXING_OPS = ["cutmix", "mixup"]

ANNOTATE_OPS = ["blur", "solarize", "translate_x", "posterize"]

# Hand-tuned (dx, dy) text offsets for THIS dataset's actual point positions,
# chosen to avoid overlapping nearby clustered points (e.g. posterize sits
# right next to flip/sharpness/auto_contrast near y=0; translate_x sits right
# next to translate_y near y=0.5). Revisit these if the underlying data changes.
_ANNOTATION_OFFSETS = {
    "blur": (0.18, 0.0),
    "solarize": (0.20, 0.18),
    "translate_x": (0.22, -0.05),
    "posterize": (0.20, -0.05),
}


def load_summary() -> pd.DataFrame:
    path = SRC_DIR / "summary_by_operation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run "
            "`python -m experiments.run_augmentation_inference_disruption` first."
        )
    return pd.read_csv(path)


def parse_statistics(stats_path: Path) -> dict:
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found.")
    text = stats_path.read_text()

    rho_m = re.search(r"rho=(-?[\d.]+)\s+p=([\d.]+)", text)
    if not rho_m:
        raise ValueError(f"Could not find 'rho=... p=...' line in {stats_path}")
    rho, p = float(rho_m.group(1)), float(rho_m.group(2))

    violations = []
    for m in re.finditer(
        r"^\s*(\w+): actual Tier (\d+), empirical rank (\d+), rank-band expects Tier (\d+)",
        text,
        re.MULTILINE,
    ):
        actual, rank, expected = int(m.group(2)), int(m.group(3)), int(m.group(4))
        violations.append(
            {
                "operation": m.group(1),
                "actual_tier": actual,
                "empirical_rank": rank,
                "expected_tier": expected,
                "severity": abs(actual - expected),
            }
        )

    n_m = re.search(
        r"Operations violating expected monotonic ordering:\s*(\d+)/(\d+)", text
    )
    n_violations, n_total = (
        (int(n_m.group(1)), int(n_m.group(2))) if n_m else (len(violations), None)
    )

    return {
        "rho": rho,
        "p": p,
        "violations": violations,
        "n_violations": n_violations,
        "n_total": n_total,
    }


def plot_tier_scatter(df: pd.DataFrame, rho: float, p: float, use_tex: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    rng = np.random.default_rng(1)

    for _, row in df.iterrows():
        op, tier, y = row["operation"], int(row["tier"]), float(row["mean_delta_loss"])
        x = tier + rng.normal(0, 0.06)
        is_mix = op in MIXING_OPS

        if is_mix:
            ax.scatter(
                x,
                y,
                s=170,
                marker="*",
                color=TIER_COLORS[tier],
                edgecolor="black",
                linewidth=0.9,
                zorder=4,
            )
        else:
            ax.scatter(
                x,
                y,
                s=55,
                marker="o",
                color=TIER_COLORS[tier],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

        if op in ANNOTATE_OPS:
            dx, dy = _ANNOTATION_OFFSETS[op]
            label = op.replace("_", r"\_") if use_tex else op
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x + dx, y + dy),
                fontsize=9.5,
                ha="left",
                va="center",
                zorder=5,
                arrowprops=dict(
                    arrowstyle="-", color="black", lw=0.7, shrinkA=3, shrinkB=5
                ),
            )

    ax.axhline(0, color="#888888", linewidth=0.9, zorder=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([TIER_NAMES[t] for t in (1, 2, 3)])
    ax.set_xlim(0.7, 3.75)
    ax.set_ylabel(_tex("Mean inference-time $\\Delta L$ per operation", use_tex))
    ax.set_title(
        _tex(
            f"Existing tier assignment vs. measured inference-time disruption\n"
            f"Spearman $\\rho={rho:.3f}$, $p={p:.3f}$ (frozen model, no further training)",
            use_tex,
        )
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TIER_COLORS[t],
            markeredgecolor="white",
            markersize=8,
            label=TIER_NAMES[t],
        )
        for t in (1, 2, 3)
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="grey",
            markeredgecolor="black",
            markersize=15,
            label="Mixing op (CutMix / MixUp)",
        )
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_OUT_DIR / f"tier_scatter.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  Saved -> {path}")
    plt.close(fig)


def make_latex_table(stats: dict, top_n: int, out_path: Path) -> None:
    violations = sorted(
        stats["violations"], key=lambda v: (-v["severity"], v["empirical_rank"])
    )[:top_n]
    n_total = stats["n_total"] if stats["n_total"] is not None else "N"

    lines = []
    lines.append("% Auto-generated by analysis/plot_inference_disruption_thesis.py")
    lines.append(
        "% Source data: results/augmentation_inference_disruption/statistics.txt"
    )
    lines.append(
        "% Requires \\usepackage{booktabs} in the preamble. Paste directly into the manuscript."
    )
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Association between the existing (code-defined) augmentation tier "
        f"assignment and measured inference-time validation-loss disruption on a frozen "
        f"model. Spearman rank correlation across all {n_total} operations: "
        f"$\\rho = {stats['rho']:.3f}$, $p = {stats['p']:.3f}$ "
        f"({'not ' if stats['p'] >= 0.05 else ''}significant at $\\alpha = 0.05$). "
        f"{stats['n_violations']} of {n_total} operations violate the monotonic ordering "
        "their assigned tier predicts; the operations below are the most severe violations, "
        "ranked by tier-distance (assigned tier vs. the tier the operation's empirical rank "
        "would predict).}"
    )
    lines.append("\\label{tab:inference-disruption-summary}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "Operation & Assigned Tier & Empirical Rank & Rank-Predicted Tier \\\\"
    )
    lines.append("\\midrule")
    for v in violations:
        op_tex = v["operation"].replace("_", "\\_")
        lines.append(
            f"{op_tex} & {v['actual_tier']} & {v['empirical_rank']} & {v['expected_tier']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-ready tier-scatter figure + LaTeX summary table for the inference-time disruption diagnostic"
    )
    parser.add_argument("--top_n_violations", type=int, default=6)
    args = parser.parse_args()

    df = load_summary()
    stats = parse_statistics(SRC_DIR / "statistics.txt")
    use_tex = configure_style()

    plot_tier_scatter(df, stats["rho"], stats["p"], use_tex)

    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_latex_table(
        stats, args.top_n_violations, TABLE_OUT_DIR / "inference_disruption_summary.tex"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
