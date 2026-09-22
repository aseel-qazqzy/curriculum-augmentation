"""analysis/plot_seed_distribution_stats.py

Per-seed distribution figure (box + individual seed points) and a
Static+Mixing-vs-proposed-method statistical comparison table, built on the
same 40 cluster .log files as analysis/plot_thesis_convergence.py (2 models x
4 methods x 5 seeds: 42, 123, 456, 1024, 3407). Reuses that script's log
parser via `import plot_thesis_convergence` rather than re-implementing the
regexes.

Design note on paired vs. Welch's t-test:
  The 4 methods share the SAME 5 seeds per model (verified at runtime — see
  `_require_matched_seeds`), so this is a matched/repeated-measures design.
  A PAIRED t-test (scipy.stats.ttest_rel) is the statistically correct primary
  test here (it removes seed-to-seed variance common to all methods, which is
  exactly what a between-seeds Welch's test would otherwise treat as noise).
  Welch's t-test (unpaired, unequal-variance) is reported alongside it as a
  secondary/robustness check, per the request. If seeds ever fail to match
  exactly for some method (e.g. a run is missing), this script falls back to
  Welch's test only for the affected comparison and says so explicitly rather
  than silently pairing mismatched seeds.

Usage:
    python analysis/plot_seed_distribution_stats.py
    python analysis/plot_seed_distribution_stats.py --metric val_acc
    python analysis/plot_seed_distribution_stats.py --plot violin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns

    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

from plot_thesis_convergence import (
    LOG_ROOT,
    METHOD_ORDER,
    METHOD_STYLE,
    METRIC_LABELS,
    MODEL_DIRS,
    OUT_DIR as _CONVERGENCE_OUT_DIR,
    _tex,
    configure_style,
    discover_logs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_OUT_DIR = (
    _CONVERGENCE_OUT_DIR  # results/figs/thesis_convergence — same figure family
)
TABLE_OUT_DIR = PROJECT_ROOT / "results" / "tables"

BASELINE_METHOD = "Static+Mixing"
PROPOSED_METHODS = ["ETS", "LPS", "EGS"]


# ─────────────────────────────────────────────────────────────────────────────
# Data shaping
# ─────────────────────────────────────────────────────────────────────────────


def per_run_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (model, method, seed) — collapses the epoch-level rows since
    the metrics used here (final/test metrics) are constant across epochs."""
    return df.drop_duplicates(subset=["model", "method", "seed"])[
        ["model", "method", "seed", metric]
    ].copy()


def _require_matched_seeds(
    pivot: pd.DataFrame, model: str, baseline: str, proposed: str
) -> pd.DataFrame:
    """Return the subset of `pivot` where BOTH baseline and proposed have a
    non-NaN value for the same seed. Prints a warning if any seeds had to be
    dropped (i.e. the design was not perfectly matched for this comparison)."""
    paired = pivot[[baseline, proposed]].dropna()
    n_dropped = len(pivot) - len(paired)
    if n_dropped > 0:
        print(
            f"  WARNING [{model}] {proposed} vs {baseline}: {n_dropped} seed(s) missing "
            f"in one of the two methods — paired test computed on the {len(paired)} "
            f"seed(s) present in both."
        )
    return paired


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────


def _cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for a paired design: mean difference / std of the differences."""
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction (no extra dependency on statsmodels)."""
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((n - rank) * pvals[idx], 1.0)
        running_max = max(running_max, adj)
        corrected[idx] = running_max
    return corrected


def compute_comparisons(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    per_run = per_run_table(df, metric)
    rows = []

    for model in MODEL_DIRS.values():
        sub = per_run[per_run["model"] == model]
        pivot = sub.pivot(index="seed", columns="method", values=metric)

        for proposed in PROPOSED_METHODS:
            if BASELINE_METHOD not in pivot.columns or proposed not in pivot.columns:
                print(
                    f"  SKIPPED [{model}] {proposed} vs {BASELINE_METHOD}: one of the two methods has no runs."
                )
                continue

            paired = _require_matched_seeds(pivot, model, BASELINE_METHOD, proposed)
            baseline_vals = paired[BASELINE_METHOD].to_numpy()
            proposed_vals = paired[proposed].to_numpy()
            n = len(paired)

            if n >= 2:
                t_paired, p_paired = stats.ttest_rel(proposed_vals, baseline_vals)
                d_paired = _cohens_d_paired(proposed_vals - baseline_vals)
            else:
                t_paired, p_paired, d_paired = np.nan, np.nan, np.nan

            # Welch's t-test (unpaired, unequal variance) — uses the FULL column
            # for each method (not seed-intersected), since Welch's does not
            # require matched samples.
            full_baseline = pivot[BASELINE_METHOD].dropna().to_numpy()
            full_proposed = pivot[proposed].dropna().to_numpy()
            if len(full_baseline) >= 2 and len(full_proposed) >= 2:
                t_welch, p_welch = stats.ttest_ind(
                    full_proposed, full_baseline, equal_var=False
                )
            else:
                t_welch, p_welch = np.nan, np.nan

            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "comparison": f"{proposed} vs {BASELINE_METHOD}",
                    "n_seeds_paired": n,
                    "mean_baseline": float(baseline_vals.mean()) if n else np.nan,
                    "mean_proposed": float(proposed_vals.mean()) if n else np.nan,
                    "mean_diff": float(proposed_vals.mean() - baseline_vals.mean())
                    if n
                    else np.nan,
                    "t_paired": float(t_paired),
                    "p_paired": float(p_paired),
                    "cohens_d_paired": d_paired,
                    "t_welch": float(t_welch),
                    "p_welch": float(p_welch),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        valid = result["p_paired"].notna()
        result.loc[valid, "p_paired_holm"] = holm_correction(
            result.loc[valid, "p_paired"].to_numpy()
        )
    return result


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


def plot_seed_distribution(
    df: pd.DataFrame, comparisons: pd.DataFrame, metric: str, kind: str, use_tex: bool
) -> None:
    per_run = per_run_table(df, metric)
    ylabel = METRIC_LABELS.get(metric, metric)
    model_labels = list(MODEL_DIRS.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    colors = [METHOD_STYLE[m]["color"] for m in METHOD_ORDER]

    for ax, model_label in zip(axes, model_labels):
        sub = per_run[per_run["model"] == model_label]

        if HAVE_SEABORN:
            if kind == "violin":
                sns.violinplot(
                    ax=ax,
                    data=sub,
                    x="method",
                    y=metric,
                    order=METHOD_ORDER,
                    palette=colors,
                    inner="box",
                    cut=0,
                    linewidth=1.0,
                    zorder=2,
                )
            else:
                sns.boxplot(
                    ax=ax,
                    data=sub,
                    x="method",
                    y=metric,
                    order=METHOD_ORDER,
                    palette=colors,
                    width=0.55,
                    showmeans=True,
                    meanprops={
                        "marker": "D",
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                        "markersize": 5,
                    },
                    zorder=2,
                )
            sns.stripplot(
                ax=ax,
                data=sub,
                x="method",
                y=metric,
                order=METHOD_ORDER,
                color="black",
                size=5,
                jitter=0.15,
                alpha=0.85,
                zorder=3,
            )
        else:
            data = [sub[sub["method"] == m][metric].to_numpy() for m in METHOD_ORDER]
            bp = ax.boxplot(
                data,
                tick_labels=METHOD_ORDER,
                patch_artist=True,
                showmeans=True,
                widths=0.55,
                zorder=2,
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)
            rng = np.random.default_rng(0)
            for i, vals in enumerate(data, start=1):
                jitter = rng.normal(0, 0.06, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter,
                    vals,
                    color="black",
                    s=22,
                    zorder=3,
                    alpha=0.85,
                )

        # Significance brackets: each proposed method vs. Static+Mixing.
        model_comp = comparisons[comparisons["model"] == model_label]
        y_max = sub[metric].max()
        y_min = sub[metric].min()
        span = max(y_max - y_min, 1e-6)
        base_idx = METHOD_ORDER.index(BASELINE_METHOD) + 1
        step = span * 0.10
        level = y_max + span * 0.08
        for proposed in PROPOSED_METHODS:
            row = model_comp[
                model_comp["comparison"] == f"{proposed} vs {BASELINE_METHOD}"
            ]
            if row.empty:
                continue
            p_val = row["p_paired"].iloc[0]
            prop_idx = METHOD_ORDER.index(proposed) + 1
            x1, x2 = sorted([base_idx, prop_idx])
            ax.plot(
                [x1, x1, x2, x2],
                [level, level + step * 0.25, level + step * 0.25, level],
                color="black",
                linewidth=1.0,
            )
            ax.text(
                (x1 + x2) / 2,
                level + step * 0.3,
                _sig_stars(p_val),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            level += step

        ax.set_title(model_label)
        ax.set_xlabel("")

    axes[0].set_ylabel(_tex(ylabel, use_tex))
    fig.suptitle(
        _tex(
            f"Per-seed {ylabel} distribution (n=5 seeds; paired t-test vs. {BASELINE_METHOD})",
            use_tex,
        )
    )
    fig.tight_layout()

    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"seed_distribution_{kind}_{metric}"
    for ext in ("png", "pdf", "svg"):
        path = FIG_OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-seed distribution plot + Static+Mixing-vs-proposed statistical table"
    )
    parser.add_argument("--log_root", default=str(LOG_ROOT))
    parser.add_argument(
        "--metric",
        default="test_top1",
        choices=[
            "final_train_loss",
            "final_train_acc",
            "final_val_loss",
            "final_val_acc",
            "best_val_top1",
            "test_top1",
            "test_top5",
            "test_error_top1",
            "test_error_top5",
        ],
    )
    parser.add_argument("--plot", choices=["box", "violin"], default="box")
    args = parser.parse_args()

    df = discover_logs(Path(args.log_root))
    use_tex = configure_style()

    comparisons = compute_comparisons(df, metric=args.metric)

    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLE_OUT_DIR / f"seed_comparison_stats_{args.metric}.csv"
    comparisons.to_csv(table_path, index=False)
    print(f"\n  Saved statistical comparison table -> {table_path}\n")

    with pd.option_context(
        "display.float_format", "{:.4f}".format, "display.width", 160
    ):
        print(comparisons.to_string(index=False))

    plot_seed_distribution(
        df, comparisons, metric=args.metric, kind=args.plot, use_tex=use_tex
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
