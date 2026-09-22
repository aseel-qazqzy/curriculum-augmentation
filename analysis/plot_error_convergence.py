"""analysis/plot_error_convergence.py

Accuracy/error convergence plot, styled identically to
analysis/plot_thesis_convergence.py (same palette, line styles, LaTeX config,
1x2 layout) — reuses that script's parser/aggregator/plotter directly instead
of duplicating them.

IMPORTANT DATA-AVAILABILITY NOTE (read before using this for the thesis):
The cluster logs evaluate the TEST set exactly ONCE, at the very end of
training (the "FINAL RESULTS" block: `Test Top-1`, `Test Error (Top-1)`,
etc.). There is no per-epoch test accuracy/error anywhere in these files —
every "Epoch [...]" line only ever reports Train/Val metrics (confirmed by
inspecting `plot_thesis_convergence._EPOCH_RE`, which has no test-metric
group because none exists in the log text). A genuine "Test Top-1 Error over
Epochs" trajectory therefore CANNOT be reconstructed from these logs without
re-running training with per-epoch test-set evaluation.

This script plots the closest honest substitute instead: VALIDATION Top-1
Error (%) = 100 - val_acc, which IS logged every 10 epochs for every run. The
single, correct final TEST error value is already reported elsewhere as a
point estimate (not a curve) — see
`analysis/plot_seed_distribution_stats.py --metric test_error_top1`, which
produces both the bar-chart-equivalent comparison table and a box/strip plot
of the 5 per-seed final test-error values.

Usage:
    python analysis/plot_error_convergence.py
    python analysis/plot_error_convergence.py --band ci95
    python analysis/plot_error_convergence.py --metric train_error
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_thesis_convergence import (
    LOG_ROOT,
    METRIC_LABELS,
    aggregate_curves,
    configure_style,
    discover_logs,
    plot_convergence,
)

# Extend (not modify) the shared label dict with the two derived error metrics.
METRIC_LABELS.setdefault("val_error", "Validation Top-1 Error (%)")
METRIC_LABELS.setdefault("train_error", "Training Top-1 Error (%)")


def add_error_columns(df):
    df = df.copy()
    df["val_error"] = 100.0 - df["val_acc"]
    df["train_error"] = 100.0 - df["train_acc"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-error convergence plot (see module docstring: test error is only ever a single final value, not a trajectory, in these logs)"
    )
    parser.add_argument("--log_root", default=str(LOG_ROOT))
    parser.add_argument("--band", choices=["sd", "ci95"], default="sd")
    parser.add_argument(
        "--metric", choices=["val_error", "train_error"], default="val_error"
    )
    args = parser.parse_args()

    df = discover_logs(Path(args.log_root))
    df = add_error_columns(df)
    use_tex = configure_style()

    agg = aggregate_curves(df, metric=args.metric, band=args.band)
    plot_convergence(
        agg,
        metric=args.metric,
        use_tex=use_tex,
        name=f"error_convergence_{args.metric}",
    )

    print(
        "\nNOTE: this plots VALIDATION error over epochs — the only per-epoch error "
        "signal available in these logs. TEST error is only computed once, at the "
        "final epoch; see analysis/plot_seed_distribution_stats.py --metric "
        "test_error_top1 for that value (bar/box comparison across seeds)."
    )


if __name__ == "__main__":
    main()
