"""analysis/plot_thesis_convergence.py

Parses the raw .log files produced by the cluster runs under
`results/cluster/logs/{resnet,wideresnet}/` and generates publication-quality
convergence and final-metric comparison figures for the thesis's main results
matrix: 2 models (ResNet-50, WideResNet-28-10) x 4 methods
(Static+Mixing, ETS, LPS, EGS) x 5 seeds (42, 123, 456, 1024, 3407).

Log format (inspected directly from the actual files before writing this
parser — see the two regexes below):

    Epoch [ 10/100] Train: 1.5006 / 57.32% | Val: 1.8243 / 50.14% | Top-5: 81.60% | LR: 0.09932 | <method-specific trailing fields> | Time: 415s

    FINAL RESULTS: ...
      Train Loss (last ep)        0.7405
      Train Acc  (last ep)        85.98%
      Val Loss   (last ep)        0.7319
      Val Acc    (last ep)        80.94%
      Best Val Top-1              80.94%  (epoch 100)
      Test Top-1                  80.41%
      Test Top-5                  95.32%
      Test Error (Top-1)          19.59%
      Test Error (Top-5)           4.68%
      Val-Test Gap                 0.53%

Note: `log_every=10` in these runs, so epoch lines only exist for
epoch 1 and every 10th epoch after (11 points per run: 1, 10, 20, ..., 100).
The convergence curves below are necessarily built from that resolution.

Usage:
    python analysis/plot_thesis_convergence.py
    python analysis/plot_thesis_convergence.py --band ci95
    python analysis/plot_thesis_convergence.py --metric val_loss --final_metric test_error_top1
"""

from __future__ import annotations

import argparse
import re
import shutil
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "results" / "cluster" / "logs"
OUT_DIR = PROJECT_ROOT / "results" / "figs" / "thesis_convergence"

# Subfolder name -> display label, in the order they should appear (left, right).
MODEL_DIRS: dict[str, str] = {
    "resnet": "ResNet-50",
    "wideresnet": "WideResNet-28-10",
}

METHOD_ORDER = ["Static+Mixing", "ETS", "LPS", "EGS"]

# Okabe-Ito colorblind-safe palette + distinct dashes/markers so figures stay
# readable in grayscale print.
METHOD_STYLE = {
    "Static+Mixing": {"color": "#E69F00", "ls": "-", "marker": "o"},  # orange
    "ETS": {"color": "#0072B2", "ls": "--", "marker": "s"},  # blue
    "LPS": {"color": "#009E73", "ls": "-.", "marker": "^"},  # green
    "EGS": {"color": "#D55E00", "ls": (0, (1, 1)), "marker": "D"},  # vermillion, dotted
}

METRIC_LABELS = {
    "train_loss": "Training Loss",
    "train_acc": "Training Accuracy (%)",
    "val_loss": "Validation Loss",
    "val_acc": "Validation Accuracy (%)",
    "val_top5": "Validation Top-5 Accuracy (%)",
    "test_top1": "Test Top-1 Accuracy (%)",
    "test_top5": "Test Top-5 Accuracy (%)",
    "test_error_top1": "Test Top-1 Error (%)",
    "test_error_top5": "Test Top-5 Error (%)",
    "best_val_top1": "Best Validation Top-1 Accuracy (%)",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Filename parsing
# ─────────────────────────────────────────────────────────────────────────────

# Order matters: "egs" must be checked before "tiered_ets"/"tiered_lps" since
# egs_v2 filenames don't contain "tiered_" at all (e.g.
# egs_v2_resnet50_19op_100ep_s42_..._p19_....log vs
# resnet50_tiered_ets_mix_both_..._s42_..._p19_....log).
_METHOD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"egs"), "EGS"),
    (re.compile(r"tiered_ets"), "ETS"),
    (re.compile(r"tiered_lps"), "LPS"),
    (re.compile(r"static_mixing"), "Static+Mixing"),
]
_SEED_RE = re.compile(r"_s(\d+)_p\d+")


def parse_method(filename: str) -> str | None:
    for pattern, label in _METHOD_PATTERNS:
        if pattern.search(filename):
            return label
    return None


def parse_seed(filename: str) -> int | None:
    m = _SEED_RE.search(filename)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text regex extraction
# ─────────────────────────────────────────────────────────────────────────────

# Common prefix shared by every method's epoch line (ETS/LPS/Static+Mixing/EGS
# all differ only in the trailing tier/mixing fields, which are intentionally
# NOT captured here since they aren't needed for the thesis figures).
_EPOCH_RE = re.compile(
    r"Epoch\s*\[\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\]\s*"
    r"Train:\s*(?P<train_loss>[\d.]+)\s*/\s*(?P<train_acc>[\d.]+)%\s*\|\s*"
    r"Val:\s*(?P<val_loss>[\d.]+)\s*/\s*(?P<val_acc>[\d.]+)%\s*\|\s*"
    r"Top-5:\s*(?P<val_top5>[\d.]+)%\s*\|\s*"
    r"LR:\s*(?P<lr>[\d.]+)"
)

# The "FINAL RESULTS" summary block at the end of every log.
_FINAL_PATTERNS: dict[str, re.Pattern] = {
    "final_train_loss": re.compile(r"Train Loss\s*\(last ep\)\s+([\d.]+)"),
    "final_train_acc": re.compile(r"Train Acc\s*\(last ep\)\s+([\d.]+)%"),
    "final_val_loss": re.compile(r"Val Loss\s*\(last ep\)\s+([\d.]+)"),
    "final_val_acc": re.compile(r"Val Acc\s*\(last ep\)\s+([\d.]+)%"),
    "best_val_top1": re.compile(r"Best Val Top-1\s+([\d.]+)%"),
    "test_top1": re.compile(r"Test Top-1\s+([\d.]+)%"),
    "test_top5": re.compile(r"Test Top-5\s+([\d.]+)%"),
    "test_error_top1": re.compile(r"Test Error\s*\(Top-1\)\s+([\d.]+)%"),
    "test_error_top5": re.compile(r"Test Error\s*\(Top-5\)\s+([\d.]+)%"),
    # en-dash in "Val–Test Gap" in the source logs; "." matches it regardless
    # of which dash character a given log actually used.
    "val_test_gap": re.compile(r"Val.Test Gap\s+([\d.]+)%"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modular data loader
# ─────────────────────────────────────────────────────────────────────────────


def load_log_file(file_path: Path) -> pd.DataFrame:
    """Parse one .log file into a per-epoch DataFrame with metadata columns
    (model, method, seed) and the final-results block broadcast onto every row."""
    model_dir = file_path.parent.name
    model_label = MODEL_DIRS.get(model_dir, model_dir)
    method = parse_method(file_path.name)
    seed = parse_seed(file_path.name)
    if method is None or seed is None:
        raise ValueError(
            f"Could not parse method/seed from filename: {file_path.name} "
            f"(method={method}, seed={seed})"
        )

    text = file_path.read_text()

    rows = []
    for m in _EPOCH_RE.finditer(text):
        rows.append(
            {
                "model": model_label,
                "model_dir": model_dir,
                "method": method,
                "seed": seed,
                "epoch": int(m["epoch"]),
                "total_epochs": int(m["total"]),
                "train_loss": float(m["train_loss"]),
                "train_acc": float(m["train_acc"]),
                "val_loss": float(m["val_loss"]),
                "val_acc": float(m["val_acc"]),
                "val_top5": float(m["val_top5"]),
                "lr": float(m["lr"]),
            }
        )
    if not rows:
        raise ValueError(f"No epoch lines matched the expected format in {file_path}")

    df = pd.DataFrame(rows)

    for key, pattern in _FINAL_PATTERNS.items():
        fm = pattern.search(text)
        df[key] = float(fm.group(1)) if fm else np.nan

    return df


def discover_logs(log_root: Path = LOG_ROOT) -> pd.DataFrame:
    """Walk every model subdirectory under log_root, parse every .log file,
    and return one concatenated long-format DataFrame."""
    if not log_root.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_root}")

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for model_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for log_file in sorted(model_dir.glob("*.log")):
            try:
                frames.append(load_log_file(log_file))
            except ValueError as e:
                skipped.append(str(e))

    if skipped:
        print(f"WARNING: skipped {len(skipped)} file(s) that did not parse cleanly:")
        for s in skipped:
            print(f"  - {s}")

    if not frames:
        raise RuntimeError(f"No parsable .log files found under {log_root}")

    df = pd.concat(frames, ignore_index=True)
    n_runs = df.drop_duplicates(subset=["model", "method", "seed"]).shape[0]
    print(
        f"Parsed {len(frames)} log file(s) -> {n_runs} unique (model, method, seed) runs, {len(df)} epoch rows."
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical aggregation across seeds
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_curves(df: pd.DataFrame, metric: str, band: str) -> pd.DataFrame:
    """Mean trajectory + variance band across seeds, per (model, method, epoch)."""
    g = df.groupby(["model", "method", "epoch"])[metric]
    agg = g.agg(mean="mean", std="std", n="count").reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg["std"] / np.sqrt(agg["n"])
    if band == "ci95":
        # t-based 95% CI (safer than a normal z-score at n=5 seeds).
        agg["band"] = agg.apply(
            lambda r: r["sem"] * stats.t.ppf(0.975, r["n"] - 1) if r["n"] > 1 else 0.0,
            axis=1,
        )
    else:
        agg["band"] = agg["std"]
    return agg


def aggregate_final(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- SEM across seeds for a final/summary metric, per (model, method)."""
    per_run = df.drop_duplicates(subset=["model", "method", "seed"])[
        ["model", "method", "seed", metric]
    ]
    g = per_run.groupby(["model", "method"])[metric]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    out["sem"] = out["std"] / np.sqrt(out["n"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Styling
# ─────────────────────────────────────────────────────────────────────────────


def configure_style() -> bool:
    """Clean scientific styling (seaborn 'paper' context if available, plain
    matplotlib equivalent otherwise) + best-effort LaTeX text rendering with
    automatic fallback. Returns whether LaTeX ended up enabled."""
    if HAVE_SEABORN:
        sns.set_context("paper")
        sns.set_style("whitegrid")
    else:
        print(
            "seaborn not installed in this environment — using a matplotlib-only "
            "styling fallback that mimics seaborn's 'paper'/'whitegrid' look. "
            "Install seaborn (e.g. `pip install seaborn`) for full parity."
        )

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white" if HAVE_SEABORN else "#EAEAF2",
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.color": "white" if not HAVE_SEABORN else "#DDDDDD",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.95,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )

    use_tex = False
    if shutil.which("latex") and shutil.which("dvipng"):
        try:
            matplotlib.rcParams["text.usetex"] = True
            matplotlib.rcParams["font.family"] = "serif"
            fig = plt.figure()
            plt.text(0.5, 0.5, r"$\alpha\,\Delta$")
            fig.canvas.draw()
            plt.close(fig)
            use_tex = True
        except Exception as e:
            print(
                f"LaTeX rendering test failed ({e}) — falling back to standard Matplotlib text."
            )
            matplotlib.rcParams["text.usetex"] = False
    if not use_tex:
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["font.family"] = "DejaVu Serif"
    return use_tex


def _tex(s: str, use_tex: bool) -> str:
    """Escape LaTeX special characters (%, _, &, #) when usetex is active, since
    matplotlib does not do this automatically and an un-escaped '%' silently
    comments out the rest of the label. No-op when usetex is off."""
    if not use_tex:
        return s
    for ch in ("%", "_", "&", "#"):
        s = s.replace(ch, f"\\{ch}")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 6. Figures
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")


def plot_convergence(
    agg: pd.DataFrame, metric: str, use_tex: bool, name: str = "main_convergence"
) -> None:
    ylabel = METRIC_LABELS.get(metric, metric)
    model_labels = list(MODEL_DIRS.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, model_label in zip(axes, model_labels):
        sub_model = agg[agg["model"] == model_label]
        for method in METHOD_ORDER:
            sub = sub_model[sub_model["method"] == method].sort_values("epoch")
            if sub.empty:
                continue
            style = METHOD_STYLE[method]
            ax.plot(
                sub["epoch"],
                sub["mean"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=4,
                markevery=1,
                label=method,
                zorder=3,
            )
            ax.fill_between(
                sub["epoch"],
                sub["mean"] - sub["band"],
                sub["mean"] + sub["band"],
                color=style["color"],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        ax.set_title(model_label)
        ax.set_xlabel(_tex("Epoch", use_tex))
        ax.margins(x=0.02)

    axes[0].set_ylabel(_tex(ylabel, use_tex))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
    )
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)


def plot_final_bars(
    final_df: pd.DataFrame,
    metric: str,
    use_tex: bool,
    name: str = "final_epoch_comparison",
) -> None:
    ylabel = METRIC_LABELS.get(metric, metric)
    model_labels = list(MODEL_DIRS.values())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)

    for ax, model_label in zip(axes, model_labels):
        sub = (
            final_df[final_df["model"] == model_label]
            .set_index("method")
            .reindex(METHOD_ORDER)
        )
        x = np.arange(len(METHOD_ORDER))
        colors = [METHOD_STYLE[m]["color"] for m in METHOD_ORDER]
        ax.bar(
            x,
            sub["mean"],
            yerr=sub["sem"],
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right")
        ax.set_title(model_label)
        ax.grid(axis="y", zorder=0)

    axes[0].set_ylabel(_tex(ylabel, use_tex))
    fig.suptitle(
        _tex(
            f"Final-epoch {ylabel} (mean $\\pm$ SEM over seeds)"
            if use_tex
            else f"Final-epoch {ylabel} (mean +/- SEM over seeds)",
            use_tex,
        )
    )
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    global LOG_ROOT, OUT_DIR
    parser = argparse.ArgumentParser(
        description="Generate thesis convergence + final-metric figures from cluster .log files"
    )
    parser.add_argument("--log_root", default=str(LOG_ROOT))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument(
        "--band",
        choices=["sd", "ci95"],
        default="sd",
        help="Shaded band: +/-1 SD or 95%% t-based CI",
    )
    parser.add_argument(
        "--metric",
        default="val_acc",
        choices=["train_loss", "train_acc", "val_loss", "val_acc", "val_top5"],
        help="Metric for the convergence plot",
    )
    parser.add_argument(
        "--final_metric",
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
        help="Metric for the final-epoch comparison bar chart",
    )
    args = parser.parse_args()

    LOG_ROOT = Path(args.log_root)
    OUT_DIR = Path(args.out_dir)

    df = discover_logs(LOG_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "parsed_epoch_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved parsed data -> {csv_path}")

    use_tex = configure_style()

    agg = aggregate_curves(df, metric=args.metric, band=args.band)
    plot_convergence(agg, metric=args.metric, use_tex=use_tex)

    final_df = aggregate_final(df, metric=args.final_metric)
    plot_final_bars(final_df, metric=args.final_metric, use_tex=use_tex)

    print("\nDone.")


if __name__ == "__main__":
    main()
