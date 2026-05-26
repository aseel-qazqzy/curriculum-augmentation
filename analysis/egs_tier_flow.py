"""
analysis/egs_tier_flow.py
Parse EGS log files and plot per-sample tier distribution over epochs.

Reads two types of log lines:
  Epoch [  1/100] ... EGS T1:45000 T2:0 T3:0 ...
  EGS tiers — Tier1: 40,500  Tier2: 4,500  Tier3: 0  (0.0% in T3)

Usage:
    python analysis/egs_tier_flow.py --log results/logs/<egs_run>.log
    python analysis/egs_tier_flow.py --log log1.log log2.log log3.log  # multi-seed overlay
"""

import re
import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

FIGURES_DIR = str(Path(__file__).resolve().parent.parent / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

TIER_COLORS = {
    "T1": "#A8D5A2",  # light green — easy
    "T2": "#F9C74F",  # amber — medium
    "T3": "#F3722C",  # orange-red — hard
}


# ── parser ────────────────────────────────────────────────────────────────────

_RE_EPOCH_LINE = re.compile(
    r"Epoch\s*\[\s*(\d+)/\d+\].*EGS\s+T1:(\d+)\s+T2:(\d+)\s+T3:(\d+)"
)
_RE_UPDATE_LINE = re.compile(
    r"EGS tiers\s*[—-]\s*Tier1:\s*([\d,]+)\s+Tier2:\s*([\d,]+)\s+Tier3:\s*([\d,]+)"
)
_RE_MIX_LINE = re.compile(r"mix:both")


def parse_log(path: str):
    """Return list of (epoch, t1, t2, t3) tuples and the mixing-start epoch."""
    records = {}  # epoch → (t1, t2, t3)
    mix_epoch = None
    current_epoch = None

    with open(path) as f:
        for line in f:
            # epoch summary line — most reliable epoch anchor
            m = _RE_EPOCH_LINE.search(line)
            if m:
                ep = int(m.group(1))
                t1, t2, t3 = int(m.group(2)), int(m.group(3)), int(m.group(4))
                records[ep] = (t1, t2, t3)
                current_epoch = ep
                if (
                    mix_epoch is None
                    and _RE_MIX_LINE.search(line)
                    and "pending" not in line
                ):
                    mix_epoch = ep
                continue

            # EGS tier update line (between epoch lines)
            m = _RE_UPDATE_LINE.search(line)
            if m and current_epoch is not None:
                t1 = int(m.group(1).replace(",", ""))
                t2 = int(m.group(2).replace(",", ""))
                t3 = int(m.group(3).replace(",", ""))
                # associate with next epoch (update happened after current_epoch)
                records[current_epoch + 1] = (t1, t2, t3)

    if not records:
        raise ValueError(f"No EGS tier data found in {path}")

    epochs = sorted(records)
    t1_arr = np.array([records[e][0] for e in epochs], dtype=float)
    t2_arr = np.array([records[e][1] for e in epochs], dtype=float)
    t3_arr = np.array([records[e][2] for e in epochs], dtype=float)
    total = t1_arr + t2_arr + t3_arr
    total = np.where(total == 0, 1, total)

    return (
        np.array(epochs),
        t1_arr / total * 100,
        t2_arr / total * 100,
        t3_arr / total * 100,
        mix_epoch,
    )


# ── figures ───────────────────────────────────────────────────────────────────


def fig_single(log_path: str, fname: str = None):
    """Stacked area chart for one EGS run."""
    epochs, t1, t2, t3, mix_ep = parse_log(log_path)
    seed = _extract_seed(log_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"EGS Sample Tier Distribution Over Training{f'  ·  Seed {seed}' if seed else ''}",
        fontsize=12,
        fontweight="bold",
    )

    ax.stackplot(
        epochs,
        t1,
        t2,
        t3,
        labels=["Tier 1 (Easy)", "Tier 2 (Medium)", "Tier 3 (Hard)"],
        colors=[TIER_COLORS["T1"], TIER_COLORS["T2"], TIER_COLORS["T3"]],
        alpha=0.85,
    )

    # 50% Tier-3 threshold line
    ax.axhline(
        50,
        color="#333333",
        lw=1.0,
        ls=":",
        alpha=0.6,
        label="50% threshold (mix:both activates)",
    )

    if mix_ep:
        ax.axvline(mix_ep, color="#D62728", lw=1.5, ls="--", alpha=0.7)
        ax.text(
            mix_ep + 0.8,
            52,
            f"CutMix+MixUp\nactivates (ep {mix_ep})",
            fontsize=8,
            color="#D62728",
            fontweight="bold",
            va="bottom",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("% of Training Samples")
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", fontsize=9)

    _annotate_final(ax, epochs, t1, t2, t3)

    out = fname or f"egs_tier_flow{'_s' + str(seed) if seed else ''}.png"
    path = os.path.join(FIGURES_DIR, out)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"  → saved: {out}")


def fig_multi_seed(log_paths: list, fname: str = "egs_tier_flow_multiseed.png"):
    """T3 fraction curves for multiple seeds — shows consistency."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "EGS Tier 3 Adoption Rate Across Seeds",
        fontsize=12,
        fontweight="bold",
    )

    log_paths = _deduplicate_logs(log_paths)
    log_paths = sorted(log_paths, key=lambda p: _extract_seed(p) or "")

    palette = ["#009E73", "#E69F00", "#CC79A7", "#0072B2", "#D55E00"]
    all_mix_eps = []

    for i, path in enumerate(log_paths):
        try:
            epochs, t1, t2, t3, mix_ep = parse_log(path)
        except ValueError as e:
            print(f"  WARN: {e}")
            continue
        seed = _extract_seed(path)
        color = palette[i % len(palette)]
        label = f"Seed {seed}" if seed else f"Run {i + 1}"
        ax.plot(epochs, t3, color=color, lw=2.0, label=label)
        if mix_ep:
            all_mix_eps.append(mix_ep)
            ax.axvline(mix_ep, color=color, lw=1.0, ls="--", alpha=0.4)

    ax.axhline(
        50, color="#333333", lw=1.0, ls=":", alpha=0.6, label="50% → mix:both activates"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("% of Samples in Tier 3")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)

    if all_mix_eps:
        ax.text(
            min(all_mix_eps) - 1,
            52,
            f"mix:both\n(ep {min(all_mix_eps)}–{max(all_mix_eps)})",
            fontsize=8,
            color="#D62728",
            ha="right",
            va="bottom",
        )

    path = os.path.join(FIGURES_DIR, fname)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# ── helpers ───────────────────────────────────────────────────────────────────


def _extract_seed(path: str):
    name = Path(path).name
    # try most-specific pattern first: cifar*_s42_ or tiny*_s42_
    for pattern in [
        r"(?:cifar|tiny)[^_]*_s(\d+)_",
        r"_s(\d+)_p\d+",
        r"_s(\d+)(?:_|\.)",
    ]:
        m = re.search(pattern, name)
        if m:
            return m.group(1)
    return None


def _deduplicate_logs(paths: list):
    """Keep only one log per seed — prefer longer filename (more specific run)."""
    by_seed = {}
    no_seed = []
    for p in paths:
        seed = _extract_seed(p)
        if seed is None:
            no_seed.append(p)
            continue
        if seed not in by_seed or len(p) > len(by_seed[seed]):
            by_seed[seed] = p
    if no_seed:
        print(f"  WARN: {len(no_seed)} log(s) skipped — seed not detected in filename:")
        for p in no_seed:
            print(f"    {Path(p).name}")
    return list(by_seed.values())


def _annotate_final(ax, epochs, t1, t2, t3):
    """Label final tier percentages on the right edge."""
    last = epochs[-1]
    cumulative = [t3[-1], t3[-1] + t2[-1]]  # top of T3, top of T2
    for pct, label, color in [
        (t3[-1] / 2, f"{t3[-1]:.0f}%", "white"),
        (t3[-1] + t2[-1] / 2, f"{t2[-1]:.0f}%", "#333333"),
        (t3[-1] + t2[-1] + t1[-1] / 2, f"{t1[-1]:.0f}%", "#333333"),
    ]:
        if pct > 3:
            ax.text(
                last,
                pct,
                label,
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", nargs="+", required=True, help="Path(s) to EGS log file(s)"
    )
    parser.add_argument("--out", default=None, help="Output filename override")
    args = parser.parse_args()

    if len(args.log) == 1:
        fig_single(args.log[0], fname=args.out)
    else:
        fig_multi_seed(args.log, fname=args.out or "egs_tier_flow_multiseed.png")


if __name__ == "__main__":
    main()
