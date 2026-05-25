"""
analysis/seed_band_plot.py
Multi-seed mean±std band plot — the primary thesis committee figure.

Produces three publication-quality figures:
  fig_seed_bands.png       — val accuracy mean ± 1σ bands + best-acc bar chart
  fig_tier_zoom.png        — zoomed view of tier-transition dip/recovery (ep 10-60)
  fig_seed_distribution.png — box plot of best acc per seed (consistency check)

Usage:
    python analysis/seed_band_plot.py
    python analysis/seed_band_plot.py --checkpoint_dir /path/to/checkpoints

Pull histories from cluster first (one-time):
    rsync -av user@cluster:/scratch/checkpoints/*_p19*_history.pt checkpoints/
"""

import os
import sys
import argparse
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

CHECKPOINT_DIR = str(_ROOT / "checkpoints")
FIGURES_DIR = str(_ROOT / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── experiment config ─────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 3407, 1024]

# ETS fixed boundaries: t1=0.20, t2=0.45 at 100 epochs
TIER_EPOCHS = [20, 45]

# One entry per method.  `globs` are tried in order; first match wins.
# {seed} is replaced with the actual seed integer.
METHODS = {
    "Static Mixing": {
        "globs": [
            "wideresnet_static_mixing_mix_both_sgd_cosine*ep100*cifar100*s{seed}*p19*history.pt",
            "wideresnet_static_mixing*ep100*cifar100*s{seed}*p19*history.pt",
            "wideresnet_static_mixing*ep100*cifar100*s{seed}*history.pt",
        ],
        "color": "#0072B2",  # blue
        "ls": "-",
        "lw": 1.8,
        "zorder": 2,
    },
    "ETS (ours)": {
        "globs": [
            "wideresnet_tiered_ets_mix_both_sgd_cosine*ep100*cifar100*s{seed}*p19*history.pt",
            "wideresnet_tiered_ets*ep100*cifar100*s{seed}*p19*history.pt",
        ],
        "color": "#009E73",  # green
        "ls": "--",
        "lw": 2.2,
        "zorder": 4,
    },
    "LPS (ours)": {
        "globs": [
            "wideresnet_tiered_lps_mix_both_sgd_cosine*ep100*cifar100*s{seed}*p19*history.pt",
            "wideresnet_tiered_lps*ep100*cifar100*s{seed}*p19*history.pt",
        ],
        "color": "#E69F00",  # orange
        "ls": "--",
        "lw": 2.2,
        "zorder": 3,
    },
    "EGS v2 (ours)": {
        "globs": [
            "egs_v2*ep100*cifar100*s{seed}*p19*history.pt",
            "wideresnet_tiered_egs*ep100*cifar100*s{seed}*p19*history.pt",
            "*egs*ep100*cifar100*s{seed}*p19*history.pt",
        ],
        "color": "#CC79A7",  # pink
        "ls": ":",
        "lw": 2.0,
        "zorder": 3,
    },
}

# Pre-computed from 5-seed Wilcoxon + Cohen's d (thesis_results_tables.md)
STATS = {
    "ETS (ours)": {"p": "<0.001", "d": "9.83", "delta": "+3.79 pp"},
    "LPS (ours)": {"p": "<0.001", "d": "10.12", "delta": "+3.84 pp"},
    "EGS v2 (ours)": {"p": "<0.001", "d": "4.57", "delta": "+2.15 pp"},
}

# ── matplotlib style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update(
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
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.titlepad": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.12,
    }
)


# ── data loading ──────────────────────────────────────────────────────────────


def _find_history(cfg, seed, checkpoint_dir):
    for pattern in cfg["globs"]:
        matches = glob(
            os.path.join(checkpoint_dir, pattern.replace("{seed}", str(seed)))
        )
        if matches:
            return matches[0]
    return None


def _load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"    WARNING: {os.path.basename(path)}: {e}")
        return None


def gather_seeds(method_name, cfg, seeds, checkpoint_dir):
    histories, found = [], []
    for seed in seeds:
        path = _find_history(cfg, seed, checkpoint_dir)
        if path is None:
            print(f"    [{method_name}] seed {seed}: not found  (pull from cluster)")
            continue
        h = _load_pt(path)
        if h is None:
            continue
        histories.append(h)
        found.append(seed)
        n = len(h["val_acc"])
        bv = max(h["val_acc"]) * 100
        print(
            f"    [{method_name}] seed {seed}: {n} ep, best={bv:.2f}%  ← {os.path.basename(path)}"
        )
    if not histories:
        print(f"    [{method_name}] — no seeds found, method skipped")
    else:
        print(
            f"    [{method_name}] {len(histories)}/{len(seeds)} seeds loaded  {found}"
        )
    return histories, found


def compute_bands(histories, metric="val_acc", smooth_w=5):
    """Return (epochs_1d, mean_1d, std_1d, list_of_smoothed_curves)."""
    arrays = []
    for h in histories:
        v = np.array(h[metric], dtype=float)
        if v.max() <= 1.0:
            v *= 100
        arrays.append(v)
    n = min(len(a) for a in arrays)
    arrays = [uniform_filter1d(a[:n], size=smooth_w) for a in arrays]
    stacked = np.stack(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if len(arrays) > 1 else np.zeros(n)
    return np.arange(1, n + 1), mean, std, arrays


# ── drawing helpers ───────────────────────────────────────────────────────────

TIER_BG = ["#EBF5FB", "#E9F7EF", "#FEF9E7"]  # blue / green / yellow
TIER_LABELS = ["Tier 1\n(flip, crop)", "Tier 2\n(+jitter, rot.)", "Tier 3\n(+cutout)"]
TIER_VCOLORS = ["#27AE60", "#E67E22"]


def draw_tiers(ax, ylo, yhi, tier_epochs=TIER_EPOCHS):
    boundaries = [1] + tier_epochs + [100]
    for j in range(3):
        ax.axvspan(
            boundaries[j], boundaries[j + 1], alpha=0.18, color=TIER_BG[j], zorder=0
        )
        mid = (boundaries[j] + boundaries[j + 1]) / 2
        ax.text(
            mid,
            ylo + (yhi - ylo) * 0.015,
            TIER_LABELS[j],
            ha="center",
            fontsize=6,
            color="#999999",
            va="bottom",
            zorder=1,
        )
    for ep_t, tc in zip(tier_epochs, TIER_VCOLORS):
        ax.axvline(ep_t, color=tc, lw=1.0, ls=":", alpha=0.80, zorder=5)


def save_fig(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path)
    print(f"  → saved: {fname}")


# ── Figure 1: main band plot ──────────────────────────────────────────────────


def fig_seed_bands(md, fname="fig_seed_bands.png"):
    """
    Left  — val accuracy curves: mean line + ±1σ shaded band + ghost seed lines.
    Right — bar chart of best val accuracy with ±1σ error bars + stat annotations.
    """
    if not md:
        print("  fig_seed_bands: no data — skipping.")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        "Validation Accuracy — Mean ± 1σ  ·  5 Seeds  ·  WideResNet-28-10  ·  CIFAR-100  ·  19-op Pool  ·  100 Epochs",
        fontsize=9,
        fontweight="bold",
    )

    # ── left: learning curves ─────────────────────────────────────────────────
    all_lo, all_hi = [], []
    for m in md.values():
        all_lo.append(float((m["mean"] - m["std"]).min()))
        all_hi.append(float((m["mean"] + m["std"]).max()))
    ylo = max(0, min(all_lo) - 2.0)
    yhi = min(100, max(all_hi) + 3.5)

    draw_tiers(ax_left, ylo, yhi)

    # tier-boundary labels above plot
    for ep_t, tc, label in zip(
        TIER_EPOCHS, TIER_VCOLORS, ["↑ Tier 2 unlocked", "↑ Tier 3 unlocked"]
    ):
        ax_left.text(
            ep_t + 0.6,
            yhi - 0.4,
            label,
            fontsize=6.5,
            color=tc,
            va="top",
            ha="left",
            zorder=6,
        )

    for method_name, m in md.items():
        cfg = m["cfg"]
        x, mean, std = m["epochs"], m["mean"], m["std"]

        # ghost individual seed lines
        for curve in m["curves"]:
            n = min(len(x), len(curve))
            ax_left.plot(
                x[:n],
                curve[:n],
                color=cfg["color"],
                lw=0.5,
                alpha=0.15,
                zorder=cfg["zorder"],
            )

        # ±1σ band
        ax_left.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.17,
            color=cfg["color"],
            zorder=cfg["zorder"],
        )

        # mean line
        n_s = m["n_seeds"]
        ax_left.plot(
            x,
            mean,
            color=cfg["color"],
            lw=cfg["lw"],
            ls=cfg["ls"],
            label=f"{method_name}  (n={n_s})",
            zorder=cfg["zorder"] + 1,
        )

        # best-epoch dot
        bi = int(np.argmax(mean))
        ax_left.scatter(
            [x[bi]],
            [mean[bi]],
            color=cfg["color"],
            s=40,
            zorder=10,
            edgecolors="white",
            linewidths=0.8,
        )

    ax_left.set_xlim(0, 102)
    ax_left.set_ylim(ylo, yhi)
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Validation Accuracy (%)")
    ax_left.set_title("(a)  Validation Accuracy — Mean ± 1σ  (shaded = ±1 std)")
    ax_left.legend(loc="lower right", fontsize=7.5)
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # ── right: bar chart ──────────────────────────────────────────────────────
    names = list(md.keys())
    colors = [md[n]["cfg"]["color"] for n in names]

    best_means, best_stds = [], []
    for n in names:
        m = md[n]
        bi = int(np.argmax(m["mean"]))
        best_means.append(float(m["mean"][bi]))
        best_stds.append(float(m["std"][bi]))

    x_pos = np.arange(len(names))
    bars = ax_right.bar(
        x_pos,
        best_means,
        yerr=best_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.55,
        capsize=5,
        error_kw={"linewidth": 1.3, "ecolor": "black"},
        zorder=3,
    )

    for bar, bm, bs, name in zip(bars, best_means, best_stds, names):
        stat_line = ""
        if name in STATS:
            s = STATS[name]
            stat_line = f"\np{s['p']}  d={s['d']}  {s['delta']}"
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bm + bs + 0.25,
            f"{bm:.2f}% ±{bs:.2f}{stat_line}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    short = [n.replace(" (ours)", "").replace(" v2", "") for n in names]
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(short, rotation=12, ha="right")
    ax_right.set_ylabel("Best Validation Accuracy (%)")
    ax_right.set_title("(b)  Best Val Accuracy ± 1σ  (vs Static: p<0.001)")
    y_bot = max(0, min(best_means) - 5)
    y_top = min(100, max(best_means) + max(best_stds) + 4)
    ax_right.set_ylim(y_bot, y_top)

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── Figure 2: tier-transition zoom ────────────────────────────────────────────


def fig_tier_zoom(md, zoom=(10, 62), fname="fig_tier_zoom.png"):
    """
    Zoomed view of epochs 10-62: shows the dip immediately after each tier
    unlock in curriculum methods — direct mechanistic evidence.
    """
    if not md:
        print("  fig_tier_zoom: no data — skipping.")
        return

    z0, z1 = zoom
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.suptitle(
        "Tier-Transition Detail  ·  Epochs 10 – 62  ·  CIFAR-100  ·  WideResNet-28-10\n"
        "Each tier unlock causes a brief accuracy dip as the model adapts — "
        "then recovers above the static baseline",
        fontsize=9,
        fontweight="bold",
    )

    all_lo, all_hi = [], []
    for m in md.values():
        mask = (m["epochs"] >= z0) & (m["epochs"] <= z1)
        lo_v = float((m["mean"] - m["std"])[mask].min())
        hi_v = float((m["mean"] + m["std"])[mask].max())
        all_lo.append(lo_v)
        all_hi.append(hi_v)

    ylo = max(0, min(all_lo) - 1.5)
    yhi = min(100, max(all_hi) + 2.5)

    draw_tiers(ax, ylo, yhi)

    annotated_dip = False
    for method_name, m in md.items():
        cfg = m["cfg"]
        x, mean, std = m["epochs"], m["mean"], m["std"]
        mask = (x >= z0) & (x <= z1)
        xz, mz, sz = x[mask], mean[mask], std[mask]
        if len(xz) == 0:
            continue

        ax.fill_between(xz, mz - sz, mz + sz, alpha=0.35, color=cfg["color"])
        ax.plot(
            xz, mz, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"], label=method_name
        )

        # annotate the tier-2 dip on the first CL method found
        if not annotated_dip and "(ours)" in method_name:
            # find the local minimum in a 10-epoch window after tier-2 transition
            t2 = TIER_EPOCHS[0]
            win_mask = (xz >= t2) & (xz <= t2 + 12)
            if win_mask.any():
                dip_rel = int(np.argmin(mz[win_mask]))
                dip_x = float(xz[win_mask][dip_rel])
                dip_y = float(mz[win_mask][dip_rel])
                ax.annotate(
                    "Accuracy dip:\nmodel adapts\nto harder ops",
                    xy=(dip_x, dip_y),
                    xytext=(dip_x + 6, dip_y - (yhi - ylo) * 0.12),
                    fontsize=7.5,
                    color=cfg["color"],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=cfg["color"], lw=1.0),
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",
                        ec=cfg["color"],
                        alpha=0.92,
                    ),
                    zorder=10,
                )
                annotated_dip = True

    # tier-boundary labels
    for ep_t, tc, label in zip(
        TIER_EPOCHS,
        TIER_VCOLORS,
        [
            "Tier 2 unlocked\n(+jitter, rotation)",
            "Tier 3 unlocked\n(+cutout, grayscale)",
        ],
    ):
        if z0 <= ep_t <= z1:
            ax.text(
                ep_t + 0.5,
                yhi - 0.3,
                label,
                fontsize=7,
                color=tc,
                va="top",
                ha="left",
                fontweight="bold",
            )

    ax.set_xlim(z0, z1)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── Figure 3: seed-distribution box plot ─────────────────────────────────────


def fig_seed_distribution(md, fname="fig_seed_distribution.png"):
    """
    Bar + seed-dot overlay per method.
    Bar = mean (low alpha fill).  Bold hline = mean.  Dots = individual seeds.
    Thin error bar (no caps) = ±1σ.
    """
    multi = {k: v for k, v in md.items() if v["n_seeds"] > 1}
    if not multi:
        print("  fig_seed_distribution: need ≥2 seeds per method — skipping.")
        return

    rng = np.random.default_rng(0)
    names = list(multi.keys())
    colors = [multi[n]["cfg"]["color"] for n in names]
    data = [[float(np.max(c)) for c in multi[n]["curves"]] for n in names]
    means = [float(np.mean(d)) for d in data]
    stds = [float(np.std(d, ddof=1)) if len(d) > 1 else 0.0 for d in data]

    all_vals = [v for d in data for v in d]
    ylo = min(all_vals) - 0.8
    yhi = max(all_vals) + 3.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        "Seed Consistency — Best Val Accuracy  ·  5 Seeds  ·  CIFAR-100  ·  WideResNet-28-10",
        fontsize=10,
        fontweight="bold",
    )

    x_pos = np.arange(len(names))
    bar_w = 0.55
    static_mu = means[0]

    for xi, (d, mu, std, color) in enumerate(zip(data, means, stds, colors)):
        # filled bar from ylo to mean (low alpha for background effect)
        ax.bar(
            xi,
            mu - ylo,
            bottom=ylo,
            width=bar_w,
            color=color,
            alpha=0.20,
            edgecolor=color,
            linewidth=1.5,
            zorder=1,
        )

        # bold mean line spanning the bar width
        ax.hlines(
            mu,
            xi - bar_w / 2 + 0.02,
            xi + bar_w / 2 - 0.02,
            color=color,
            linewidth=3.0,
            zorder=4,
        )

        # thin ±1σ line, no caps
        ax.vlines(xi, mu - std, mu + std, color="black", linewidth=1.8, zorder=3)

        # individual seed dots, jittered
        jitter = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(
            x_pos[xi] + jitter,
            d,
            color=color,
            s=90,
            zorder=5,
            edgecolors="white",
            linewidths=1.2,
        )

        # label above the ±1σ top
        delta_str = f"\nvs Static +{mu - static_mu:.2f} pp" if xi > 0 else ""
        ax.text(
            xi,
            mu + std + 0.45,
            f"{mu:.2f}% ± {std:.2f}{delta_str}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    short = [n.replace(" (ours)", "").replace(" v2", "") for n in names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=10)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.text(
        0.99,
        0.02,
        "● individual seed   — mean   | ±1σ",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#777777",
        style="italic",
    )

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-seed band plot")
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Gaussian smoothing window in epochs (default 5)",
    )
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not found — activate your venv first.")
        sys.exit(1)

    print(f"\n  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  seeds          : {args.seeds}")
    print(f"  smooth window  : {args.smooth}")
    print(f"  figures → {FIGURES_DIR}\n")

    md = {}
    for method_name, cfg in METHODS.items():
        print(f"  ── {method_name}")
        histories, found = gather_seeds(
            method_name, cfg, args.seeds, args.checkpoint_dir
        )
        if not histories:
            print()
            continue

        epochs, mean, std, curves = compute_bands(histories, "val_acc", args.smooth)

        md[method_name] = {
            "epochs": epochs,
            "mean": mean,
            "std": std,
            "curves": curves,
            "n_seeds": len(histories),
            "cfg": cfg,
        }
        print()

    if not md:
        print("  No history files found locally.")
        print("  Pull them from the cluster with:")
        print(
            "    rsync -av user@cluster:/path/checkpoints/*_p19*_history.pt checkpoints/"
        )
        sys.exit(0)

    print(f"  Methods loaded: {list(md.keys())}\n")
    print("  Generating figures...\n")

    fig_seed_bands(md)
    fig_tier_zoom(md)
    fig_seed_distribution(md)

    print(f"\n  Done. Figures saved to: {os.path.abspath(FIGURES_DIR)}\n")


if __name__ == "__main__":
    main()
