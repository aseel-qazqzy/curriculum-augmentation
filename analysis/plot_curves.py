"""
analysis/plot_curves.py  —  Publication-quality training analysis
Thesis: Curriculum-Style Data Augmentation — CIFAR-100

Usage:
    python analysis/plot_curves.py --mode baselines
    python analysis/plot_curves.py --mode all
    python analysis/plot_curves.py --mode ablation
"""

import os, sys, argparse, json, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.ndimage import uniform_filter1d
from pathlib import Path

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

warnings.filterwarnings("ignore")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

CHECKPOINT_DIR = str(_PROJECT_ROOT / "checkpoints")
FIGURES_DIR = str(_PROJECT_ROOT / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# PALETTE — colorblind-friendly (Wong 2011, Nature Methods)
PALETTE = {
    "no_aug": "#999999",  # neutral gray
    "static": "#0072B2",  # blue
    "cl": "#009E73",  # green
    "cosine": "#E69F00",  # orange
    "adam": "#CC79A7",  # pink
}

# Research paper style — clean, minimal, publication-ready
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
        "grid.linewidth": 0.6,
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
        "lines.antialiased": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.12,
    }
)

# Tier transition epochs: ETS t1=0.20, t2=0.45 over 100 epochs
TIER_EPOCHS = [20, 45]  # end of Tier 1, end of Tier 2


# DATA LOADING
def _merge_histories(parts):
    """Concatenate a list of history dicts in order (chronological)."""
    keys = ["train_loss", "train_acc", "val_loss", "val_acc", "val_top5"]
    merged = {k: [] for k in keys}
    for p in parts:
        for k in keys:
            if k in p:
                merged[k].extend(p[k])
    return merged


def load_history(name, checkpoint_dir=CHECKPOINT_DIR):
    """
    Dynamically finds ALL history files whose filename starts with `{name}`
    (e.g. resnet50_static_aug_cifar100_v2_cifar100_history.pt and
          resnet50_static_aug_cifar100_v2_cifar100_resumed_history.pt),
    sorts them by file modification time (oldest first), and merges them
    into one continuous run automatically.

    Nothing needs to change in EXPERIMENTS when a run is resumed under a
    new name — just make sure the new name starts with the same base.
    """
    from glob import glob as _glob

    found = []
    for ext in [".pt", ".json"]:
        pattern = os.path.join(checkpoint_dir, f"{name}*_history{ext}")
        found.extend(_glob(pattern))

    # Sort by modification time so part-1 always comes before part-2
    found = sorted(set(found), key=os.path.getmtime)

    if not found:
        print(f"  {name}  — not found")
        return None

    parts = []
    for path in found:
        ext = os.path.splitext(path)[1]
        h = _load_pt(path) if ext == ".pt" else _load_json(path)
        if h is not None:
            parts.append((path, h))

    if not parts:
        return None

    if len(parts) == 1:
        path, h = parts[0]
        print(
            f"  {os.path.basename(path):<55}  {len(h['val_acc']):>3} epochs  "
            f"best_val={max(h['val_acc']) * 100:.2f}%"
        )
        return h

    # Multiple parts — merge and report
    ep_counts = " + ".join(str(len(h["val_acc"])) for _, h in parts)
    merged = _merge_histories([h for _, h in parts])
    print(
        f"  {name}  — auto-merged {len(parts)} parts  "
        f"({ep_counts} ep  ->  {len(merged['val_acc'])} total  "
        f"best_val={max(merged['val_acc']) * 100:.2f}%)"
    )
    for path, h in parts:
        print(f"    [{len(h['val_acc']):>3} ep]  {os.path.basename(path)}")
    return merged


def _load_pt(path):
    return torch.load(path, map_location="cpu") if HAVE_TORCH else None


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# UTILITY
def smooth(values, w=7):
    a = np.array(values, dtype=float)
    return a if len(a) < w else uniform_filter1d(a, size=w)


def ep(h):
    return np.arange(1, len(h["val_acc"]) + 1)


def best(h):
    arr = np.array(h["val_acc"])
    idx = int(np.argmax(arr))
    return idx + 1, float(arr[idx]) * 100


def milestones(n=100):
    # kept for legacy callers; new runs use CosineAnnealingLR (no milestones)
    return [int(n * f) for f in (0.33, 0.66, 0.83)]


def set_epoch_axis(ax, n):
    ax.set_xlim(0, n + 2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))


def set_epoch_labels(ax, ylabel=""):
    ax.set_xlabel("Epoch", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel)


def _ylim_padded(values_list, pad_frac=0.06, lo_floor=None, hi_ceil=None):
    flat = [v for vals in values_list for v in vals]
    lo, hi = min(flat), max(flat)
    span = max(hi - lo, 1e-6)
    lo_pad = lo - span * pad_frac
    hi_pad = hi + span * pad_frac
    if lo_floor is not None:
        lo_pad = max(lo_pad, lo_floor)
    if hi_ceil is not None:
        hi_pad = min(hi_pad, hi_ceil)
    return lo_pad, hi_pad


def save(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path)
    print(f"  -> saved: {fname}")
    return path


def lr_vlines(ax, n, color="#888888", alpha=0.35):
    """No-op for CosineAnnealingLR runs (no discrete milestones).
    Kept so old call sites don't break."""
    pass


def tier_vlines(ax, color="#009E73", alpha=0.60):
    """Draw vertical dashed lines at Tier 2 and Tier 3 transition epochs."""
    labels = ["Tier 2", "Tier 3"]
    for i, ep_t in enumerate(TIER_EPOCHS):
        ax.axvline(
            ep_t,
            color=color,
            lw=1.1,
            ls="--",
            alpha=alpha,
            label=f"↑ {labels[i]}" if i == 0 else "_",
        )
        ax.text(
            ep_t + 0.8,
            ax.get_ylim()[1] * 0.97,
            labels[i],
            fontsize=7,
            color=color,
            va="top",
            alpha=0.85,
        )


# FIGURE 1  —  Validation Comparison  (overlaid, all three methods)
def fig1_val_comparison(runs, title="", fname="fig1_val_comparison.png"):
    """Two panels: Val Loss | Val Accuracy, all runs overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.suptitle(
        f"Validation Loss & Accuracy — CIFAR-100  ·  ResNet-50  ·  SGD\n{title}",
        fontsize=9,
        fontweight="bold",
    )

    all_val_loss, all_val_acc = [], []
    max_n = 1

    for label, (h, _) in runs.items():
        if h is None:
            continue
        max_n = max(max_n, len(h["val_acc"]))
        all_val_loss.extend(smooth(h["val_loss"]))
        all_val_acc.extend(smooth(h["val_acc"]) * 100)

    for label, (h, color) in runs.items():
        if h is None:
            continue
        x = ep(h)
        ls = "--" if "CL" in label or "Tier" in label else "-"
        lw = 2.2 if "CL" in label or "Tier" in label else 1.8
        axes[0].plot(x, smooth(h["val_loss"]), color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(
            x, smooth(h["val_acc"]) * 100, color=color, lw=lw, ls=ls, label=label
        )
        ep_b, val_b = best(h)
        axes[1].scatter(
            [ep_b],
            [val_b],
            color=color,
            s=45,
            zorder=6,
            edgecolors="white",
            linewidths=0.8,
        )

    for ax in axes:
        set_epoch_axis(ax, max_n)
        set_epoch_labels(ax)
        lr_vlines(ax, max_n)

    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("(a)  Validation Loss")
    if all_val_loss:
        lo, hi = _ylim_padded([all_val_loss], pad_frac=0.08, lo_floor=0.0)
        axes[0].set_ylim(lo, hi)

    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("(b)  Validation Accuracy")
    if all_val_acc:
        lo, hi = _ylim_padded([all_val_acc], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        axes[1].set_ylim(lo, hi)

    handles, labs = axes[1].get_legend_handles_labels()
    lr_p = mpatches.Patch(color="#888888", alpha=0.5, label="MultiStepLR decay")
    fig.legend(
        handles + [lr_p],
        labs + ["MultiStepLR decay"],
        loc="lower center",
        ncol=min(len(runs) + 1, 4),
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save(fig, fname)
    plt.show()


# FIGURE 2  —  Overfitting Analysis  (Train vs Val Loss, one subplot per method)
def fig2_overfitting(runs, fname="fig2_overfitting.png"):
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    n = len(valid)
    if n == 0:
        print("  fig2: no valid runs, skipping.")
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Overfitting Analysis — Train vs Validation Loss  ·  CIFAR-100  ·  ResNet-50",
        fontsize=9,
        fontweight="bold",
    )

    for i, (label, h, color) in enumerate(valid):
        x = ep(h)
        tl = smooth(h["train_loss"], w=3)
        vl = smooth(h["val_loss"], w=7)

        axes[i].plot(x, tl, color=color, lw=1.8, ls="--", alpha=0.80, label="Train")
        axes[i].plot(x, vl, color=color, lw=2.0, label="Val")
        axes[i].fill_between(x, tl, vl, alpha=0.12, color=color)
        set_epoch_axis(axes[i], len(x))
        lr_vlines(axes[i], len(x))

        all_loss = np.concatenate([tl, vl])
        lo, hi = _ylim_padded([all_loss], pad_frac=0.12, lo_floor=0.0)
        axes[i].set_ylim(lo, hi)

        gap = float(vl[-1] - tl[-1])
        mid_y = (float(tl[-1]) + float(vl[-1])) / 2
        axes[i].annotate(
            f"gap = {abs(gap):.3f}",
            xy=(len(x), mid_y),
            xycoords="data",
            xytext=(0.40, 0.80),
            textcoords="axes fraction",
            fontsize=8,
            color=color,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=0.9, connectionstyle="arc3,rad=-0.15"
            ),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.88),
        )

        axes[i].set_title(f"({chr(97 + i)})  {label}")
        set_epoch_labels(axes[i], "Loss" if i == 0 else "")
        axes[i].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 3  —  Generalization Gap  (Train vs Val Accuracy, one subplot per method)
def fig3_generalization(runs, fname="fig3_generalization.png"):
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    n = len(valid)
    if n == 0:
        print("  fig3: no valid runs, skipping.")
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Generalization Gap — Train vs Validation Accuracy  ·  CIFAR-100  ·  ResNet-50",
        fontsize=9,
        fontweight="bold",
    )

    for i, (label, h, color) in enumerate(valid):
        x = ep(h)
        ta = smooth(h["train_acc"], w=3) * 100
        va = smooth(h["val_acc"], w=7) * 100

        axes[i].plot(x, ta, color=color, lw=1.8, ls="--", alpha=0.80, label="Train")
        axes[i].plot(x, va, color=color, lw=2.0, label="Val")
        axes[i].fill_between(
            x,
            va,
            ta,
            alpha=0.12,
            color=color,
            label=f"Gap = {float(ta[-1] - va[-1]):.1f}%",
        )

        ep_b, val_b = best(h)
        axes[i].scatter(
            [ep_b],
            [val_b],
            color=color,
            s=55,
            zorder=6,
            edgecolors="white",
            lw=0.8,
            label=f"Best {val_b:.2f}%",
        )

        all_acc = np.concatenate([ta, va])
        lo, hi = _ylim_padded([all_acc], pad_frac=0.12, lo_floor=0.0, hi_ceil=100.0)
        axes[i].set_ylim(lo, hi)

        axes[i].annotate(
            f"{val_b:.2f}%",
            xy=(ep_b, val_b),
            xycoords="data",
            xytext=(0.55, 0.20),
            textcoords="axes fraction",
            fontsize=8,
            color=color,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=0.8, connectionstyle="arc3,rad=0.2"
            ),
        )

        lr_vlines(axes[i], len(x))
        axes[i].set_title(f"({chr(97 + i)})  {label}")
        set_epoch_labels(axes[i], "Accuracy (%)" if i == 0 else "")
        axes[i].legend(loc="lower right", fontsize=8)
        set_epoch_axis(axes[i], len(x))

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 4  —  LR Schedule  (CosineAnnealingLR with warmup + tier markers)
def fig4_lr_schedule(
    n_epochs=100, warmup=5, lr_init=0.1, eta_min=1e-6, fname="fig4_lr_schedule.png"
):
    """CosineAnnealingLR + linear warmup for 100 epochs, with tier transition markers."""
    ep_x = np.arange(1, n_epochs + 1)

    lr = np.zeros(n_epochs)
    for i, e in enumerate(ep_x):
        if e <= warmup:
            lr[i] = lr_init * (0.1 + 0.9 * e / warmup)
        else:
            t = (e - warmup) / max(1, n_epochs - warmup)
            lr[i] = eta_min + 0.5 * (lr_init - eta_min) * (1 + np.cos(np.pi * t))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.suptitle(
        "Learning Rate Schedule — CosineAnnealingLR + 5-ep Warmup  ·  CIFAR-100  ·  100 epochs",
        fontsize=9,
        fontweight="bold",
    )

    ax.plot(
        ep_x,
        lr,
        color=PALETTE["static"],
        lw=2.2,
        label=f"CosineAnnealingLR  (warmup={warmup}, η_min={eta_min})",
    )

    tier_labels = ["Tier 2 unlocked", "Tier 3 unlocked"]
    tier_colors = [PALETTE["cl"], PALETTE["cosine"]]
    for j, ep_t in enumerate(TIER_EPOCHS):
        ax.axvline(
            ep_t,
            color=tier_colors[j],
            lw=1.2,
            ls="--",
            alpha=0.75,
            label=f"ep {ep_t}: {tier_labels[j]}",
        )
        ax.text(
            ep_t + 0.5,
            lr_init * 0.6,
            f"ep {ep_t}\n{tier_labels[j]}",
            fontsize=7,
            color=tier_colors[j],
            va="bottom",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 0.5)
    ax.set_xlim(0, n_epochs + 2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 5  —  Summary Dashboard (4-panel)
def fig5_summary(runs, fname="fig5_summary.png"):
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    if not valid:
        print("  fig5: no valid runs, skipping.")
        return

    labels = [k for k, _, _ in valid]
    colors = [c for _, _, c in valid]
    best_v = [best(h)[1] for _, h, _ in valid]
    gaps = [(h["train_acc"][-1] - max(h["val_acc"])) * 100 for _, h, _ in valid]
    max_epochs = max(len(h["val_acc"]) for _, h, _ in valid)
    short = [l.split("(")[0].strip() for l in labels]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        "Experiment Summary Dashboard — CIFAR-100  ·  WideResNet-28-10  ·  SGD  ·  Cosine",
        fontsize=10,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel A: Best Val Acc
    ax0 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(valid))
    bars = ax0.bar(
        x_pos,
        best_v,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
        zorder=3,
    )
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(short, rotation=15, ha="right", fontsize=8)
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title("(a)  Best Validation Accuracy")
    ax0.set_ylim(max(0, min(best_v) - 5), min(100, max(best_v) + 6))
    for bar, val in zip(bars, best_v):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    # Panel B: Generalization gap
    ax1 = fig.add_subplot(gs[0, 1])
    bars2 = ax1.bar(
        x_pos,
        gaps,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
        zorder=3,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Train − Val Gap (%)")
    ax1.set_title("(b)  Generalization Gap  (↓ lower is better)")
    ax1.set_ylim(0, max(gaps) + 8)
    for bar, val in zip(bars2, gaps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    # Panel C: Val accuracy curves overlaid
    ax2 = fig.add_subplot(gs[1, 0])
    all_va = []
    for label, h, color in valid:
        x = ep(h)
        ls = "--" if "CL" in label or "Tier" in label else "-"
        lw = 2.2 if "CL" in label or "Tier" in label else 1.8
        va = smooth(h["val_acc"]) * 100
        ax2.plot(x, va, color=color, lw=lw, ls=ls, label=label.split("(")[0].strip())
        all_va.extend(va)
    lr_vlines(ax2, max_epochs)
    set_epoch_axis(ax2, max_epochs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("(c)  Validation Accuracy Curves")
    ax2.legend(fontsize=7, loc="lower right")
    if all_va:
        lo, hi = _ylim_padded([all_va], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        ax2.set_ylim(lo, hi)

    # Panel D: Gap over epochs (train − val accuracy)
    ax3 = fig.add_subplot(gs[1, 1])
    all_gaps_data = []
    for label, h, color in valid:
        x = ep(h)
        ta = np.array(h["train_acc"]) * 100
        va = np.array(h["val_acc"]) * 100
        g = smooth(ta - va, w=5)
        ls = "--" if "CL" in label or "Tier" in label else "-"
        lw = 2.2 if "CL" in label or "Tier" in label else 1.8
        ax3.plot(x, g, color=color, lw=lw, ls=ls, label=label.split("(")[0].strip())
        all_gaps_data.extend(g)
    lr_vlines(ax3, max_epochs)
    set_epoch_axis(ax3, max_epochs)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train − Val Accuracy (pp)")
    ax3.set_title("(d)  Overfitting Gap Over Epochs  (↓ lower is better)")
    ax3.legend(fontsize=7, loc="upper left")
    if all_gaps_data:
        # lo_floor=None allows negative values (train < val with heavy augmentation)
        lo, hi = _ylim_padded([all_gaps_data], pad_frac=0.10, lo_floor=None)
        ax3.set_ylim(lo, hi)

    save(fig, fname)
    plt.show()


# FIGURE 6  —  Tiered CL Learning Curves with Tier Markers
def fig6_cl_with_tiers(runs, fname="fig6_cl_tier_transitions.png"):
    """
    Tiered CL vs Static Aug: val accuracy curves with vertical tier-transition
    markers at epoch 20 (Tier 2) and epoch 45 (Tier 3) — ETS t1=0.20, t2=0.45.
    """
    cl_key = next((k for k in runs if "CL" in k or "Tier" in k), None)
    static_key = next((k for k in runs if "Static" in k), None)

    if cl_key is None:
        print("  fig6: Tiered CL not in runs, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(
        "Tiered CL: Val Accuracy with Tier Transition Markers\n"
        "CIFAR-100  ·  ResNet-50  ·  100 epochs  ·  SGD",
        fontsize=9,
        fontweight="bold",
    )

    all_acc = []
    for key, color, ls, lw in [
        (static_key, PALETTE["static"], "-", 1.8),
        (cl_key, PALETTE["cl"], "--", 2.2),
    ]:
        if key is None:
            continue
        h, _ = runs[key]
        if h is None:
            continue
        x = ep(h)
        va = smooth(h["val_acc"]) * 100
        ax.plot(x, va, color=color, lw=lw, ls=ls, label=key)
        ep_b, val_b = best(h)
        ax.scatter(
            [ep_b], [val_b], color=color, s=55, zorder=6, edgecolors="white", lw=0.8
        )
        ax.annotate(
            f"  {val_b:.2f}%",
            xy=(ep_b, val_b),
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )
        all_acc.extend(va)

    # Tier transition vertical markers
    tier_colors = ["#009E73", "#E69F00"]
    tier_names = [
        "Tier 2 unlocked\n(+jitter, rotation)",
        "Tier 3 unlocked\n(+cutout, grayscale)",
    ]
    ylo, yhi = _ylim_padded([all_acc], pad_frac=0.12, lo_floor=0.0, hi_ceil=100.0)
    ax.set_ylim(ylo, yhi)

    for ep_t, tcolor, tname in zip(TIER_EPOCHS, tier_colors, tier_names):
        ax.axvline(ep_t, color=tcolor, lw=1.1, ls=":", alpha=0.70)
        ax.text(
            ep_t + 0.8,
            yhi - (yhi - ylo) * 0.04,
            tname,
            fontsize=7,
            color=tcolor,
            va="top",
        )

    # Shaded regions for each tier
    tier_boundaries = [1] + TIER_EPOCHS + [100]
    tier_bg_colors = ["#F0F8FF", "#F0FFF0", "#FFFFF0"]
    tier_labels_bg = [
        "Tier 1\n(flip, crop)",
        "Tier 2\n(+jitter, rot.)",
        "Tier 3\n(+cutout)",
    ]
    for j in range(len(tier_bg_colors)):
        ax.axvspan(
            tier_boundaries[j],
            tier_boundaries[j + 1],
            alpha=0.18,
            color=tier_bg_colors[j],
            zorder=0,
        )
        mid = (tier_boundaries[j] + tier_boundaries[j + 1]) / 2
        ax.text(
            mid,
            ylo + (yhi - ylo) * 0.04,
            tier_labels_bg[j],
            ha="center",
            fontsize=6.5,
            color="#555555",
            va="bottom",
        )

    set_epoch_axis(ax, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 7  —  All Methods: Gap Over Epochs
def fig7_gap_over_epochs(runs, fname="fig7_gap_over_epochs.png"):
    """
    All three methods overlaid: train−val accuracy gap vs epoch.
    Lower gap = better generalisation.
    """
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    if not valid:
        print("  fig7: no valid runs, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(
        "Generalisation Gap Over Training — All Methods\n"
        "CIFAR-100  ·  ResNet-50  ·  100 epochs  ·  SGD  (lower is better)",
        fontsize=9,
        fontweight="bold",
    )

    all_gaps = []
    for label, h, color in valid:
        x = ep(h)
        ta = np.array(h["train_acc"]) * 100
        va = np.array(h["val_acc"]) * 100
        g = smooth(ta - va, w=5)
        ls = "--" if "CL" in label or "Tier" in label else "-"
        lw = 2.2 if "CL" in label or "Tier" in label else 1.8
        ax.plot(x, g, color=color, lw=lw, ls=ls, label=label)
        all_gaps.extend(g)

    lr_vlines(ax, 100)
    set_epoch_axis(ax, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train − Val Accuracy (pp)")
    ax.legend(fontsize=8, loc="upper left")
    if all_gaps:
        lo, hi = _ylim_padded([all_gaps], pad_frac=0.10, lo_floor=0.0)
        ax.set_ylim(lo, hi)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 8  —  Top-5 Accuracy Comparison  (bar chart)
def fig8_top5(runs, fname="fig8_top5_comparison.png"):
    """
    Bar chart of best Top-5 validation accuracy per method.
    Skipped silently if no run has val_top5 data.
    """
    valid = [
        (k, h, c)
        for k, (h, c) in runs.items()
        if h is not None and "val_top5" in h and len(h["val_top5"]) > 0
    ]
    if not valid:
        print("  fig8: val_top5 not available in histories, skipping.")
        return

    labels = [k for k, _, _ in valid]
    colors = [c for _, _, c in valid]
    top5v = [max(h["val_top5"]) * 100 for _, h, _ in valid]
    top1v = [best(h)[1] for _, h, _ in valid]
    short = [l.split("(")[0].strip() for l in labels]
    x_pos = np.arange(len(valid))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.suptitle(
        "Top-1 vs Top-5 Validation Accuracy — CIFAR-100  ·  ResNet-50",
        fontsize=9,
        fontweight="bold",
    )

    bars1 = ax.bar(
        x_pos - width / 2,
        top1v,
        width,
        label="Top-1",
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        top5v,
        width,
        label="Top-5",
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.50,
        hatch="///",
        zorder=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, rotation=10, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(top1v) - 8), min(100, max(top5v) + 6))

    for bar, val in zip(bars1, top1v):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val in zip(bars2, top5v):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ANALYSIS PRINTOUT
def print_analysis(runs):
    valid = {k: h for k, (h, _) in runs.items() if h}
    rows = []
    for label, h in valid.items():
        ep_b, val_b = best(h)
        ft = h["train_acc"][-1] * 100
        gap = ft - val_b
        n = len(h["val_acc"])
        ep90 = next((i + 1 for i, v in enumerate(h["val_acc"]) if v * 100 >= 70), None)
        ep95 = next((i + 1 for i, v in enumerate(h["val_acc"]) if v * 100 >= 80), None)
        rows.append(
            dict(
                label=label,
                val_b=val_b,
                ep_b=ep_b,
                ft=ft,
                gap=gap,
                n=n,
                ep90=ep90,
                ep95=ep95,
            )
        )

    W = 82
    bar = "=" * W

    print(f"\n{bar}")
    print(
        "THOROUGH ANALYSIS — CIFAR-100  ·  WideResNet-28-10  ·  SGD  ·  CosineAnnealingLR"
    )
    print(bar)

    # TABLE 1: Core metrics
    print("\n  TABLE 1 — Core Metrics\n")
    h1 = f"  {'Method':<42} {'BestVal':>8} {'@Ep':>5} {'TrainAcc':>9} {'Gap':>7} {'Eps':>5}"
    print(h1)
    print("  " + "-" * (W - 2))
    for r in rows:
        tag = ""
        if "static" in r["label"].lower() and "cl" not in r["label"].lower():
            tag = "  <- baseline"
        if "CL" in r["label"] or "Tier" in r["label"]:
            tag = "  <- YOUR METHOD *"
        print(
            f"  {r['label']:<42} {r['val_b']:>7.2f}%  {r['ep_b']:>4}  "
            f"{r['ft']:>8.2f}%  {r['gap']:>6.1f}%  {r['n']:>4}{tag}"
        )
    print("  " + "-" * (W - 2))
    print("  Gap = Final Train Acc - Best Val Acc  (lower = better generalization)")

    # TABLE 2: Deltas vs static baseline
    bl = next(
        (
            r
            for r in rows
            if "static" in r["label"].lower()
            and "cl" not in r["label"].lower()
            and "Tier" not in r["label"]
        ),
        None,
    )
    if bl and len(rows) > 1:
        print(
            f"\n  TABLE 2 — Delta vs Static Augmentation Baseline  ({bl['val_b']:.2f}%)\n"
        )
        print(f"  {'Method':<42} {'Dv Val':>8} {'Dv Gap':>8}  Verdict")
        print("  " + "-" * (W - 2))
        for r in rows:
            if r["label"] == bl["label"]:
                continue
            dv, dg = r["val_b"] - bl["val_b"], r["gap"] - bl["gap"]
            sv, sg = ("+" if dv >= 0 else ""), ("+" if dg >= 0 else "")
            if dv > 1.0:
                verdict = "Strong improvement"
            elif dv > 0.3:
                verdict = "Marginal improvement"
            elif dv > -0.3:
                verdict = "On par"
            else:
                verdict = "Below baseline"
            print(f"  {r['label']:<42} {sv}{dv:>6.2f}%  {sg}{dg:>6.1f}%  {verdict}")

    # TABLE 3: Convergence
    print("\n  TABLE 3 — Convergence Speed\n")
    print(f"  {'Method':<42} {'->70% @':>8} {'->80% @':>8} {'Stable @':>10}")
    print("  " + "-" * (W - 2))
    for r in rows:
        s90 = f"ep {r['ep90']}" if r["ep90"] else "N/A"
        s95 = f"ep {r['ep95']}" if r["ep95"] else "N/A"
        h = valid[r["label"]]
        best_so, last_gain = 0, 1
        for i, v in enumerate(h["val_acc"]):
            if v * 100 - best_so > 0.5:
                last_gain, best_so = i + 1, v * 100
        print(f"  {r['label']:<42} {s90:>8} {s95:>8} {f'ep {last_gain}':>10}")

    # TABLE 4: Overfitting metrics
    print("\n  TABLE 4 — Overfitting Metrics\n")
    print(
        f"  {'Method':<42} {'ValLoss@Best':>13} {'TrainLoss@End':>14} {'LossRatio':>10}"
    )
    print("  " + "-" * (W - 2))
    for r in rows:
        h = valid[r["label"]]
        ep_i = r["ep_b"] - 1
        vl_b = h["val_loss"][ep_i]
        tl_e = h["train_loss"][-1]
        ratio = vl_b / (tl_e + 1e-8)
        print(f"  {r['label']:<42} {vl_b:>12.4f}  {tl_e:>13.4f}  {ratio:>9.1f}x")

    # Key Findings
    print(f"\n{bar}")
    print("  KEY FINDINGS\n")

    no_aug = next(
        (r for r in rows if "no" in r["label"].lower() and "aug" in r["label"].lower()),
        None,
    )
    cl_r = next((r for r in rows if "CL" in r["label"] or "Tier" in r["label"]), None)

    idx = 1
    if no_aug and bl:
        dv = bl["val_b"] - no_aug["val_b"]
        dg = no_aug["gap"] - bl["gap"]
        print(
            f"  {idx}. Augmentation effect\n"
            f"     Static augmentation adds +{dv:.2f}% val accuracy over no-augmentation\n"
            f"     ({no_aug['val_b']:.2f}% -> {bl['val_b']:.2f}%) and reduces the generalization\n"
            f"     gap by {dg:.1f}pp ({no_aug['gap']:.1f}% -> {bl['gap']:.1f}%).\n"
        )
        idx += 1

    if bl and cl_r:
        dv = cl_r["val_b"] - bl["val_b"]
        dg = bl["gap"] - cl_r["gap"]
        sv = "+" if dv >= 0 else ""
        print(
            f"  {idx}. Curriculum Learning effect\n"
            f"     Tiered CL {sv}{dv:.2f}% vs static augmentation\n"
            f"     ({bl['val_b']:.2f}% -> {cl_r['val_b']:.2f}%) with gap reduced by {dg:.1f}pp.\n"
        )
        idx += 1

    if no_aug:
        print(
            f"  {idx}. No-augmentation overfitting\n"
            f"     Train accuracy: {no_aug['ft']:.2f}%  |  Val: {no_aug['val_b']:.2f}%.\n"
            f"     The {no_aug['gap']:.1f}pp gap demonstrates severe memorization.\n"
        )
        idx += 1

    print(bar)
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print(bar + "\n")


# MODES
def build_runs(names_colors, checkpoint_dir):
    return {
        label: (load_history(name, checkpoint_dir), color)
        for label, name, color in names_colors
    }


def mode_baselines(ckpt):
    print("\n  Loading baseline histories...")
    runs = build_runs(
        [
            (
                "No Augmentation",
                "wideresnet_none_sgd_cosine_ep100_cifar100_s42",
                PALETTE["no_aug"],
            ),
            (
                "Static Aug",
                "wideresnet_static_sgd_cosine_ep100_cifar100_s42",
                PALETTE["static"],
            ),
            (
                "RandAugment (N=2, M=9)",
                "wideresnet_randaugment_sgd_cosine_ep100_cifar100_s42",
                PALETTE["cosine"],
            ),
        ],
        ckpt,
    )
    print("\n  Generating figures + analysis...")
    fig1_val_comparison(
        runs,
        title="Baseline Comparison — CIFAR-100 · WideResNet-28-10",
        fname="fig1_val_comparison_baselines.png",
    )
    fig2_overfitting(runs, fname="fig2_overfitting_baselines.png")
    fig3_generalization(runs, fname="fig3_generalization_baselines.png")
    fig4_lr_schedule()
    fig5_summary(runs, fname="fig5_summary_baselines.png")
    fig7_gap_over_epochs(runs, fname="fig7_gap_baselines.png")
    fig8_top5(runs, fname="fig8_top5_baselines.png")
    print_analysis(runs)


def mode_all(ckpt):
    print("\n  Loading all WideResNet-28-10 histories (seed=42)...")
    runs = build_runs(
        [
            (
                "No Augmentation",
                "wideresnet_none_sgd_cosine_ep100_cifar100_s42",
                PALETTE["no_aug"],
            ),
            (
                "Static Aug",
                "wideresnet_static_sgd_cosine_ep100_cifar100_s42",
                PALETTE["static"],
            ),
            (
                "RandAugment (N=2, M=9)",
                "wideresnet_randaugment_sgd_cosine_ep100_cifar100_s42",
                PALETTE["cosine"],
            ),
            (
                "ETS + mix (ours)",
                "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42",
                PALETTE["cl"],
            ),
            (
                "LPS + mix (ours)",
                "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42",
                PALETTE["adam"],
            ),
        ],
        ckpt,
    )
    print("\n  Generating figures + analysis...")
    fig1_val_comparison(
        runs,
        title="All Methods — CIFAR-100 · WideResNet-28-10",
        fname="fig1_val_comparison_all.png",
    )
    fig2_overfitting(runs, fname="fig2_overfitting_all.png")
    fig3_generalization(runs, fname="fig3_generalization_all.png")
    fig4_lr_schedule()
    fig5_summary(runs, fname="fig5_summary_all.png")
    fig6_cl_with_tiers(runs, fname="fig6_cl_tier_transitions.png")
    fig7_gap_over_epochs(runs, fname="fig7_gap_all.png")
    fig8_top5(runs, fname="fig8_top5_all.png")
    print_analysis(runs)


def mode_ablation(ckpt):
    print("\n  Loading ablation histories (seed=42)...")
    runs = build_runs(
        [
            (
                "Static Aug",
                "wideresnet_static_sgd_cosine_ep100_cifar100_s42",
                PALETTE["static"],
            ),
            (
                "Static + Mixing",
                "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42",
                PALETTE["cosine"],
            ),
            (
                "ETS + mix (ours)",
                "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42",
                PALETTE["cl"],
            ),
            (
                "ETS no-mix (ablation)",
                "wideresnet_tiered_ets_nomix_sgd_cosine_ep100_cifar100_s42",
                PALETTE["adam"],
            ),
        ],
        ckpt,
    )
    print("\n  Generating ablation figures + analysis...")
    fig1_val_comparison(
        runs,
        title="Ablation Study — CIFAR-100 · WideResNet-28-10",
        fname="fig1_val_comparison_ablation.png",
    )
    fig3_generalization(runs, fname="fig3_generalization_ablation.png")
    fig5_summary(runs, fname="fig5_summary_ablation.png")
    fig6_cl_with_tiers(runs, fname="fig6_cl_tier_transitions_ablation.png")
    fig7_gap_over_epochs(runs, fname="fig7_gap_ablation.png")
    print_analysis(runs)


# FIGURE 9  —  Single-run training curves with tier markers
def fig9_single_run(h, label, tier_epochs, title="", fname="fig9_single_run.png"):
    """
    4-panel plot for one run:
      (a) Train & Val Loss
      (b) Train & Val Accuracy
      (c) Val Accuracy with tier shading
      (d) Train−Val generalization gap
    Tier transition vertical lines at tier_epochs = [t1, t2].
    """
    x = ep(h)
    n = len(x)

    tl = smooth(h["train_loss"], w=5)
    vl = smooth(h["val_loss"], w=7)
    ta = smooth(h["train_acc"], w=5) * 100
    va = smooth(h["val_acc"], w=7) * 100
    gap = smooth(np.array(h["train_acc"]) * 100 - np.array(h["val_acc"]) * 100, w=5)

    ep_b, val_b = best(h)

    tier_colors = ["#009E73", "#E69F00"]
    tier_names = ["Tier 2", "Tier 3"]
    tier_bg = ["#F0F8FF", "#F0FFF0", "#FFFFF0"]
    tier_bounds = [1] + tier_epochs + [n]

    def _add_tier_markers(ax, ylo, yhi):
        for j, ep_t in enumerate(tier_epochs):
            ax.axvline(
                ep_t,
                color=tier_colors[j],
                lw=1.1,
                ls="--",
                alpha=0.65,
                label=f"↑ {tier_names[j]}",
            )
            ax.text(
                ep_t + 0.8,
                yhi - (yhi - ylo) * 0.03,
                tier_names[j],
                fontsize=7,
                color=tier_colors[j],
                va="top",
            )
        for j in range(len(tier_bg)):
            ax.axvspan(
                tier_bounds[j],
                tier_bounds[j + 1],
                alpha=0.10,
                color=tier_bg[j],
                zorder=0,
            )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Training Curves — {title}\n{label}",
        fontsize=10,
        fontweight="bold",
    )

    # (a) Loss curves
    ax = axes[0, 0]
    ax.plot(
        x, tl, color=PALETTE["static"], lw=1.8, ls="--", alpha=0.80, label="Train Loss"
    )
    ax.plot(x, vl, color=PALETTE["static"], lw=2.0, label="Val Loss")
    ax.fill_between(x, tl, vl, alpha=0.10, color=PALETTE["static"])
    ylo, yhi = _ylim_padded([tl, vl], pad_frac=0.10, lo_floor=0.0)
    ax.set_ylim(ylo, yhi)
    _add_tier_markers(ax, ylo, yhi)
    set_epoch_axis(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a)  Train & Validation Loss")
    ax.legend(fontsize=7, loc="upper right")

    # (b) Accuracy curves
    ax = axes[0, 1]
    ax.plot(x, ta, color=PALETTE["cl"], lw=1.8, ls="--", alpha=0.80, label="Train Acc")
    ax.plot(x, va, color=PALETTE["cl"], lw=2.0, label="Val Acc")
    ax.scatter(
        [ep_b],
        [val_b],
        color=PALETTE["cl"],
        s=60,
        zorder=6,
        edgecolors="white",
        lw=0.8,
        label=f"Best {val_b:.2f}%",
    )
    ax.annotate(
        f"  {val_b:.2f}%",
        xy=(ep_b, val_b),
        fontsize=8,
        color=PALETTE["cl"],
        fontweight="bold",
    )
    ylo, yhi = _ylim_padded([ta, va], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
    ax.set_ylim(ylo, yhi)
    _add_tier_markers(ax, ylo, yhi)
    set_epoch_axis(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b)  Train & Validation Accuracy")
    ax.legend(fontsize=7, loc="lower right")

    # (c) Val accuracy with tier shading (clean, thesis-facing)
    ax = axes[1, 0]
    ax.plot(x, va, color=PALETTE["cl"], lw=2.2, label="Val Accuracy")
    ax.scatter(
        [ep_b], [val_b], color=PALETTE["cl"], s=60, zorder=6, edgecolors="white", lw=0.8
    )
    ax.annotate(
        f"  Best: {val_b:.2f}%",
        xy=(ep_b, val_b),
        fontsize=8,
        color=PALETTE["cl"],
        fontweight="bold",
    )
    ylo, yhi = _ylim_padded([va], pad_frac=0.12, lo_floor=0.0, hi_ceil=100.0)
    ax.set_ylim(ylo, yhi)
    # Shaded tier regions with labels
    tier_bg_labels = [
        "Tier 1\n(flip, crop,\ntranslate)",
        "Tier 2\n(+jitter, rot.,\nshear…)",
        "Tier 3\n(+cutout,\ngrayscale…)",
    ]
    for j in range(3):
        ax.axvspan(
            tier_bounds[j], tier_bounds[j + 1], alpha=0.13, color=tier_bg[j], zorder=0
        )
        mid = (tier_bounds[j] + tier_bounds[j + 1]) / 2
        ax.text(
            mid,
            ylo + (yhi - ylo) * 0.03,
            tier_bg_labels[j],
            ha="center",
            fontsize=6.5,
            color="#555555",
            va="bottom",
        )
    for j, ep_t in enumerate(tier_epochs):
        ax.axvline(ep_t, color=tier_colors[j], lw=1.3, ls="--", alpha=0.70)
        ax.text(
            ep_t + 0.8,
            yhi - (yhi - ylo) * 0.03,
            tier_names[j],
            fontsize=7,
            color=tier_colors[j],
            va="top",
        )
    set_epoch_axis(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("(c)  Validation Accuracy with Tier Regions")
    ax.legend(fontsize=7, loc="lower right")

    # (d) Generalization gap
    ax = axes[1, 1]
    ax.plot(x, gap, color=PALETTE["adam"], lw=2.0, label="Train − Val Gap")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ylo, yhi = _ylim_padded([gap], pad_frac=0.12)
    ax.set_ylim(ylo, yhi)
    _add_tier_markers(ax, ylo, yhi)
    set_epoch_axis(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train − Val Accuracy (pp)")
    ax.set_title("(d)  Generalization Gap  (↓ lower is better)")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    save(fig, fname)
    plt.show()


def mode_single(name, ckpt, tier_t1, tier_t2, title, label):
    """Plot training curves for a single checkpoint by name prefix."""
    print(f"\n  Loading: {name}")
    h = load_history(name, ckpt)
    if h is None:
        print(f"  ERROR: no history found for '{name}' in {ckpt}")
        return

    n_epochs = len(h["val_acc"])
    t1 = int(round(tier_t1 * n_epochs)) if tier_t1 <= 1.0 else int(tier_t1)
    t2 = int(round(tier_t2 * n_epochs)) if tier_t2 <= 1.0 else int(tier_t2)
    tier_epochs = [t1, t2]

    slug = name.replace("/", "_")[:60]
    fname = f"fig9_{slug}.png"
    print(f"  Tier transitions at epochs {tier_epochs}  |  {n_epochs} total epochs")
    fig9_single_run(h, label=label, tier_epochs=tier_epochs, title=title, fname=fname)
    print_analysis({"Run": (h, PALETTE["cl"])})


# ─────────────────────────────────────────────────────────────
# HELPERS — multi-seed loading
# ─────────────────────────────────────────────────────────────
def load_multiseed(base_name, seeds, checkpoint_dir=CHECKPOINT_DIR):
    results = []
    for s in seeds:
        h = load_history(f"{base_name}_s{s}", checkpoint_dir)
        if h is not None:
            results.append((s, h))
    return results


def _multiseed_stats(histories):
    min_len = min(len(h["val_acc"]) for h in histories)
    stats = {}
    for key in ["val_acc", "val_loss", "train_acc", "train_loss"]:
        arr = np.array([h[key][:min_len] for h in histories], dtype=float)
        stats[key] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
    return stats, min_len


# ─────────────────────────────────────────────────────────────
# FIGURE 10 — Multi-seed confidence bands
# ─────────────────────────────────────────────────────────────
def fig10_multiseed(
    method_seeds, tier_epochs=None, title="", fname="fig10_multiseed.png"
):
    """
    method_seeds: {label: {"histories": [h1,h2,...], "color": "#..."}}
    Val accuracy and loss with mean ± 1σ shaded bands across seeds.
    """
    if tier_epochs is None:
        tier_epochs = TIER_EPOCHS
    valid = {k: v for k, v in method_seeds.items() if v["histories"]}
    if not valid:
        print("  fig10: no multi-seed data found, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Multi-Seed Stability (mean ± 1σ) — {title}",
        fontsize=10,
        fontweight="bold",
    )
    tier_colors_ms = ["#009E73", "#E69F00"]
    tier_names_ms = ["Tier 2", "Tier 3"]
    all_loss, all_acc = [], []
    max_n = 1

    for label, v in valid.items():
        stats, n = _multiseed_stats(v["histories"])
        max_n = max(max_n, n)
        x = np.arange(1, n + 1)
        color = v["color"]

        mean_l = smooth(stats["val_loss"]["mean"], w=7)
        std_l = stats["val_loss"]["std"]
        axes[0].plot(x, mean_l, color=color, lw=2.2, label=label)
        axes[0].fill_between(x, mean_l - std_l, mean_l + std_l, color=color, alpha=0.18)
        all_loss.extend((mean_l + std_l).tolist())
        all_loss.extend((mean_l - std_l).tolist())

        mean_a = smooth(stats["val_acc"]["mean"], w=7) * 100
        std_a = stats["val_acc"]["std"] * 100
        axes[1].plot(x, mean_a, color=color, lw=2.2, label=label)
        axes[1].fill_between(x, mean_a - std_a, mean_a + std_a, color=color, alpha=0.18)
        ep_b = int(np.argmax(stats["val_acc"]["mean"])) + 1
        val_b = float(stats["val_acc"]["mean"][ep_b - 1]) * 100
        axes[1].scatter(
            [ep_b], [val_b], color=color, s=50, zorder=6, edgecolors="white", lw=0.8
        )
        axes[1].annotate(
            f" {val_b:.2f}%",
            xy=(ep_b, val_b),
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )
        all_acc.extend((mean_a + std_a).tolist())
        all_acc.extend((mean_a - std_a).tolist())

    for ax in axes:
        set_epoch_axis(ax, max_n)
        ax.set_xlabel("Epoch")

    if all_loss:
        lo, hi = _ylim_padded([all_loss], pad_frac=0.08, lo_floor=0.0)
        axes[0].set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            axes[0].axvline(ep_t, color=tier_colors_ms[j], lw=1.1, ls="--", alpha=0.65)
            axes[0].text(
                ep_t + 0.8,
                hi * 0.97,
                tier_names_ms[j],
                fontsize=7,
                color=tier_colors_ms[j],
                va="top",
            )
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("(a)  Validation Loss (mean ± 1σ)")
    axes[0].legend(fontsize=8, loc="upper right")

    if all_acc:
        lo, hi = _ylim_padded([all_acc], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        axes[1].set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            axes[1].axvline(ep_t, color=tier_colors_ms[j], lw=1.1, ls="--", alpha=0.65)
            axes[1].text(
                ep_t + 0.8,
                hi * 0.97,
                tier_names_ms[j],
                fontsize=7,
                color=tier_colors_ms[j],
                va="top",
            )
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("(b)  Validation Accuracy (mean ± 1σ)")
    axes[1].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 11 — Reverse vs Forward Curriculum
# ─────────────────────────────────────────────────────────────
def fig11_reverse_vs_forward(
    h_forward,
    h_reverse,
    tier_epochs=None,
    title="",
    fname="fig11_reverse_vs_forward.png",
):
    """Forward (Tier 1→2→3) vs Reverse (Tier 3→2→1). If forward > reverse, ordering is causal."""
    if tier_epochs is None:
        tier_epochs = TIER_EPOCHS
    if h_forward is None and h_reverse is None:
        print("  fig11: no reverse/forward histories found, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Curriculum Order Ablation: Forward vs Reverse — {title}\n"
        "Forward > Reverse confirms tier ordering is causal, not incidental.",
        fontsize=9,
        fontweight="bold",
    )
    tier_colors_rv = ["#009E73", "#E69F00"]
    tier_names_rv = ["Tier 2", "Tier 3"]
    configs = []
    if h_forward is not None:
        configs.append(("Forward  (T1→T2→T3)", h_forward, PALETTE["cl"], "-", 2.2))
    if h_reverse is not None:
        configs.append(("Reverse  (T3→T2→T1)", h_reverse, PALETTE["adam"], "--", 2.0))

    max_n = max(len(c[1]["val_acc"]) for c in configs)
    all_acc, all_loss = [], []

    for label, h, color, ls, lw in configs:
        x = ep(h)
        va = smooth(h["val_acc"], w=7) * 100
        vl = smooth(h["val_loss"], w=7)
        ep_b, val_b = best(h)
        axes[0].plot(x, vl, color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(x, va, color=color, lw=lw, ls=ls, label=label)
        axes[1].scatter(
            [ep_b], [val_b], color=color, s=55, zorder=6, edgecolors="white", lw=0.8
        )
        axes[1].annotate(
            f"  {val_b:.2f}%",
            xy=(ep_b, val_b),
            fontsize=8,
            color=color,
            fontweight="bold",
        )
        all_loss.extend(vl.tolist())
        all_acc.extend(va.tolist())

    for ax in axes:
        set_epoch_axis(ax, max_n)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    if all_loss:
        lo, hi = _ylim_padded([all_loss], pad_frac=0.08, lo_floor=0.0)
        axes[0].set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            axes[0].axvline(ep_t, color=tier_colors_rv[j], lw=1.0, ls=":", alpha=0.60)
            axes[0].text(
                ep_t + 0.8,
                hi * 0.97,
                tier_names_rv[j],
                fontsize=7,
                color=tier_colors_rv[j],
                va="top",
            )
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("(a)  Validation Loss")

    if all_acc:
        lo, hi = _ylim_padded([all_acc], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        axes[1].set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            axes[1].axvline(ep_t, color=tier_colors_rv[j], lw=1.0, ls=":", alpha=0.60)
            axes[1].text(
                ep_t + 0.8,
                hi * 0.97,
                tier_names_rv[j],
                fontsize=7,
                color=tier_colors_rv[j],
                va="top",
            )
    axes[1].set_ylabel("Validation Accuracy (%)")
    axes[1].set_title("(b)  Validation Accuracy")

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 12 — Tier Transition Zoom
# ─────────────────────────────────────────────────────────────
def fig12_tier_zoom(
    h, tier_epochs=None, window=12, title="", fname="fig12_tier_zoom.png"
):
    """Zoomed ±window epochs around each tier boundary. Makes the model's reaction visible."""
    if tier_epochs is None:
        tier_epochs = TIER_EPOCHS
    if h is None:
        print("  fig12: no history, skipping.")
        return

    x = ep(h)
    n = len(x)
    va_raw = np.array(h["val_acc"]) * 100
    vl_raw = np.array(h["val_loss"])
    ta_raw = np.array(h["train_acc"]) * 100
    tier_colors_z = ["#009E73", "#E69F00"]
    transition_names = ["Tier 1 → Tier 2", "Tier 2 → Tier 3"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(
        f"Tier Transition Zoom (±{window} epochs) — {title}",
        fontsize=9,
        fontweight="bold",
    )

    for col, (ep_t, tname, tcolor) in enumerate(
        zip(tier_epochs, transition_names, tier_colors_z)
    ):
        lo_ep = max(1, ep_t - window)
        hi_ep = min(n, ep_t + window)
        mask = (x >= lo_ep) & (x <= hi_ep)
        xz = x[mask]

        ax_a = axes[0, col]
        ax_a.plot(
            xz,
            ta_raw[mask],
            color=PALETTE["static"],
            lw=1.5,
            ls="--",
            alpha=0.70,
            label="Train Acc",
        )
        ax_a.plot(xz, va_raw[mask], color=tcolor, lw=2.2, label="Val Acc")
        ax_a.axvline(
            ep_t,
            color=tcolor,
            lw=1.5,
            ls="--",
            alpha=0.80,
            label=f"↑ {tname} (ep {ep_t})",
        )
        ax_a.fill_between(xz, va_raw[mask], ta_raw[mask], alpha=0.10, color=tcolor)
        idx_t = ep_t - 1
        if 0 <= idx_t < n:
            span = va_raw[mask].max() - va_raw[mask].min()
            ax_a.annotate(
                f"ep {ep_t}: {tname}\nVal={va_raw[idx_t]:.1f}%",
                xy=(ep_t, va_raw[idx_t]),
                xytext=(ep_t + 1.5, va_raw[mask].min() + span * 0.2),
                fontsize=7.5,
                color=tcolor,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=tcolor, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=tcolor, alpha=0.85),
            )
        ylo, yhi = _ylim_padded(
            [va_raw[mask], ta_raw[mask]], pad_frac=0.15, lo_floor=0.0, hi_ceil=100.0
        )
        ax_a.set_ylim(ylo, yhi)
        ax_a.set_xlim(lo_ep - 0.5, hi_ep + 0.5)
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_a.set_title(f"({'ab'[col]})  {tname} — Accuracy")
        ax_a.legend(fontsize=7, loc="lower right")
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        ax_l = axes[1, col]
        ax_l.plot(xz, vl_raw[mask], color=tcolor, lw=2.2, label="Val Loss")
        ax_l.axvline(
            ep_t,
            color=tcolor,
            lw=1.5,
            ls="--",
            alpha=0.80,
            label=f"↑ {tname} (ep {ep_t})",
        )
        ylo_l, yhi_l = _ylim_padded([vl_raw[mask]], pad_frac=0.15, lo_floor=0.0)
        ax_l.set_ylim(ylo_l, yhi_l)
        ax_l.set_xlim(lo_ep - 0.5, hi_ep + 0.5)
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Validation Loss" if col == 0 else "")
        ax_l.set_title(f"({'cd'[col]})  {tname} — Loss")
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 13 — Cross-Dataset Delta Bar Chart
# ─────────────────────────────────────────────────────────────
def fig13_cross_dataset(dataset_results, fname="fig13_cross_dataset.png"):
    """
    dataset_results: {dataset_label: {method_label: best_val_acc_float}}
    e.g. {"CIFAR-100": {"Static": 0.712, "ETS + mix": 0.741}}
    Plots Δ over Static baseline (panel a) and absolute accuracy (panel b).
    """
    datasets = list(dataset_results.keys())
    if not datasets:
        print("  fig13: no cross-dataset results, skipping.")
        return

    all_methods = sorted(
        {m for d in dataset_results.values() for m in d if m.lower() != "static"}
    )
    method_colors_cd = [
        PALETTE["cl"],
        PALETTE["adam"],
        PALETTE["cosine"],
        PALETTE["no_aug"],
        "#CC79A7",
    ]
    n_ds = len(datasets)
    n_m = len(all_methods)
    if n_m == 0:
        print("  fig13: no non-static methods found, skipping.")
        return

    x = np.arange(n_ds)
    width = 0.7 / n_m

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_ds * 3), 4.5))
    fig.suptitle(
        "Cross-Dataset Generalisation — Δ over Static Augmentation Baseline\n"
        "Consistent positive delta confirms the method generalises across datasets.",
        fontsize=9,
        fontweight="bold",
    )

    ax = axes[0]
    for i, (method, color) in enumerate(zip(all_methods, method_colors_cd)):
        deltas = []
        for ds_label in datasets:
            res = dataset_results[ds_label]
            static_acc = res.get("Static", res.get("static", None))
            method_acc = res.get(method, None)
            deltas.append(
                (method_acc - static_acc) * 100
                if static_acc is not None and method_acc is not None
                else float("nan")
            )
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            deltas,
            width,
            label=method,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        for bar, val in zip(bars, deltas):
            if not np.isnan(val):
                sign = "+" if val >= 0 else ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.1 if val >= 0 else -0.4),
                    f"{sign}{val:.1f}pp",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7,
                    fontweight="bold",
                )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("Δ Val Accuracy over Static (pp)")
    ax.set_title("(a)  Improvement over Static Baseline")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    all_methods_abs = ["Static"] + all_methods
    all_colors_abs = [PALETTE["static"]] + method_colors_cd[:n_m]
    n_m2 = len(all_methods_abs)
    width2 = 0.7 / n_m2
    for i, (method, color) in enumerate(zip(all_methods_abs, all_colors_abs)):
        accs = []
        for ds_label in datasets:
            res = dataset_results[ds_label]
            a = res.get(method, res.get(method.lower(), None))
            accs.append(a * 100 if a is not None else float("nan"))
        offset = (i - n_m2 / 2 + 0.5) * width2
        bars = ax2.bar(
            x + offset,
            accs,
            width2,
            label=method,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85 if method == "Static" else 1.0,
            zorder=3,
        )
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=9)
    ax2.set_ylabel("Best Validation Accuracy (%)")
    ax2.set_title("(b)  Absolute Accuracy per Dataset")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 14 — Ablation Component Curves
# ─────────────────────────────────────────────────────────────
def fig14_ablation_components(
    runs, tier_epochs=None, fname="fig14_ablation_components.png"
):
    """
    Progressive ablation: each condition adds one component.
    3-panel: val loss curves | val acc curves | best-val horizontal bars.
    """
    if tier_epochs is None:
        tier_epochs = TIER_EPOCHS
    valid = [(k, h, c) for k, (h, c) in runs.items() if h is not None]
    if not valid:
        print("  fig14: no ablation data, skipping.")
        return

    tier_colors_ab = ["#009E73", "#E69F00"]
    tier_names_ab = ["Tier 2", "Tier 3"]
    max_n = max(len(h["val_acc"]) for _, h, _ in valid)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        "Ablation Study — Component Attribution\n"
        "Each bar adds one component; cumulative positive delta confirms independent contribution.",
        fontsize=9,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    all_vl = []
    for label, h, color in valid:
        vl = smooth(h["val_loss"], w=7)
        ax0.plot(ep(h), vl, color=color, lw=2.0, label=label)
        all_vl.extend(vl.tolist())
    if all_vl:
        lo, hi = _ylim_padded([all_vl], pad_frac=0.08, lo_floor=0.0)
        ax0.set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            ax0.axvline(ep_t, color=tier_colors_ab[j], lw=1.0, ls="--", alpha=0.55)
            ax0.text(
                ep_t + 0.5,
                hi * 0.97,
                tier_names_ab[j],
                fontsize=6.5,
                color=tier_colors_ab[j],
                va="top",
            )
    set_epoch_axis(ax0, max_n)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Validation Loss")
    ax0.set_title("(a)  Validation Loss")
    ax0.legend(fontsize=7, loc="upper right")

    ax1 = fig.add_subplot(gs[1])
    all_va = []
    for label, h, color in valid:
        va = smooth(h["val_acc"], w=7) * 100
        ep_b, val_b = best(h)
        ax1.plot(ep(h), va, color=color, lw=2.0, label=label)
        ax1.scatter(
            [ep_b], [val_b], color=color, s=45, zorder=6, edgecolors="white", lw=0.8
        )
        all_va.extend(va.tolist())
    if all_va:
        lo, hi = _ylim_padded([all_va], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        ax1.set_ylim(lo, hi)
        for j, ep_t in enumerate(tier_epochs):
            ax1.axvline(ep_t, color=tier_colors_ab[j], lw=1.0, ls="--", alpha=0.55)
            ax1.text(
                ep_t + 0.5,
                hi * 0.97,
                tier_names_ab[j],
                fontsize=6.5,
                color=tier_colors_ab[j],
                va="top",
            )
    set_epoch_axis(ax1, max_n)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("(b)  Validation Accuracy")
    ax1.legend(fontsize=7, loc="lower right")

    ax2 = fig.add_subplot(gs[2])
    labels_short = [k.split("(")[0].strip() for k, _, _ in valid]
    best_vals = [best(h)[1] for _, h, _ in valid]
    colors_ab = [c for _, _, c in valid]
    x_pos = np.arange(len(valid))
    bars = ax2.barh(
        x_pos,
        best_vals,
        color=colors_ab,
        edgecolor="black",
        linewidth=0.6,
        height=0.55,
        zorder=3,
    )
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(labels_short, fontsize=8)
    ax2.set_xlabel("Best Val Accuracy (%)")
    ax2.set_title("(c)  Best Val Acc")
    lo_b = max(0, min(best_vals) - 3)
    hi_b = min(100, max(best_vals) + 4)
    ax2.set_xlim(lo_b, hi_b)
    for bar, val in zip(bars, best_vals):
        ax2.text(
            val + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 15 — Scheduling Strategy Comparison (ETS vs LPS vs EGS)
# ─────────────────────────────────────────────────────────────
def fig15_scheduling_comparison(runs, fname="fig15_scheduling_comparison.png"):
    """ETS vs LPS vs EGS: val loss, val accuracy, and generalization gap."""
    valid = [(k, h, c) for k, (h, c) in runs.items() if h is not None]
    if not valid:
        print("  fig15: no scheduling comparison data, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Scheduling Strategy Comparison: ETS vs LPS vs EGS\n"
        "WideResNet-28-10 · CIFAR-100 · 100 epochs · SGD · CosineAnnealingLR",
        fontsize=9,
        fontweight="bold",
    )
    max_n = max(len(h["val_acc"]) for _, h, _ in valid)
    all_vl, all_va, all_gap = [], [], []

    for label, h, color in valid:
        x = ep(h)
        va = smooth(h["val_acc"], w=7) * 100
        vl = smooth(h["val_loss"], w=7)
        gap = smooth(np.array(h["train_acc"]) * 100 - np.array(h["val_acc"]) * 100, w=5)
        ep_b, val_b = best(h)
        axes[0].plot(x, vl, color=color, lw=2.0, label=label)
        axes[1].plot(x, va, color=color, lw=2.0, label=label)
        axes[1].scatter(
            [ep_b], [val_b], color=color, s=50, zorder=6, edgecolors="white", lw=0.8
        )
        axes[1].annotate(
            f"  {val_b:.2f}%",
            xy=(ep_b, val_b),
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )
        axes[2].plot(x, gap, color=color, lw=2.0, label=label)
        all_vl.extend(vl.tolist())
        all_va.extend(va.tolist())
        all_gap.extend(gap.tolist())

    panels = [
        (
            axes[0],
            all_vl,
            "Validation Loss",
            "(a)  Validation Loss",
            0.0,
            None,
            "upper right",
        ),
        (
            axes[1],
            all_va,
            "Validation Accuracy (%)",
            "(b)  Validation Accuracy",
            0.0,
            100.0,
            "lower right",
        ),
        (
            axes[2],
            all_gap,
            "Train − Val Accuracy (pp)",
            "(c)  Generalization Gap (↓ better)",
            None,
            None,
            "upper left",
        ),
    ]
    for ax, data, ylabel, title_s, lo_fl, hi_cl, leg_loc in panels:
        set_epoch_axis(ax, max_n)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title_s)
        ax.legend(fontsize=7.5, loc=leg_loc)
        if data:
            lo, hi = _ylim_padded([data], pad_frac=0.08, lo_floor=lo_fl, hi_ceil=hi_cl)
            ax.set_ylim(lo, hi)
    axes[2].axhline(0, color="black", lw=0.7, ls=":")

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FIGURE 16 — Convergence Speed (epochs to accuracy threshold)
# ─────────────────────────────────────────────────────────────
def fig16_convergence_speed(runs, thresholds=None, fname="fig16_convergence_speed.png"):
    """Bar chart matrix: columns = accuracy thresholds, bars = methods. Lower = faster."""
    if thresholds is None:
        thresholds = [20, 30, 40, 50, 60, 70]
    valid = [(k, h, c) for k, (h, c) in runs.items() if h is not None]
    if not valid:
        print("  fig16: no convergence data, skipping.")
        return

    reachable = [
        t for t in thresholds if any(max(h["val_acc"]) * 100 >= t for _, h, _ in valid)
    ]
    if not reachable:
        print("  fig16: no thresholds reachable, skipping.")
        return

    fig, axes = plt.subplots(1, len(reachable), figsize=(3.5 * len(reachable), 4.5))
    if len(reachable) == 1:
        axes = [axes]
    fig.suptitle(
        "Convergence Speed — Epochs to Reach Accuracy Threshold\n"
        "(Lower = faster; N/A = never reached within training)",
        fontsize=9,
        fontweight="bold",
    )
    for col, thresh in enumerate(reachable):
        ax = axes[col]
        labels_cs = [k.split("(")[0].strip() for k, _, _ in valid]
        colors_cs = [c for _, _, c in valid]
        ep_vals = []
        for _, h, _ in valid:
            idx = next(
                (i + 1 for i, v in enumerate(h["val_acc"]) if v * 100 >= thresh), None
            )
            ep_vals.append(idx if idx is not None else len(h["val_acc"]) + 5)

        x_pos = np.arange(len(valid))
        bars = ax.bar(
            x_pos,
            ep_vals,
            color=colors_cs,
            edgecolor="black",
            linewidth=0.6,
            width=0.55,
            zorder=3,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_cs, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Epochs" if col == 0 else "")
        ax.set_title(f"({chr(97 + col)})  → {thresh}% Val Acc")
        ax.set_ylim(0, max(ep_vals) + 10)
        for bar, val, (_, h, _) in zip(bars, ep_vals, valid):
            reached = max(h["val_acc"]) * 100 >= thresh
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val}" if reached else "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black" if reached else "#CC0000",
            )

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# ─────────────────────────────────────────────────────────────
# MODE: thesis — generate ALL figures
# ─────────────────────────────────────────────────────────────
def mode_thesis(
    ckpt, dataset="cifar100", model="wideresnet", seeds=(42, 43, 44), n_ep=100
):
    """
    Generate every thesis figure for the given dataset + model.
    Gracefully skips figures whose checkpoints are not yet available.
    """
    ds_str = {
        "cifar100": "CIFAR-100",
        "cifar10": "CIFAR-10",
        "tiny_imagenet": "Tiny-ImageNet",
    }.get(dataset, dataset)
    model_str = {
        "wideresnet": "WideResNet-28-10",
        "resnet50": "ResNet-50",
        "resnet18": "ResNet-18",
    }.get(model, model)
    title = f"{model_str} · {ds_str} · {n_ep} ep · SGD"
    base = model
    ds = dataset

    print(f"\n  Thesis figure generation")
    print(f"  Dataset: {ds_str}  |  Model: {model_str}  |  Epochs: {n_ep}")
    print(f"  Seeds  : {list(seeds)}")
    print(f"  Output : {FIGURES_DIR}\n")

    # ── 1. All-methods comparison (figs 1–3, 5–8) ──────────────
    runs_all = build_runs(
        [
            ("No Aug", f"{base}_none_sgd_cosine_ep{n_ep}_{ds}_s42", PALETTE["no_aug"]),
            (
                "Static Aug",
                f"{base}_static_sgd_cosine_ep{n_ep}_{ds}_s42",
                PALETTE["static"],
            ),
            (
                "RandAugment",
                f"{base}_randaugment_sgd_cosine_ep{n_ep}_{ds}_s42",
                PALETTE["cosine"],
            ),
            (
                "ETS + mix (ours)",
                f"{base}_tiered_ets_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["cl"],
            ),
            (
                "LPS + mix (ours)",
                f"{base}_tiered_lps_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["adam"],
            ),
            (
                "EGS + mix (ours)",
                f"{base}_tiered_egs_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                "#CC79A7",
            ),
        ],
        ckpt,
    )

    fig1_val_comparison(runs_all, title=title, fname=f"fig1_val_comparison_{ds}.png")
    fig2_overfitting(runs_all, fname=f"fig2_overfitting_{ds}.png")
    fig3_generalization(runs_all, fname=f"fig3_generalization_{ds}.png")
    fig5_summary(runs_all, fname=f"fig5_summary_{ds}.png")
    fig6_cl_with_tiers(runs_all, fname=f"fig6_cl_tier_transitions_{ds}.png")
    fig7_gap_over_epochs(runs_all, fname=f"fig7_gap_{ds}.png")
    fig8_top5(runs_all, fname=f"fig8_top5_{ds}.png")
    fig16_convergence_speed(runs_all, fname=f"fig16_convergence_{ds}.png")
    print_analysis(runs_all)

    # ── 2. Single-run 4-panel + zoom (figs 9, 12) ──────────────
    cl_h = next(
        (v[0] for k, v in runs_all.items() if v[0] is not None and "ETS" in k),
        next((v[0] for v in runs_all.values() if v[0] is not None), None),
    )
    if cl_h is not None:
        fig9_single_run(
            cl_h,
            label="ETS + mix (ours)",
            tier_epochs=TIER_EPOCHS,
            title=title,
            fname=f"fig9_single_run_{ds}.png",
        )
        fig12_tier_zoom(
            cl_h,
            tier_epochs=TIER_EPOCHS,
            window=12,
            title=title,
            fname=f"fig12_tier_zoom_{ds}.png",
        )

    # ── 3. Multi-seed confidence bands (fig 10) ─────────────────
    method_seeds = {}
    for m_label, base_name, color in [
        ("Static Aug", f"{base}_static_sgd_cosine_ep{n_ep}_{ds}", PALETTE["static"]),
        (
            "ETS + mix (ours)",
            f"{base}_tiered_ets_mix_both_sgd_cosine_ep{n_ep}_{ds}_p19",
            PALETTE["cl"],
        ),
        (
            "LPS + mix (ours)",
            f"{base}_tiered_lps_mix_both_sgd_cosine_ep{n_ep}_{ds}_p19",
            PALETTE["adam"],
        ),
    ]:
        seed_data = load_multiseed(base_name, list(seeds), ckpt)
        if seed_data:
            method_seeds[m_label] = {
                "histories": [h for _, h in seed_data],
                "color": color,
            }
    if method_seeds:
        fig10_multiseed(
            method_seeds,
            tier_epochs=TIER_EPOCHS,
            title=title,
            fname=f"fig10_multiseed_{ds}.png",
        )

    # ── 4. Reverse vs Forward curriculum (fig 11) ───────────────
    h_forward = runs_all.get("ETS + mix (ours)", (None, None))[0]
    h_reverse = load_history(
        f"{base}_tiered_ets_mix_both_reverse_sgd_cosine_ep{n_ep}_{ds}_s42_p19", ckpt
    ) or load_history(
        f"{base}_reverse_tiered_ets_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19", ckpt
    )
    fig11_reverse_vs_forward(
        h_forward,
        h_reverse,
        tier_epochs=TIER_EPOCHS,
        title=title,
        fname=f"fig11_reverse_{ds}.png",
    )

    # ── 5. Ablation components (fig 14) ─────────────────────────
    runs_ablation = build_runs(
        [
            (
                "Static Aug",
                f"{base}_static_sgd_cosine_ep{n_ep}_{ds}_s42",
                PALETTE["static"],
            ),
            (
                "Static + Mixing",
                f"{base}_static_mixing_sgd_cosine_ep{n_ep}_{ds}_s42",
                PALETTE["cosine"],
            ),
            (
                "ETS no-mix",
                f"{base}_tiered_ets_nomix_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["adam"],
            ),
            (
                "ETS + mix (ours)",
                f"{base}_tiered_ets_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["cl"],
            ),
        ],
        ckpt,
    )
    fig14_ablation_components(
        runs_ablation, tier_epochs=TIER_EPOCHS, fname=f"fig14_ablation_{ds}.png"
    )

    # ── 6. Scheduling comparison: ETS vs LPS vs EGS (fig 15) ────
    runs_sched = build_runs(
        [
            (
                "ETS + mix",
                f"{base}_tiered_ets_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["cl"],
            ),
            (
                "LPS + mix",
                f"{base}_tiered_lps_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                PALETTE["adam"],
            ),
            (
                "EGS + mix",
                f"{base}_tiered_egs_mix_both_sgd_cosine_ep{n_ep}_{ds}_s42_p19",
                "#CC79A7",
            ),
        ],
        ckpt,
    )
    fig15_scheduling_comparison(runs_sched, fname=f"fig15_scheduling_{ds}.png")

    # ── 7. Cross-dataset delta bar (fig 13) ─────────────────────
    datasets_cfg = {
        "CIFAR-10": "cifar10",
        "CIFAR-100": "cifar100",
        "Tiny-ImageNet": "tiny_imagenet",
    }
    methods_cfg = {
        "Static": f"{base}_static_sgd_cosine_ep{{n}}_{{ds}}_s42",
        "ETS + mix": f"{base}_tiered_ets_mix_both_sgd_cosine_ep{{n}}_{{ds}}_s42_p19",
        "LPS + mix": f"{base}_tiered_lps_mix_both_sgd_cosine_ep{{n}}_{{ds}}_s42_p19",
    }
    dataset_results = {}
    for ds_label, ds_key in datasets_cfg.items():
        res = {}
        for m_label, m_pattern in methods_cfg.items():
            h = load_history(m_pattern.format(n=n_ep, ds=ds_key), ckpt)
            if h:
                res[m_label] = max(h["val_acc"])
        if len(res) >= 2:
            dataset_results[ds_label] = res
    if dataset_results:
        fig13_cross_dataset(dataset_results, fname="fig13_cross_dataset.png")

    print(f"\n  All thesis figures saved to: {os.path.abspath(FIGURES_DIR)}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Publication-quality training analysis"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baselines", "all", "ablation", "single", "thesis"],
    )
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    # --mode single
    parser.add_argument(
        "--name", default="", help="Checkpoint name prefix (--mode single)"
    )
    parser.add_argument(
        "--tier_t1",
        type=float,
        default=0.20,
        help="Tier 1→2 boundary: fraction ≤1.0 or absolute epoch >1.0",
    )
    parser.add_argument(
        "--tier_t2",
        type=float,
        default=0.45,
        help="Tier 2→3 boundary: fraction ≤1.0 or absolute epoch >1.0",
    )
    parser.add_argument(
        "--title",
        default="WideResNet-28-10 · ETS · 19-op pool · 100 epochs",
        help="Figure suptitle",
    )
    parser.add_argument(
        "--label", default="ETS + mix (ours)", help="Run label in legend"
    )
    # --mode thesis
    parser.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar100", "cifar10", "tiny_imagenet"],
        help="Dataset for --mode thesis",
    )
    parser.add_argument(
        "--model",
        default="wideresnet",
        choices=["wideresnet", "resnet50", "resnet18"],
        help="Model for --mode thesis",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seed list for multi-seed plots (--mode thesis)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total epochs for --mode thesis checkpoint lookup",
    )
    args = parser.parse_args()

    print(f"\n  Mode           : {args.mode}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print(f"  Figures dir    : {FIGURES_DIR}")

    if args.mode == "single":
        if not args.name:
            parser.error("--mode single requires --name <checkpoint_prefix>")
        mode_single(
            args.name,
            args.checkpoint_dir,
            args.tier_t1,
            args.tier_t2,
            args.title,
            args.label,
        )
    elif args.mode == "thesis":
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
        mode_thesis(
            args.checkpoint_dir,
            dataset=args.dataset,
            model=args.model,
            seeds=seeds,
            n_ep=args.epochs,
        )
    else:
        {"baselines": mode_baselines, "all": mode_all, "ablation": mode_ablation}[
            args.mode
        ](args.checkpoint_dir)
