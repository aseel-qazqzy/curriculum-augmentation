"""
analysis/plot_curves.py  —  Publication-quality training analysis
Thesis: Curriculum Learning Based on Loss — CIFAR-10

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
FIGURES_DIR    = str(_PROJECT_ROOT / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# PUBLICATION STYLE  (thesis / ICML aesthetic)
PALETTE = {
    "no_aug":  "#D62728",   # deep red
    "static":  "#1F77B4",   # steel blue
    "cl":      "#2CA02C",   # forest green  ← YOUR METHOD
    "cosine":  "#FF7F0E",   # amber
    "adam":    "#9467BD",   # purple
}

matplotlib.rcParams.update({
    "figure.facecolor":      "white",
    "figure.dpi":            150,
    "axes.facecolor":        "#FAFAFA",
    "axes.edgecolor":        "#CCCCCC",
    "axes.linewidth":        1.0,
    "axes.labelsize":        12,
    "axes.labelweight":      "normal",
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.titlepad":         14,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.color":            "#E0E0E0",
    "grid.linewidth":        0.7,
    "grid.alpha":            1.0,
    "grid.linestyle":        "-",
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      4,
    "ytick.major.size":      4,
    "legend.fontsize":       10,
    "legend.framealpha":     0.95,
    "legend.edgecolor":      "#CCCCCC",
    "legend.fancybox":       False,
    "legend.borderpad":      0.6,
    "legend.labelspacing":   0.4,
    "lines.linewidth":       2.0,
    "lines.antialiased":     True,
    "font.family":           "DejaVu Sans",
    "font.size":             11,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "white",
    "savefig.pad_inches":    0.15,
})


# DATA LOADING
def load_history(name, checkpoint_dir=CHECKPOINT_DIR):
    for ext, loader in [(".pt", _load_pt), (".json", _load_json)]:
        path = os.path.join(checkpoint_dir, f"{name}_history{ext}")
        if os.path.exists(path):
            h = loader(path)
            if h is not None:
                print(f"  ✅  {name:<48}  {len(h['val_acc']):>3} epochs  "
                      f"best_val={max(h['val_acc'])*100:.2f}%")
                return h
    print(f"  ⚠️   {name}  — not found")
    return None

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

def milestones(n=150):
    return [int(n * f) for f in (0.33, 0.66, 0.83)]

def vlines(ax, n, color="#888888", alpha=0.30, label=True):
    ms = milestones(n)
    # Only draw milestone lines that fall within the actual x range
    xlo, xhi = ax.get_xlim() if ax.get_xlim()[1] > 1 else (0, n + 2)
    for i, m in enumerate(ms):
        if m < xhi:
            ax.axvline(m, color=color, lw=1.0, ls=":", alpha=alpha,
                       label="LR decay" if (i == 0 and label) else "_")

def set_epoch_axis(ax, n):
    ax.set_xlim(0, n + 2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

def set_epoch_labels(ax, ylabel=""):
    ax.set_xlabel("Epoch", labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel)

def _ylim_padded(values_list, pad_frac=0.06, lo_floor=None, hi_ceil=None):
    """Compute y-axis limits from data with fractional padding on both sides."""
    flat = [v for vals in values_list for v in vals]
    lo, hi = min(flat), max(flat)
    span   = max(hi - lo, 1e-6)
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
    print(f"  → saved: {fname}")
    return path


# FIGURE 1  ─  Validation Comparison  (overlaid)
def fig1_val_comparison(runs, title="", fname="fig1_val_comparison.png"):
    """One row: val loss | val accuracy.  All runs overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    fig.suptitle(
        f"Validation Loss & Accuracy — CIFAR-10  ·  ResNet-18  ·  SGD  ·  MultiStepLR\n{title}",
        fontsize=13, fontweight="bold", y=1.02,
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
        x  = ep(h)
        ls = "--" if "CL" in label else "-"
        lw = 2.4 if "CL" in label else 2.0
        axes[0].plot(x, smooth(h["val_loss"]),     color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(x, smooth(h["val_acc"]) * 100, color=color, lw=lw, ls=ls, label=label)
        ep_b, val_b = best(h)
        axes[1].scatter([ep_b], [val_b], color=color, s=55, zorder=6,
                        edgecolors="white", linewidths=0.8)

    for ax in axes:
        set_epoch_axis(ax, max_n)
        set_epoch_labels(ax)
        vlines(ax, max_n)

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

    # Shared legend below figure
    handles, labs = axes[1].get_legend_handles_labels()
    lr_p = mpatches.Patch(color="#888888", alpha=0.5, label="MultiStepLR decay")
    fig.legend(handles + [lr_p], labs + ["MultiStepLR decay"],
               loc="lower center", ncol=min(len(runs) + 1, 4),
               bbox_to_anchor=(0.5, -0.04), frameon=True,
               fontsize=10, edgecolor="#CCCCCC")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, fname)
    plt.show()


# FIGURE 2  ─  Overfitting Analysis  (Train vs Val Loss)
def fig2_overfitting(runs, fname="fig2_overfitting.png"):
    """One subplot per experiment.  Shaded = overfitting gap."""
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    n     = len(valid)
    if n == 0:
        print("  fig2: no valid runs, skipping.")
        return
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Overfitting Analysis — Train vs Validation Loss  ·  ResNet-18  ·  SGD",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for i, (label, h, color) in enumerate(valid):
        x  = ep(h)
        tl = smooth(h["train_loss"], w=3)
        vl = smooth(h["val_loss"],   w=7)

        axes[i].plot(x, tl, color=color, lw=2.0, ls="--", alpha=0.80, label="Train")
        axes[i].plot(x, vl, color=color, lw=2.2,            label="Val")
        axes[i].fill_between(x, tl, vl, alpha=0.13, color=color)
        set_epoch_axis(axes[i], len(x))
        vlines(axes[i], len(x), label=(i == 0))

        # Data-driven y-limits
        all_loss = np.concatenate([tl, vl])
        lo, hi = _ylim_padded([all_loss], pad_frac=0.12, lo_floor=0.0)
        axes[i].set_ylim(lo, hi)

        # Annotate final gap — text in axes-fraction coords so it never escapes
        gap   = float(vl[-1] - tl[-1])
        mid_y = (float(tl[-1]) + float(vl[-1])) / 2
        # Choose annotation side: text on left half if the curve ends near right edge
        text_x_frac = 0.40
        text_y_frac = 0.82
        axes[i].annotate(
            f"gap = {abs(gap):.3f}",
            xy=(len(x), mid_y),
            xycoords="data",
            xytext=(text_x_frac, text_y_frac),
            textcoords="axes fraction",
            fontsize=9, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                            connectionstyle="arc3,rad=-0.15"),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, alpha=0.88),
        )

        axes[i].set_title(f"({chr(97+i)})  {label}")
        set_epoch_labels(axes[i], "Loss" if i == 0 else "")
        axes[i].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 3  ─  Generalization Gap  (Train vs Val Accuracy)
def fig3_generalization(runs, fname="fig3_generalization.png"):
    """One subplot per experiment.  Shaded = generalization gap."""
    valid = [(k, h, c) for k, (h, c) in runs.items() if h]
    n     = len(valid)
    if n == 0:
        print("  fig3: no valid runs, skipping.")
        return
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Generalization Gap — Train vs Validation Accuracy  ·  ResNet-18  ·  SGD",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for i, (label, h, color) in enumerate(valid):
        x  = ep(h)
        ta = smooth(h["train_acc"], w=3) * 100
        va = smooth(h["val_acc"],   w=7) * 100

        axes[i].plot(x, ta, color=color, lw=2.0, ls="--", alpha=0.80, label="Train")
        axes[i].plot(x, va, color=color, lw=2.2,            label="Val")
        axes[i].fill_between(x, va, ta, alpha=0.13, color=color,
                             label=f"Gap = {float(ta[-1]-va[-1]):.1f}%")

        ep_b, val_b = best(h)
        axes[i].scatter([ep_b], [val_b], color=color, s=70, zorder=6,
                        edgecolors="white", lw=0.8, label=f"Best {val_b:.2f}%")

        # Data-driven y-limits with room for the "Best" label
        all_acc = np.concatenate([ta, va])
        lo, hi = _ylim_padded([all_acc], pad_frac=0.12, lo_floor=0.0, hi_ceil=100.0)
        axes[i].set_ylim(lo, hi)

        # Annotate best — use axes-fraction coords so it never escapes the plot
        axes[i].annotate(
            f"{val_b:.2f}%",
            xy=(ep_b, val_b),
            xycoords="data",
            xytext=(0.55, 0.20),
            textcoords="axes fraction",
            fontsize=9, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                            connectionstyle="arc3,rad=0.2"),
        )

        vlines(axes[i], len(x), label=(i == 0))
        axes[i].set_title(f"({chr(97+i)})  {label}")
        set_epoch_labels(axes[i], "Accuracy (%)" if i == 0 else "")
        axes[i].legend(loc="lower right", fontsize=9)
        set_epoch_axis(axes[i], len(x))

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 4  ─  LR Schedule Comparison
def fig4_lr_schedule(n_epochs=150, fname="fig4_lr_schedule.png"):
    """MultiStepLR vs CosineAnnealingLR — the thesis motivation figure."""
    ms   = milestones(n_epochs)
    ep_x = np.arange(1, n_epochs + 1)

    # Simulate MultiStep
    lr_ms  = np.full(n_epochs, 0.1)
    for m in ms:
        lr_ms[m:] *= 0.1

    # Simulate Cosine
    lr_cos = 0.1 * 0.5 * (1 + np.cos(np.pi * ep_x / n_epochs))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    fig.suptitle(
        "Learning Rate Schedule Comparison — Thesis Motivation",
        fontsize=13, fontweight="bold",
    )

    ax.plot(ep_x, lr_ms,  color=PALETTE["static"],  lw=2.5,
            label=f"MultiStepLR  (milestones={ms},  γ=0.1)")
    ax.plot(ep_x, lr_cos, color=PALETTE["no_aug"],   lw=2.5, ls="--",
            label="CosineAnnealingLR")

    for j, m in enumerate(ms):
        ax.axvline(m, color=PALETTE["static"], ls=":", lw=1.1, alpha=0.45)
        ax.text(m + 1, lr_ms[0] * 1.08, f"ep {m}", fontsize=8,
                color=PALETTE["static"], va="bottom")

    ax.axvspan(ms[2], n_epochs, alpha=0.07, color=PALETTE["cl"],
               label="CL hard samples arrive here")

    # Annotations: place at 85% of x-range to stay inside axes
    annot_ep = int(n_epochs * 0.88)
    ax.annotate("LR = 0.0001\nstill active for CL",
                xy=(annot_ep, lr_ms[annot_ep - 1]),
                xytext=(int(n_epochs * 0.68), 4e-3),
                fontsize=9, color=PALETTE["static"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["static"], lw=1.1),
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=PALETTE["static"], alpha=0.92))

    ax.annotate("LR ≈ 0\nCL signal wasted",
                xy=(annot_ep, lr_cos[annot_ep - 1]),
                xytext=(int(n_epochs * 0.65), 6e-5),
                fontsize=9, color=PALETTE["no_aug"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["no_aug"], lw=1.1),
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=PALETTE["no_aug"], alpha=0.92))

    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlim(0, n_epochs + 2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    save(fig, fname)
    plt.show()


# FIGURE 5  ─  Summary Dashboard
def fig5_summary(runs, fname="fig5_summary.png"):
    """4-panel summary dashboard."""
    valid  = [(k, h, c) for k, (h, c) in runs.items() if h]
    if not valid:
        print("  fig5: no valid runs, skipping.")
        return

    labels = [k          for k, _, _ in valid]
    colors = [c          for _, _, c in valid]
    best_v = [best(h)[1] for _, h, _ in valid]
    gaps   = [(h["train_acc"][-1] - max(h["val_acc"])) * 100 for _, h, _ in valid]
    ep_b   = [best(h)[0] for _, h, _ in valid]
    max_epochs = max(len(h["val_acc"]) for _, h, _ in valid)
    ep90   = [next((i+1 for i, v in enumerate(h["val_acc"]) if v >= 0.9), None)
              for _, h, _ in valid]
    short  = [l.split("|")[0].strip() if "|" in l else l.split("(")[0].strip()
              for l in labels]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Experiment Summary Dashboard — CIFAR-10  ·  ResNet-18  ·  SGD",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel A: Best Val Acc bar
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(range(len(valid)), best_v, color=colors,
                   alpha=0.85, edgecolor="white", lw=1.4, width=0.55)
    ax0.set_xticks(range(len(valid)))
    ax0.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title("(a)  Best Validation Accuracy")
    # Data-driven y-limits: enough headroom for bar labels
    label_margin = (max(best_v) - min(best_v) + 2) * 0.20
    ax0.set_ylim(max(0, min(best_v) - 3),
                 min(100, max(best_v) + label_margin + 1))
    for bar, val in zip(bars, best_v):
        # Place label inside if bar is near the top limit
        ylo, yhi = ax0.get_ylim()
        label_y = bar.get_height() + (yhi - ylo) * 0.012
        ax0.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{val:.2f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 clip_on=False)

    # Panel B: Generalization gap bar
    ax1 = fig.add_subplot(gs[0, 1])
    bars2 = ax1.bar(range(len(valid)), gaps, color=colors,
                    alpha=0.85, edgecolor="white", lw=1.4, width=0.55)
    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Gap (%)")
    ax1.set_title("(b)  Final Generalization Gap  (↓ lower is better)")
    # Data-driven y-limits for gap bars
    gap_margin = (max(gaps) - min(gaps) + 1) * 0.20 if len(gaps) > 1 else max(gaps) * 0.2
    ax1.set_ylim(0, max(gaps) + gap_margin + 1)
    for bar, val in zip(bars2, gaps):
        ylo, yhi = ax1.get_ylim()
        label_y = bar.get_height() + (yhi - ylo) * 0.012
        ax1.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 clip_on=False)

    # Panel C: Val accuracy curves overlaid
    ax2 = fig.add_subplot(gs[1, 0])
    all_va = []
    for label, h, color in valid:
        x  = ep(h)
        ls = "--" if "CL" in label else "-"
        lw = 2.3 if "CL" in label else 1.8
        va = smooth(h["val_acc"]) * 100
        ax2.plot(x, va, color=color, lw=lw, ls=ls,
                 label=label.split("(")[0].strip())
        all_va.extend(va)
    vlines(ax2, max_epochs)
    set_epoch_axis(ax2, max_epochs)
    ax2.set_xlabel("Epoch", labelpad=6)
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("(c)  Validation Accuracy Curves")
    ax2.legend(fontsize=8, loc="lower right")
    if all_va:
        lo, hi = _ylim_padded([all_va], pad_frac=0.08, lo_floor=0.0, hi_ceil=100.0)
        ax2.set_ylim(lo, hi)

    # Panel D: Convergence scatter
    ax3 = fig.add_subplot(gs[1, 1])
    # Use max_epochs + 5 as fallback when 90% is never reached
    fallback_x = max_epochs + int(max_epochs * 0.05)
    x90_vals   = [ep90[i] if ep90[i] else fallback_x for i in range(len(valid))]
    for i, (label, h, color) in enumerate(valid):
        x90 = x90_vals[i]
        ax3.scatter([x90], [best(h)[1]], color=color, s=120, zorder=5,
                    edgecolors="white", lw=1.0, label=short[i])
        ax3.annotate(f"  {short[i]}", (x90, best(h)[1]),
                     fontsize=8, color=color, va="center")

    # Data-driven limits for scatter panel
    if x90_vals and best_v:
        x_lo, x_hi = _ylim_padded([x90_vals], pad_frac=0.15)
        y_lo, y_hi = _ylim_padded([best_v], pad_frac=0.10, lo_floor=0.0, hi_ceil=100.0)
        ax3.set_xlim(max(0, x_lo), x_hi)
        ax3.set_ylim(y_lo, y_hi)

    ax3.set_xlabel("Epoch to reach 90% Val Acc  (↓ faster is better)", labelpad=6)
    ax3.set_ylabel("Best Val Accuracy (%)")
    ax3.set_title("(d)  Speed vs Quality")
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    save(fig, fname)
    plt.show()


# ANALYSIS PRINTOUT
def print_analysis(runs):
    valid = {k: h for k, (h, _) in runs.items() if h}
    rows  = []
    for label, h in valid.items():
        ep_b, val_b = best(h)
        ft  = h["train_acc"][-1] * 100
        gap = ft - val_b
        n   = len(h["val_acc"])
        ep90 = next((i+1 for i, v in enumerate(h["val_acc"]) if v*100 >= 90), None)
        ep95 = next((i+1 for i, v in enumerate(h["val_acc"]) if v*100 >= 95), None)
        rows.append(dict(label=label, val_b=val_b, ep_b=ep_b,
                         ft=ft, gap=gap, n=n, ep90=ep90, ep95=ep95))

    W   = 82
    bar = "═" * W

    print(f"\n{bar}")
    print("THOROUGH ANALYSIS — CIFAR-10  ·  ResNet-18  ·  SGD  ·  MultiStepLR [49,99,124]")
    print(bar)

    # TABLE 1: Core metrics
    print("\n  TABLE 1 — Core Metrics\n")
    h1 = f"  {'Method':<42} {'BestVal':>8} {'@Ep':>5} {'TrainAcc':>9} {'Gap':>7} {'Eps':>5}"
    print(h1)
    print("  " + "─" * (W - 2))
    for r in rows:
        tag = "  ◄ baseline" if "static" in r["label"].lower() and "cl" not in r["label"].lower() else ""
        tag = "  ◄ YOUR METHOD ★" if ("CL" in r["label"] or "cl_loss" in r["label"].lower()) else tag
        print(f"  {r['label']:<42} {r['val_b']:>7.2f}%  {r['ep_b']:>4}  "
              f"{r['ft']:>8.2f}%  {r['gap']:>6.1f}%  {r['n']:>4}{tag}")
    print("  " + "─" * (W - 2))
    print("  Gap = Final Train Acc − Best Val Acc  (lower = better generalization)")

    # TABLE 2: Deltas vs static baseline
    bl = next((r for r in rows if "static" in r["label"].lower()
               and "cl" not in r["label"].lower()), None)
    if bl and len(rows) > 1:
        print(f"\n  TABLE 2 — Δ vs Static Augmentation Baseline  ({bl['val_b']:.2f}%)\n")
        print(f"  {'Method':<42} {'Δ Val':>8} {'Δ Gap':>8}  Verdict")
        print("  " + "─" * (W - 2))
        for r in rows:
            if r["label"] == bl["label"]:
                continue
            dv, dg = r["val_b"] - bl["val_b"], r["gap"] - bl["gap"]
            sv, sg = ("+" if dv >= 0 else ""), ("+" if dg >= 0 else "")
            if   dv >  1.0: verdict = "✅  Strong improvement"
            elif dv >  0.3: verdict = "✅  Marginal improvement"
            elif dv > -0.3: verdict = "➡️   On par"
            else:           verdict = "❌  Below baseline"
            print(f"  {r['label']:<42} {sv}{dv:>6.2f}%  {sg}{dg:>6.1f}%  {verdict}")

    # TABLE 3: Convergence
    print("\n  TABLE 3 — Convergence Speed\n")
    print(f"  {'Method':<42} {'→90% @':>8} {'→95% @':>8} {'Stable @':>10}")
    print("  " + "─" * (W - 2))
    for r in rows:
        s90 = f"ep {r['ep90']}" if r["ep90"] else "N/A"
        s95 = f"ep {r['ep95']}" if r["ep95"] else "N/A"
        h   = valid[r["label"]]
        best_so, last_gain = 0, 1
        for i, v in enumerate(h["val_acc"]):
            if v * 100 - best_so > 0.5:
                last_gain, best_so = i + 1, v * 100
        print(f"  {r['label']:<42} {s90:>8} {s95:>8} {f'ep {last_gain}':>10}")

    # TABLE 4: Overfitting metrics
    print("\n  TABLE 4 — Overfitting Metrics\n")
    print(f"  {'Method':<42} {'ValLoss@Best':>13} {'TrainLoss@End':>14} {'LossRatio':>10}")
    print("  " + "─" * (W - 2))
    for r in rows:
        h    = valid[r["label"]]
        ep_i = r["ep_b"] - 1
        vl_b = h["val_loss"][ep_i]
        tl_e = h["train_loss"][-1]
        ratio = vl_b / (tl_e + 1e-8)
        print(f"  {r['label']:<42} {vl_b:>12.4f}  {tl_e:>13.4f}  {ratio:>9.1f}x")

    # Key Findings
    print(f"\n{bar}")
    print("  KEY FINDINGS\n")

    no_aug = next((r for r in rows if "no_aug" in r["label"].lower()
                   or "no aug" in r["label"].lower()), None)
    cl_ms  = next((r for r in rows if ("CL" in r["label"] or "cl_loss" in r["label"].lower())
                   and "cosine" not in r["label"].lower()), None)

    idx = 1
    if no_aug and bl:
        dv = bl["val_b"] - no_aug["val_b"]
        dg = no_aug["gap"] - bl["gap"]
        print(f"  {idx}. Augmentation effect\n"
              f"     Static augmentation adds +{dv:.2f}% val accuracy over no-augmentation\n"
              f"     ({no_aug['val_b']:.2f}% → {bl['val_b']:.2f}%) and reduces the generalization\n"
              f"     gap by {dg:.1f}pp ({no_aug['gap']:.1f}% → {bl['gap']:.1f}%).\n"); idx += 1

    if bl and cl_ms:
        dv = cl_ms["val_b"] - bl["val_b"]
        dg = bl["gap"] - cl_ms["gap"]
        print(f"  {idx}. Curriculum Learning effect\n"
              f"     Loss-guided CL adds +{dv:.2f}% over static augmentation\n"
              f"     ({bl['val_b']:.2f}% → {cl_ms['val_b']:.2f}%) and reduces the\n"
              f"     generalization gap by {dg:.1f}pp.\n"); idx += 1

    if no_aug:
        print(f"  {idx}. Overfitting in no-augmentation baseline\n"
              f"     Train accuracy reaches {no_aug['ft']:.2f}% vs Val {no_aug['val_b']:.2f}%.\n"
              f"     The {no_aug['gap']:.1f}pp gap demonstrates severe memorization and motivates\n"
              f"     both augmentation and curriculum learning.\n"); idx += 1

    st_ms  = next((r for r in rows if "static" in r["label"].lower()
                   and "multistep" in r["label"].lower()), None)
    st_cos = next((r for r in rows if "static" in r["label"].lower()
                   and "cosine" in r["label"].lower()), None)
    if st_ms and st_cos:
        dv = st_ms["val_b"] - st_cos["val_b"]
        sv = "+" if dv >= 0 else ""
        print(f"  {idx}. Scheduler ablation (Static Aug)\n"
              f"     MultiStepLR vs CosineAnnealingLR: {sv}{dv:.2f}% val accuracy difference.\n"
              f"     MultiStepLR provides explicit, reproducible LR drops that are\n"
              f"     better suited for curriculum learning.\n"); idx += 1

    print(bar)
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print(bar + "\n")


# MODES
def build_runs(names_colors, checkpoint_dir):
    return {label: (load_history(name, checkpoint_dir), color)
            for label, name, color in names_colors}


def mode_baselines(ckpt):
    print("\n  Loading baseline histories...")
    runs = build_runs([
        ("No Augmentation  (MultiStep)", "resnet18_no_aug_sgd_multistep_cifar10",    PALETTE["no_aug"]),
        ("Static Aug       (MultiStep)", "resnet18_static_aug_sgd_multistep_cifar10", PALETTE["static"]),
    ], ckpt)
    print("\n  Generating 5 figures + analysis...")
    fig1_val_comparison(runs, title="Baselines")
    fig2_overfitting(runs)
    fig3_generalization(runs)
    fig4_lr_schedule()
    fig5_summary(runs)
    print_analysis(runs)


def mode_all(ckpt):
    print("\n  Loading all histories (baselines + CL)...")
    runs = build_runs([
        ("No Aug    | MultiStep",   "resnet18_no_aug_sgd_multistep_cifar10",    PALETTE["no_aug"]),
        ("Static    | MultiStep",   "resnet18_static_aug_sgd_multistep_cifar10", PALETTE["static"]),
        ("CL (Loss) | MultiStep ★", "resnet18_cl_loss_sgd_multistep_cifar10",    PALETTE["cl"]),
    ], ckpt)
    print("\n  Generating 5 figures + analysis...")
    fig1_val_comparison(runs, title="Baselines vs CL",
                        fname="fig1_val_comparison_all.png")
    fig2_overfitting(runs,    fname="fig2_overfitting_all.png")
    fig3_generalization(runs, fname="fig3_generalization_all.png")
    fig4_lr_schedule()
    fig5_summary(runs,        fname="fig5_summary_all.png")
    print_analysis(runs)


def mode_ablation(ckpt):
    print("\n  Loading ablation histories...")
    runs = build_runs([
        ("Static | MultiStep (main)", "resnet18_static_aug_sgd_multistep_cifar10", PALETTE["static"]),
        ("Static | Cosine",           "resnet18_static_aug_sgd_cosine_cifar10",    PALETTE["cosine"]),
        ("CL     | MultiStep ★",      "resnet18_cl_loss_sgd_multistep_cifar10",    PALETTE["cl"]),
        ("CL     | Cosine",           "resnet18_cl_loss_sgd_cosine_cifar10",       PALETTE["adam"]),
    ], ckpt)
    print("\n  Generating ablation figures + analysis...")
    fig1_val_comparison(runs, title="Scheduler Ablation",
                        fname="fig1_ablation.png")
    fig5_summary(runs, fname="fig5_ablation.png")
    print_analysis(runs)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publication-quality training analysis")
    parser.add_argument("--mode", default="baselines",
                        choices=["baselines", "all", "ablation"])
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    print(f"\n  Mode           : {args.mode}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print(f"  Figures dir    : {FIGURES_DIR}")

    {"baselines": mode_baselines,
     "all":       mode_all,
     "ablation":  mode_ablation}[args.mode](args.checkpoint_dir)
