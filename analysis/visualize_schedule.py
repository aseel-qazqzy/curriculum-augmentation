"""
analysis/visualize_schedule.py
Visualizes the curriculum learning schedule — how augmentation
difficulty increases over training epochs.

Usage:
    python analysis/visualize_schedule.py
"""

import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = "./results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "legend.fontsize": 10,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

PALETTE = {
    "static": "#1F77B4",
    "cl": "#2CA02C",
    "easy": "#A8D5A2",
    "medium": "#F9C74F",
    "hard": "#F3722C",
    "lr": "#D62728",
}

N_EPOCHS = 100

# Actual ETS tier boundaries (t1=0.20, t2=0.45)
TIER_T1 = int(N_EPOCHS * 0.20)  # epoch 20
TIER_T2 = int(N_EPOCHS * 0.45)  # epoch 45
STRENGTH_RAMP = 5  # epochs for linear ramp at each boundary
TIER_STRENGTHS = (0.40, 0.70, 1.00)  # Tier 1/2/3 as fraction of strength ceiling


# CURRICULUM SCHEDULES
def ets_difficulty(epoch, n=N_EPOCHS, t1=0.20, t2=0.45, ramp=STRENGTH_RAMP):
    """
    Actual ETS schedule used in all experiments.
    Tier strengths: 40% / 70% / 100% of ceiling.
    5-epoch linear ramp at each tier boundary.
    """
    ep1 = int(n * t1)
    ep2 = int(n * t2)
    s1, s2, s3 = TIER_STRENGTHS
    if epoch <= ep1:
        return s1
    elif epoch <= ep1 + ramp:
        return s1 + (epoch - ep1) / ramp * (s2 - s1)
    elif epoch <= ep2:
        return s2
    elif epoch <= ep2 + ramp:
        return s2 + (epoch - ep2) / ramp * (s3 - s2)
    else:
        return s3


def cosine_lr(epoch, n=N_EPOCHS, lr_init=0.1, eta_min=1e-6, warmup=5):
    """CosineAnnealingLR + linear warmup — matches actual training runs."""
    if epoch <= warmup:
        return lr_init * (0.1 + 0.9 * epoch / warmup)
    t = (epoch - warmup) / max(1, n - warmup)
    return eta_min + 0.5 * (lr_init - eta_min) * (1 + np.cos(np.pi * t))


def linear_difficulty(epoch, n=N_EPOCHS):
    return min(epoch / n, 1.0)


def step_difficulty(epoch, n=N_EPOCHS):
    if epoch < int(n * 0.33):
        return 0.0
    if epoch < int(n * 0.66):
        return 0.33
    if epoch < int(n * 0.83):
        return 0.66
    return 1.0


def static_difficulty(epoch, n=N_EPOCHS):
    return 0.5  # flat — same augmentation every epoch


# AUGMENTATION PARAMETER SCHEDULES
def aug_params_over_time(epochs):
    """
    Simulate how individual augmentation parameters scale with CL difficulty.
    Returns dict of param_name → list of values per epoch.
    """
    diff = np.array([ets_difficulty(e, epochs[-1]) for e in epochs])

    return {
        "ColorJitter brightness": 0.1 + 0.5 * diff,
        "ColorJitter contrast": 0.05 + 0.3 * diff,
        "RandomCrop padding": 1 + 3 * diff,  # 1 → 4
        "Rotation (°)": 0 + 15 * diff,
        "Cutout probability": 0 + 0.5 * diff,
    }


# FIGURE 1 — Schedule comparison
def fig_schedule_comparison(fname="figS1_schedule_comparison.png"):
    epochs = np.arange(1, N_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Curriculum Learning Schedule Comparison — CIFAR-100",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    schedules = {
        "ETS (ours)": (
            [ets_difficulty(e) for e in epochs],
            PALETTE["cl"],
            "-",
            2.8,
        ),
        "Linear": (
            [linear_difficulty(e) for e in epochs],
            PALETTE["static"],
            "--",
            1.8,
        ),
        "Step": ([step_difficulty(e) for e in epochs], PALETTE["lr"], ":", 1.8),
        "Static (baseline)": (
            [static_difficulty(e) for e in epochs],
            "#888888",
            "-.",
            1.8,
        ),
    }

    for label, (vals, color, ls, lw) in schedules.items():
        axes[0].plot(epochs, vals, color=color, lw=lw, ls=ls, label=label)

    # Shade tiers
    axes[0].axvspan(1, TIER_T1, alpha=0.06, color=PALETTE["easy"], label="_")
    axes[0].axvspan(TIER_T1, TIER_T2, alpha=0.06, color=PALETTE["medium"], label="_")
    axes[0].axvspan(TIER_T2, N_EPOCHS, alpha=0.06, color=PALETTE["hard"], label="_")

    for m in [TIER_T1, TIER_T2]:
        axes[0].axvline(m, color="#888888", lw=1.0, ls=":", alpha=0.4)

    axes[0].set_xlabel("Epoch", labelpad=6)
    axes[0].set_ylabel("Augmentation Difficulty  (0 = easy, 1 = hard)")
    axes[0].set_title("(a)  Schedule Strategies")
    axes[0].set_ylim(-0.05, 1.10)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # Cumulative hard samples (area under curve = total hard exposure)
    for label, (vals, color, ls, lw) in schedules.items():
        cumulative = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        axes[1].plot(epochs, cumulative, color=color, lw=lw, ls=ls, label=label)

    axes[1].set_xlabel("Epoch", labelpad=6)
    axes[1].set_ylabel("Mean Difficulty So Far")
    axes[1].set_title("(b)  Cumulative Difficulty Exposure")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # Tier labels on both axes
    phase_labels = [
        (1, TIER_T1, "Tier 1", PALETTE["easy"]),
        (TIER_T1, TIER_T2, "Tier 2", PALETTE["medium"]),
        (TIER_T2, N_EPOCHS, "Tier 3", PALETTE["hard"]),
    ]
    for ax in axes:
        for start, end, label, color in phase_labels:
            ax.text(
                (start + end) / 2,
                1.06,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# FIGURE 2 — Augmentation parameters over time
def fig_aug_params(fname="figS2_aug_params.png"):
    epochs = np.arange(1, N_EPOCHS + 1)
    params = aug_params_over_time(epochs)

    colors = ["#1F77B4", "#2CA02C", "#D62728", "#FF7F0E", "#9467BD"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Augmentation Parameter Progression — ETS 3-Tier Curriculum Schedule",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # Left: all params overlaid (normalised 0-1)
    for i, (name, vals) in enumerate(params.items()):
        norm = (np.array(vals) - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-8)
        axes[0].plot(epochs, norm, color=colors[i], lw=2.0, label=name)

    for m in [TIER_T1, TIER_T2]:
        axes[0].axvline(m, color="#888888", lw=1.0, ls=":", alpha=0.4)

    axes[0].set_xlabel("Epoch", labelpad=6)
    axes[0].set_ylabel("Normalised Parameter Strength  (0 → 1)")
    axes[0].set_title("(a)  All Parameters — Normalised")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylim(-0.05, 1.12)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # Right: absolute values for key params
    key_params = ["ColorJitter brightness", "Rotation (°)"]
    ax2 = axes[1]
    ax3 = ax2.twinx()

    ax2.plot(
        epochs,
        params["ColorJitter brightness"],
        color="#1F77B4",
        lw=2.2,
        label="ColorJitter brightness (left)",
    )
    ax2.plot(
        epochs,
        params["Cutout probability"],
        color="#2CA02C",
        lw=2.2,
        ls="--",
        label="Cutout probability (left)",
    )
    ax3.plot(
        epochs,
        params["Rotation (°)"],
        color="#D62728",
        lw=2.2,
        ls=":",
        label="Rotation ° (right)",
    )

    for m in [TIER_T1, TIER_T2]:
        ax2.axvline(m, color="#888888", lw=1.0, ls=":", alpha=0.4)

    ax2.set_xlabel("Epoch", labelpad=6)
    ax2.set_ylabel("Jitter / Probability")
    ax3.set_ylabel("Rotation (°)")
    ax2.set_title("(b)  Key Parameters — Absolute Values")
    ax2.set_ylim(0, 0.8)
    ax3.set_ylim(0, 20)

    lines2, labs2 = ax2.get_legend_handles_labels()
    lines3, labs3 = ax3.get_legend_handles_labels()
    ax2.legend(lines2 + lines3, labs2 + labs3, fontsize=8, loc="upper left")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax2.spines["top"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# FIGURE 3 — LR + Difficulty combined (thesis key figure)
def fig_lr_and_difficulty(fname="figS3_lr_difficulty_combined.png"):
    epochs = np.arange(1, N_EPOCHS + 1)

    lr = np.array([cosine_lr(e) for e in epochs])
    diff = np.array([ets_difficulty(e) for e in epochs])

    fig, ax1 = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "Learning Rate & CL Difficulty Schedule — Combined View  ·  CosineAnnealingLR + ETS 3-Tier CL",
        fontsize=13,
        fontweight="bold",
    )

    ax2 = ax1.twinx()

    ax1.plot(epochs, lr, color=PALETTE["lr"], lw=2.5, label="Learning Rate (left)")
    ax2.plot(epochs, diff, color=PALETTE["cl"], lw=2.5, label="CL Difficulty (right)")
    ax2.fill_between(epochs, 0, diff, alpha=0.08, color=PALETTE["cl"])

    # Tier shading
    ax1.axvspan(1, TIER_T1, alpha=0.05, color=PALETTE["easy"])
    ax1.axvspan(TIER_T1, TIER_T2, alpha=0.05, color=PALETTE["medium"])
    ax1.axvspan(TIER_T2, N_EPOCHS, alpha=0.05, color=PALETTE["hard"])

    for m in [TIER_T1, TIER_T2]:
        ax1.axvline(m, color="#888888", lw=1.2, ls=":", alpha=0.5)
        ax1.text(m + 1, 0.085, f"ep {m}", fontsize=8, color="#888888")

    # Annotations
    ax2.annotate(
        "Tier 3 active\n(full augmentation)",
        xy=(N_EPOCHS, diff[-1]),
        xytext=(int(N_EPOCHS * 0.75), 0.55),
        fontsize=9,
        color=PALETTE["cl"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["cl"], lw=1.1),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PALETTE["cl"], alpha=0.9),
    )

    ax1.annotate(
        "LR ≈ 0\n(cosine end)",
        xy=(N_EPOCHS, lr[-1]),
        xytext=(int(N_EPOCHS * 0.75), 0.003),
        fontsize=9,
        color=PALETTE["lr"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["lr"], lw=1.1),
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PALETTE["lr"], alpha=0.9),
    )

    # Tier labels
    for start, end, label, color in [
        (1, TIER_T1, "Tier 1", PALETTE["easy"]),
        (TIER_T1, TIER_T2, "Tier 2", PALETTE["medium"]),
        (TIER_T2, N_EPOCHS, "Tier 3", PALETTE["hard"]),
    ]:
        ax1.text(
            (start + end) / 2,
            0.075,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax1.set_xlabel("Epoch", labelpad=6)
    ax1.set_ylabel("Learning Rate", color=PALETTE["lr"])
    ax2.set_ylabel("Augmentation Difficulty", color=PALETTE["cl"])
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 0.5)
    ax2.set_ylim(-0.05, 1.15)
    ax1.set_xlim(0, N_EPOCHS + 2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax1.tick_params(axis="y", colors=PALETTE["lr"])
    ax2.tick_params(axis="y", colors=PALETTE["cl"])
    ax1.spines["left"].set_color(PALETTE["lr"])
    ax2.spines["right"].set_color(PALETTE["cl"])
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center left", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# ─────────────────────────────────────────────────────────────
# PRINT SCHEDULE ANALYSIS
# ─────────────────────────────────────────────────────────────
def print_schedule_analysis():
    W = 80
    print(f"\n{'═' * W}")
    print(f"  CL SCHEDULE ANALYSIS  ·  ETS 3-Tier Curriculum  ·  {N_EPOCHS} epochs")
    print(f"{'═' * W}\n")

    epochs = list(range(1, N_EPOCHS + 1))
    diff = [ets_difficulty(e) for e in epochs]
    # checkpoints: tier boundaries + even fractions, all clamped to [1, N_EPOCHS]
    raw = [
        1,
        int(N_EPOCHS * 0.10),
        TIER_T1,
        TIER_T1 + STRENGTH_RAMP,
        int(N_EPOCHS * 0.35),
        TIER_T2,
        TIER_T2 + STRENGTH_RAMP,
        int(N_EPOCHS * 0.75),
        N_EPOCHS,
    ]
    checkpoints = sorted(set(max(1, min(e, N_EPOCHS)) for e in raw))

    print(f"  {'Epoch':>6}  {'Difficulty':>11}  {'Tier':>8}  {'Augmentation Level'}")
    print("  " + "─" * 65)

    for ep in checkpoints:
        d = diff[ep - 1]
        if ep <= TIER_T1:
            tier, aug = "Tier 1", "flip, crop, translate"
        elif ep <= TIER_T2:
            tier, aug = "Tier 2", "+ jitter, rotation, shear"
        else:
            tier, aug = "Tier 3", "+ cutout, grayscale, contrast"

        bar = "█" * int(d * 20)
        print(f"  {ep:>6}  {d:>10.3f}  {tier:>8}  {bar}")

    print(f"\n  ETS tier boundaries: ep {TIER_T1} (t1=0.20), ep {TIER_T2} (t2=0.45)")
    print(f"  Strength ramp: {STRENGTH_RAMP} epochs linear at each boundary.\n")

    print(
        "  "
        + "Schedule".ljust(20)
        + "Mean Diff".rjust(10)
        + "Final Diff".rjust(11)
        + "Ramp Style"
    )
    print("  " + "─" * 60)
    schedules = {
        "ETS (ours)": [ets_difficulty(e) for e in epochs],
        "Linear": [linear_difficulty(e) for e in epochs],
        "Step": [step_difficulty(e) for e in epochs],
        "Static": [static_difficulty(e) for e in epochs],
    }
    ramp_styles = {
        "ETS (ours)": f"Step + {STRENGTH_RAMP}-ep linear ramp at tier boundaries",
        "Linear": "Constant rate from ep 1 to ep 100",
        "Step": "Discrete jumps at milestones",
        "Static": "Flat — no curriculum",
    }
    for name, vals in schedules.items():
        mean_d = np.mean(vals)
        final_d = vals[-1]
        print(f"  {name:<20} {mean_d:>10.3f} {final_d:>11.3f}  {ramp_styles[name]}")

    print(f"\n{'═' * W}\n")


# MAIN
if __name__ == "__main__":
    print("\n📈  Generating curriculum schedule visualizations...")
    fig_schedule_comparison()
    fig_aug_params()
    fig_lr_and_difficulty()
    print_schedule_analysis()
    print(f"  All figures saved to: {os.path.abspath(FIGURES_DIR)}\n")
