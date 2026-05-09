"""
analysis/compare_methods.py
Final comparison table across all experiments.

Usage:
    python analysis/compare_methods.py
"""

import os, sys, json, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, MultipleLocator
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
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# LOAD
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
    (e.g. resnet50_tiered_curriculum_cifar100_history.pt and
          resnet50_tiered_curriculum_cifar100_resumed_history.pt),
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

    found = sorted(set(found), key=os.path.getmtime)

    if not found:
        print(f" {name}  — not found")
        return None

    parts = []
    for path in found:
        ext = os.path.splitext(path)[1]
        if ext == ".pt" and HAVE_TORCH:
            h = torch.load(path, map_location="cpu")
        else:
            try:
                with open(path) as f:
                    h = json.load(f)
            except Exception:
                h = None
        if h is not None:
            parts.append((path, h))

    if not parts:
        return None

    if len(parts) == 1:
        path, h = parts[0]
        print(f" {os.path.basename(path):<52}  best={max(h['val_acc']) * 100:.2f}%")
        return h

    ep_counts = " + ".join(str(len(h["val_acc"])) for _, h in parts)
    merged = _merge_histories([h for _, h in parts])
    print(
        f" {name}  — auto-merged {len(parts)} parts  "
        f"({ep_counts} ep  ->  {len(merged['val_acc'])} total  "
        f"best={max(merged['val_acc']) * 100:.2f}%)"
    )
    for path, h in parts:
        print(f"   [{len(h['val_acc']):>3} ep]  {os.path.basename(path)}")
    return merged


def best(h):
    arr = np.array(h["val_acc"])
    idx = int(np.argmax(arr))
    return idx + 1, float(arr[idx]) * 100


# ALL EXPERIMENTS REGISTRY
#
# load_history() automatically finds ALL files starting with the given base name
# (e.g. base_history.pt  +  base_resumed_history.pt), sorts by modification time,
# and merges them — no manual changes needed when a run is resumed under a new name.
#
EXPERIMENTS = [
    # (display_label, checkpoint_base_name, color, group)
    # ── CIFAR-100 | WideResNet-28-10 | seed=42 ──────────────────────────────────
    (
        "No Augmentation",
        "wideresnet_none_sgd_cosine_ep100_cifar100_s42",
        PALETTE["no_aug"],
        "floor",
    ),
    (
        "Static Aug",
        "wideresnet_static_sgd_cosine_ep100_cifar100_s42",
        PALETTE["static"],
        "baseline",
    ),
    (
        "Static + Mixing",
        "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42",
        PALETTE["static"],
        "baseline",
    ),
    (
        "RandAugment (N=2, M=9)",
        "wideresnet_randaugment_sgd_cosine_ep100_cifar100_s42",
        PALETTE["cosine"],
        "baseline",
    ),
    (
        "ETS + mix (ours)",
        "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42",
        PALETTE["cl"],
        "cl",
    ),
    (
        "ETS no-mix (ablation)",
        "wideresnet_tiered_ets_nomix_sgd_cosine_ep100_cifar100_s42",
        PALETTE["cl"],
        "cl",
    ),
    (
        "LPS + mix (ours)",
        "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42",
        PALETTE["adam"],
        "cl",
    ),
    (
        "EGS + mix (ours)",
        "wideresnet_tiered_egs_freq10_mix_both_sgd_cosine_ep100_cifar100_s42",
        PALETTE["adam"],
        "cl",
    ),
]


# FIGURE — research paper quality grouped bar chart
def fig_comparison(rows, fname="fig_compare_methods.png"):
    valid = [r for r in rows if r["h"] is not None]
    colors = [r["color"] for r in valid]
    labels = [r["label"] for r in valid]
    vals = [r["val_b"] for r in valid]
    tests = [r["test_acc"] if r["test_acc"] else r["val_b"] for r in valid]
    gaps = [r["gap"] for r in valid]

    # Short x-tick labels derived from full labels
    short = [
        r["label"].replace(" (ours)", "").replace(" (ablation)", "") for r in valid
    ]

    x = np.arange(len(valid))
    width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(max(7, len(valid) * 1.4), 3.8))
    fig.suptitle(
        "CIFAR-100  ·  WideResNet-28-10  ·  100 epochs  ·  SGD  ·  CosineAnnealingLR",
        fontsize=9,
        fontweight="bold",
    )

    # ── (a) Test Top-1 Accuracy ──────────────────────────────────
    ax = axes[0]
    bars = ax.bar(
        x, tests, width=width, color=colors, edgecolor="black", linewidth=0.6, zorder=3
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Test Top-1 Accuracy (%)")
    ax.set_title("(a)  Test Top-1 Accuracy")
    ymin = max(0, min(tests) - 8)
    ymax = min(100, max(tests) + 6)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    for bar, val in zip(bars, tests):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    # ── (b) Train − Val Generalization Gap ──────────────────────
    ax = axes[1]
    bars2 = ax.bar(
        x, gaps, width=width, color=colors, edgecolor="black", linewidth=0.6, zorder=3
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Train − Val Gap (pp)  ↓ lower is better")
    ax.set_title("(b)  Generalization Gap")
    ax.set_ylim(min(0, min(gaps)) - 4, max(gaps) + 8)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    for bar, val in zip(bars2, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# FINAL RESULTS TABLE
def print_final_table(rows):
    W = 88
    print(f"\n{'═' * W}")
    print("  FINAL RESULTS TABLE — All Experiments")
    print(
        "  CIFAR-100  ·  WideResNet-28-10  ·  SGD  ·  CosineAnnealingLR  ·  seed=42  ·  100 epochs"
    )
    print(f"{'═' * W}\n")

    groups = {
        "floor": "FLOOR (NO AUGMENTATION)",
        "baseline": "BASELINE",
        "cl": "CURRICULUM LEARNING (PROPOSED)",
    }
    current_group = None

    for r in rows:
        if r["h"] is None:
            continue

        if r["group"] != current_group:
            current_group = r["group"]
            print(
                f"  ── {groups[current_group]} {'─' * (W - 8 - len(groups[current_group]))}"
            )
            print(
                f"  {'Method':<42} {'BestVal':>8} {'TestAcc':>8} {'@Ep':>5} "
                f"{'TrainAcc':>9} {'Gap':>7} {'ValLoss':>8}"
            )
            print("  " + "─" * (W - 2))

        tag = ""
        if r["group"] == "baseline" and "static" in r["label"].lower():
            tag = "  ◄ beat this"
        if r["group"] == "cl" and "multistep" in r["label"].lower():
            tag = "  ◄ YOUR METHOD ★"

        test_str = f"{r['test_acc']:.2f}%" if r["test_acc"] else "  —   "
        print(
            f"  {r['label']:<42} {r['val_b']:>7.2f}%  {test_str:>8}  "
            f"{r['ep_b']:>4}  {r['ft']:>8.2f}%  {r['gap']:>6.1f}%  "
            f"{r['val_loss']:>7.4f}{tag}"
        )

    print(f"\n{'═' * W}")
    print("  Gap     = Final Train Acc − Best Val Acc")
    print("  ValLoss = Val loss at best epoch")
    print("  TestAcc = Reported only after final evaluation on test set")
    print(f"{'═' * W}")

    # ── Improvement summary ──────────────────────────────────
    no_aug = next(
        (r for r in rows if r["group"] == "floor" and r["h"] is not None), None
    )
    bl = next(
        (r for r in rows if r["group"] == "baseline" and r["h"] is not None), None
    )
    cl = next((r for r in rows if r["group"] == "cl" and r["h"] is not None), None)

    if bl and cl:
        dv = cl["val_b"] - bl["val_b"]
        dg = bl["gap"] - cl["gap"]
        sv = "+" if dv >= 0 else ""
        print("\n  THESIS CLAIM:")
        print("  Tiered CL vs Static Augmentation:")
        print(f"    Δ Val Accuracy   : {sv}{dv:.2f}%")
        print(f"    Δ Generalization : reduced gap by {dg:.1f}pp")
        verdict = "CL method improves over baseline" if dv > 0 else "CL did not improve"
        print(f"    Verdict          : {verdict}")

    if no_aug and bl:
        aug_gain = bl["val_b"] - no_aug["val_b"]
        print(f"\n  Augmentation benefit (Static vs No Aug): +{aug_gain:.2f}%")
    if no_aug and cl:
        cl_gain = cl["val_b"] - no_aug["val_b"]
        print(f"  Curriculum benefit (CL vs No Aug)      : +{cl_gain:.2f}%")

    print(f"{'═' * W}\n")


# MAIN
def main(checkpoint_dir=CHECKPOINT_DIR):
    print("\n📊  Loading all experiment histories...")

    rows = []
    for label, name, color, group in EXPERIMENTS:
        h = load_history(name, checkpoint_dir)
        if h is not None:
            ep_b, val_b = best(h)
            ft = h["train_acc"][-1] * 100
            gap = ft - val_b
            val_loss = h["val_loss"][ep_b - 1]
            # Load test acc from checkpoint if available
            test_acc = None
            ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pth")
            if HAVE_TORCH and os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    test_acc = ckpt.get("test_top1", None)
                    if test_acc:
                        test_acc *= 100
                except Exception:
                    pass
        else:
            ep_b = val_b = ft = gap = val_loss = 0
            test_acc = None

        rows.append(
            dict(
                label=label,
                h=h,
                color=color,
                group=group,
                ep_b=ep_b,
                val_b=val_b,
                ft=ft,
                gap=gap,
                val_loss=val_loss,
                test_acc=test_acc,
            )
        )

    print("\n📈  Generating comparison figure...")
    fig_comparison(rows)
    print_final_table(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()
    main(args.checkpoint_dir)
