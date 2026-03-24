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
FIGURES_DIR    = str(_PROJECT_ROOT / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# PALETTE & STYLE
PALETTE = {
    "no_aug":  "#D62728",
    "static":  "#1F77B4",
    "cl":      "#2CA02C",
    "cosine":  "#FF7F0E",
    "adam":    "#9467BD",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "legend.fontsize":   10,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#CCCCCC",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# LOAD
def load_history(name, checkpoint_dir=CHECKPOINT_DIR):
    for ext in [".pt", ".json"]:
        path = os.path.join(checkpoint_dir, f"{name}_history{ext}")
        if not os.path.exists(path):
            continue
        if ext == ".pt" and HAVE_TORCH:
            h = torch.load(path, map_location="cpu")
        else:
            with open(path) as f:
                h = json.load(f)
        print(f" {name:<48}  best={max(h['val_acc'])*100:.2f}%")
        return h
    print(f" {name}  — not found")
    return None

def best(h):
    arr = np.array(h["val_acc"])
    idx = int(np.argmax(arr))
    return idx + 1, float(arr[idx]) * 100



# ALL EXPERIMENTS REGISTRY
EXPERIMENTS = [
    # (display_label, checkpoint_name, color, group)
    ("No Aug    | SGD | MultiStep",   "resnet18_no_aug_sgd_multistep_cifar10",    PALETTE["no_aug"], "baseline"),
    ("Static    | SGD | MultiStep",   "resnet18_static_aug_sgd_multistep_cifar10", PALETTE["static"], "baseline"),
    ("Static    | SGD | Cosine",      "resnet18_static_aug_sgd_cosine_cifar10",    PALETTE["cosine"], "ablation"),
    ("Static    | Adam| MultiStep",   "resnet18_static_aug_adam_multistep_cifar10",PALETTE["adam"],   "ablation"),
    ("CL (Loss) | SGD | MultiStep ★", "resnet18_cl_loss_sgd_multistep_cifar10",    PALETTE["cl"],     "cl"),
    ("CL (Loss) | SGD | Cosine",      "resnet18_cl_loss_sgd_cosine_cifar10",       PALETTE["cosine"], "cl"),
]


# FIGURE — grouped bar chart comparison
def fig_comparison(rows, fname="fig_compare_methods.png"):
    valid  = [r for r in rows if r["h"] is not None]
    labels = [r["label"] for r in valid]
    colors = [r["color"] for r in valid]
    vals   = [r["val_b"]  for r in valid]
    gaps   = [r["gap"]    for r in valid]
    short  = [l.split("|")[0].strip() for l in labels]

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        "Method Comparison — CIFAR-10  ·  ResNet-18  ·  All Experiments",
        fontsize=14, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # Val Accuracy
    ax0 = fig.add_subplot(gs[0])
    bars = ax0.bar(range(len(valid)), vals, color=colors, alpha=0.85, edgecolor="white", lw=1.4, width=0.6)
    ax0.set_xticks(range(len(valid)))
    ax0.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_title("(a)  Best Validation Accuracy")
    # Data-driven limits: enough headroom for bar labels
    val_span    = max(max(vals) - min(vals), 2)
    val_top     = min(100, max(vals) + val_span * 0.25)
    ax0.set_ylim(max(0, min(vals) - val_span * 0.15), val_top)
    ax0.yaxis.set_major_locator(MultipleLocator(max(1, round(val_span / 8))))
    for bar, val in zip(bars, vals):
        ylo, yhi = ax0.get_ylim()
        label_y = bar.get_height() + (yhi - ylo) * 0.012
        ax0.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{val:.2f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", clip_on=False)

    # Generalization gap
    ax1 = fig.add_subplot(gs[1])
    bars2 = ax1.bar(range(len(valid)), gaps, color=colors, alpha=0.85, edgecolor="white", lw=1.4, width=0.6)
    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Gap (%)")
    ax1.set_title("(b)  Generalization Gap  (Train − Val  ↓ lower is better)")
    gap_span = max(max(gaps) - min(gaps), 1)
    ax1.set_ylim(0, max(gaps) + gap_span * 0.25)
    for bar, val in zip(bars2, gaps):
        ylo, yhi = ax1.get_ylim()
        label_y = bar.get_height() + (yhi - ylo) * 0.012
        ax1.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", clip_on=False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path)
    plt.show()
    print(f"  → saved: {fname}")


# FINAL RESULTS TABLE
def print_final_table(rows):
    W = 88
    print(f"\n{'═'*W}")
    print("  FINAL RESULTS TABLE — All Experiments")
    print("  CIFAR-10  ·  ResNet-18  ·  SGD  ·  seed=42  ·  150 epochs")
    print(f"{'═'*W}\n")

    groups = {"baseline": "BASELINES", "ablation": "ABLATIONS", "cl": "CURRICULUM LEARNING"}
    current_group = None

    for r in rows:
        if r["h"] is None:
            continue

        if r["group"] != current_group:
            current_group = r["group"]
            print(f"  ── {groups[current_group]} {'─'*(W-8-len(groups[current_group]))}")
            print(f"  {'Method':<42} {'BestVal':>8} {'TestAcc':>8} {'@Ep':>5} "
                  f"{'TrainAcc':>9} {'Gap':>7} {'ValLoss':>8}")
            print("  " + "─" * (W - 2))

        tag = ""
        if r["group"] == "baseline" and "static" in r["label"].lower():
            tag = "  ◄ beat this"
        if r["group"] == "cl" and "multistep" in r["label"].lower():
            tag = "  ◄ YOUR METHOD ★"

        test_str = f"{r['test_acc']:.2f}%" if r["test_acc"] else "  —   "
        print(f"  {r['label']:<42} {r['val_b']:>7.2f}%  {test_str:>8}  "
              f"{r['ep_b']:>4}  {r['ft']:>8.2f}%  {r['gap']:>6.1f}%  "
              f"{r['val_loss']:>7.4f}{tag}")

    print(f"\n{'═'*W}")
    print("  Gap     = Final Train Acc − Best Val Acc")
    print("  ValLoss = Val loss at best epoch")
    print("  TestAcc = Reported only after final evaluation on test set")
    print(f"{'═'*W}")

    # ── Improvement summary ──────────────────────────────────
    bl = next((r for r in rows if "static" in r["label"].lower()
               and "multistep" in r["label"].lower()
               and "cl" not in r["label"].lower()
               and r["h"] is not None), None)
    cl = next((r for r in rows if "CL" in r["label"]
               and "multistep" in r["label"].lower()
               and r["h"] is not None), None)

    if bl and cl:
        dv  = cl["val_b"] - bl["val_b"]
        dg  = bl["gap"]   - cl["gap"]
        sv  = "+" if dv >= 0 else ""
        print("\n  THESIS CLAIM:")
        print("  CL (Loss-Guided) vs Static Augmentation:")
        print(f"    Δ Val Accuracy   : {sv}{dv:.2f}%")
        print(f"    Δ Generalization : reduced gap by {dg:.1f}pp")
        verdict = "✅ CL method improves over baseline" if dv > 0 else "❌ CL did not improve"
        print(f"    Verdict          : {verdict}")
        print(f"{'═'*W}\n")



# MAIN
def main(checkpoint_dir=CHECKPOINT_DIR):
    print("\n📊  Loading all experiment histories...")

    rows = []
    for label, name, color, group in EXPERIMENTS:
        h = load_history(name, checkpoint_dir)
        if h is not None:
            ep_b, val_b = best(h)
            ft      = h["train_acc"][-1] * 100
            gap     = ft - val_b
            val_loss = h["val_loss"][ep_b - 1]
            # Load test acc from checkpoint if available
            test_acc = None
            ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pth")
            if HAVE_TORCH and os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    test_acc = ckpt.get("test_acc", None)
                    if test_acc:
                        test_acc *= 100
                except Exception:
                    pass
        else:
            ep_b = val_b = ft = gap = val_loss = 0
            test_acc = None

        rows.append(dict(label=label, h=h, color=color, group=group,
                         ep_b=ep_b, val_b=val_b, ft=ft, gap=gap,
                         val_loss=val_loss, test_acc=test_acc))

    print("\n📈  Generating comparison figure...")
    fig_comparison(rows)
    print_final_table(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()
    main(args.checkpoint_dir)