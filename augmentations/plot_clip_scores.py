"""augmentations/plot_clip_scores.py — visualise CLIP augmentation difficulty scores.

Usage:
    python -m augmentations.plot_clip_scores --dataset cifar100
    python -m augmentations.plot_clip_scores --dataset tiny_imagenet
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Colorblind-safe palette (Wong 2011)
_T1_COLOR = "#0072B2"  # blue
_T2_COLOR = "#E69F00"  # amber
_T3_COLOR = "#D55E00"  # vermilion
_MIX_COLOR = "#CC79A7"  # pink

_TIER_COLORS = {
    "flip": _T1_COLOR,
    "crop": _T1_COLOR,
    "translate_x": _T1_COLOR,
    "translate_y": _T1_COLOR,
    "auto_contrast": _T2_COLOR,
    "equalize": _T2_COLOR,
    "sharpness": _T2_COLOR,
    "color_jitter": _T2_COLOR,
    "rotation": _T2_COLOR,
    "shear": _T2_COLOR,
    "perspective": _T2_COLOR,
    "grayscale": _T3_COLOR,
    "contrast": _T3_COLOR,
    "brightness": _T3_COLOR,
    "blur": _T3_COLOR,
    "cutout": _T3_COLOR,
    "invert": _T3_COLOR,
    "solarize": _T3_COLOR,
    "posterize": _T3_COLOR,
}

# Human-readable labels
_LABELS = {
    "flip": "Flip",
    "crop": "Crop",
    "translate_x": "Translate X",
    "translate_y": "Translate Y",
    "auto_contrast": "Auto Contrast",
    "equalize": "Equalize",
    "sharpness": "Sharpness",
    "color_jitter": "Color Jitter",
    "rotation": "Rotation",
    "shear": "Shear",
    "perspective": "Perspective",
    "grayscale": "Grayscale",
    "contrast": "Contrast",
    "brightness": "Brightness",
    "blur": "Blur",
    "cutout": "Cutout",
    "invert": "Invert",
    "solarize": "Solarize",
    "posterize": "Posterize",
    "cutmix": "CutMix",
    "mixup": "MixUp",
}


def plot(dataset: str, scores_path: str, output_dir: str) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    data = torch.load(scores_path, map_location="cpu", weights_only=False)
    scores = data["scores"]
    ranked = data["ranked"]
    mixing_scores = data.get("mixing_scores", {})

    n_mix = len(mixing_scores)

    # ── group by tier, sort by CLIP score within each tier ────────────────────
    _T1 = {"flip", "crop", "translate_x", "translate_y"}
    _T2 = {
        "auto_contrast",
        "equalize",
        "sharpness",
        "color_jitter",
        "rotation",
        "shear",
        "perspective",
    }
    _T3 = set(scores.keys()) - _T1 - _T2

    def _sorted(group):
        return sorted([op for op in group if op in scores], key=lambda o: scores[o])

    ordered = _sorted(_T1) + _sorted(_T2) + _sorted(_T3)
    ordered += list(mixing_scores.keys())

    n_ops = len(ordered) - n_mix  # ops only (excl. mixing)
    ops = ordered
    vals = [scores.get(op, mixing_scores.get(op, 0.0)) for op in ops]
    colors = [_TIER_COLORS.get(op, _MIX_COLOR) for op in ops]
    labels = [_LABELS.get(op, op) for op in ops]

    # tier boundary positions (for horizontal dividers)
    t1_end = len(_sorted(_T1)) - 0.5
    t2_end = len(_sorted(_T1)) + len(_sorted(_T2)) - 0.5

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))

    y = list(range(len(ops)))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.72)

    # light vertical grid on x-axis only
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="gray")
    ax.set_axisbelow(True)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("CLIP Semantic Distance  (1 − cosine similarity)", fontsize=11)

    ds_label = dataset.replace("_", "-").upper()
    ax.set_title(
        f"CLIP-Validated Augmentation Difficulty\n"
        f"WideResNet-28-10  ·  {ds_label}  ·  strength = {data['strength']}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # tier dividers (solid thin lines between T1/T2, T2/T3, and ops/mixing)
    for boundary in (t1_end, t2_end):
        ax.axhline(boundary, color="#888888", linestyle="-", linewidth=0.8, alpha=0.5)
    if n_mix:
        ax.axhline(n_ops - 0.5, color="#aaaaaa", linestyle=":", linewidth=1.0)

    # ── legend below the plot ─────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=_T1_COLOR, label="Tier 1 — Geometric"),
        mpatches.Patch(color=_T2_COLOR, label="Tier 2 — Colour / Texture"),
        mpatches.Patch(color=_T3_COLOR, label="Tier 3 — Information Removal"),
    ]
    if n_mix:
        patches.append(mpatches.Patch(color=_MIX_COLOR, label="Batch Mixing (Tier 3)"))

    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # ── save ─────────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = out / f"clip_difficulty_{dataset}.{fmt}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved → {path}")

    plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["cifar100", "tiny_imagenet"]
    )
    parser.add_argument("--scores_dir", default="./configs")
    parser.add_argument("--output_dir", default="./results/figs/clip_validation")
    args = parser.parse_args()

    scores_path = Path(args.scores_dir) / f"aug_difficulty_scores_{args.dataset}.pt"
    if not scores_path.exists():
        print(
            f"Scores file not found: {scores_path}\n"
            f"Run first: python -m augmentations.clip_calibration --dataset {args.dataset}"
        )
        sys.exit(1)

    print(f"\nPlotting CLIP difficulty scores for {args.dataset}...")
    plot(args.dataset, str(scores_path), args.output_dir)


if __name__ == "__main__":
    main()
