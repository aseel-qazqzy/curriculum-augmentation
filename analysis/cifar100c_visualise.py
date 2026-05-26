"""
analysis/cifar100c_visualise.py

Show sample images from CIFAR-100-C for a fixed image index across
selected corruptions and all 5 severity levels.

Run:
    python analysis/cifar100c_visualise.py --c_root data/CIFAR-100-C
Output:
    results/figs/cifar100c_samples.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CORRUPTIONS_TO_SHOW = [
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "fog",
    "frost",
    "snow",
    "brightness",
    "contrast",
    "pixelate",
    "jpeg_compression",
]

SEVERITIES = [1, 2, 3, 4, 5]
IMG_IDX = 42  # which image to show (0–9999 within the clean set)


def load_image(c_root, corruption, severity, idx):
    data = np.load(c_root / f"{corruption}.npy")
    offset = (severity - 1) * 10000
    return data[offset + idx]  # (32, 32, 3) uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_root", default="data/CIFAR-100-C")
    parser.add_argument("--idx", type=int, default=IMG_IDX)
    args = parser.parse_args()

    c_root = Path(args.c_root)
    save_dir = Path("results/figs")
    save_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(CORRUPTIONS_TO_SHOW)
    n_cols = len(SEVERITIES) + 1  # +1 for label column

    fig, axes = plt.subplots(
        n_rows, len(SEVERITIES), figsize=(len(SEVERITIES) * 1.8, n_rows * 1.8)
    )
    fig.suptitle(
        f"CIFAR-100-C sample corruptions — image index {args.idx}\n"
        "Columns: severity 1 (mild) → 5 (severe)",
        fontsize=11,
        fontweight="bold",
    )

    for r, corruption in enumerate(CORRUPTIONS_TO_SHOW):
        c_file = c_root / f"{corruption}.npy"
        if not c_file.exists():
            print(f"  Skipping {corruption} — file not found")
            continue
        for c, sev in enumerate(SEVERITIES):
            img = load_image(c_root, corruption, sev, args.idx)
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_title(corruption.replace("_", "\n"), fontsize=7, loc="left", pad=2)
            if r == 0:
                ax.set_title(f"s{sev}", fontsize=8)

    plt.tight_layout()
    out = save_dir / "cifar100c_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
