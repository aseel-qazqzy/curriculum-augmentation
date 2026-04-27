"""
analysis/visualize_augmentations.py
Visualize augmented images for each curriculum tier.

Each column = one image.  Rows = Original | Tier 1 | Tier 2 | Tier 3
(or whichever tiers/modes you select).

Usage:
    # CIFAR-100 (default)
    python analysis/visualize_augmentations.py
    python analysis/visualize_augmentations.py --dataset cifar100 --aug curriculum
    python analysis/visualize_augmentations.py --dataset cifar100 --aug static
    python analysis/visualize_augmentations.py --dataset cifar100 --tier 2

    # Tiny-ImageNet
    python analysis/visualize_augmentations.py --dataset tiny_imagenet
    python analysis/visualize_augmentations.py --dataset tiny_imagenet --aug static
    python analysis/visualize_augmentations.py --dataset tiny_imagenet --tier 3

    # Control number of images shown
    python analysis/visualize_augmentations.py --n 8
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms

from data.datasets import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    CIFAR_STATS,
)

FIGURES_DIR = str(Path(__file__).resolve().parent.parent / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TIER_EPOCHS = {1: 1, 2: 40, 3: 70}
TIER_LABELS = {
    1: "Tier 1\nflip, crop",
    2: "Tier 2\n+ jitter, rotation, shear",
    3: "Tier 3\n+ grayscale, cutout",
}

DATASET_NAMES = {
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
}


# HELPERS


def get_stats(dataset):
    return CIFAR_STATS[dataset]["mean"], CIFAR_STATS[dataset]["std"]


def denorm(tensor, mean, std):
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor)


def pil_to_normed_tensor(pil, mean, std):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )(pil)


def get_raw_images(n, dataset, data_root):
    """Load n raw images as [0,1] tensors from the chosen dataset."""
    if dataset == "cifar100":
        ds = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif dataset == "tiny_imagenet":
        train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {train_dir}.\n"
                f"Run:  python data/download_tiny_imagenet.py"
            )
        ds = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
    images, _ = next(iter(loader))
    return images


def apply_transform(raw_images, transform, mean, std, seed=42):
    """
    Apply a PIL-based transform to a batch of [0,1] tensors.
    A fixed seed is used so that the same random decisions (flip, crop)
    are made consistently across all tiers — making only the newly added
    ops visually distinct in each tier row.
    """
    import random

    out = []
    for i, img in enumerate(raw_images):
        # Same seed per image index across all tiers
        random.seed(seed + i)
        torch.manual_seed(seed + i)
        pil = tensor_to_pil(img)
        result = transform(pil)
        if isinstance(result, torch.Tensor):
            out.append(result)
        else:
            out.append(pil_to_normed_tensor(result, mean, std))
    return torch.stack(out)


def show_image(ax, tensor, mean, std):
    img = denorm(tensor, mean, std).permute(1, 2, 0).numpy()
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# MAIN


def visualize(
    dataset="cifar100", aug_type="curriculum", tiers=None, n=6, data_root="data"
):
    from augmentations.policies import ThreeTierCurriculumAugmentation
    from data.datasets import get_static_transforms

    mean, std = get_stats(dataset)
    ds_label = DATASET_NAMES.get(dataset, dataset)

    print(f"\nLoading {n} {ds_label} images...")
    raw = get_raw_images(n, dataset, data_root)

    # Original images — reused as the left image in every pair
    original = torch.stack(
        [pil_to_normed_tensor(tensor_to_pil(img), mean, std) for img in raw]
    )

    # rows: (row_label, augmented_batch | None)
    # Layout per row: [orig_1 | aug_1]  [orig_2 | aug_2]  ...
    rows = []

    if aug_type == "curriculum":
        tiers_to_show = tiers or [1, 2, 3]
        policy = ThreeTierCurriculumAugmentation(dataset=dataset)
        train_tf = policy.get_train_transform()
        for tier in tiers_to_show:
            train_tf.set_epoch(TIER_EPOCHS[tier])
            rows.append((TIER_LABELS[tier], apply_transform(raw, train_tf, mean, std)))
        fname = f"aug_preview_{dataset}_curriculum_n{n}.png"
        suptitle = (
            f"Curriculum Augmentation — {ds_label}  "
            f"|  grey = Original   green = Augmented"
        )

    elif aug_type == "static":
        train_tf, _ = get_static_transforms(dataset)
        rows.append(
            ("Static Aug\n(all ops ep 1)", apply_transform(raw, train_tf, mean, std))
        )
        fname = f"aug_preview_{dataset}_static_n{n}.png"
        suptitle = (
            f"Static Augmentation — {ds_label}  |  grey = Original   green = Augmented"
        )

    else:  # none — only originals
        rows.append(("Original", None))
        fname = f"aug_preview_{dataset}_original_n{n}.png"
        suptitle = f"Original {ds_label} Images (no augmentation)"

    # FIGURE
    # Each image shown as a pair: [original | augmented]
    # So we need 2*n columns when augmented exists, n columns otherwise
    has_aug = rows[0][1] is not None
    n_cols = n * 2 if has_aug else n
    n_rows = len(rows)
    cell_size = 1.8 if dataset == "tiny_imagenet" else 1.5

    fig = plt.figure(figsize=(n_cols * cell_size + 2.0, n_rows * cell_size + 0.8))
    fig.suptitle(suptitle, fontsize=9, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.10,
        wspace=0.03,
        left=0.18,
        right=0.98,
        top=0.93,
        bottom=0.02,
    )

    for r, (label, aug_batch) in enumerate(rows):
        for i in range(n):
            if has_aug:
                # Left = original, Right = augmented
                ax_o = fig.add_subplot(gs[r, i * 2])
                ax_a = fig.add_subplot(gs[r, i * 2 + 1])
                show_image(ax_o, original[i], mean, std)
                show_image(ax_a, aug_batch[i], mean, std)
                if r == 0:
                    ax_o.set_title(
                        f"orig {i + 1}", fontsize=6.5, pad=3, color="#888888"
                    )
                    ax_a.set_title(
                        f"aug {i + 1}",
                        fontsize=6.5,
                        pad=3,
                        color="#009E73",
                        fontweight="bold",
                    )
            else:
                ax = fig.add_subplot(gs[r, i])
                show_image(ax, original[i], mean, std)
                if r == 0:
                    ax.set_title(f"img {i + 1}", fontsize=7, pad=3)

        fig.text(
            0.01,
            1 - (r + 0.5) / n_rows,
            label,
            va="center",
            ha="left",
            fontsize=8,
            fontweight="normal",
            color="#333333",
        )

    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  -> saved: {path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar100", "tiny_imagenet"],
        help="Dataset to load images from (default: cifar100)",
    )
    parser.add_argument(
        "--aug",
        default="curriculum",
        choices=["curriculum", "static", "none"],
        help="Augmentation type to visualize (default: curriculum)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Show only this curriculum tier. Omit to show all 3.",
    )
    parser.add_argument(
        "--n", type=int, default=6, help="Number of image columns (default: 6)"
    )
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()

    # --tier 3 shows tiers 1, 2, and 3 so you can see the full cumulative progression
    tiers = list(range(1, args.tier + 1)) if args.tier else None
    visualize(
        dataset=args.dataset,
        aug_type=args.aug,
        tiers=tiers,
        n=args.n,
        data_root=args.data_root,
    )
