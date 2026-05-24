"""
analysis/perclass_accuracy.py

Per-class accuracy breakdown for all five methods.
Reports top-10 / bottom-10 classes per method and a class-level
gain table (Static → ETS/LPS/EGS).

Run on cluster (all checkpoints present):
    python analysis/perclass_accuracy.py

Optional flags:
    --ckpt_dir  path to checkpoints folder  (default: checkpoints/)
    --data_dir  path to data folder         (default: data/raw)
    --out_dir   where to save outputs       (default: results/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import get_model

# ── Checkpoint names (seed 42, 19-op pool, 100 epochs) ───────────────────────
CHECKPOINTS = {
    "No Augmentation": "wideresnet_none_sgd_cosine_ep100_cifar100_s42_best.pth",
    "Static Mixing": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19_best.pth",
}

# Fallback names for local dev machine (different naming convention)
CHECKPOINTS_LOCAL = {
    "No Augmentation": "wideresnet_none_sgd_cosine_wr_ep100_cifar100_s42_best.pth",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_wr_ep100_cifar100_s42_p19_best.pth",
}

NORMALIZE = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

# CIFAR-100 class names in label-index order
CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model("wideresnet", num_classes=100).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_acc", 0) * 100
    print(f"  Loaded: {ckpt_path.name}  (val_acc={val_acc:.2f}%)")
    return model


def per_class_accuracy(model, loader, device, n_classes=100):
    correct = np.zeros(n_classes, dtype=int)
    total = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for c in range(n_classes):
                mask = y == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    acc = np.where(total > 0, correct / total * 100, np.nan)
    return acc


def print_top_bottom(accs, method, n=10):
    ranked = sorted(range(len(accs)), key=lambda i: accs[i], reverse=True)
    print(f"\n  Top {n} classes — {method}")
    print(f"  {'Rank':<5} {'Class':<22} {'Acc':>6}")
    print(f"  {'─' * 38}")
    for rank, idx in enumerate(ranked[:n], 1):
        print(f"  {rank:<5} {CIFAR100_CLASSES[idx]:<22} {accs[idx]:>5.1f}%")

    print(f"\n  Bottom {n} classes — {method}")
    print(f"  {'Rank':<5} {'Class':<22} {'Acc':>6}")
    print(f"  {'─' * 38}")
    for rank, idx in enumerate(reversed(ranked[-n:]), 1):
        print(f"  {rank:<5} {CIFAR100_CLASSES[idx]:<22} {accs[idx]:>5.1f}%")


def print_gain_table(base_accs, compare_accs, base_name, compare_name, n=10):
    gains = compare_accs - base_accs
    ranked_gain = sorted(range(len(gains)), key=lambda i: gains[i], reverse=True)

    print(f"\n  Largest gains: {base_name} → {compare_name}")
    print(f"  {'Class':<22} {base_name:>8} {compare_name:>8} {'Δ':>7}")
    print(f"  {'─' * 50}")
    for idx in ranked_gain[:n]:
        print(
            f"  {CIFAR100_CLASSES[idx]:<22} {base_accs[idx]:>7.1f}% {compare_accs[idx]:>7.1f}%  {gains[idx]:>+6.1f}pp"
        )

    print(f"\n  Largest regressions: {base_name} → {compare_name}")
    print(f"  {'Class':<22} {base_name:>8} {compare_name:>8} {'Δ':>7}")
    print(f"  {'─' * 50}")
    for idx in ranked_gain[-n:]:
        print(
            f"  {CIFAR100_CLASSES[idx]:<22} {base_accs[idx]:>7.1f}% {compare_accs[idx]:>7.1f}%  {gains[idx]:>+6.1f}pp"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "perclass_accuracy.txt"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # CIFAR-100 test set (no augmentation, only normalise)
    test_tf = T.Compose([T.ToTensor(), NORMALIZE])
    test_ds = CIFAR100(
        root=args.data_dir, train=False, download=False, transform=test_tf
    )
    num_workers = 4 if torch.cuda.is_available() else 0
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test set: {len(test_ds)} samples\n")

    # resolve checkpoints — try cluster names first, then local fallbacks
    resolved = {}
    for method, fname in CHECKPOINTS.items():
        p = ckpt_dir / fname
        if p.exists():
            resolved[method] = p
        elif method in CHECKPOINTS_LOCAL:
            p2 = ckpt_dir / CHECKPOINTS_LOCAL[method]
            if p2.exists():
                resolved[method] = p2
                print(f"  Using local fallback for {method}: {p2.name}")

    if not resolved:
        print("No checkpoints found. Run on the cluster or copy checkpoints locally.")
        return

    all_accs = {}
    for method, ckpt_path in resolved.items():
        print(f"\n{'=' * 60}\n  {method}\n{'=' * 60}")
        model = load_model(ckpt_path, device)
        accs = per_class_accuracy(model, loader, device)
        all_accs[method] = accs
        print(f"  Overall mean: {np.mean(accs):.2f}%")
        print_top_bottom(accs, method)

    # Per-class gain vs Static Mixing
    if "Static Mixing" in all_accs:
        print(f"\n{'=' * 60}\n  CLASS-LEVEL GAINS vs Static Mixing\n{'=' * 60}")
        for method in ["ETS", "LPS", "EGS"]:
            if method in all_accs:
                print_gain_table(
                    all_accs["Static Mixing"], all_accs[method], "Static", method
                )

    # Save raw per-class numbers
    out_lines = [
        "Per-class accuracy — WideResNet-28-10 · CIFAR-100 · seed 42 · 19-op · 100ep\n"
    ]
    header = f"{'Class':<22}" + "".join(f"  {m:>8}" for m in all_accs)
    out_lines.append(header)
    out_lines.append("─" * len(header))
    for i, cls in enumerate(CIFAR100_CLASSES):
        row = f"{cls:<22}" + "".join(
            f"  {all_accs[m][i]:>7.1f}%" if m in all_accs else f"  {'—':>8}"
            for m in all_accs
        )
        out_lines.append(row)

    out_lines.append("\nMean")
    means_row = f"{'Mean':22}" + "".join(
        f"  {np.mean(all_accs[m]):>7.2f}%" if m in all_accs else f"  {'—':>8}"
        for m in all_accs
    )
    out_lines.append(means_row)

    with open(log_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nSaved: {log_path}")


if __name__ == "__main__":
    main()
