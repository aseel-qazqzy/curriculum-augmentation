"""augmentations/clip_calibration.py — offline CLIP augmentation difficulty calibration.

Runs ONCE before training to measure how semantically disruptive each
augmentation operation is, using a frozen CLIP ViT-B/32.

Scores both individual ops AND batch mixing strategies (CutMix, MixUp):

  Per-op scoring:
    1. Apply op at fixed strength to 1000 random training images.
    2. Compute 1 - cosine_similarity(CLIP(original), CLIP(augmented)).
    3. Average across images → per-op difficulty score.

  Mixing scoring:
    1. Sample random image pairs from the 1000 images.
    2. Apply CutMix / MixUp (alpha=1.0, same as training default).
    3. Score 1 - cosine_similarity(CLIP(image_A), CLIP(mixed_A_B)).
    4. Average across pairs → per-mixing-strategy difficulty score.

Results saved to configs/aug_difficulty_scores_{dataset}.pt.

Usage:
    python -m augmentations.clip_calibration --dataset cifar100
    python -m augmentations.clip_calibration --dataset tiny_imagenet --n_images 2000
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentations.clip_scorer import CLIPDifficultyScorer
from augmentations.policies import _TIER_OPS
from augmentations.primitives import AUGMENTATION_REGISTRY
from experiments.utils import get_device, set_seed


# ── defaults ──────────────────────────────────────────────────────────────────
_N_IMAGES = 1000
_STRENGTH = 0.7  # ceiling strength — same as Tier 3 in training
_BATCH = 64  # images per CLIP forward pass


# ── dataset loading ───────────────────────────────────────────────────────────


def _load_pil_images(dataset: str, root: str, n: int, seed: int) -> List[Image.Image]:
    """Return n random training PIL images from the requested dataset."""
    from torchvision import datasets as tvd

    rng = random.Random(seed)

    if dataset == "cifar100":
        ds = tvd.CIFAR100(root, train=True, download=True, transform=None)
    elif dataset == "tiny_imagenet":
        train_dir = Path(root) / "tiny-imagenet-200" / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {train_dir}. "
                "Download and extract it first."
            )
        ds = tvd.ImageFolder(str(train_dir), transform=None)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose cifar100 or tiny_imagenet."
        )

    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    images = []
    for idx in indices:
        img, _ = ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    return images


# ── per-op scoring ────────────────────────────────────────────────────────────


def _score_op(
    op_name: str,
    images: List[Image.Image],
    scorer: CLIPDifficultyScorer,
    strength: float,
    batch_size: int,
) -> float:
    """Apply op to all images, score with CLIP, return mean difficulty."""
    fn, _, _ = AUGMENTATION_REGISTRY[op_name]
    all_scores: List[torch.Tensor] = []

    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        orig_tensors, aug_tensors = [], []

        for img in chunk:
            try:
                aug = fn(img.copy(), strength)
                orig_tensors.append(TF.to_tensor(img))  # [0, 1]
                aug_tensors.append(TF.to_tensor(aug))  # [0, 1]
            except Exception:
                continue  # skip images where the op fails

        if not orig_tensors:
            continue

        orig_batch = torch.stack(orig_tensors)  # (B, 3, H, W)
        aug_batch = torch.stack(aug_tensors)  # (B, 3, H, W)
        all_scores.append(scorer(orig_batch, aug_batch).cpu())

    if not all_scores:
        return 0.0
    return float(torch.cat(all_scores).mean())


# ── mixing scoring ────────────────────────────────────────────────────────────


def _score_mixing(
    mix_mode: str,
    images: List[Image.Image],
    scorer: CLIPDifficultyScorer,
    batch_size: int,
    alpha: float = 1.0,
) -> float:
    """Score CutMix or MixUp by comparing CLIP(image_A) vs CLIP(mixed_A_B).

    Args:
        mix_mode  : "cutmix" or "mixup"
        images    : list of PIL images — random pairs are drawn from this list
        scorer    : CLIPDifficultyScorer
        batch_size: images per CLIP forward pass (must be even)
        alpha     : Beta distribution parameter (default 1.0 = training default)

    Returns:
        mean 1 - cosine_similarity across all scored pairs
    """
    from augmentations.mixing import cutmix, mixup

    mix_fn = cutmix if mix_mode == "cutmix" else mixup
    bs = batch_size if batch_size % 2 == 0 else batch_size - 1  # must be even
    all_scores: List[torch.Tensor] = []

    for start in range(0, len(images) - bs, bs):
        chunk = images[start : start + bs]
        tensors = torch.stack([TF.to_tensor(img) for img in chunk])  # (B, 3, H, W)

        # Dummy labels — only images matter for CLIP scoring
        dummy_labels = torch.zeros(len(chunk), dtype=torch.long)
        mixed, _, _, _ = mix_fn(tensors, dummy_labels, alpha)

        # Score: how different is the mixed image from the first half (the "original")?
        orig = tensors[: bs // 2]
        mixed_half = mixed[: bs // 2]
        all_scores.append(scorer(orig, mixed_half).cpu())

    if not all_scores:
        return 0.0
    return float(torch.cat(all_scores).mean())


# ── tier label helper ─────────────────────────────────────────────────────────


def _tier_label(op_name: str) -> str:
    if op_name in _TIER_OPS[1]:
        return "T1"
    if op_name in _TIER_OPS[2]:
        return "T2"
    return "T3"


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline CLIP-based augmentation difficulty calibration"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar100", "tiny_imagenet"],
        help="Dataset to sample images from",
    )
    parser.add_argument("--root", default="./data", help="Data root directory")
    parser.add_argument(
        "--n_images",
        type=int,
        default=_N_IMAGES,
        help=f"Number of random training images to use (default: {_N_IMAGES})",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=_STRENGTH,
        help=f"Augmentation strength for scoring (default: {_STRENGTH})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_BATCH,
        help=f"Images per CLIP forward pass (default: {_BATCH})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./configs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"\n{'=' * 62}")
    print(f"  CLIP Augmentation Difficulty Calibration")
    print(f"  Dataset    : {args.dataset}")
    print(f"  N images   : {args.n_images}")
    print(f"  Strength   : {args.strength}")
    print(f"  Device     : {device}")
    print(f"{'=' * 62}\n")

    # 1. Load images ──────────────────────────────────────────────────────────
    print(f"Loading {args.n_images} random training images...")
    images = _load_pil_images(args.dataset, args.root, args.n_images, args.seed)
    h, w = images[0].size[1], images[0].size[0]
    print(f"  Loaded {len(images)} images  ({images[0].size[0]}×{h} px)\n")

    # 2. Load CLIP ────────────────────────────────────────────────────────────
    print("Loading CLIP ViT-B/32 (frozen)...")
    scorer = CLIPDifficultyScorer(device=device)
    print("  Ready\n")

    # 3. Score every augmentation op ─────────────────────────────────────────
    all_ops = list(AUGMENTATION_REGISTRY.keys())
    scores: Dict[str, float] = {}

    print(f"  {'Op':<22} {'CLIP Difficulty':>16}  {'Tier':>5}")
    print(f"  {'-' * 22} {'-' * 16}  {'-' * 5}")

    for op_name in all_ops:
        score = _score_op(op_name, images, scorer, args.strength, args.batch_size)
        scores[op_name] = score
        tier = _tier_label(op_name)
        print(f"  {op_name:<22} {score:>16.4f}  {tier:>5}")

    # 4. Score mixing strategies ───────────────────────────────────────────────
    mixing_scores: Dict[str, float] = {}
    print(f"\n  {'Mixing strategy':<22} {'CLIP Difficulty':>16}  {'Note':>5}")
    print(f"  {'-' * 22} {'-' * 16}  {'-' * 12}")

    for mix_mode in ("cutmix", "mixup"):
        score = _score_mixing(mix_mode, images, scorer, args.batch_size)
        mixing_scores[mix_mode] = score
        print(f"  {mix_mode:<22} {score:>16.4f}  alpha=1.0")

    # 5. Ranked summary (ops only — mixing reported separately) ───────────────
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print(f"\n{'=' * 62}")
    print("  Ranked easiest → hardest — augmentation ops:")
    print(f"{'=' * 62}")
    for rank, (op, score) in enumerate(ranked, 1):
        tier = _tier_label(op)
        registry_tier = AUGMENTATION_REGISTRY[op][1]
        match = "✓" if int(tier[1]) == registry_tier else "≠"
        print(f"  {rank:>2}. {op:<22}  {score:.4f}  [{tier}] {match}")

    print(f"\n  Mixing strategies:")
    for mix_mode, score in sorted(mixing_scores.items(), key=lambda x: x[1]):
        print(f"      {mix_mode:<22}  {score:.4f}  [T3 only]")

    print(f"\n  ✓ = CLIP tier matches manual tier  ≠ = CLIP suggests different tier\n")

    # 6. Save ─────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"aug_difficulty_scores_{args.dataset}.pt"

    payload = {
        "dataset": args.dataset,
        "strength": args.strength,
        "n_images": len(images),
        "seed": args.seed,
        "model": "CLIP ViT-B/32",
        "scores": scores,  # {op_name: float}
        "mixing_scores": mixing_scores,  # {"cutmix": float, "mixup": float}
        "ranked": [op for op, _ in ranked],  # ops ordered easiest → hardest
    }
    torch.save(payload, out_path)
    print(f"  Saved → {out_path}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
