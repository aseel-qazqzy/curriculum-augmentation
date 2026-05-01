import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # must come before local imports

from data.datasets import (
    CIFAR_STATS,
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
)

from models.registry import get_model

from experiments.utils import get_device, set_seed

ENTROPY_DIR = Path("checkpoints/entropy")


def build_raw_train_subset(dataset: str, root: str, val_split=0.1):
    """Returns training subset with NO augmentation — only normalisation."""
    if dataset in CIFAR_STATS:
        mean, std = CIFAR_STATS[dataset]["mean"], CIFAR_STATS[dataset]["std"]
    else:
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset == "cifar100":
        full = datasets.CIFAR100(root, train=True, download=True, transform=transform)
    elif dataset == "cifar10":
        full = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    elif dataset == "tiny_imagenet":
        train_dir = Path(root) / "tiny-imagenet-200" / "train"
        full = datasets.ImageFolder(train_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n = len(full)
    val_size = int(n * val_split)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    return torch.utils.data.Subset(full, indices[: n - val_size])


def build_raw_entropy_loader(
    dataset: str,
    root: str,
    val_split: float = 0.1,
    batch_size: int = 128,
    debug: bool = False,
):
    """
    Build a DataLoader for entropy computation during EGS training.

    Rules:
      - NO augmentation (only ToTensor + Normalize) — entropy must reflect
        genuine sample difficulty, not augmentation randomness
      - shuffle=False — preserves index order so entropy_scores[i]
        maps correctly to CurriculumDataset.difficulties[i]
      - Same split as main training (seed=42) — index i refers to
        the same image in both this loader and the training loader
      - Built once before the epoch loop, reused every update

    Args:
        dataset    : "cifar10", "cifar100", "tiny_imagenet"
        root       : data root (same as cfg["data_root"])
        val_split  : same value used in main training (default 0.1)
        batch_size : same as training batch_size
        debug      : if True, limit to 512 samples to match debug training loader

    Returns:
        DataLoader — shuffle=False, no augmentation, same training split
    """
    train_subset = build_raw_train_subset(dataset, root, val_split)

    if debug:
        train_subset = torch.utils.data.Subset(train_subset, range(512))

    return DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,  # MUST be False — preserves index order
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def assign_egs_difficulties(
    entropy_scores: np.ndarray,
    max_tier_reached: np.ndarray,
    strength: float = 0.7,
    num_classes: int = 100,
    epoch: int = 0,
    force_t2_epoch: int = 9999,
    force_t3_epoch: int = 9999,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Convert entropy scores to per-sample difficulty values for CurriculumDataset.

    Uses entropy THRESHOLD assignment — a sample advances only when its entropy
    is genuinely low enough. Safety fallback forces advancement at ETS-equivalent
    epoch boundaries so the full curriculum is used regardless of model capacity
    or how conservative the entropy thresholds are.

    Thresholds (fraction of maximum entropy log(num_classes)):
      Tier 2: H < 0.60 * log(C)  — moderately confident
      Tier 3: H < 0.30 * log(C)  — highly confident

    Safety fallback (force_t2_epoch / force_t3_epoch):
      At force_t2_epoch: all samples advance to at least Tier 2
      At force_t3_epoch: all samples advance to Tier 3
      Defaults to 9999 (disabled) — set via egs_state from tier_t1/tier_t2 config.

    Args:
        entropy_scores   : np.ndarray [N] — H(x) per sample from current model
        max_tier_reached : np.ndarray [N] — highest tier each sample has reached
        strength         : augmentation strength ceiling (default 0.7)
        num_classes      : number of classes — scales entropy thresholds
        epoch            : current training epoch — used for safety fallback
        force_t2_epoch   : force all samples to Tier 2 at this epoch
        force_t3_epoch   : force all samples to Tier 3 at this epoch

    Returns:
        max_tier_reached : np.ndarray [N] — updated
        difficulties     : torch.Tensor [N] — for CurriculumDataset.set_difficulties()
    """
    log_C = np.log(num_classes)

    # Entropy threshold assignment — only advance when genuinely confident
    new_tiers = np.ones(len(entropy_scores), dtype=np.int32)
    new_tiers[entropy_scores < 0.60 * log_C] = 2
    new_tiers[entropy_scores < 0.30 * log_C] = 3

    # Safety fallback: force advancement at ETS-equivalent boundaries
    # so any model/dataset is guaranteed to use the full curriculum
    if epoch >= force_t3_epoch:
        new_tiers[:] = 3
        print(
            f"  EGS fallback: all samples forced to Tier 3 (epoch {epoch} >= {force_t3_epoch})"
        )
    elif epoch >= force_t2_epoch:
        new_tiers = np.maximum(new_tiers, 2)
        print(
            f"  EGS fallback: all samples at least Tier 2 (epoch {epoch} >= {force_t2_epoch})"
        )

    # Monotonic: only advance, never retreat
    max_tier_reached = np.maximum(max_tier_reached, new_tiers)

    tier_to_diff = {1: strength * 0.40, 2: strength * 0.70, 3: strength * 1.00}
    difficulties = np.vectorize(tier_to_diff.get)(max_tier_reached)

    n1 = int((max_tier_reached == 1).sum())
    n2 = int((max_tier_reached == 2).sum())
    n3 = int((max_tier_reached == 3).sum())
    pct3 = n3 / len(entropy_scores) * 100
    print(
        f"  EGS tiers — Tier1: {n1:,}  Tier2: {n2:,}  Tier3: {n3:,}  ({pct3:.1f}% in T3)"
    )

    return max_tier_reached, torch.tensor(difficulties, dtype=torch.float32)


# Load model from checkpoint
def load_model(model_name: str, dataset: str, checkpoint: str, device):
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}[dataset]
    model = get_model(model_name, num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint, map_location=device)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {model_name} from {checkpoint}")
    return model


def train_proxy_from_scratch(
    model_name: str,
    dataset: str,
    root: str,
    proxy_epochs: int,
    device,
    batch_size: int = 128,
):
    """
    Train a small proxy model with standard augmentation and stop early.
    Model weights are discarded after entropy is collected — not saved.
    """
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}[dataset]

    if dataset in CIFAR_STATS:
        mean, std = CIFAR_STATS[dataset]["mean"], CIFAR_STATS[dataset]["std"]
    else:
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD

    crop_size = 64 if dataset == "tiny_imagenet" else 32
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset == "cifar100":
        full = datasets.CIFAR100(
            root, train=True, download=True, transform=train_transform
        )
    elif dataset == "cifar10":
        full = datasets.CIFAR10(
            root, train=True, download=True, transform=train_transform
        )
    elif dataset == "tiny_imagenet":
        full = datasets.ImageFolder(
            Path(root) / "tiny-imagenet-200" / "train", transform=train_transform
        )

    n = len(full)
    val_size = int(n * 0.1)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_ds = torch.utils.data.Subset(full, indices[: n - val_size])
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = get_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    print(
        f"  Training proxy {model_name} for {proxy_epochs} epochs "
        f"(early stop for best entropy spread)..."
    )

    model.train()
    for epoch in range(1, proxy_epochs + 1):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"    Proxy epoch [{epoch:2d}/{proxy_epochs}]  "
            f"loss={total_loss / len(loader):.4f}"
        )

    model.eval()
    return model


# Compute per-sample entropy
def compute_entropy(model, train_subset, dataset: str, device, batch_size: int = 256):
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}[dataset]
    log_C = np.log(num_classes)
    print(f"Log for classes {log_C}")

    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    N = len(train_subset)
    entropy_scores = np.zeros(N, dtype=np.float32)
    pos = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            probs = torch.softmax(model(images).float(), dim=1)
            H = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)
            n = H.size(0)
            entropy_scores[pos : pos + n] = H.cpu().numpy()
            pos += n

    print(
        f"  Entropy range: [{entropy_scores.min():.4f}, {entropy_scores.max():.4f}]"
        f"  mean={entropy_scores.mean():.4f}  (max={log_C:.4f})"
    )
    return entropy_scores, log_C


def entropy_to_difficulty(entropy_scores: np.ndarray, log_C: float) -> torch.Tensor:
    """
    High entropy (uncertain) → low difficulty  → shallower augmentation
    Low  entropy (confident) → high difficulty → deeper augmentation
    """
    normalised = entropy_scores / log_C  # [0, 1]
    difficulty = 1.0 - normalised  # invert
    return torch.tensor(difficulty, dtype=torch.float32)


def save_entropy(entropy_scores, log_C, dataset, model_name):
    ENTROPY_DIR.mkdir(parents=True, exist_ok=True)
    out = ENTROPY_DIR / f"entropy_{dataset}_{model_name}.npy"
    np.save(out, np.array([log_C, *entropy_scores], dtype=np.float32))
    print(f"Saved -> {out}({len(entropy_scores):,} samples)")


def load_entropy(dataset: str, model_name: str) -> tuple[np.ndarray, float]:
    path = ENTROPY_DIR / f"entropy_{dataset}_{model_name}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No entropy file for ({dataset}, {model_name}).\n"
            f"Run: python -m experiments.compute_entropy "
            f"--dataset {dataset} --model {model_name} --checkpoint <path>"
        )
    data = np.load(path)
    log_C = float(data[0])
    entropy_scores = data[1:]
    return entropy_scores, log_C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "tiny_imagenet"]
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Backbone to use (e.g. wideresnet, resnet50) — same as your main training run",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to existing trained .pth checkpoint"
    )
    parser.add_argument(
        "--proxy_epochs",
        type=int,
        default=None,
        help="Train model from scratch and stop at this epoch (e.g. 15)",
    )
    parser.add_argument("--root", default="./data")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.checkpoint and args.proxy_epochs:
        raise ValueError("Provide either --checkpoint OR --proxy_epochs, not both.")
    if not args.checkpoint and not args.proxy_epochs:
        raise ValueError("Must provide either --checkpoint or --proxy_epochs.")

    print(f"Dataset : {args.dataset}  |  Model : {args.model}")

    if args.checkpoint:
        model = load_model(args.model, args.dataset, args.checkpoint, device)
    else:
        model = train_proxy_from_scratch(
            args.model,
            args.dataset,
            args.root,
            args.proxy_epochs,
            device,
            args.batch_size,
        )
    model_tag = args.model

    train_subset = build_raw_train_subset(args.dataset, args.root, args.val_split)
    scores, logC = compute_entropy(
        model, train_subset, args.dataset, device, args.batch_size
    )
    save_entropy(scores, logC, args.dataset, model_tag)


if __name__ == "__main__":
    main()
