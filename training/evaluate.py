"""
evaluate.py — Robustness Evaluation on CIFAR-10-C / CIFAR-100-C
Curriculum-Style Augmentation for Image Classification

Evaluates a trained model against 19 corruption types at 5 severity levels.
Computes mean Corruption Error (mCE) — the standard robustness metric.

Usage:
    # Download CIFAR-10-C (once)
    wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    tar -xf CIFAR-10-C.tar -C /Users/aseelahmedal-qazqzy/Documents/Hildeshiem University/Thesis/curriculum-augmentation/data/raw

    # Then evaluate your best checkpoint
    python training/evaluate.py \
    --checkpoint checkpoints/resnet18_static_aug_sgd_multistep_best.pth \
    --data_dir   data/raw/CIFAR-10-C \
    --dataset    cifar10
    # Evaluate a checkpoint on CIFAR-10-C
    python training/evaluate.py \
        --checkpoint checkpoints/cifar10_baseline_best.pth \
        --data_dir   data/raw/CIFAR-10-C \
        --dataset    cifar10

    # Evaluate and compare multiple checkpoints
    python training/evaluate.py \
        --checkpoint checkpoints/cifar10_curriculum_cosine_best.pth \
        --data_dir   data/raw/CIFAR-10-C \
        --dataset    cifar10 \
        --baseline_acc 75.28   # clean accuracy of AlexNet baseline (for mCE)

Download CIFAR-10-C from:
    https://zenodo.org/record/2535967   → CIFAR-10-C.tar
    https://zenodo.org/record/3555552   → CIFAR-100-C.tar

Extract into:
    data/raw/CIFAR-10-C/   → contains .npy files + labels.npy
    data/raw/CIFAR-100-C/  → contains .npy files + labels.npy
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from rich.console import Console
from rich.table import Table
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import get_device

console = Console()
# CIFAR-C CORRUPTION TYPES (19 total)


CORRUPTIONS = [
    # Noise
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    # Blur
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    # Weather
    "snow",
    "frost",
    "fog",
    "brightness",
    # Digital
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    # Extra (CIFAR-10-C only)
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

SEVERITIES = [1, 2, 3, 4, 5]

# AlexNet baseline errors on CIFAR-10-C (used to compute normalized mCE)
# Source: Hendrycks & Dietterich (2019)
ALEXNET_BASELINE = {
    "gaussian_noise": 0.886,
    "shot_noise": 0.860,
    "impulse_noise": 0.922,
    "defocus_blur": 0.820,
    "glass_blur": 0.826,
    "motion_blur": 0.786,
    "zoom_blur": 0.798,
    "snow": 0.867,
    "frost": 0.827,
    "fog": 0.819,
    "brightness": 0.565,
    "contrast": 0.853,
    "elastic_transform": 0.646,
    "pixelate": 0.718,
    "jpeg_compression": 0.607,
    "speckle_noise": 0.845,
    "gaussian_blur": 0.787,
    "spatter": 0.717,
    "saturate": 0.658,
}

# 1. MODEL LOADING


def load_model(checkpoint_path: str, num_classes: int, device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})

    model_name = cfg.get("model", {}).get("name", "resnet18").lower()

    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    console.print(f"Loaded checkpoint: {checkpoint_path}")
    console.print(
        f"   Trained for {ckpt.get('epoch', '?')} epochs | "
        f"Best val acc: {ckpt.get('val_acc', 0) * 100:.2f}%"
    )
    return model, cfg


# 2. CIFAR-C DATASET
def get_normalization(dataset: str):
    from data.datasets import CIFAR_STATS

    if dataset not in CIFAR_STATS:
        raise ValueError(f"Unknown dataset: {dataset}")
    s = CIFAR_STATS[dataset]
    return transforms.Normalize(s["mean"], s["std"])


def load_corruption_data(
    data_dir: str, corruption: str, severity: int, normalize, batch_size: int = 256
):
    """
    Load a single corruption + severity combination from .npy files.
    Returns a DataLoader.
    """
    images_path = Path(data_dir) / f"{corruption}.npy"
    labels_path = Path(data_dir) / "labels.npy"

    if not images_path.exists():
        return None  # corruption not available, skip

    images = np.load(
        images_path
    )  # shape: (50000*5, 32, 32, 3) — all severities stacked
    labels = np.load(labels_path)  # shape: (50000*5,)

    # Each severity is 10,000 samples, stacked in order 1→5
    start = (severity - 1) * 10000
    end = severity * 10000
    images = images[start:end]
    labels = labels[start:end]

    # Convert to tensors
    images_tensor = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    labels_tensor = torch.from_numpy(labels).long()

    # Normalize
    images_tensor = normalize(images_tensor)
    dataset = TensorDataset(images_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# 3. EVALUATION
@torch.no_grad()
def evaluate_loader(model, loader, device):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total  # accuracy


def evaluate_clean(
    model, dataset: str, data_dir: str, normalize, device, batch_size=256
):
    """Evaluate on the clean (uncorrupted) test set."""
    from torchvision import datasets

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    root = str(Path(data_dir).parent)

    if dataset == "cifar10":
        test_dataset = datasets.CIFAR10(
            root, train=False, download=True, transform=transform
        )
    else:
        test_dataset = datasets.CIFAR100(
            root, train=False, download=True, transform=transform
        )
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return evaluate_loader(model, loader, device)


def run_corruption_evaluation(
    model, data_dir: str, dataset: str, normalize, device, batch_size=256
):
    """
    Evaluate model on all 19 corruptions × 5 severities.
    Returns results dict and mean corruption error (mCE).
    """
    results = {}  # {corruption: {severity: error_rate}}

    console.print(
        "\n[bold cyan]Running corruption robustness evaluation...[/bold cyan]\n"
    )

    for corruption in CORRUPTIONS:
        results[corruption] = {}
        severity_errors = []

        for severity in SEVERITIES:
            loader = load_corruption_data(
                data_dir, corruption, severity, normalize, batch_size
            )
            if loader is None:
                continue

            acc = evaluate_loader(model, loader, device)
            error = 1.0 - acc
            results[corruption][severity] = error
            severity_errors.append(error)

        if severity_errors:
            avg_error = np.mean(severity_errors)
            results[corruption]["mean"] = avg_error

    return results


def compute_mce(results: dict) -> float:
    """
    Compute mean Corruption Error (mCE) normalized against AlexNet baseline.
    Lower is better.
    """
    ce_scores = []
    for corruption, severity_results in results.items():
        if "mean" not in severity_results:
            continue

        model_error = severity_results["mean"]
        if baseline_error := ALEXNET_BASELINE.get(corruption):
            ce = model_error / baseline_error
            ce_scores.append(ce)

    return np.mean(ce_scores) if ce_scores else None


# 4. REPORTING
def print_results_table(results: dict, clean_acc: float, mce: float):
    table = Table(title="Corruption Robustness Results", show_lines=True)
    table.add_column("Corruption", style="cyan", width=22)
    table.add_column("Sev 1", justify="right")
    table.add_column("Sev 2", justify="right")
    table.add_column("Sev 3", justify="right")
    table.add_column("Sev 4", justify="right")
    table.add_column("Sev 5", justify="right")
    table.add_column("Mean Error", justify="right", style="magenta")

    for corruption, severity_results in results.items():
        row = [corruption]
        for s in SEVERITIES:
            err = severity_results.get(s, None)
            row.append(f"{err * 100:.1f}%" if err is not None else "—")
        mean = severity_results.get("mean", None)
        row.append(f"{mean * 100:.2f}%" if mean is not None else "—")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold]Clean Accuracy:[/bold]  {clean_acc * 100:.2f}%")
    console.print(
        f"[bold]mCE (↓ better):[/bold]  {mce * 100:.2f}%" if mce else "mCE: N/A"
    )


def save_results(
    results: dict,
    clean_acc: float,
    mce: float,
    checkpoint_path: str,
    output_dir: str = "results",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    name = Path(checkpoint_path).stem

    # ── Save CSV
    rows = []
    for corruption, severity_results in results.items():
        row = {"corruption": corruption}
        for s in SEVERITIES:
            row[f"sev_{s}"] = severity_results.get(s, None)
        row["mean_error"] = severity_results.get("mean", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "corruption": "SUMMARY",
        "mean_error": 1 - clean_acc,
    }  # clean error
    csv_path = os.path.join(output_dir, f"{name}_corruption_results.csv")
    df.to_csv(csv_path, index=False)
    console.print(f"\n[green]💾 Results saved → {csv_path}[/green]")

    # ── Save bar chart
    corruptions = [r["corruption"] for r in rows]
    mean_errors = [r["mean_error"] * 100 for r in rows if r["mean_error"] is not None]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(corruptions, mean_errors, color="steelblue", edgecolor="white")
    ax.axhline(
        y=(1 - clean_acc) * 100,
        color="green",
        linestyle="--",
        label=f"Clean error ({(1 - clean_acc) * 100:.1f}%)",
    )
    ax.axhline(
        y=mce * 100 if mce else 0,
        color="red",
        linestyle="--",
        label=f"mCE ({mce * 100:.1f}%)" if mce else "mCE",
    )
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Mean Error Rate (%)")
    ax.set_title(f"Corruption Robustness — {name}")
    ax.set_xticklabels(corruptions, rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{name}_corruption_results.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    console.print(f"[green]📊 Chart saved  → {fig_path}[/green]")


# 5. MAIN
def main():
    parser = argparse.ArgumentParser(description="Robustness Evaluation on CIFAR-C")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to CIFAR-10-C or CIFAR-100-C folder",
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    normalize = get_normalization(args.dataset)

    # Load model
    model, cfg = load_model(args.checkpoint, num_classes, device)

    # Clean accuracy
    console.print("\n[bold]Evaluating on clean test set...[/bold]")
    clean_acc = evaluate_clean(
        model, args.dataset, args.data_dir, normalize, device, args.batch_size
    )
    console.print(f"Clean Accuracy: [green]{clean_acc * 100:.2f}%[/green]")

    # Corruption robustness
    results = run_corruption_evaluation(
        model, args.data_dir, args.dataset, normalize, device, args.batch_size
    )

    # mCE
    mce = compute_mce(results)

    # Print table
    print_results_table(results, clean_acc, mce)

    # Save CSV + chart
    save_results(results, clean_acc, mce, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
