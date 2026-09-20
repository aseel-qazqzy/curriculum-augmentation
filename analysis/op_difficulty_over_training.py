"""
analysis/op_difficulty_over_training.py

Measure how much each augmentation operation disrupts validation loss at
three training stages (epoch 5, 30, 80), using no-augmentation checkpoints.

This produces Table A0 and the Δval_loss heatmap figure.

Setup on cluster:
    First run a no-aug training that saves checkpoints at epochs 5, 30, 80:

        python -m experiments.train_baseline \\
            --augmentation none --dataset cifar100 --model wideresnet \\
            --epochs 100 --seed 42 --save_at_epochs 5,30,80

    Then run this script:

        python analysis/op_difficulty_over_training.py \\
            --checkpoint_ep5  checkpoints/wideresnet_none_..._ep5_cifar100_s42_best.pth \\
            --checkpoint_ep30 checkpoints/wideresnet_none_..._ep30_cifar100_s42_best.pth \\
            --checkpoint_ep80 checkpoints/wideresnet_none_..._ep80_cifar100_s42_best.pth

Outputs:
    results/figs/op_difficulty_heatmap.png
    results/figs/op_difficulty_heatmap_hd.png
    results/logs/op_difficulty_YYYYMMDD.log
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import get_cifar100_loaders
from models.registry import get_model
from augmentations.primitives import (
    random_flip,
    random_crop,
    translate_x,
    translate_y,
    color_jitter,
    random_rotation,
    random_shear,
    auto_contrast,
    equalize,
    sharpness,
    random_grayscale,
    cutout,
    enhance_contrast,
    enhance_brightness,
    posterize,
    invert,
    gaussian_blur,
    solarize,
    random_perspective,
)

FIGURES_DIR = Path("results/figs")
LOGS_DIR = Path("results/logs")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STRENGTH = 0.7

# Each entry: (display_name, primitive_fn, tier)
OPS = [
    ("flip", random_flip, "T1"),
    ("crop", random_crop, "T1"),
    ("translate_x", translate_x, "T1"),
    ("translate_y", translate_y, "T1"),
    ("sharpness", sharpness, "T2"),
    ("auto_contrast", auto_contrast, "T2"),
    ("shear", random_shear, "T2"),
    ("equalize", equalize, "T2"),
    ("color_jitter", color_jitter, "T2"),
    ("perspective", random_perspective, "T2"),
    ("rotation", random_rotation, "T2"),
    ("grayscale", random_grayscale, "T3"),
    ("contrast", enhance_contrast, "T3"),
    ("brightness", enhance_brightness, "T3"),
    ("posterize", posterize, "T3"),
    ("invert", invert, "T3"),
    ("cutout", cutout, "T3"),
    ("blur", gaussian_blur, "T3"),
    ("solarize", solarize, "T3"),
]

TIER_COLORS = {"T1": "#4878CF", "T2": "#E8A838", "T3": "#C44E52"}
SAFE_THRESHOLD = 0.05  # nats — ops below this are considered safe at that stage

# CIFAR-100 normalisation constants (must match training)
_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
_STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)


def _apply_pil_aug(imgs: torch.Tensor, op_fn, strength: float) -> torch.Tensor:
    """Apply a PIL-based primitive to a normalised tensor batch (N,C,H,W)."""
    mean = _MEAN.to(imgs.device)
    std = _STD.to(imgs.device)
    out = []
    for img in imgs:
        pil = TF.to_pil_image((img * std + mean).clamp(0, 1).cpu())
        aug = op_fn(pil, strength)
        t = TF.to_tensor(aug).to(imgs.device)
        out.append((t - mean.cpu()) / std.cpu())
    return torch.stack(out).to(imgs.device)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = 100
    model = get_model("wideresnet", num_classes=num_classes)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def compute_val_loss(model, loader, aug_fn, device):
    """Return mean cross-entropy loss over val set, optionally with aug_fn applied."""
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if aug_fn is not None:
                imgs = _apply_pil_aug(imgs, aug_fn, STRENGTH)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_n += labels.size(0)
    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_ep5", required=True)
    parser.add_argument("--checkpoint_ep30", required=True)
    parser.add_argument("--checkpoint_ep80", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"op_difficulty_{ts}.log"

    checkpoints = {
        "ep5": args.checkpoint_ep5,
        "ep30": args.checkpoint_ep30,
        "ep80": args.checkpoint_ep80,
    }
    stage_labels = ["Epoch 5\n(early)", "Epoch 30\n(mid)", "Epoch 80\n(late)"]

    _, val_loader, _ = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=4,
    )

    results = {}  # op_name -> [delta_ep5, delta_ep30, delta_ep80]

    with open(log_path, "w") as log:

        def pr(msg):
            print(msg)
            log.write(msg + "\n")

        pr(f"Op difficulty over training stages — {ts}")
        pr(f"Strength: {STRENGTH}  |  Threshold τ: {SAFE_THRESHOLD} nats")
        pr("=" * 70)

        for stage_key, ckpt_path in checkpoints.items():
            pr(f"\n--- {stage_key.upper()} checkpoint: {ckpt_path} ---")
            model = load_model(ckpt_path, device)

            baseline = compute_val_loss(model, val_loader, None, device)
            pr(f"  Baseline (clean) val_loss: {baseline:.4f}")

            for op_name, op_fn, tier in OPS:
                aug_loss = compute_val_loss(model, val_loader, op_fn, device)
                delta = aug_loss - baseline
                if op_name not in results:
                    results[op_name] = []
                results[op_name].append(delta)
                safe = "SAFE" if delta < SAFE_THRESHOLD else "HARM"
                pr(f"  {op_name:<16} {tier}  Δ={delta:+.4f}  [{safe}]")

    # ── Build delta matrix (ops × 3 stages) ──────────────────────────────────
    op_names = [r[0] for r in OPS]
    op_tiers = [r[2] for r in OPS]
    delta_mat = np.array([results[n] for n in op_names])  # shape (19, 3)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05}
    )

    # Left: heatmap
    ax = axes[0]
    vmax = max(0.3, float(np.percentile(delta_mat, 95)))
    im = ax.imshow(
        delta_mat,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_yticks(range(len(op_names)))
    ax.set_yticklabels(op_names, fontsize=9)
    ax.set_xlabel("Training Stage", fontsize=11)
    ax.set_title(
        "Δval_loss per Operation per Training Stage\n"
        "(higher = more disruptive; τ = 0.05 nats threshold)",
        fontsize=11,
    )

    # Tier colour bars on the y-axis labels
    for i, (name, tier) in enumerate(zip(op_names, op_tiers)):
        ax.get_yticklabels()[i].set_color(TIER_COLORS[tier])

    # Threshold line markers (cells above τ get a hatch)
    for i in range(len(op_names)):
        for j in range(3):
            val = delta_mat[i, j]
            cell_txt = f"{val:+.3f}"
            ax.text(
                j,
                i,
                cell_txt,
                ha="center",
                va="center",
                fontsize=7,
                color="black" if val < vmax * 0.6 else "white",
            )
            if val > SAFE_THRESHOLD:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.8,
                        linestyle="--",
                    )
                )

    plt.colorbar(im, ax=ax, label="Δval_loss (nats)")

    # Right panel: tier legend + "Stage first safe" column
    axes[1].axis("off")
    legend_text = "Tier colours:\n"
    for tier, col in TIER_COLORS.items():
        legend_text += f"  {tier}: ops in tier {tier[-1]}\n"
    legend_text += f"\n── = above τ={SAFE_THRESHOLD}"
    axes[1].text(
        0.05,
        0.95,
        legend_text,
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    for suffix, dpi in [("", 150), ("_hd", 300)]:
        out = FIGURES_DIR / f"op_difficulty_heatmap{suffix}.png"
        fig.savefig(out, dpi=dpi)
        print(f"Saved: {out}")

    plt.close(fig)

    # ── Print markdown table for thesis ──────────────────────────────────────
    print("\n\n--- Markdown table (paste into thesis_results_tables.md) ---")
    print(
        f"| Op | Tier | Δval_loss ep 5 | Δval_loss ep 30 | Δval_loss ep 80 | Stage first safe |"
    )
    print(f"|:---|:---:|:---:|:---:|:---:|:---:|")
    for op_name, tier, deltas in zip(op_names, op_tiers, delta_mat):
        d5, d30, d80 = deltas
        safe_ep = (
            "ep 1"
            if d5 < SAFE_THRESHOLD
            else (
                "ep 30"
                if d30 < SAFE_THRESHOLD
                else ("ep 80" if d80 < SAFE_THRESHOLD else ">ep 80")
            )
        )
        print(
            f"| {op_name} | {tier} | {d5:+.3f} | {d30:+.3f} | {d80:+.3f} | {safe_ep} |"
        )


if __name__ == "__main__":
    main()
