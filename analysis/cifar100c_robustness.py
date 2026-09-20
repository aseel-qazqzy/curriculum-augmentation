"""
analysis/cifar100c_robustness.py

Evaluate all 5 trained checkpoints on CIFAR-100-C corruptions.
No new training — inference only.

Setup on cluster:
    wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    tar -xf CIFAR-100-C.tar -C data/

Run:
    python analysis/cifar100c_robustness.py --c_root data/CIFAR-100-C

Outputs:
    results/logs/cifar100c_YYYYMMDD_HHMMSS.log
    results/figs/cifar100c_robustness.png
    results/figs/cifar100c_robustness_hd.png
"""

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import get_model

# ── Checkpoints (seed 42) ─────────────────────────────────────────────────────
CHECKPOINTS = {
    "No Augmentation": "wideresnet_none_sgd_cosine_ep100_cifar100_s42_best.pth",
    "Static Mixing": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19_best.pth",
}

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "fog",
    "frost",
    "snow",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "saturate",
    "gaussian_blur",
    "spatter",
]

SEVERITIES = [1, 2, 3, 4, 5]
BATCH_SIZE = 256

NORMALIZE = T.Normalize(
    mean=[0.5071, 0.4867, 0.4408],
    std=[0.2675, 0.2565, 0.2761],
)

CLEAN_ACCS = {
    "No Augmentation": 72.86,
    "Static Mixing": 77.60,
    "ETS": 81.39,
    "LPS": 81.35,
    "EGS": 79.75,
}

COLORS = {
    "No Augmentation": "#999999",
    "Static Mixing": "#D55E00",
    "ETS": "#0072B2",
    "LPS": "#009E73",
    "EGS": "#E69F00",
}


# ── Logging ───────────────────────────────────────────────────────────────────


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Model ─────────────────────────────────────────────────────────────────────


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model("wideresnet", num_classes=100).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {ckpt_path.name}  (val_acc={ckpt.get('val_acc', 0) * 100:.2f}%)")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_corruption(model, c_root, corruption, severity, device):
    data = np.load(c_root / f"{corruption}.npy")  # (50000, 32, 32, 3)
    labels = np.load(c_root / "labels.npy")  # (50000,)

    idx = (severity - 1) * 10000
    images = data[idx : idx + 10000]
    targets = labels[idx : idx + 10000]

    images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    images = torch.stack([NORMALIZE(img) for img in images])
    targets = torch.from_numpy(targets).long()

    loader = DataLoader(
        TensorDataset(images, targets),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(dim=1).eq(y).sum().item()

    return correct / len(targets) * 100


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_results(results, save_dir):
    methods = list(CHECKPOINTS.keys())
    mean_corr = [np.mean(list(results[m].values())) for m in methods]
    drops = [CLEAN_ACCS[m] - mc for m, mc in zip(methods, mean_corr)]
    colors = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "CIFAR-100-C Robustness · WideResNet-28-10 · Seed 42\n"
        "19 corruptions × 5 severity levels",
        fontsize=12,
        fontweight="bold",
    )

    # Left — mean corrupted accuracy
    bars = axes[0].barh(
        methods, mean_corr, color=colors, edgecolor="white", linewidth=0.8
    )
    axes[0].set_xlabel("Mean Accuracy on Corrupted Data (%) ↑", fontsize=10)
    axes[0].set_title("Corruption Robustness", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, max(mean_corr) * 1.12)
    for bar, v in zip(bars, mean_corr):
        axes[0].text(
            v + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%",
            va="center",
            fontsize=9,
        )

    # Right — robustness drop
    bars2 = axes[1].barh(methods, drops, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_xlabel("Accuracy Drop: Clean → Corrupted (pp) ↓", fontsize=10)
    axes[1].set_title(
        "Sensitivity to Corruption (lower = better)", fontsize=11, fontweight="bold"
    )
    axes[1].set_xlim(0, max(drops) * 1.12)
    for bar, v in zip(bars2, drops):
        axes[1].text(
            v + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}pp",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_dir / "cifar100c_robustness.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "cifar100c_robustness_hd.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {save_dir}/cifar100c_robustness.png")
    print(f"Saved: {save_dir}/cifar100c_robustness_hd.png")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_root", default="data/CIFAR-100-C")
    args = parser.parse_args()

    c_root = Path(args.c_root)
    ckpt_dir = Path("results/cluster/checkpoints")
    save_dir = Path("results/figs")
    log_dir = Path("results/logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        log_dir / f"cifar100c_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"Log file : {log_path}")
    print(f"C-root   : {c_root}")
    print(f"Corruptions: {len(CORRUPTIONS)} × {len(SEVERITIES)} severities\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}\n")

    results = {}

    for method, ckpt_name in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"  {method}")
        print(f"{'=' * 60}")

        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"  SKIPPED — checkpoint not found: {ckpt_path}")
            continue

        model = load_model(ckpt_path, device)
        method_results = {}

        for corruption in CORRUPTIONS:
            c_file = c_root / f"{corruption}.npy"
            if not c_file.exists():
                print(f"  SKIPPED corruption: {corruption} (file not found)")
                continue
            sev_accs = []
            for severity in SEVERITIES:
                acc = evaluate_corruption(model, c_root, corruption, severity, device)
                sev_accs.append(acc)
            mean_acc = np.mean(sev_accs)
            method_results[corruption] = mean_acc
            sev_str = "  ".join(f"s{s}={a:.1f}" for s, a in zip(SEVERITIES, sev_accs))
            print(f"  {corruption:<25}  mean={mean_acc:.2f}%  [{sev_str}]")

        results[method] = method_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Method':<20} {'Clean Acc':>10} {'Corr. Acc':>10} {'Drop':>8}")
    print("─" * 52)
    for method in CHECKPOINTS:
        if method not in results:
            continue
        mc = np.mean(list(results[method].values()))
        drop = CLEAN_ACCS[method] - mc
        print(f"{method:<20} {CLEAN_ACCS[method]:>9.2f}%  {mc:>8.2f}%  {drop:>6.2f}pp")

    plot_results(results, save_dir)

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
