"""
analysis/gradcam_comparison.py

Run on the cluster (full 100-epoch checkpoints live there):
    source venv/bin/activate
    python analysis/gradcam_comparison.py --checkpoint_dir checkpoints

Either grid is skipped (with a warning) if none of its checkpoints are found,
rather than aborting the whole run — e.g. to render only the CIFAR-100 grid:
    python analysis/gradcam_comparison.py --checkpoint_dir checkpoints --skip_tiny_imagenet

Override/add a column without editing the file (repeatable):
    python analysis/gradcam_comparison.py --checkpoint_dir checkpoints \\
        --run "Static Mixing=wideresnet_static_mixing_sgd_cosine_ep2_cifar100_s42" \\
        --run_tiny "LPS=wideresnet_tiered_lps_mix_both_sgd_cosine_ep2_tiny_imagenet_s123"

Output:
    results/figs/gradcam/gradcam_grid_cifar100.pdf        (+ .png, _hd.png)
    results/figs/gradcam/gradcam_grid_tinyimagenet.pdf    (+ .png, _hd.png)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

from models.registry import get_model
from data.datasets import CIFAR_STATS, _reorganize_tiny_imagenet_val

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAVE_DIR = PROJECT_ROOT / "results" / "figs" / "gradcam"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── CIFAR-100 grid ───────────────────────────────────────────────────────────
# Full-length seed-42 checkpoints (cluster). Same names + known accuracies as
# analysis/tsne_features.py's RUNS dict (minus "No Augmentation" — mixing is
# the variable under test here, not augmentation presence). Dict order is
# column order in the figure.
RUNS_CIFAR100 = {
    "Static Mixing": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19",
}
KNOWN_ACCS_CIFAR100 = {
    "Static Mixing": 77.43,
    "ETS": 81.35,
    "LPS": 81.36,
    "EGS": 79.83,
}

# 6 semantically diverse CIFAR-100 classes (subset of tsne_features.py's 15) —
# one test image per class becomes one row, labelled by class name.
SELECTED_CLASSES_CIFAR100 = {
    8: "bicycle",
    30: "dolphin",
    31: "elephant",
    72: "pickup_truck",
    81: "mushroom",
    96: "sunflower",
}

# ── Tiny-ImageNet grid ───────────────────────────────────────────────────────

RUNS_TINY_IMAGENET = {
    "Static Mixing": "wideresnet_static_mixing_sgd_cosine_ep100_tiny_imagenet_s123_p19",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_tiny_imagenet_s123_p19",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_tiny_imagenet_s123_p19",
    "EGS": "wideresnet_tiered_egs_freq5_mix_both_sgd_cosine_ep100_tiny_imagenet_s42_p19",
}
# No published numbers yet for these runs — subtitles are omitted until filled in.
KNOWN_ACCS_TINY_IMAGENET: dict[str, float] = {}

# 6 diverse Tiny-ImageNet classes (wnid -> readable name, from words.txt),
# chosen to be easily recognisable at 64x64.
SELECTED_CLASSES_TINY_IMAGENET = {
    "n02504458": "African elephant",
    "n02085620": "Chihuahua",
    "n04285008": "sports car",
    "n07873807": "pizza",
    "n07734744": "mushroom",
    "n04356056": "sunglasses",
}

# Fallback row indices for datasets with no class-name lookup defined above.
IMAGE_INDICES = [10, 42, 100, 250]

# Grad-CAM's backward hooks are unreliable on MPS; these models are small
# enough (32x32 / 64x64 input) that CPU is plenty fast for a handful of
# forward+backward passes.
DEVICE = torch.device("cpu")


def _target_layer(model: torch.nn.Module, model_name: str):
    """Last conv block before global pooling, keyed by architecture name."""
    name = model_name.lower()
    if name in {"wideresnet", "wrn28_10", "wrn16_8", "wrn-28-10", "wrn-16-8"}:
        return [model.group3[-1]]
    if name in {"resnet18", "resnet50"}:
        return [model.backbone.layer4[-1]]
    if name in {"pyramidnet", "pyramidnet272", "pyramidnet272_sd"}:
        return [model.layer3[-1]]
    if name == "baseline_cnn":
        return [model.block3]
    raise ValueError(f"No known Grad-CAM target layer for model '{model_name}'")


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip a leading 'module.' left by DataParallel-saved checkpoints."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def resolve_checkpoint(checkpoint_dir: Path, name: str) -> Path:
    path = checkpoint_dir / f"{name}_best.pth"
    if path.exists():
        return path
    return checkpoint_dir / f"{name}.pth"


def load_checkpoint_model(
    ckpt_path: Path, model_override: str | None, num_classes_override: int | None
):
    """Load a trained model from a train_baseline.py checkpoint.

    Reads architecture + dataset from the checkpoint's own `cfg` dict so callers
    don't need to hardcode them; --model/--num_classes cover older checkpoints
    saved without `cfg`.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("cfg", {})

    model_name = model_override or cfg.get("model")
    dataset = cfg.get("dataset")
    if model_name is None:
        raise ValueError(
            f"{ckpt_path.name} has no 'cfg' with a model name — pass --model explicitly."
        )
    num_classes = num_classes_override or {"cifar100": 100, "tiny_imagenet": 200}.get(
        dataset, 10
    )

    model = get_model(model_name, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(_strip_module_prefix(ckpt["model_state_dict"]))
    model.eval()

    print(
        f"  Loaded {ckpt_path.name}  (model={model_name}, dataset={dataset}, "
        f"epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', 0) * 100:.2f}%)"
    )
    return model, model_name, dataset, cfg.get("seed")


def _default_data_root(dataset: str) -> Path:
    """Matches data/datasets.py's own conventions: CIFAR/SVHN default to
    './data/raw', Tiny-ImageNet (via get_tiny_imagenet_loaders) to './data'."""
    if dataset == "tiny_imagenet":
        return PROJECT_ROOT / "data"
    return PROJECT_ROOT / "data" / "raw"


def _raw_val_dataset(dataset: str, root: str):
    """Un-normalised (0-1) test/val set — used both for GradCAM's RGB overlay
    and, after normalising, as the actual model input (matching training-time
    preprocessing)."""
    tf = torchvision.transforms.ToTensor()
    if dataset == "cifar100":
        return torchvision.datasets.CIFAR100(
            root, train=False, download=True, transform=tf
        )
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=tf
        )
    if dataset == "svhn":
        return torchvision.datasets.SVHN(
            root, split="test", download=True, transform=tf
        )
    if dataset == "tiny_imagenet":
        val_dir = Path(root) / "tiny-imagenet-200" / "val"
        _reorganize_tiny_imagenet_val(str(val_dir))  # no-op if already done
        return torchvision.datasets.ImageFolder(str(val_dir), transform=tf)
    raise ValueError(f"Unsupported dataset for Grad-CAM: '{dataset}'")


def _select_rows(valset, dataset: str, indices_override: list[int] | None):
    """Return list of (dataset_index, row_label).

    Uses one image per curated class (row label = class name) for CIFAR-100 /
    Tiny-ImageNet; falls back to plain numeric IMAGE_INDICES for any other
    dataset, or when an explicit indices override is passed.
    """
    if indices_override:
        return [(i, f"idx {i}") for i in indices_override]

    if dataset == "cifar100":
        targets = valset.targets
        rows = []
        for class_id, class_name in SELECTED_CLASSES_CIFAR100.items():
            idx = next((i for i, t in enumerate(targets) if t == class_id), None)
            if idx is None:
                print(f"  WARNING: no sample found for class {class_name!r}, skipping")
                continue
            rows.append((idx, class_name))
        return rows

    if dataset == "tiny_imagenet":
        rows = []
        for wnid, class_name in SELECTED_CLASSES_TINY_IMAGENET.items():
            class_idx = valset.class_to_idx.get(wnid)
            if class_idx is None:
                print(
                    f"  WARNING: wnid {wnid!r} ({class_name}) not in this val set, skipping"
                )
                continue
            idx = next(
                (i for i, t in enumerate(valset.targets) if t == class_idx), None
            )
            if idx is None:
                print(f"  WARNING: no sample found for class {class_name!r}, skipping")
                continue
            rows.append((idx, class_name))
        return rows

    return [(i, f"idx {i}") for i in IMAGE_INDICES]


def build_grid(
    *,
    grid_name: str,
    runs: dict[str, str],
    known_accs: dict[str, float],
    checkpoint_dir: Path,
    data_root: str | None,
    model_override: str | None,
    num_classes_override: int | None,
    indices_override: list[int] | None,
    output_path: Path,
):
    """Load `runs`' checkpoints, render one Original+methods grid, save it.

    Skips gracefully (prints a message, returns) if none of `runs`' checkpoints
    are found — callers loop over multiple grids without one missing dataset
    aborting the others.
    """
    print(f"\n=== {grid_name} ===")
    print("Loading checkpoints...")
    loaded = {}  # label -> (model, model_name, dataset, seed)
    for label, ckpt_name in runs.items():
        try:
            loaded[label] = load_checkpoint_model(
                resolve_checkpoint(checkpoint_dir, ckpt_name),
                model_override,
                num_classes_override,
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED {label!r}: {e}\n")

    if not loaded:
        available = "\n".join(
            f"  {p.name}" for p in sorted(checkpoint_dir.glob("*_best.pth"))
        )
        print(
            f"No checkpoints found for {grid_name} — skipping this grid.\n"
            f"Available in {checkpoint_dir}:\n{available}"
        )
        return

    datasets_used = {dataset for _, _, dataset, _ in loaded.values()}
    if len(datasets_used) > 1:
        raise ValueError(
            f"{grid_name}: checkpoints span different datasets ({datasets_used}) — "
            "Grad-CAM comparison requires the same validation images for every column."
        )
    dataset = datasets_used.pop()
    seed = next(iter(loaded.values()))[3]
    mean, std = CIFAR_STATS[dataset]["mean"], CIFAR_STATS[dataset]["std"]
    normalize = torchvision.transforms.Normalize(mean, std)

    root = data_root or str(_default_data_root(dataset))
    valset = _raw_val_dataset(dataset, root)
    rows = _select_rows(valset, dataset, indices_override)
    if not rows:
        print(f"No rows to visualise for {grid_name}.")
        return

    cams = {
        label: GradCAM(model=model, target_layers=_target_layer(model, model_name))
        for label, (model, model_name, _, _) in loaded.items()
    }

    n_rows, n_cols = len(rows), 1 + len(loaded)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Original Image", fontsize=13, fontweight="bold")
    for col, label in enumerate(loaded, start=1):
        title = label
        if label in known_accs:
            title += f"\n({known_accs[label]:.2f}%)"
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    print(
        f"Generating Grad-CAM overlays for {n_rows} images x {len(loaded)} methods..."
    )
    for row, (idx, row_label) in enumerate(rows):
        raw_tensor, label = valset[idx]  # (3, H, W), values in [0, 1]
        rgb_img = raw_tensor.permute(1, 2, 0).numpy()  # (H, W, 3) for display/overlay
        # Normalise for the model — matches training-time preprocessing; the raw
        # (un-normalised) tensor is only used for display, never fed to the model.
        input_tensor = normalize(raw_tensor).unsqueeze(0)
        targets = [ClassifierOutputTarget(int(label))]

        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_ylabel(row_label, fontsize=11, rotation=90)

        for col, method_label in enumerate(loaded, start=1):
            heatmap = cams[method_label](input_tensor=input_tensor, targets=targets)[0]
            axes[row, col].imshow(show_cam_on_image(rgb_img, heatmap, use_rgb=True))

        for c in range(n_cols):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            for spine in axes[row, c].spines.values():
                spine.set_visible(False)

    model_name = next(iter(loaded.values()))[1]
    model_display = {"wideresnet": "WideResNet-28-10"}.get(model_name, model_name)
    seed_str = f" · Seed {seed}" if seed is not None else ""
    fig.suptitle(
        f"Grad-CAM · Final Residual Block · {model_display} · {dataset.upper()}{seed_str}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    hd_path = png_path.with_name(f"{png_path.stem}_hd.png")
    plt.savefig(hd_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")
    print(f"Saved -> {png_path}")
    print(f"Saved -> {hd_path}")


def _parse_run_overrides(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    overrides = {}
    for item in items:
        label, _, ckpt_name = item.partition("=")
        overrides[label] = ckpt_name
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="LABEL=CKPT_NAME",
        help="Override/add a CIFAR-100 grid column. Repeatable; replaces "
        "RUNS_CIFAR100 entirely if given.",
    )
    parser.add_argument(
        "--run_tiny",
        action="append",
        dest="runs_tiny",
        metavar="LABEL=CKPT_NAME",
        help="Override/add a Tiny-ImageNet grid column. Repeatable; replaces "
        "RUNS_TINY_IMAGENET entirely if given.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="CIFAR-100 data root (default: data/raw)",
    )
    parser.add_argument(
        "--data_root_tiny",
        type=str,
        default=None,
        help="Tiny-ImageNet data root (default: data — expects data/tiny-imagenet-200/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name — only needed if a checkpoint has no 'cfg'",
    )
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit row indices for the CIFAR-100 grid (overrides SELECTED_CLASSES_CIFAR100)",
    )
    parser.add_argument(
        "--indices_tiny",
        type=int,
        nargs="+",
        default=None,
        help="Explicit row indices for the Tiny-ImageNet grid (overrides SELECTED_CLASSES_TINY_IMAGENET)",
    )
    parser.add_argument(
        "--skip_cifar100", action="store_true", help="Don't render the CIFAR-100 grid"
    )
    parser.add_argument(
        "--skip_tiny_imagenet",
        action="store_true",
        help="Don't render the Tiny-ImageNet grid",
    )
    parser.add_argument(
        "--output", type=str, default=str(SAVE_DIR / "gradcam_grid_cifar100.pdf")
    )
    parser.add_argument(
        "--output_tiny",
        type=str,
        default=str(SAVE_DIR / "gradcam_grid_tinyimagenet.pdf"),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not args.skip_cifar100:
        build_grid(
            grid_name="CIFAR-100",
            runs=_parse_run_overrides(args.runs) or RUNS_CIFAR100,
            known_accs=KNOWN_ACCS_CIFAR100,
            checkpoint_dir=checkpoint_dir,
            data_root=args.data_root,
            model_override=args.model,
            num_classes_override=args.num_classes,
            indices_override=args.indices,
            output_path=Path(args.output),
        )

    if not args.skip_tiny_imagenet:
        build_grid(
            grid_name="Tiny-ImageNet",
            runs=_parse_run_overrides(args.runs_tiny) or RUNS_TINY_IMAGENET,
            known_accs=KNOWN_ACCS_TINY_IMAGENET,
            checkpoint_dir=checkpoint_dir,
            data_root=args.data_root_tiny,
            model_override=args.model,
            num_classes_override=args.num_classes,
            indices_override=args.indices_tiny,
            output_path=Path(args.output_tiny),
        )


if __name__ == "__main__":
    main()
