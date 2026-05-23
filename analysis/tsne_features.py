"""
analysis/tsne_features.py

Extract 640-dim penultimate-layer features from trained WideResNet checkpoints
and produce two publication-quality t-SNE comparison figures:

  Grid figure (thesis):  2×3 grid — 5 method panels + separation-ratio bar chart
  Row figure (slides):   1×5 row — all methods side-by-side

Methods compared:
    No Augmentation | Static Mixing | ETS | LPS | EGS

The hook attaches to model.fc (Linear layer) and captures its INPUT —
the 640-dim global-average-pooled feature vector, identical to what the
classifier sees. No model architecture changes required.

Run on the cluster (checkpoints live there):
    source venv/bin/activate
    python analysis/tsne_features.py

Outputs:
    results/figs/tsne/tsne_grid.png          (150 dpi — thesis)
    results/figs/tsne/tsne_grid_hd.png       (300 dpi — print)
    results/figs/tsne/tsne_row.png           (150 dpi — slides)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE

from models.registry import get_model
from data.datasets import get_cifar100_loaders

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAVE_DIR = PROJECT_ROOT / "results" / "figs" / "tsne"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Checkpoints to compare ────────────────────────────────────────────────────
# Keys become panel titles. Values are checkpoint name prefixes (without _best.pth).
# Update these if your cluster checkpoint names differ.
# Layout: Row 1 = baselines, Row 2 = curriculum methods
RUNS = {
    "No Augmentation": "wideresnet_none_sgd_cosine_ep100_cifar100_s42",
    "Static Mixing": "wideresnet_static_mixing_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19",
}

# ── Class selection ───────────────────────────────────────────────────────────
# 15 semantically distinct CIFAR-100 classes (indices 0–99).
# Chosen for visual separation: mix of animals, vehicles, objects, nature.
SELECTED_CLASS_IDS = [4, 8, 13, 14, 23, 30, 31, 40, 55, 66, 72, 80, 81, 90, 96]
SELECTED_CLASS_NAMES = [
    "beaver",
    "bicycle",
    "bus",
    "butterfly",
    "cloud",
    "dolphin",
    "elephant",
    "forest",
    "man",
    "keyboard",
    "pickup_truck",
    "motorcycle",
    "mushroom",
    "rose",
    "sunflower",
]

N_PER_CLASS = 60  # 60 × 15 = 900 test points total (dense enough for t-SNE)
TSNE_PERPLEX = 30  # perplexity: 5–50 typical; 30 is standard
TSNE_ITER = 1000
RANDOM_STATE = 42

# ── Wong 2011 colorblind-safe palette (15 colours) ───────────────────────────
PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
    "#999999",
    "#117733",
    "#AA4499",
    "#44AA99",
    "#88CCEE",
    "#DDCC77",
    "#332288",
]


# ── Feature extraction ────────────────────────────────────────────────────────


def load_model(ckpt_name: str, device: torch.device) -> torch.nn.Module:
    """Load WideResNet weights from checkpoint. Raises FileNotFoundError if missing."""
    path = CHECKPOINT_DIR / f"{ckpt_name}_best.pth"
    if not path.exists():
        # Try without _best suffix (some runs save as plain .pth)
        path = CHECKPOINT_DIR / f"{ckpt_name}.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {CHECKPOINT_DIR / ckpt_name}_best.pth\n"
            f"Available checkpoints:\n"
            + "\n".join(f"  {p.stem}" for p in CHECKPOINT_DIR.glob("*_best.pth"))
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = get_model("wideresnet", num_classes=100).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {path.name}  (val_acc={ckpt.get('val_acc', 0) * 100:.2f}%)")
    return model


def extract_features(
    model: torch.nn.Module,
    loader,
    selected_ids: list,
    n_per_class: int,
    device: torch.device,
) -> tuple:
    """
    Run test images through the model and capture the 640-dim feature vector
    that feeds into model.fc (the penultimate representation).

    Hook strategy: register_forward_hook on model.fc captures its INPUT,
    which is the flattened global-average-pooled tensor — exactly what the
    linear classifier sees. No architecture changes needed.
    """
    collected_feats = {c: [] for c in selected_ids}
    hook_buffer = []

    def _hook(module, input, output):
        # input is a tuple; input[0] is the (B, 640) feature tensor
        hook_buffer.append(input[0].detach().cpu())

    handle = model.fc.register_forward_hook(_hook)
    id_set = set(selected_ids)

    with torch.no_grad():
        for images, labels in loader:
            hook_buffer.clear()
            images = images.to(device)
            _ = model(images)

            feats = hook_buffer[0]  # (B, 640)
            labels = labels.cpu().numpy()

            for i, lbl in enumerate(labels):
                if lbl in id_set and len(collected_feats[lbl]) < n_per_class:
                    collected_feats[lbl].append(feats[i].numpy())

            if all(len(v) >= n_per_class for v in collected_feats.values()):
                break

    handle.remove()

    all_feats, all_labels = [], []
    for c in selected_ids:
        all_feats.extend(collected_feats[c])
        all_labels.extend([c] * len(collected_feats[c]))

    return np.array(all_feats), np.array(all_labels)


# ── t-SNE ─────────────────────────────────────────────────────────────────────


def run_tsne(features: np.ndarray) -> np.ndarray:
    print(f"  Running t-SNE on {features.shape} …", flush=True)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEX,
        n_iter=TSNE_ITER,
        random_state=RANDOM_STATE,
        init="pca",  # PCA init is more stable than random
        learning_rate="auto",
        verbose=0,
    )
    return tsne.fit_transform(features)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _draw_tsne_panel(ax, emb, labels, title, id_to_colour, id_to_name, acc=None):
    """Draw one t-SNE scatter panel onto ax."""
    for cid in SELECTED_CLASS_IDS:
        mask = labels == cid
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=id_to_colour[cid],
            s=14,
            alpha=0.75,
            linewidths=0,
        )
    title_str = title if acc is None else f"{title}\n({acc:.2f}%)"
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f8f8f8")


def _legend_handles():
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE[i],
            markersize=9,
            label=SELECTED_CLASS_NAMES[i],
        )
        for i in range(len(SELECTED_CLASS_IDS))
    ]


def plot_grid(results: dict, metrics: dict, accs: dict):
    """
    2×3 grid figure for the thesis document.
    Row 1: No Augmentation | Static Mixing | ETS
    Row 2: LPS             | EGS           | Separation-ratio bar chart
    """
    id_to_colour = {cid: PALETTE[i] for i, cid in enumerate(SELECTED_CLASS_IDS)}
    id_to_name = dict(zip(SELECTED_CLASS_IDS, SELECTED_CLASS_NAMES))

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    titles = list(results.keys())
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]  # 5 panels

    for (r, c), title in zip(positions, titles):
        emb, labels = results[title]
        _draw_tsne_panel(
            axes[r, c],
            emb,
            labels,
            title,
            id_to_colour,
            id_to_name,
            acc=accs.get(title),
        )

    # 6th panel: separation ratio bar chart
    ax_bar = axes[1, 2]
    names = list(metrics.keys())
    ratios = [metrics[n]["separation_ratio"] for n in names]
    colours = ["#999999", "#D55E00", "#0072B2", "#009E73", "#E69F00"]
    bars = ax_bar.barh(
        names, ratios, color=colours[: len(names)], edgecolor="white", height=0.6
    )
    ax_bar.set_xlabel("Separation Ratio (Inter / Intra ↑)", fontsize=10)
    ax_bar.set_title("Representation Quality", fontsize=12, fontweight="bold", pad=8)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_facecolor("#f8f8f8")
    for bar, ratio in zip(bars, ratios):
        ax_bar.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # Shared class legend
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "t-SNE of Penultimate-Layer Features · WideResNet-28-10 · CIFAR-100 · "
        "15 classes · 60 samples each",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout(h_pad=2.5, w_pad=1.5)

    for dpi, suffix in [(150, ""), (300, "_hd")]:
        out = SAVE_DIR / f"tsne_grid{suffix}.png"
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close()


def plot_row(results: dict, accs: dict):
    """
    1×5 row figure for presentation slides.
    All methods side-by-side for direct visual comparison.
    """
    id_to_colour = {cid: PALETTE[i] for i, cid in enumerate(SELECTED_CLASS_IDS)}
    id_to_name = dict(zip(SELECTED_CLASS_IDS, SELECTED_CLASS_NAMES))

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0))
    if n == 1:
        axes = [axes]

    for ax, (title, (emb, labels)) in zip(axes, results.items()):
        _draw_tsne_panel(
            ax, emb, labels, title, id_to_colour, id_to_name, acc=accs.get(title)
        )

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Feature Space: t-SNE · WideResNet-28-10 · CIFAR-100",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out = SAVE_DIR / "tsne_row.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ── Intra-cluster distance metric ─────────────────────────────────────────────


def cluster_compactness(emb: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean intra-class distance in t-SNE space.
    Lower = tighter clusters = better representations.
    """
    dists = []
    for cid in SELECTED_CLASS_IDS:
        pts = emb[labels == cid]
        if len(pts) < 2:
            continue
        centre = pts.mean(axis=0)
        dists.append(np.linalg.norm(pts - centre, axis=1).mean())
    return float(np.mean(dists))


def inter_cluster_distance(emb: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean inter-class centroid distance.
    Higher = better-separated classes.
    """
    centres = np.array([emb[labels == c].mean(axis=0) for c in SELECTED_CLASS_IDS])
    n = len(centres)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(centres[i] - centres[j]))
    return float(np.mean(dists))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to data directory containing CIFAR-100",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Must match the val_split used in training (default: 0.1)",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    # Load test loader (standard CIFAR-100 normalisation, no augmentation)
    from torchvision import transforms

    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    _, _, test_loader = get_cifar100_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        train_transform=test_tf,
        test_transform=test_tf,
        num_workers=4,
    )
    print(f"Test set: {len(test_loader.dataset):,} images\n")

    results = {}
    metrics = {}
    accs = {}  # test accuracy per method for panel subtitles

    # Known test accuracies (seed 42, 19-op, 100ep) — shown under each panel title
    KNOWN_ACCS = {
        "No Augmentation": 72.86,
        "Static Mixing": 77.43,
        "ETS": 81.35,
        "LPS": 81.36,
        "EGS": 79.83,
    }

    for title, ckpt_name in RUNS.items():
        print(f"── {title} ──")
        try:
            model = load_model(ckpt_name, device)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}\n")
            continue

        feats, labels = extract_features(
            model, test_loader, SELECTED_CLASS_IDS, N_PER_CLASS, device
        )
        print(f"  Features extracted: {feats.shape}")

        emb = run_tsne(feats)
        results[title] = (emb, labels)
        accs[title] = KNOWN_ACCS.get(title)

        intra = cluster_compactness(emb, labels)
        inter = inter_cluster_distance(emb, labels)
        ratio = inter / max(intra, 1e-8)
        metrics[title] = {
            "intra_dist": intra,
            "inter_dist": inter,
            "separation_ratio": ratio,
        }
        print(f"  Intra: {intra:.2f}  |  Inter: {inter:.2f}  |  Ratio: {ratio:.2f}\n")

    if not results:
        print("No checkpoints found. Check RUNS dict and CHECKPOINT_DIR.")
        return

    print("\nGenerating figures …")
    plot_grid(results, metrics, accs)
    plot_row(results, accs)

    # Print summary table
    print("\n── Representation Quality Summary ──")
    print(f"{'Method':<22}  {'Acc':>7}  {'Intra↓':>8}  {'Inter↑':>8}  {'Ratio↑':>8}")
    print("─" * 60)
    for title, m in metrics.items():
        acc_str = f"{accs[title]:.2f}%" if accs.get(title) else "  —"
        print(
            f"{title:<22}  {acc_str:>7}  {m['intra_dist']:>8.2f}  "
            f"{m['inter_dist']:>8.2f}  {m['separation_ratio']:>8.2f}"
        )
    print()
    print(
        "Intra↓ = tighter clusters (better)  |  "
        "Inter↑ = more separated classes (better)  |  "
        "Ratio↑ = combined quality"
    )


if __name__ == "__main__":
    main()
