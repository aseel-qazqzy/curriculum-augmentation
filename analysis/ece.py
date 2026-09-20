# Expected Calibration Error
import argparse
import sys
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")

from data.datasets import get_cifar100_loaders
from models.registry import get_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
FIGURES_DIR = _PROJECT_ROOT / "results" / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BINS = 10

# Full-length seed-42 checkpoints (cluster) — same names as
# analysis/tsne_features.py's RUNS dict and analysis/gradcam_comparison.py's
# RUNS_CIFAR100 (minus "No Augmentation"), so all three figures describe the
# exact same set of runs. Any entry not found in --checkpoint_dir is skipped
# with a warning rather than aborting the whole comparison.
RUNS = {
    "Static Mixing": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19",
}


def resolve_checkpoint(checkpoint_dir: Path, name: str) -> Path:
    path = checkpoint_dir / f"{name}_best.pth"
    if path.exists():
        return path
    return checkpoint_dir / f"{name}.pth"


def bin_calibration_stats(confidences, correct, n_bins=N_BINS):
    """Bin (confidence, correctness) pairs into n_bins equal-width bins.

    The last bin is right-inclusive ([0.9, 1.0]) so samples with confidence
    exactly 1.0 (common with float32 softmax on confidently-trained models)
    are counted instead of silently dropped.

    Returns a list of dicts: {lo, hi, count, avg_conf, avg_acc}, one per bin
    (bins with zero samples are omitted).
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = [j for j, c in enumerate(confidences) if lo <= c <= hi]
        else:
            in_bin = [j for j, c in enumerate(confidences) if lo <= c < hi]
        if not in_bin:
            continue
        avg_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        avg_acc = sum(correct[j] for j in in_bin) / len(in_bin)
        stats.append(
            {
                "lo": lo,
                "hi": hi,
                "count": len(in_bin),
                "avg_conf": avg_conf,
                "avg_acc": avg_acc,
            }
        )
    return stats


def compute_ece(confidences, correct, n_bins=N_BINS):
    stats = bin_calibration_stats(confidences, correct, n_bins)
    total = len(confidences)
    return sum(b["count"] / total * abs(b["avg_conf"] - b["avg_acc"]) for b in stats)


def fit_temperature(model, val_loader, device, max_iter=50, lr=0.01):
    """Fit a single scalar T minimizing NLL on held-out (validation) logits.

    Dividing logits by a positive scalar doesn't change argmax, so this only
    rescales confidence — accuracy is unaffected. T is fit on val, not test,
    for the same reason any other hyperparameter is: it must not see test data.
    """
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            logits_list.append(model(images.to(device)).cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = torch.nn.Parameter(torch.ones(1))
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


# Wong (2011) colorblind-safe pair — reused from analysis/plot_curves.py's PALETTE.
ACC_COLOR = "#0072B2"  # observed accuracy bar
GAP_COLOR = "#D55E00"  # miscalibration gap (|confidence - accuracy|)

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
    }
)


def _plot_panel(ax, confidences, correct, n_bins):
    """Draw one reliability-diagram panel: accuracy bar + miscalibration-gap
    hatch bar per confidence bin, against the perfect-calibration diagonal.
    """
    bin_width = 1.0 / n_bins
    stats = bin_calibration_stats(confidences, correct, n_bins)
    centers = [(b["lo"] + b["hi"]) / 2 for b in stats]
    accs = [b["avg_acc"] for b in stats]
    confs = [b["avg_conf"] for b in stats]
    gaps = [c - a for c, a in zip(confs, accs)]

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2, zorder=1)
    ax.bar(
        centers,
        accs,
        width=bin_width * 0.9,
        color=ACC_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Accuracy",
        zorder=2,
    )
    ax.bar(
        centers,
        gaps,
        width=bin_width * 0.9,
        bottom=accs,
        color=GAP_COLOR,
        alpha=0.6,
        hatch="//",
        edgecolor="black",
        linewidth=0.5,
        label="Gap",
        zorder=3,
    )

    ece = compute_ece(confidences, correct, n_bins)
    acc = sum(correct) / len(correct) * 100
    ax.text(
        0.05,
        0.95,
        f"ECE={ece:.4f}\nAcc={acc:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_reliability_diagram(
    results: dict,
    results_calibrated: dict = None,
    n_bins: int = N_BINS,
    save_path: Path = None,
):
    """Reliability-diagram grid, one column per method.

    `results` / `results_calibrated` map method name -> {"confidences": [...], "correct": [...]}.
    If `results_calibrated` is given, plots two rows (raw on top, temperature-
    scaled below) so the calibration fix can be compared directly.
    """
    n_rows = 2 if results_calibrated else 1
    fig, axes = plt.subplots(
        n_rows,
        len(results),
        figsize=(4 * len(results), 4 * n_rows),
        sharey=True,
        squeeze=False,
    )

    for col, (name, data) in enumerate(results.items()):
        _plot_panel(axes[0][col], data["confidences"], data["correct"], n_bins)
        axes[0][col].set_title(name)
        if results_calibrated:
            cdata = results_calibrated[name]
            _plot_panel(axes[1][col], cdata["confidences"], cdata["correct"], n_bins)
            axes[1][col].set_xlabel("Confidence")
        else:
            axes[0][col].set_xlabel("Confidence")

    axes[0][0].set_ylabel("Raw" if results_calibrated else "Accuracy")
    if results_calibrated:
        axes[1][0].set_ylabel("Temperature-scaled")
    axes[0][-1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()

    save_path = save_path or (FIGURES_DIR / "reliability_diagram.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {save_path}")


def _parse_run_overrides(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    overrides = {}
    for item in items:
        label, _, ckpt_name = item.partition("=")
        overrides[label] = ckpt_name
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="LABEL=CKPT_NAME",
        help="Override/add a method, e.g. --run 'EGS=my_egs_checkpoint'. "
        "Repeatable; replaces the default RUNS dict entirely if given.",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--output", type=str, default=str(FIGURES_DIR / "reliability_diagram.png")
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    runs = _parse_run_overrides(args.runs) or RUNS

    _, val_loader, test_loader = get_cifar100_loaders(
        root=args.data_root,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=0,
    )

    print(f"{'Method':<14}  {'ECE':>6}  {'ECE(T)':>7}  {'T':>5}  {'Acc':>6}")
    print("-" * 46)

    all_results, all_results_calibrated = {}, {}
    for name, ckpt_name in runs.items():
        ckpt_path = resolve_checkpoint(checkpoint_dir, ckpt_name)
        if not ckpt_path.exists():
            print(f"{name:<14}  (checkpoint not found: {ckpt_path.name})")
            continue

        model = get_model("wideresnet", num_classes=100).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        T = fit_temperature(model, val_loader, DEVICE)

        all_confidences, all_confidences_T, all_correct = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                confidence, predicted = F.softmax(logits, dim=1).max(dim=1)
                confidence_T, _ = F.softmax(logits / T, dim=1).max(
                    dim=1
                )  # argmax unchanged by scaling
                all_confidences.extend(confidence.cpu().tolist())
                all_confidences_T.extend(confidence_T.cpu().tolist())
                all_correct.extend((predicted == labels).cpu().tolist())

        ece = compute_ece(all_confidences, all_correct)
        ece_T = compute_ece(all_confidences_T, all_correct)
        acc = sum(all_correct) / len(all_correct) * 100
        print(f"{name:<14}  {ece:>6.4f}  {ece_T:>7.4f}  {T:>5.3f}  {acc:>5.2f}%")
        all_results[name] = {"confidences": all_confidences, "correct": all_correct}
        all_results_calibrated[name] = {
            "confidences": all_confidences_T,
            "correct": all_correct,
        }

    if all_results:
        plot_reliability_diagram(
            all_results, all_results_calibrated, save_path=Path(args.output)
        )
    else:
        print("\nNo checkpoints found. Check RUNS / --run and --checkpoint_dir.")


if __name__ == "__main__":
    main()