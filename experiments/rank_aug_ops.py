"""experiments/rank_aug_ops.py — rank augmentation ops by val-loss delta.

Two-phase evaluation:
  Phase 1 — sweep: evaluate every op at STRENGTHS levels, store the full
             Δloss-vs-strength curve.
  Phase 2 — calibrate: compute per-tier targets as the mean Δloss@1.0 of
             each manual tier (tier centroid), then interpolate to find each
             op's recommended_strength — the strength that makes it hit its
             tier's average difficulty.

No arbitrary thresholds: targets are derived entirely from the data.

The JSON is read by ThreeTierCurriculumTransform (--op_ranking_file) to enable
loss-based tier ordering and per-op calibrated strengths.

Usage:
    python -m experiments.rank_aug_ops \\
        --checkpoint checkpoints/<name>_best.pth \\
        --dataset cifar100
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from augmentations.primitives import AUGMENTATION_REGISTRY
from data.datasets import CIFAR_STATS, get_cifar10_loaders, get_cifar100_loaders
from experiments.utils import get_device, set_seed
from models.registry import get_model

STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Cumulative pool sizes matching the manual tier design (T1=4, T1+T2=11, T3=rest).
_T1_SIZE = 4
_T2_CUMULATIVE = 11


# ── helpers ──────────────────────────────────────────────────────────────────


class _SingleOpValTransform:
    """Apply one aug op at a given strength then normalise; op_name=None = baseline."""

    def __init__(self, dataset: str, op_name: str | None, strength: float = 1.0):
        stats = CIFAR_STATS[dataset]
        self.normalize = T.Normalize(stats["mean"], stats["std"])
        self.to_tensor = T.ToTensor()
        self.fn = AUGMENTATION_REGISTRY[op_name][0] if op_name else None
        self.strength = strength

    def __call__(self, img):
        if self.fn is not None:
            img = self.fn(img, strength=self.strength)
        return self.normalize(self.to_tensor(img))


def _get_val_loader(dataset, op_name, strength, data_root, batch_size):
    transform = _SingleOpValTransform(dataset, op_name, strength)
    loader_fn = get_cifar100_loaders if dataset == "cifar100" else get_cifar10_loaders
    _, val_loader, _ = loader_fn(
        root=data_root,
        batch_size=batch_size,
        val_split=0.1,
        train_transform=transform,
        test_transform=transform,
        num_workers=2,
    )
    return val_loader


def _eval_mean_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            total += F.cross_entropy(model(imgs), labels, reduction="sum").item()
            n += labels.size(0)
    return total / n


def _compute_tier_targets(
    raw_results: list[dict], z_thresh: float = 2.0
) -> tuple[dict[int, float], dict[str, float]]:
    """Tier centroid targets — mean Δloss@1.0 per manual tier, outlier-filtered.

    Ops whose Δloss@1.0 exceeds mean + z_thresh * std within their tier are
    excluded from the centroid to prevent a single extreme op (e.g. blur at
    max strength on 32×32 images) from inflating the target for the whole tier.

    Returns:
        targets  — {tier: centroid_delta}
        outliers — {op_name: delta_loss} ops excluded per tier
    """
    tier_data: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in raw_results:
        tier_data[r["manual_tier"]].append((r["name"], r["delta_loss"]))

    targets: dict[int, float] = {}
    outliers: dict[str, float] = {}

    for tier, items in sorted(tier_data.items()):
        deltas = [d for _, d in items]
        mean = sum(deltas) / len(deltas)
        std = (sum((d - mean) ** 2 for d in deltas) / len(deltas)) ** 0.5
        threshold = mean + z_thresh * std

        clean = [(name, d) for name, d in items if d <= threshold]
        excluded = [(name, d) for name, d in items if d > threshold]

        for name, d in excluded:
            outliers[name] = d

        clean_deltas = [d for _, d in clean]
        targets[tier] = round(sum(clean_deltas) / len(clean_deltas), 4)

    return targets, outliers


def _calibrate_strength(strengths: list, deltas: list, target: float) -> float:
    """Interpolate to find the strength where delta_loss first reaches target."""
    for i, (s, d) in enumerate(zip(strengths, deltas)):
        if d >= target:
            if i == 0:
                return round(s, 3)
            s0, d0 = strengths[i - 1], deltas[i - 1]
            frac = (target - d0) / max(1e-8, d - d0)
            return round(s0 + frac * (s - s0), 3)
    return 1.0  # never reached target — use max strength


def _manual_tier(op_name: str) -> int:
    return AUGMENTATION_REGISTRY[op_name][1]


# ── main evaluation ───────────────────────────────────────────────────────────


def rank_ops(
    checkpoint_path: str,
    dataset: str,
    data_root: str,
    batch_size: int,
    model_override: str | None = None,
) -> dict:
    device = get_device()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = model_override or ckpt.get("model_name") or "resnet18"
    num_classes = 100 if dataset == "cifar100" else 10
    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"\n  Model : {model_name}  |  Dataset : {dataset}  |  Device : {device}")

    baseline_loader = _get_val_loader(dataset, None, 1.0, data_root, batch_size)
    baseline_loss = _eval_mean_loss(model, baseline_loader, device)
    print(f"  Baseline loss (no aug): {baseline_loss:.4f}")
    print(
        f"\n  Phase 1 — sweeping {len(AUGMENTATION_REGISTRY)} ops × {len(STRENGTHS)} strengths\n"
    )

    # ── Phase 1: sweep ────────────────────────────────────────────────────────
    raw_results = []
    for op_name in AUGMENTATION_REGISTRY:
        m_tier = _manual_tier(op_name)
        delta_curve = []
        for s in STRENGTHS:
            loader = _get_val_loader(dataset, op_name, s, data_root, batch_size)
            loss = _eval_mean_loss(model, loader, device)
            delta_curve.append(round(loss - baseline_loss, 6))
        delta_at_max = delta_curve[-1]
        print(f"  {op_name:<22} Δ@1.0={delta_at_max:+.4f}  tier={m_tier}")
        raw_results.append(
            {
                "name": op_name,
                "delta_loss": delta_at_max,
                "manual_tier": m_tier,
                "strengths_tested": STRENGTHS,
                "delta_losses_by_strength": delta_curve,
            }
        )

    # ── Phase 2: calibrate ────────────────────────────────────────────────────
    tier_targets, outliers = _compute_tier_targets(raw_results)
    print("\n  Phase 2 — tier centroid targets (outlier-filtered mean Δloss@1.0):")
    for t, v in tier_targets.items():
        included = [
            r["name"]
            for r in raw_results
            if r["manual_tier"] == t and r["name"] not in outliers
        ]
        excluded = [
            r["name"]
            for r in raw_results
            if r["manual_tier"] == t and r["name"] in outliers
        ]
        print(f"    T{t} target = {v:.4f}  (from: {', '.join(included)})")
        if excluded:
            print(
                f"         outliers excluded (>2σ): {', '.join(f'{n}={outliers[n]:.3f}' for n in excluded)}"
            )

    results = []
    for r in raw_results:
        target = tier_targets[r["manual_tier"]]
        rec_s = _calibrate_strength(STRENGTHS, r["delta_losses_by_strength"], target)
        results.append(
            {
                **r,
                "recommended_strength": rec_s,
                "tier_target_used": target,
                "outlier_excluded": r["name"] in outliers,
            }
        )

    results.sort(key=lambda x: x["delta_loss"])
    for i, r in enumerate(results):
        r["loss_tier"] = 1 if i < _T1_SIZE else (2 if i < _T2_CUMULATIVE else 3)

    return {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "model": model_name,
            "dataset": dataset,
            "date": str(date.today()),
            "baseline_loss": round(baseline_loss, 6),
            "t1_pool_size": _T1_SIZE,
            "t2_cumulative_size": _T2_CUMULATIVE,
            "tier_delta_targets": tier_targets,
            "target_method": "tier_centroid_z2",
            "outliers_excluded": outliers,
        },
        "ops": results,
        "ranked_ops": [r["name"] for r in results],
    }


def _recompute_from_json(json_path: str) -> dict:
    """Reload an existing ranking JSON and redo Phase 2 only (no model/data needed)."""
    with open(json_path) as f:
        existing = json.load(f)

    raw_results = [
        {
            "name": r["name"],
            "delta_loss": r["delta_loss"],
            "manual_tier": r["manual_tier"],
            "strengths_tested": r["strengths_tested"],
            "delta_losses_by_strength": r["delta_losses_by_strength"],
        }
        for r in existing["ops"]
    ]

    tier_targets, outliers = _compute_tier_targets(raw_results)
    print("\n  Phase 2 (recomputed) — outlier-filtered tier centroid targets:")
    for t, v in tier_targets.items():
        included = [
            r["name"]
            for r in raw_results
            if r["manual_tier"] == t and r["name"] not in outliers
        ]
        excluded = [
            r["name"]
            for r in raw_results
            if r["manual_tier"] == t and r["name"] in outliers
        ]
        print(f"    T{t} target = {v:.4f}  (from: {', '.join(included)})")
        if excluded:
            print(
                f"         outliers excluded (>2σ): {', '.join(f'{n}={outliers[n]:.3f}' for n in excluded)}"
            )

    results = []
    for r in raw_results:
        target = tier_targets[r["manual_tier"]]
        rec_s = _calibrate_strength(STRENGTHS, r["delta_losses_by_strength"], target)
        results.append(
            {
                **r,
                "recommended_strength": rec_s,
                "tier_target_used": target,
                "outlier_excluded": r["name"] in outliers,
            }
        )

    results.sort(key=lambda x: x["delta_loss"])
    for i, r in enumerate(results):
        r["loss_tier"] = 1 if i < _T1_SIZE else (2 if i < _T2_CUMULATIVE else 3)

    existing["metadata"]["tier_delta_targets"] = tier_targets
    existing["metadata"]["target_method"] = "tier_centroid_z2"
    existing["metadata"]["outliers_excluded"] = outliers
    existing["ops"] = results
    existing["ranked_ops"] = [r["name"] for r in results]
    return existing


def main():
    parser = argparse.ArgumentParser(description="Rank aug ops by val-loss delta")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--dataset", default="cifar100", choices=["cifar10", "cifar100"]
    )
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--output", default="results/aug_op_ranking.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "resnet18",
            "resnet50",
            "wideresnet",
            "wrn16_8",
            "pyramidnet",
            "pyramidnet272",
        ],
        help="Override model architecture (needed when checkpoint lacks model_name)",
    )
    parser.add_argument(
        "--recompute",
        type=str,
        default=None,
        metavar="JSON",
        help="Skip sweep — reload an existing JSON and redo Phase 2 only (fast)",
    )
    args = parser.parse_args()

    set_seed(42)

    if args.recompute:
        print(f"\nRecomputing Phase 2 from existing JSON: {args.recompute}")
        ranking = _recompute_from_json(args.recompute)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --recompute is set")
        print(f"\nRanking ops — checkpoint: {args.checkpoint}")
        ranking = rank_ops(
            args.checkpoint, args.dataset, args.data_root, args.batch_size, args.model
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ranking, f, indent=2)

    targets = ranking["metadata"]["tier_delta_targets"]
    print(f"\n{'─' * 68}")
    print(f"Saved → {out}")
    print(
        f"\n  Tier targets (centroid): T1={targets[1]}  T2={targets[2]}  T3={targets[3]}"
    )
    print(
        f"\n  {'Op':<22} {'Δ@1.0':>8}  {'target':>7}  {'rec_s':>6}  loss_t  manual_t  match?"
    )
    print(f"  {'─' * 70}")
    matches = 0
    for r in ranking["ops"]:
        ok = r["loss_tier"] == r["manual_tier"]
        matches += ok
        print(
            f"  {r['name']:<22} {r['delta_loss']:>+8.4f}"
            f"  {r['tier_target_used']:>7.4f}  {r['recommended_strength']:>6.2f}"
            f"      {r['loss_tier']}       {r['manual_tier']}     {'✓' if ok else '✗'}"
        )
    n = len(ranking["ops"])
    print(f"\n  Tier agreement: {matches}/{n} ({100 * matches / n:.0f}%)")


if __name__ == "__main__":
    main()
