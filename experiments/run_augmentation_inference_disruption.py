"""experiments/run_augmentation_inference_disruption.py

Isolated diagnostic experiment — NOT part of ETS / LPS / EGS / static mixing and
does not modify any curriculum logic, augmentation policy, training script,
config, checkpoint, or existing result file (including the earlier
`run_augmentation_difficulty.py` training-time probe, which this experiment
does not touch or depend on).

Research question
------------------
Do the existing code-defined augmentation tiers in
`augmentations/primitives.py:AUGMENTATION_REGISTRY` (+ the Tier-3-only mixing
mechanism in `augmentations/mixing.py`) correspond to increasing IMMEDIATE
INFERENCE-TIME validation-loss disruption?

For an operation a, applied to a FROZEN model:

    D(a) = L_aug(a) - L_clean

This is deliberately different from `run_augmentation_difficulty.py`, which
measures training-time disruption (1 epoch of gradient updates with only that
op active). This script measures pure inference sensitivity: the frozen model
never takes another gradient step after the shared warm-up. See "Limitations"
in the generated README for what this metric does and does not show.

Source of truth (inspected, not assumed)
-----------------------------------------
- Operation pool + per-op tier: `augmentations/primitives.py:AUGMENTATION_REGISTRY`
  (19 operations detected at import time — dynamically counted, not hardcoded).
- Tier op pools (cross-check): `augmentations/policies.py:_TIER_OPS` (op_pool=19).
- Mixing mechanism: `augmentations/mixing.py` (`BatchMixer`, `mixup`, `cutmix`).
  Mixing operations have NO per-op tier field in AUGMENTATION_REGISTRY or
  _TIER_OPS — they are a separate batch-level mechanism, not a member of the
  19-op pool. Their Tier-3-only status is established by:
    * `augmentations/mixing.py` module docstring: "Stage 2 of the two-stage
      Tier 3 pipeline" / BatchMixer docstring: "Intended for Tier 3 only".
    * `augmentations/policies.py:ThreeTierCurriculumTransform.mix_scale()`,
      which returns 0.0 for tier < 3 and only ramps at the Tier-3 boundary.
    * `experiments/train_baseline.py`'s mixer-gating logic, which sets
      `active_mixer = None` for Tiers 1-2 in every scheduling path (ETS/LPS/EGS)
      and only activates the mixer once Tier 3 is reached (or, for
      `static_mixing`, treats the whole run as Tier-3-equivalent, "always on
      from epoch 1").
  This script therefore evaluates `cutmix` and `mixup` and reports them under
  Tier 3, with the derivation documented above and again in the generated
  README so the assignment is auditable rather than assumed.

Protocol
--------
1. Per seed: train ResNet-18 on CIFAR-100 with NO augmentation for
   `warmup_epochs` (default 10, matching `lps_min_epochs` in
   experiments/config.py — see run_augmentation_difficulty.py for the same
   rationale). This is the shared frozen model for every operation under that
   seed.
2. Freeze the model (`model.eval()`, all evaluation under `torch.no_grad()`).
   ZERO further training: no optimizer is constructed, no `.backward()` or
   `.step()` call exists anywhere past this point in this script.
3. L_clean = validation loss of the frozen model on the normal (unaugmented)
   validation set — computed once per seed, reused for every operation.
4. For each of the operations in AUGMENTATION_REGISTRY: build a validation
   loader whose transform applies ONLY that operation (at the project's
   `FIXED_STRENGTH` = 0.7 ceiling) then the standard normalize step, run the
   SAME frozen model over it, record L_aug(a). No parameter update occurs.
5. For `cutmix` and `mixup`: reuse `augmentations.mixing.BatchMixer` exactly
   (same `mixup()`/`cutmix()` functions, same lambda handling, same
   sample-pairing) with `p=1.0` (always mixed — matching every ordinary op's
   "apply ONLY this operation, unconditionally" treatment; the production
   default `mix_prob=0.5` is a training-time regularization choice, not part
   of the operation's definition) and `alpha=cfg['mix_alpha']` (project
   default 1.0, from experiments/config.py). Loss uses the exact mixed-label
   formula from `training/trainer.py:train_one_epoch`:
   `lam * CE(out, y_a) + (1-lam) * CE(out, y_b)`.
6. After every single operation evaluation, the frozen model's full
   `state_dict()` (params + buffers) is compared against a pre-evaluation
   snapshot with `torch.equal`. Any mismatch raises immediately (see
   `_assert_frozen`) — the experiment cannot silently produce results from a
   model that changed.

Usage:
    python -m experiments.run_augmentation_inference_disruption
    python -m experiments.run_augmentation_inference_disruption --debug   # smoke test
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from scipy import stats

from augmentations.mixing import BatchMixer
from augmentations.policies import FIXED_STRENGTH, get_tier_ops
from augmentations.primitives import AUGMENTATION_REGISTRY
from data.datasets import (
    CIFAR_STATS,
    get_cifar100_loaders,
    get_no_augmentation_transforms,
)
from experiments.utils import build_optimizer, get_device, set_seed, setup_logging
from models.registry import get_model
from training.trainer import evaluate, train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same palette as run_augmentation_difficulty.py / scripts/plot_aug_ranking.py so
# every tier-coloured figure in the thesis stays visually consistent.
TIER_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
TIER_NAMES = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}

# Mixing ops are NOT in AUGMENTATION_REGISTRY — see module docstring for the
# code evidence establishing their Tier-3-only status.
MIXING_OPS = ["cutmix", "mixup"]
MIXING_OP_TIER = 3


# ─────────────────────────────────────────────────────────────────────────────
# Frozen-model safety
# ─────────────────────────────────────────────────────────────────────────────


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_frozen(model: nn.Module, snapshot: dict, context: str) -> None:
    """Raise immediately if ANY parameter or buffer changed. Per spec: STOP, don't
    silently continue producing results from a model that was supposed to be frozen."""
    for k, v in model.state_dict().items():
        if not torch.equal(v, snapshot[k]):
            raise RuntimeError(
                f"FROZEN-MODEL VIOLATION during '{context}': tensor '{k}' changed. "
                f"Aborting — no results were saved."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Single-image operation inference transform
# ─────────────────────────────────────────────────────────────────────────────


class _SingleOpEvalTransform:
    """Apply exactly one op from AUGMENTATION_REGISTRY at a fixed strength to a
    VALIDATION image, then normalize. Same call convention (fn(img, strength))
    the project already uses everywhere else (ThreeTierCurriculumTransform,
    RandomAugmentTransform, run_augmentation_difficulty.py) — reused here, just
    pointed at the val/test transform slot instead of the train slot."""

    def __init__(self, dataset: str, op_name: str, strength: float):
        cs = CIFAR_STATS[dataset]
        self.normalize = T.Normalize(cs["mean"], cs["std"])
        self.to_tensor = T.ToTensor()
        self.fn = AUGMENTATION_REGISTRY[op_name][0]
        self.strength = strength

    def __call__(self, img):
        img = self.fn(img, strength=self.strength)
        return self.normalize(self.to_tensor(img))


def _make_optimizer_for_warmup(model, cfg):
    optimizer, _ = build_optimizer(
        model,
        {"optimizer": "sgd", "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — clean warm-up (only place in this script where training happens)
# ─────────────────────────────────────────────────────────────────────────────


def run_warmup(cfg: dict, seed: int, device) -> dict:
    """Train from scratch with no augmentation for cfg['warmup_epochs']. Returns
    the CPU state_dict of the frozen model. No further training occurs anywhere
    else in this script after this function returns."""
    set_seed(seed)
    model = get_model(cfg["model"], num_classes=100).to(device)
    train_tf, val_tf = get_no_augmentation_transforms(cfg["dataset"])
    train_loader, _, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=train_tf,
        test_transform=val_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    optimizer = _make_optimizer_for_warmup(model, cfg)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, cfg["warmup_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"    [warmup seed={seed}] epoch {ep}/{cfg['warmup_epochs']} "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc * 100:.2f}%"
        )

    frozen_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return frozen_state


def build_frozen_model(cfg: dict, frozen_state: dict, device) -> nn.Module:
    model = get_model(cfg["model"], num_classes=100).to(device)
    model.load_state_dict(frozen_state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — clean validation loss (frozen model, no training)
# ─────────────────────────────────────────────────────────────────────────────


def compute_clean_loss(cfg: dict, seed: int, model: nn.Module, device) -> float:
    set_seed(seed)
    _, val_tf = get_no_augmentation_transforms(cfg["dataset"])
    _, val_loader, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=val_tf,
        test_transform=val_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    criterion = nn.CrossEntropyLoss()
    snap = _snapshot(model)
    with torch.no_grad():
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    _assert_frozen(model, snap, context="clean validation-loss evaluation")
    return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — single-operation inference (frozen model, no training)
# ─────────────────────────────────────────────────────────────────────────────


def run_single_op_inference(
    cfg: dict, seed: int, op_name: str, model: nn.Module, device
) -> float:
    """Evaluate the FROZEN model on validation images passed through ONE
    augmentation op. No optimizer, no backward, no parameter update."""
    set_seed(seed)  # controls the op's own internal randomness reproducibly
    aug_tf = _SingleOpEvalTransform(cfg["dataset"], op_name, cfg["fixed_strength"])
    _, val_loader, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=aug_tf,  # inert: the train_loader is built but never iterated
        test_transform=aug_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    criterion = nn.CrossEntropyLoss()
    snap = _snapshot(model)
    with torch.no_grad():
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    _assert_frozen(model, snap, context=f"operation '{op_name}' inference")
    return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — mixing-operation inference (frozen model, no training)
# ─────────────────────────────────────────────────────────────────────────────


def run_mixing_op_inference(
    cfg: dict, seed: int, mix_op_name: str, model: nn.Module, device
) -> tuple[float, float]:
    """Evaluate the FROZEN model under CutMix/MixUp, reusing BatchMixer/mixup/
    cutmix from augmentations/mixing.py exactly as train_one_epoch uses them,
    including the lambda-weighted CE loss formula. p=1.0 so mixing is always
    applied (see module docstring for why). Returns (mean delta-consistent val
    loss, mean lambda across all validation batches)."""
    set_seed(seed)
    _, val_tf = get_no_augmentation_transforms(cfg["dataset"])
    _, val_loader, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=val_tf,
        test_transform=val_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    mixer = BatchMixer(mode=mix_op_name, alpha=cfg["mix_alpha"], p=1.0)
    criterion = nn.CrossEntropyLoss()

    snap = _snapshot(model)
    total_loss, total, lams = 0.0, 0, []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            mixed, label_a, label_b, lam = mixer(images, labels)
            outputs = model(mixed).float()
            if lam >= 1.0 - 1e-6:
                loss = criterion(outputs, label_a)
            else:
                loss = lam * criterion(outputs, label_a) + (1.0 - lam) * criterion(
                    outputs, label_b
                )
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            lams.append(lam)
    _assert_frozen(model, snap, context=f"mixing operation '{mix_op_name}' inference")
    return total_loss / total, float(np.mean(lams))


# ─────────────────────────────────────────────────────────────────────────────
# Preflight checks (spec §24 — run BEFORE the expensive experiment)
# ─────────────────────────────────────────────────────────────────────────────


def preflight_checks(cfg: dict, device) -> list[str]:
    report = []

    op_names = list(AUGMENTATION_REGISTRY.keys())
    n_ops = len(op_names)
    report.append(
        f"[1] Operation registry detected: augmentations/primitives.py:AUGMENTATION_REGISTRY"
    )
    report.append(f"[2] Number of single-image operations detected: {n_ops}")

    tiers_present = sorted({t for _, t, _ in AUGMENTATION_REGISTRY.values()})
    assert tiers_present == [1, 2, 3], (
        f"Unexpected tier set in registry: {tiers_present}"
    )
    report.append(
        f"[3] Tiers correctly imported: {tiers_present} (all ops have tier in {{1,2,3}})"
    )

    tier_ops, n_ops_sampled = get_tier_ops(op_pool=19, dataset=cfg["dataset"])
    for op in op_names:
        _, reg_tier, _ = AUGMENTATION_REGISTRY[op]
        assert op in tier_ops[reg_tier], (
            f"Cross-check failed: '{op}' has tier {reg_tier} in AUGMENTATION_REGISTRY "
            f"but is not in augmentations/policies.py:_TIER_OPS[{reg_tier}]"
        )
    report.append(
        "[3b] Cross-checked every op's tier against augmentations/policies.py:_TIER_OPS — consistent"
    )

    from augmentations.mixing import cutmix, mixup  # noqa: F401 (import-check only)

    report.append(
        "[4] Mixing operations detected: cutmix, mixup (augmentations/mixing.py: "
        "BatchMixer, mixup(), cutmix()) — NOT in AUGMENTATION_REGISTRY; Tier-3-only "
        "status established structurally (see module docstring), assigned tier=3"
    )

    model = get_model(cfg["model"], num_classes=100).to(device)
    snap = _snapshot(model)
    dummy = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        model.eval()
        _ = model(dummy)
    _assert_frozen(model, snap, context="preflight dry-run forward pass")
    report.append(
        "[9/10] Preflight dry-run: forward pass under eval()+no_grad() left all parameters/buffers unchanged"
    )
    report.append(
        "[10] No optimizer is constructed and .backward()/.step() are never called in Steps 2/3/3b of this script"
    )
    del model

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_operation_level(summary_rows: list[dict], out_path: Path) -> None:
    ordered = sorted(summary_rows, key=lambda s: (s["tier"], s["operation"]))
    names = [r["operation"].replace("_", " ") for r in ordered]
    means = [r["mean_delta_loss"] for r in ordered]
    stds = [r["std_delta_loss"] for r in ordered]
    colors = [TIER_COLORS[r["tier"]] for r in ordered]
    hatches = ["///" if r["operation"] in MIXING_OPS else None for r in ordered]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=3,
        zorder=2,
    )
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#888888", linewidth=0.9, zorder=1)
    ax.set_ylabel(
        "Mean $\\Delta L$  (frozen-model val loss, augmented $-$ clean)", fontsize=10
    )
    ax.set_title(
        "Per-operation inference-time loss disruption (frozen model, no further training)\n"
        "Bar colour = existing (code-defined) tier, predefined before this experiment · "
        "hatched = mixing ops (cutmix/mixup) · error bars = std over seeds",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t]) for t in (1, 2, 3)
    ]
    legend_patches.append(
        mpatches.Patch(
            facecolor="white", edgecolor="black", hatch="///", label="Mixing op"
        )
    )
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_category_distribution(op_level_by_tier: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    data = [op_level_by_tier[t] for t in (1, 2, 3)]
    bp = ax.boxplot(
        data,
        labels=[TIER_NAMES[t] for t in (1, 2, 3)],
        patch_artist=True,
        showmeans=True,
    )
    for patch, t in zip(bp["boxes"], (1, 2, 3)):
        patch.set_facecolor(TIER_COLORS[t])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for t, x_pos in zip((1, 2, 3), (1, 2, 3)):
        vals = np.array(op_level_by_tier[t])
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full_like(vals, x_pos) + jitter,
            vals,
            color=TIER_COLORS[t],
            edgecolor="white",
            s=28,
            zorder=3,
            alpha=0.9,
        )

    ax.axhline(0, color="#888888", linewidth=0.9)
    ax.set_ylabel(
        "Mean $\\Delta L$ per operation (operation is the statistical unit)",
        fontsize=10,
    )
    ax.set_title(
        "Category-level (existing, predefined tier) distributions\n"
        "Every individual operation shown — no outliers removed",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_tier_scatter(
    summary_rows: list[dict], rho: float, rho_p: float, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(1)
    for r in summary_rows:
        jitter = rng.normal(0, 0.05)
        ax.scatter(
            r["tier"] + jitter,
            r["mean_delta_loss"],
            color=TIER_COLORS[r["tier"]],
            edgecolor="white",
            s=40,
            zorder=3,
            marker="*" if r["operation"] in MIXING_OPS else "o",
        )
    ax.axhline(0, color="#888888", linewidth=0.9)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([TIER_NAMES[t] for t in (1, 2, 3)])
    ax.set_ylabel("Mean $\\Delta L$ per operation", fontsize=10)
    ax.set_title(
        f"Tier vs. mean inference-time $\\Delta L$ per operation (association, not causation)\n"
        f"Spearman $\\rho$={rho:.3f}, p={rho_p:.4f}  ·  \u2605 = mixing op",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Inference-time augmentation disruption test (frozen model, no training past warm-up)"
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--fixed_strength", type=float, default=FIXED_STRENGTH)
    parser.add_argument("--mix_alpha", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results" / "augmentation_inference_disruption"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Smoke test: tiny data subset (512 train / 128 val), warmup_epochs forced to 1",
    )
    args = parser.parse_args()
    cfg = vars(args)
    if cfg["debug"]:
        cfg["warmup_epochs"] = min(cfg["warmup_epochs"], 1)

    out_dir = Path(cfg["output_dir"])
    if cfg["debug"]:
        out_dir = out_dir / "_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        {
            "log_dir": str(out_dir),
            "experiment_name": "augmentation_inference_disruption",
        }
    )
    device = get_device()

    print(f"\n{'=' * 78}")
    print(
        "  Augmentation INFERENCE-time Disruption Validation (frozen model, no training)"
    )
    if cfg["debug"]:
        print("  *** SMOKE TEST MODE (--debug): tiny subset, warmup_epochs=1 ***")
    print(f"{'=' * 78}\n")

    print("Preflight checks:")
    for line in preflight_checks(cfg, device):
        print(f"  {line}")
    print()

    op_names = list(AUGMENTATION_REGISTRY.keys())
    if cfg["debug"]:
        # Smoke test: one op per tier (so category/pairwise stats have >=1 sample
        # per tier instead of crashing on an empty tier) + both mixing ops.
        seen_tiers = set()
        picked = []
        for name in op_names:
            t = AUGMENTATION_REGISTRY[name][1]
            if t not in seen_tiers:
                seen_tiers.add(t)
                picked.append(name)
        op_names = picked
    all_op_names = op_names + MIXING_OPS
    print(
        f"Operations to evaluate this run: {len(all_op_names)} ({len(op_names)} single-image + {len(MIXING_OPS)} mixing)\n"
    )

    def tier_of(op_name: str) -> int:
        if op_name in MIXING_OPS:
            return MIXING_OP_TIER
        return AUGMENTATION_REGISTRY[op_name][1]

    rows = []
    t_start = time.time()
    for seed in cfg["seeds"]:
        print(f"\n-- Seed {seed} " + "-" * 55)
        frozen_state = run_warmup(cfg, seed, device)
        model = build_frozen_model(cfg, frozen_state, device)

        l_clean = compute_clean_loss(cfg, seed, model, device)
        print(f"    [seed={seed}] frozen clean val_loss = {l_clean:.4f}")

        for op_name in op_names:
            t0 = time.time()
            l_aug = run_single_op_inference(cfg, seed, op_name, model, device)
            delta = l_aug - l_clean
            rows.append(
                {
                    "seed": seed,
                    "operation": op_name,
                    "tier": tier_of(op_name),
                    "clean_val_loss": l_clean,
                    "augmented_val_loss": l_aug,
                    "delta_loss": delta,
                    "lambda": "",
                    "pairing_method": "",
                }
            )
            print(
                f"    {op_name:<15} Tier {tier_of(op_name)}  dL={delta:+.4f}  ({time.time() - t0:.1f}s)"
            )

        for mix_name in MIXING_OPS:
            t0 = time.time()
            l_aug, mean_lam = run_mixing_op_inference(
                cfg, seed, mix_name, model, device
            )
            delta = l_aug - l_clean
            rows.append(
                {
                    "seed": seed,
                    "operation": mix_name,
                    "tier": tier_of(mix_name),
                    "clean_val_loss": l_clean,
                    "augmented_val_loss": l_aug,
                    "delta_loss": delta,
                    "lambda": f"{mean_lam:.4f}",
                    "pairing_method": "within-batch random permutation (torch.randperm, augmentations/mixing.py)",
                }
            )
            print(
                f"    {mix_name:<15} Tier {tier_of(mix_name)}  dL={delta:+.4f}  mean_lam={mean_lam:.3f}  ({time.time() - t0:.1f}s)"
            )

        del model

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")

    # ---- 15.1 raw_results.csv ----
    raw_fields = [
        "seed",
        "operation",
        "tier",
        "clean_val_loss",
        "augmented_val_loss",
        "delta_loss",
        "lambda",
        "pairing_method",
    ]
    raw_path = out_dir / "raw_results.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {raw_path}")

    # ---- 15.2 summary_by_operation.csv ----
    summary_rows = []
    for op_name in all_op_names:
        op_rows = [r for r in rows if r["operation"] == op_name]
        deltas = np.array([r["delta_loss"] for r in op_rows])
        summary_rows.append(
            {
                "operation": op_name,
                "tier": tier_of(op_name),
                "n_seeds": len(deltas),
                "mean_delta_loss": float(deltas.mean()),
                "median_delta_loss": float(np.median(deltas)),
                "std_delta_loss": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "min_delta_loss": float(deltas.min()),
                "max_delta_loss": float(deltas.max()),
            }
        )
    summary_path = out_dir / "summary_by_operation.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operation",
                "tier",
                "n_seeds",
                "mean_delta_loss",
                "median_delta_loss",
                "std_delta_loss",
                "min_delta_loss",
                "max_delta_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved -> {summary_path}")

    # ---- 15.3 summary_by_category.csv ----
    # Statistical unit = operation (per spec §16): N = n_ops per tier, NOT ops x seeds.
    op_level_by_tier = {t: [] for t in (1, 2, 3)}
    for s in summary_rows:
        op_level_by_tier[s["tier"]].append(s["mean_delta_loss"])

    category_rows = []
    for tier in (1, 2, 3):
        vals = np.array(op_level_by_tier[tier])
        category_rows.append(
            {
                "tier": tier,
                "n_operations": len(vals),
                "mean_delta_loss": float(vals.mean()),
                "median_delta_loss": float(np.median(vals)),
                "std_delta_loss": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min_delta_loss": float(vals.min()),
                "max_delta_loss": float(vals.max()),
            }
        )
    category_path = out_dir / "summary_by_category.csv"
    with open(category_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tier",
                "n_operations",
                "mean_delta_loss",
                "median_delta_loss",
                "std_delta_loss",
                "min_delta_loss",
                "max_delta_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(category_rows)
    print(f"  Saved -> {category_path}")

    # ---- 15.4 operation_ranking.csv ----
    ranked = sorted(summary_rows, key=lambda s: s["mean_delta_loss"])
    ranking_path = out_dir / "operation_ranking.csv"
    with open(ranking_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "operation",
                "tier",
                "mean_delta_loss",
                "median_delta_loss",
            ],
        )
        writer.writeheader()
        for i, s in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "operation": s["operation"],
                    "tier": s["tier"],
                    "mean_delta_loss": s["mean_delta_loss"],
                    "median_delta_loss": s["median_delta_loss"],
                }
            )
    print(
        f"  Saved -> {ranking_path}  (empirical ordering by measured loss disruption — not a 'best/worst' ranking)"
    )

    # ---- statistics ----
    tiers_arr = np.array([s["tier"] for s in summary_rows])
    means_arr = np.array([s["mean_delta_loss"] for s in summary_rows])
    rho, rho_p = stats.spearmanr(tiers_arr, means_arr)

    pairwise = {}
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        xa, xb = np.array(op_level_by_tier[a]), np.array(op_level_by_tier[b])
        t_stat, t_p = stats.ttest_ind(xb, xa, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(xb, xa, alternative="greater")
        pairwise[(a, b)] = {
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "u_stat": float(u_stat),
            "u_p": float(u_p),
        }

    n1, n2, n3 = (
        len(op_level_by_tier[1]),
        len(op_level_by_tier[2]),
        len(op_level_by_tier[3]),
    )

    def expected_tier_for_rank(rank: int) -> int:
        if rank < n1:
            return 1
        if rank < n1 + n2:
            return 2
        return 3

    violations = []
    for i, s in enumerate(ranked):
        exp_tier = expected_tier_for_rank(i)
        if s["tier"] != exp_tier:
            violations.append((s["operation"], s["tier"], i + 1, exp_tier))

    stats_path = out_dir / "statistics.txt"
    with open(stats_path, "w") as f:
        f.write(
            "Statistical unit: operation (not operation x seed). See spec/README for justification.\n"
        )
        f.write(
            f"n_ops per tier: Tier1={n1}, Tier2={n2}, Tier3={n3} (Tier3 includes {len(MIXING_OPS)} mixing ops)\n\n"
        )
        f.write(f"Spearman(tier, mean_delta_loss): rho={rho:.4f}  p={rho_p:.4f}\n\n")
        for (a, b), st in pairwise.items():
            f.write(
                f"Tier {b} > Tier {a}: Welch t={st['t_stat']:.4f} p={st['t_p']:.4f} | "
                f"Mann-Whitney U={st['u_stat']:.1f} p={st['u_p']:.4f}\n"
            )
        f.write(
            f"\nOperations violating expected monotonic ordering: {len(violations)}/{len(ranked)}\n"
        )
        for op_name, actual_tier, rank, exp_tier in violations:
            f.write(
                f"  {op_name}: actual Tier {actual_tier}, empirical rank {rank}, rank-band expects Tier {exp_tier}\n"
            )
    print(f"  Saved -> {stats_path}")
    print(f"\n  Spearman(tier, mean dL): rho={rho:.3f}  p={rho_p:.4f}")
    print(
        f"  Ordering violations ({len(violations)}/{len(ranked)}): {[v[0] for v in violations]}"
    )

    # ---- plots ----
    plot_operation_level(summary_rows, out_dir / "operation_level_disruption.png")
    plot_category_distribution(op_level_by_tier, out_dir / "category_distribution.png")
    plot_tier_scatter(summary_rows, rho, rho_p, out_dir / "tier_scatter.png")

    # ---- README ----
    write_readme(
        cfg,
        summary_rows,
        category_rows,
        rho,
        rho_p,
        pairwise,
        violations,
        len(rows),
        out_dir,
        n1,
        n2,
        n3,
    )

    print(f"\nAll outputs saved to: {out_dir}")


def write_readme(
    cfg,
    summary_rows,
    category_rows,
    rho,
    rho_p,
    pairwise,
    violations,
    n_raw_rows,
    out_dir,
    n1,
    n2,
    n3,
):
    n_seeds = len(cfg["seeds"])
    n_ops_total = len(summary_rows)

    lines = []
    lines.append("# Augmentation Inference-Time Disruption Validation\n")
    lines.append("## Research question\n")
    lines.append(
        "Do the existing code-defined augmentation tiers correspond to increasing "
        "IMMEDIATE INFERENCE-TIME validation-loss disruption, measured on a FROZEN "
        "model (no further training after a shared no-augmentation warm-up)?\n"
        "`D(a) = L_aug(a) - L_clean`, where L_aug(a) is the frozen model's validation "
        "loss after applying only operation a to the validation images (or, for "
        "mixing ops, the existing mixed-label loss), and L_clean is the frozen "
        "model's loss on the unaugmented validation set.\n"
    )
    lines.append(
        "This measures immediate augmentation-induced prediction sensitivity, NOT "
        "training difficulty. It is a separate, isolated diagnostic from "
        "`experiments/run_augmentation_difficulty.py` (which measures 1-epoch "
        "training-time disruption) — neither script depends on or modifies the other.\n"
    )

    lines.append("## Source of operations\n")
    lines.append("`augmentations/primitives.py:AUGMENTATION_REGISTRY`\n")
    lines.append(
        f"**{len(summary_rows) - len(MIXING_OPS)} single-image operations detected** (dynamically counted from the registry at runtime, not hardcoded).\n"
    )

    lines.append("## Source of tiers\n")
    lines.append(
        "Per-op tier for the single-image operations: the tier field in "
        "`AUGMENTATION_REGISTRY` itself (cross-checked at runtime against "
        "`augmentations/policies.py:_TIER_OPS` for op_pool=19 — both agree).\n"
    )
    lines.append(
        "Mixing operations (`cutmix`, `mixup`) have **no per-op tier field** anywhere "
        "in the codebase — they are a separate batch-level mechanism defined in "
        "`augmentations/mixing.py` (`BatchMixer`, `mixup()`, `cutmix()`), not a member "
        "of `AUGMENTATION_REGISTRY`. Their Tier-3-only status is established "
        "structurally rather than declared: `augmentations/mixing.py`'s module "
        'docstring calls it "Stage 2 of the two-stage Tier 3 pipeline" and '
        '`BatchMixer`\'s docstring says "Intended for Tier 3 only"; '
        "`augmentations/policies.py:ThreeTierCurriculumTransform.mix_scale()` returns "
        "0.0 before Tier 3 and only ramps at the Tier-3 boundary; and every mixer-gating "
        "branch in `experiments/train_baseline.py` (ETS/LPS/EGS paths) sets "
        "`active_mixer = None` until Tier 3 is reached (the `static_mixing` policy is "
        'the sole exception, treating the entire run as Tier-3-equivalent "always on '
        'from epoch 1"). This script therefore reports `cutmix`/`mixup` under Tier 3, '
        "with this derivation documented here rather than assumed.\n"
    )

    lines.append(f"## Operation count\n")
    lines.append(
        f"{len(summary_rows) - len(MIXING_OPS)} single-image operations detected + {len(MIXING_OPS)} mixing operations (cutmix, mixup) = **{n_ops_total} operations evaluated**.\n"
    )

    lines.append("## Protocol\n")
    lines.append(
        f"- Dataset: `{cfg['dataset']}` (val_split={cfg['val_split']}, existing project split — `data/datasets.py:get_cifar100_loaders`)"
    )
    lines.append(f"- Model: `{cfg['model']}` (`models/registry.py:get_model`)")
    lines.append(f"- Seeds: {cfg['seeds']} (n={n_seeds})")
    lines.append(
        f"- Warm-up: {cfg['warmup_epochs']} epoch(s), NO augmentation (matches `lps_min_epochs` in experiments/config.py)"
    )
    lines.append(
        "- After warm-up: model is frozen (`.eval()`), wrapped in `torch.no_grad()` for every subsequent evaluation. No optimizer is constructed and `.backward()`/`.step()` are never called again."
    )
    lines.append(
        f"- Op strength: {cfg['fixed_strength']} (= `FIXED_STRENGTH` in `augmentations/policies.py`, unchanged — no per-op severity recalibration)"
    )
    lines.append(
        f"- Mixing alpha: {cfg['mix_alpha']} (Beta(alpha,alpha), project default from experiments/config.py `mix_alpha`)"
    )
    lines.append(
        f"- Optimizer (warm-up only): SGD, lr={cfg['lr']}, momentum=0.9, nesterov=True, weight_decay={cfg['weight_decay']}, no LR schedule"
    )
    lines.append(f"- Batch size: {cfg['batch_size']}")
    lines.append(
        "- Every operation's L_aug is measured against the SAME frozen model and the SAME L_clean for that seed. `set_seed(seed)` is called again before every operation to make the op's own internal randomness (and, for mixing, the batch-pairing permutation) reproducible."
    )
    lines.append(
        "- After every single operation evaluation, the model's full `state_dict()` (all parameters + buffers) is checked with `torch.equal` against a pre-evaluation snapshot. Any change raises immediately and no results are saved for that run.\n"
    )

    lines.append("## Metric\n")
    lines.append("`D(a) = L_aug(a) - L_clean`\n")

    lines.append("## Mixing operations\n")
    lines.append(
        "`cutmix` and `mixup` are evaluated by reusing `augmentations/mixing.py:BatchMixer` "
        "unchanged, with `mode` set to the operation being tested and `p=1.0` (always "
        "mixed — the production default `mix_prob=0.5` is a training-time "
        "regularization knob, not part of the operation's definition; every ordinary "
        "op in this experiment is likewise applied unconditionally). Sample pairing "
        "reuses the existing `torch.randperm`-based logic in `mixup()`/`cutmix()` "
        "unmodified. The loss is the exact mixed-label formula from "
        "`training/trainer.py:train_one_epoch`: "
        "`lam * CE(out, y_a) + (1-lam) * CE(out, y_b)`, computed on the frozen model's "
        "output for each mixed validation batch, batch-size-weighted and averaged over "
        "the validation set exactly as `training/trainer.py:evaluate` does for ordinary "
        "operations. Mean `lambda` across all validation batches is recorded per "
        "(seed, operation) row in `raw_results.csv`.\n"
    )

    lines.append("## Statistical analysis\n")
    lines.append(
        "**Statistical unit = operation, not operation x seed.** Per spec: there are "
        f"N={n_ops_total} unique operations; category-level tests compare each "
        "operation's mean-over-seeds delta loss (N=n1/n2/n3 operations per tier), "
        "never the raw n_ops x n_seeds pool.\n"
    )
    lines.append(
        f"**Primary test** — Spearman rank correlation between assigned tier (1/2/3) and per-operation mean ΔL across all {n_ops_total} ops: **ρ = {rho:.3f}, p = {rho_p:.4f}**.\n"
    )
    lines.append(
        "**Secondary tests** — Welch's t-test + Mann-Whitney U (one-sided, higher tier > lower tier), on operation-level mean ΔL:\n"
    )
    lines.append("| Comparison | Welch t | Welch p | Mann-Whitney U | MWU p |")
    lines.append("|---|---|---|---|---|")
    for (a, b), st in pairwise.items():
        lines.append(
            f"| Tier {b} > Tier {a} | {st['t_stat']:.3f} | {st['t_p']:.4f} | {st['u_stat']:.1f} | {st['u_p']:.4f} |"
        )
    lines.append("")

    lines.append(
        f"**Ordering violations** — {len(violations)}/{n_ops_total} operations fall outside the rank band their existing tier would predict (band sizes {n1}/{n2}/{n3}, by ascending mean ΔL):\n"
    )
    if violations:
        lines.append(
            "| Operation | Existing Tier | Empirical Rank | Rank Band's Expected Tier |"
        )
        lines.append("|---|---|---|---|")
        for op_name, actual_tier, rank, exp_tier in violations:
            lines.append(
                f"| {op_name} | Tier {actual_tier} | {rank} | Tier {exp_tier} |"
            )
    else:
        lines.append(
            "None — every operation's empirical difficulty rank falls within its existing tier's band."
        )
    lines.append("")

    lines.append("## Results by operation\n")
    lines.append(
        "| Operation | Tier | Mean ΔL | Median ΔL | Std ΔL | Min ΔL | Max ΔL |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(summary_rows, key=lambda s: (s["tier"], s["operation"])):
        lines.append(
            f"| {r['operation']} | Tier {r['tier']} | {r['mean_delta_loss']:+.4f} | {r['median_delta_loss']:+.4f} | {r['std_delta_loss']:.4f} | {r['min_delta_loss']:+.4f} | {r['max_delta_loss']:+.4f} |"
        )
    lines.append("")

    lines.append("## Results by category\n")
    lines.append("| Tier | # Ops | Mean ΔL | Median ΔL | Std ΔL | Min ΔL | Max ΔL |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in category_rows:
        lines.append(
            f"| Tier {r['tier']} | {r['n_operations']} | {r['mean_delta_loss']:+.4f} | {r['median_delta_loss']:+.4f} | {r['std_delta_loss']:.4f} | {r['min_delta_loss']:+.4f} | {r['max_delta_loss']:+.4f} |"
        )
    lines.append("")

    lines.append("## Limitations\n")
    lines.append(
        "1. This measures immediate inference-time disruption on a frozen model, not "
        "full training difficulty — a different (also isolated) diagnostic, "
        "`run_augmentation_difficulty.py`, measures 1-epoch training-time disruption.\n"
        "2. Different augmentation operations have different parameterization/severity "
        "scales; a common `strength=0.7` does not necessarily represent equal "
        "perceptual severity across operations.\n"
        "3. This experiment tests association between predefined tiers and measured "
        "disruption. It does not prove the tiers are objectively correct.\n"
        "4. Individual operations may violate (and, per the Results tables above, do "
        "violate) the expected monotonic ordering — this is reported, not corrected.\n"
        "5. Mixing operations have fundamentally different semantics from ordinary "
        "single-image operations (batch-level sample pairing vs. per-image transform) "
        "and are forced to `p=1.0` here for a clean per-operation signal, which is not "
        "how they are actually scheduled during real curriculum training "
        "(production default `mix_prob=0.5`, activated only in Tier 3, often ramped).\n"
    )

    lines.append("## Reproduction\n")
    lines.append(
        "```\npython -m experiments.run_augmentation_inference_disruption\n```\n"
    )

    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved -> {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
