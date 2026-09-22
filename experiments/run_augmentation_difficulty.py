"""experiments/run_augmentation_difficulty.py

Isolated diagnostic experiment — NOT part of the ETS / LPS / EGS curriculum and
does not modify any curriculum logic, augmentation policy, training script,
config, checkpoint, or result file.

Research question
------------------
augmentations/primitives.py:AUGMENTATION_REGISTRY assigns each of the 19
augmentation operations to a tier (1, 2, or 3) — the existing, code-defined
design. This script empirically tests whether that assignment corresponds to
increasing training-time disruption: does introducing a Tier-3 operation into
training destabilize a trained model's validation loss more than introducing a
Tier-1 operation?

For operation a:
    D(a) = L_val_after(a) - L_val_before(a)

Protocol (see README.md written into the output directory after each run for
the exact rationale)
--------------------
1. Per seed: train ResNet-18 on CIFAR-100 with NO augmentation for
   `warmup_epochs` (default 10, matching LPS's `lps_min_epochs` floor in
   experiments/config.py — the shortest period the existing curriculum design
   itself considers sufficient before allowing a tier transition). This is the
   shared controlled state for every operation under that seed.
2. L_before = clean validation loss on that checkpoint (identical for all 19
   ops of that seed — computed once).
3. For each of the 19 ops, independently, restarting from the exact same
   warm-up checkpoint with a fresh (zero-momentum) optimizer: train for
   `probe_epochs` (default 1) with the training transform set to ONLY that
   operation, applied at the project's standard strength ceiling
   (augmentations.policies.FIXED_STRENGTH = 0.7 — the same value used
   everywhere else in the repo), then measure L_after on the clean
   (unaugmented) validation transform.
4. D(a) = L_after - L_before. Restore the warm-up checkpoint before the next
   operation. All ops for a seed share the same RNG state at probe start
   (set_seed(seed) is called again before every probe) so batch order and any
   op-internal randomness are aligned across ops — the only thing that differs
   between probes is which operation was active.

Reuses (does not duplicate) existing repository code: AUGMENTATION_REGISTRY
(augmentations/primitives.py), FIXED_STRENGTH (augmentations/policies.py),
get_model (models/registry.py), get_cifar100_loaders / CIFAR_STATS /
get_no_augmentation_transforms (data/datasets.py), train_one_epoch / evaluate
(training/trainer.py), set_seed / get_device / build_optimizer / setup_logging
(experiments/utils.py).

Usage:
    python -m experiments.run_augmentation_difficulty
    python -m experiments.run_augmentation_difficulty --debug   # smoke test
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from scipy import stats

from augmentations.policies import FIXED_STRENGTH
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

# Colours kept identical to scripts/plot_aug_ranking.py's TIER_COLORS so every
# tier-coloured figure in the thesis uses the same three colours.
TIER_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
TIER_NAMES = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}


class _SingleOpTrainTransform:
    """Apply exactly one op from AUGMENTATION_REGISTRY at a fixed strength, then normalize.

    Same call signature (fn(img, strength)) that ThreeTierCurriculumTransform
    uses for an "active" op — this is what it means for op `a` to be
    introduced into training under the existing pipeline's conventions.
    """

    def __init__(self, dataset: str, op_name: str, strength: float):
        cs = CIFAR_STATS[dataset]
        self.normalize = T.Normalize(cs["mean"], cs["std"])
        self.to_tensor = T.ToTensor()
        self.fn = AUGMENTATION_REGISTRY[op_name][0]
        self.strength = strength

    def __call__(self, img):
        img = self.fn(img, strength=self.strength)
        return self.normalize(self.to_tensor(img))


def _make_optimizer(model, cfg):
    optimizer, _ = build_optimizer(
        model,
        {"optimizer": "sgd", "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
    )
    return optimizer


def run_warmup(cfg: dict, seed: int, device) -> tuple[dict, float]:
    """Train from scratch with no augmentation. Returns (cpu state_dict, clean val loss)."""
    set_seed(seed)
    model = get_model(cfg["model"], num_classes=100).to(device)
    train_tf, val_tf = get_no_augmentation_transforms(cfg["dataset"])
    train_loader, val_loader, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=train_tf,
        test_transform=val_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    optimizer = _make_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, cfg["warmup_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"    [warmup seed={seed}] epoch {ep}/{cfg['warmup_epochs']} "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc * 100:.2f}%"
        )

    val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
    print(
        f"    [warmup seed={seed}] DONE — clean val_loss={val_loss:.4f}  val_acc={val_acc * 100:.2f}%"
    )

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return base_state, val_loss


def run_probe(cfg: dict, seed: int, op_name: str, base_state: dict, device) -> float:
    """Restart from base_state, train `probe_epochs` with ONLY op_name active. Returns L_after."""
    set_seed(seed)
    model = get_model(cfg["model"], num_classes=100).to(device)
    model.load_state_dict(base_state)

    train_tf = _SingleOpTrainTransform(cfg["dataset"], op_name, cfg["fixed_strength"])
    _, val_tf = get_no_augmentation_transforms(cfg["dataset"])
    train_loader, val_loader, _ = get_cifar100_loaders(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=train_tf,
        test_transform=val_tf,
        num_workers=cfg["num_workers"],
        debug=cfg["debug"],
    )
    optimizer = _make_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg["probe_epochs"]):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    return val_loss


def plot_individual_ops(summary_rows: list[dict], out_path: Path) -> None:
    ordered = sorted(
        summary_rows,
        key=lambda s: (AUGMENTATION_REGISTRY[s["operation"]][1], s["operation"]),
    )
    names = [r["operation"].replace("_", " ") for r in ordered]
    means = [r["mean_loss_delta"] for r in ordered]
    stds = [r["std_loss_delta"] for r in ordered]
    colors = [TIER_COLORS[AUGMENTATION_REGISTRY[r["operation"]][1]] for r in ordered]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=3,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#888888", linewidth=0.9, zorder=1)
    ax.set_ylabel("Mean $\\Delta L$  (val loss after probe $-$ before)", fontsize=10)
    ax.set_title(
        "Per-operation validation-loss disruption\n"
        "Bar colour = existing (code-defined) tier · error bars = std over seeds",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t]) for t in (1, 2, 3)
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_category_comparison(category_stats: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    data = [category_stats[t]["pooled_deltas"] for t in (1, 2, 3)]
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
        vals = category_stats[t]["pooled_deltas"]
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full_like(vals, x_pos) + jitter,
            vals,
            color=TIER_COLORS[t],
            edgecolor="white",
            s=25,
            zorder=3,
            alpha=0.85,
        )

    ax.axhline(0, color="#888888", linewidth=0.9)
    ax.set_ylabel("$\\Delta L$  (per operation × seed)", fontsize=10)
    ax.set_title("Category-level (existing tier) difficulty distributions", fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def write_readme(
    cfg: dict,
    summary_rows: list[dict],
    category_rows: list[dict],
    rho: float,
    rho_p: float,
    pairwise: dict,
    violations: list[tuple],
    out_dir: Path,
) -> None:
    n_seeds = len(cfg["seeds"])

    lines = []
    lines.append("# Augmentation Difficulty Validation\n")
    lines.append(
        "Isolated diagnostic experiment testing whether the existing tier assignments "
        "in `augmentations/primitives.py:AUGMENTATION_REGISTRY` correspond to increasing "
        "empirical training-time difficulty, measured as the immediate validation-loss "
        "disruption `D(a) = L_val_after(a) - L_val_before(a)` produced by introducing each "
        "of the 19 operations into training individually.\n"
    )
    lines.append(
        "This script (`experiments/run_augmentation_difficulty.py`) does not modify ETS, "
        "LPS, EGS, static mixing, existing baselines, augmentation implementations, "
        "training scripts, configs, checkpoints, or other result files. All outputs live "
        "under `results/augmentation_difficulty/`.\n"
    )

    lines.append("## Existing tier assignment tested (from code, unmodified)\n")
    lines.append("| Operation | Existing Category/Tier |")
    lines.append("|---|---|")
    for tier in (1, 2, 3):
        for name, (_, t, _) in AUGMENTATION_REGISTRY.items():
            if t == tier:
                lines.append(f"| {name} | Tier {tier} |")
    lines.append("")

    lines.append("## Protocol\n")
    lines.append(
        f"- Dataset: `{cfg['dataset']}` (CIFAR-100, val_split={cfg['val_split']})"
    )
    lines.append(f"- Model: `{cfg['model']}`")
    lines.append(f"- Seeds: {cfg['seeds']} (n={n_seeds})")
    lines.append(
        f"- Warm-up: {cfg['warmup_epochs']} epochs, no augmentation "
        f"(matches `lps_min_epochs` in experiments/config.py — the existing curriculum's own "
        f"minimum-epochs-per-tier floor)"
    )
    lines.append(
        f"- Probe: {cfg['probe_epochs']} epoch(s) per operation, that operation only"
    )
    lines.append(
        f"- Op strength: {cfg['fixed_strength']} (= `FIXED_STRENGTH` in `augmentations/policies.py`, "
        f"the project's standard ceiling — same value used by every other policy in the repo)"
    )
    lines.append(
        f"- Optimizer: SGD, lr={cfg['lr']}, momentum=0.9, nesterov=True, weight_decay={cfg['weight_decay']}"
    )
    lines.append("  (fresh, zero-momentum optimizer at the start of every probe)")
    lines.append(
        "- LR schedule: none (fixed LR) — kept simple/controlled for this short diagnostic"
    )
    lines.append(f"- Batch size: {cfg['batch_size']}")
    lines.append(
        "- Each op's probe restarts from the exact same warm-up checkpoint; `set_seed(seed)` is "
        "called again immediately before every probe so batch order/op-internal randomness are "
        "aligned across all 19 ops of a seed — the only thing that varies between probes is which "
        "operation was active."
    )
    lines.append("")

    lines.append("## Results by operation\n")
    lines.append(
        "| Operation | Existing Category | Mean ΔL | Std ΔL | Min ΔL | Max ΔL |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(
        summary_rows,
        key=lambda s: (AUGMENTATION_REGISTRY[s["operation"]][1], s["operation"]),
    ):
        lines.append(
            f"| {r['operation']} | {r['existing_category']} | {r['mean_loss_delta']:+.4f} | "
            f"{r['std_loss_delta']:.4f} | {r['min_loss_delta']:+.4f} | {r['max_loss_delta']:+.4f} |"
        )
    lines.append("")

    lines.append("## Results by existing category\n")
    lines.append(
        "| Existing Category | # Ops | Mean ΔL | Median ΔL | Std ΔL | Min ΔL | Max ΔL |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in category_rows:
        lines.append(
            f"| {r['existing_category']} | {r['n_operations']} | {r['mean_difficulty']:+.4f} | "
            f"{r['median_difficulty']:+.4f} | {r['std_difficulty']:.4f} | "
            f"{r['min_difficulty']:+.4f} | {r['max_difficulty']:+.4f} |"
        )
    lines.append("")

    lines.append("## Statistical analysis\n")
    lines.append(
        f"**Primary test** — Spearman rank correlation between assigned tier (1/2/3) and "
        f"per-operation mean ΔL across all 19 ops: **ρ = {rho:.3f}, p = {rho_p:.4f}**. "
        f"This directly tests whether the existing tier ordering is monotonically associated "
        f"with empirical difficulty.\n"
    )
    lines.append(
        "**Secondary tests** — pairwise comparisons on per-operation mean ΔL "
        "(n=4/7/8 ops per tier; Welch's t-test + Mann-Whitney U, one-sided: higher tier > lower tier):\n"
    )
    lines.append("| Comparison | Welch t | Welch p | Mann-Whitney U | MWU p |")
    lines.append("|---|---|---|---|---|")
    for (a, b), st in pairwise.items():
        lines.append(
            f"| Tier {b} > Tier {a} | {st['t_stat']:.3f} | {st['t_p']:.4f} | "
            f"{st['u_stat']:.1f} | {st['u_p']:.4f} |"
        )
    lines.append("")

    lines.append(
        f"**Ordering violations** — {len(violations)}/19 operations fall outside the rank band "
        "their existing tier would predict (band sizes 4/7/8, by ascending mean ΔL):\n"
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

    lines.append("## Interpretation\n")
    lines.append(
        "The existing tier assignments were treated as the hypothesis under test, not ground truth; "
        "the statements below follow directly from the statistics above, computed after this script "
        "ran to completion.\n"
    )

    if rho > 0 and rho_p < 0.05:
        support = (
            f"The empirical evidence **supports** the existing categorization: assigned tier and "
            f"mean validation-loss disruption are positively and significantly correlated "
            f"(ρ = {rho:.3f}, p = {rho_p:.4f} < 0.05), i.e. operations assigned to higher tiers "
            f"tend to produce larger immediate validation-loss disruption than operations assigned "
            f"to lower tiers."
        )
    elif rho > 0:
        support = (
            f"The empirical evidence is **directionally consistent but not statistically significant**: "
            f"the correlation between assigned tier and mean ΔL is positive (ρ = {rho:.3f}) but does not "
            f"reach p < 0.05 (p = {rho_p:.4f}) at n=19 operations. The existing ordering is not "
            f"contradicted, but this sample does not provide strong statistical confirmation of it."
        )
    else:
        support = (
            f"The empirical evidence **does not support** the existing categorization: the correlation "
            f"between assigned tier and mean ΔL is non-positive (ρ = {rho:.3f}, p = {rho_p:.4f}). Higher "
            f"tiers did not, on average, produce larger disruption than lower tiers under this protocol."
        )
    lines.append(support + "\n")

    if violations:
        viol_names = ", ".join(f"`{op}`" for op, *_ in violations)
        lines.append(
            f"{len(violations)} of 19 operations ({viol_names}) violate the expected monotonic ordering — "
            f"their empirical difficulty rank does not fall inside the rank band their existing tier "
            f"assignment predicts. These are candidates for re-examination if the tier design is revisited, "
            f"but do not by themselves overturn the overall ordering result above."
        )
    lines.append("")

    lines.append("## Suggested thesis phrasing\n")
    lines.append(
        "> To empirically assess the difficulty ordering used by our curriculum, we measured the "
        "immediate validation-loss disruption produced by each of the 19 augmentation operations. "
        "Operations were evaluated individually under identical experimental conditions "
        f"(ResNet-18, CIFAR-100, {n_seeds} seeds, {cfg['warmup_epochs']}-epoch shared warm-up, "
        f"{cfg['probe_epochs']}-epoch single-operation probe) and grouped according to the tier "
        "assignments already defined in our augmentation framework. The resulting distributions were "
        f"then compared across tiers (Spearman ρ = {rho:.3f}, p = {rho_p:.4f}) to determine whether the "
        "predefined curriculum ordering corresponds to increasing empirical difficulty."
    )
    lines.append("")

    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out_dir / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate existing augmentation tier assignments via isolated per-op val-loss disruption"
    )
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--probe_epochs", type=int, default=1)
    parser.add_argument("--fixed_strength", type=float, default=FIXED_STRENGTH)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results" / "augmentation_difficulty"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny smoke test: 512 train / 128 val samples",
    )
    args = parser.parse_args()
    cfg = vars(args)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        {"log_dir": str(out_dir), "experiment_name": "augmentation_difficulty"}
    )
    device = get_device()
    op_names = list(AUGMENTATION_REGISTRY.keys())

    print(f"\n{'=' * 70}")
    print("  Augmentation Difficulty Validation — isolated 19-op probe")
    print(f"  Dataset={cfg['dataset']}  Model={cfg['model']}  Seeds={cfg['seeds']}")
    print(
        f"  Warmup={cfg['warmup_epochs']}ep  Probe={cfg['probe_epochs']}ep  "
        f"strength={cfg['fixed_strength']}"
    )
    print(f"{'=' * 70}\n")

    rows = []
    t_start = time.time()
    for seed in cfg["seeds"]:
        print(f"\n── Seed {seed} " + "─" * 45)
        base_state, l_before = run_warmup(cfg, seed, device)
        for op_name in op_names:
            t0 = time.time()
            l_after = run_probe(cfg, seed, op_name, base_state, device)
            delta = l_after - l_before
            tier = AUGMENTATION_REGISTRY[op_name][1]
            rows.append(
                {
                    "operation": op_name,
                    "existing_category": f"Tier {tier}",
                    "existing_category_id": tier,
                    "seed": seed,
                    "loss_before": l_before,
                    "loss_after": l_after,
                    "loss_delta": delta,
                }
            )
            print(
                f"    {op_name:<15} Tier {tier}  ΔL={delta:+.4f}  ({time.time() - t0:.0f}s)"
            )

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")

    raw_fields = [
        "operation",
        "existing_category",
        "existing_category_id",
        "seed",
        "loss_before",
        "loss_after",
        "loss_delta",
    ]
    raw_path = out_dir / "augmentation_difficulty_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {raw_path}")

    summary_rows = []
    for op_name in op_names:
        op_rows = [r for r in rows if r["operation"] == op_name]
        deltas = np.array([r["loss_delta"] for r in op_rows])
        tier = AUGMENTATION_REGISTRY[op_name][1]
        summary_rows.append(
            {
                "operation": op_name,
                "existing_category": f"Tier {tier}",
                "mean_loss_delta": float(deltas.mean()),
                "std_loss_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "min_loss_delta": float(deltas.min()),
                "max_loss_delta": float(deltas.max()),
            }
        )
    summary_fields = [
        "operation",
        "existing_category",
        "mean_loss_delta",
        "std_loss_delta",
        "min_loss_delta",
        "max_loss_delta",
    ]
    summary_path = out_dir / "augmentation_difficulty_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved → {summary_path}")

    category_stats = {}
    category_rows = []
    for tier in (1, 2, 3):
        ops_in_tier = [
            s for s in summary_rows if s["existing_category"] == f"Tier {tier}"
        ]
        pooled = np.array(
            [r["loss_delta"] for r in rows if r["existing_category_id"] == tier]
        )
        op_means = np.array([s["mean_loss_delta"] for s in ops_in_tier])
        category_stats[tier] = {
            "pooled_deltas": pooled,
            "op_means": op_means,
            "n_ops": len(ops_in_tier),
        }
        category_rows.append(
            {
                "existing_category": f"Tier {tier}",
                "n_operations": len(ops_in_tier),
                "mean_difficulty": float(pooled.mean()),
                "median_difficulty": float(np.median(pooled)),
                "std_difficulty": float(pooled.std(ddof=1)),
                "min_difficulty": float(pooled.min()),
                "max_difficulty": float(pooled.max()),
            }
        )
    category_fields = [
        "existing_category",
        "n_operations",
        "mean_difficulty",
        "median_difficulty",
        "std_difficulty",
        "min_difficulty",
        "max_difficulty",
    ]
    category_path = out_dir / "augmentation_difficulty_by_category.csv"
    with open(category_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=category_fields)
        writer.writeheader()
        writer.writerows(category_rows)
    print(f"  Saved → {category_path}")

    tiers_arr = np.array(
        [AUGMENTATION_REGISTRY[s["operation"]][1] for s in summary_rows]
    )
    means_arr = np.array([s["mean_loss_delta"] for s in summary_rows])
    rho, rho_p = stats.spearmanr(tiers_arr, means_arr)

    pairwise = {}
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        xa, xb = category_stats[a]["op_means"], category_stats[b]["op_means"]
        t_stat, t_p = stats.ttest_ind(xb, xa, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(xb, xa, alternative="greater")
        pairwise[(a, b)] = {
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "u_stat": float(u_stat),
            "u_p": float(u_p),
        }

    ordered = sorted(summary_rows, key=lambda s: s["mean_loss_delta"])

    def expected_tier_for_rank(rank: int) -> int:
        if rank < 4:
            return 1
        if rank < 11:
            return 2
        return 3

    violations = []
    for i, s in enumerate(ordered):
        actual_tier = AUGMENTATION_REGISTRY[s["operation"]][1]
        exp_tier = expected_tier_for_rank(i)
        if actual_tier != exp_tier:
            violations.append((s["operation"], actual_tier, i + 1, exp_tier))

    print(f"\n  Spearman(tier, mean ΔL): ρ={rho:.3f}  p={rho_p:.4f}")
    for (a, b), st in pairwise.items():
        print(
            f"  Tier {b} > Tier {a}: Welch t={st['t_stat']:.3f} p={st['t_p']:.4f} | MWU p={st['u_p']:.4f}"
        )
    print(f"  Ordering violations ({len(violations)}/19): {[v[0] for v in violations]}")

    plot_individual_ops(summary_rows, out_dir / "augmentation_difficulty.png")
    plot_category_comparison(
        category_stats, out_dir / "augmentation_difficulty_by_category.png"
    )

    write_readme(
        cfg, summary_rows, category_rows, rho, rho_p, pairwise, violations, out_dir
    )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
