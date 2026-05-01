"""
experiments/wandb_tools.py — Unified W&B management for curriculum augmentation thesis.

All W&B operations in one place:
  1. upload  — upload local checkpoints to W&B
  2. table   — build/update comparison table from finished runs
  3. rename  — set clean display names on all runs
  4. sync    — run rename + table in one shot

Usage:
    python experiments/wandb_tools.py upload
    python experiments/wandb_tools.py upload --run resnet50_tiered_lps_mix_both_sgd_multistep_ep100_cifar100
    python experiments/wandb_tools.py table
    python experiments/wandb_tools.py rename
    python experiments/wandb_tools.py sync
"""

import os
import sys
import argparse
import warnings
import numpy as np
from pathlib import Path

import wandb

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

WANDB_PROJECT = "curriculum-augmentation"
CHECKPOINT_DIR = _ROOT / "checkpoints"
FIGURES_DIR = _ROOT / "results" / "figures"
TABLE_RUN_ID_FILE = Path(__file__).parent / ".table_run_id"

SKIP_DISPLAY_NAMES = {"results-comparison-table", "analysis-figures"}


# ── Run name → clean display name ─────────────────────────────────────────────
def get_display_name(run_name: str, config: dict) -> str:
    from models.registry import MODEL_DISPLAY_NAMES

    aug = config.get("augmentation", "")
    sched = config.get("tier_schedule", "")
    mixing = config.get("mix_mode", "")
    raw_model = config.get("model", "")
    model = MODEL_DISPLAY_NAMES.get(raw_model, raw_model)
    prefix = f"{model} | " if model else ""

    if aug == "none":
        return f"{prefix}No Augmentation"
    if aug == "static":
        return f"{prefix}Static"
    if aug == "static_mixing":
        return f"{prefix}Static + Mixing"
    if aug == "random":
        return f"{prefix}Random Augmentation"
    if aug == "randaugment":
        return f"{prefix}RandAugment"
    if aug == "tiered_curriculum":
        s = sched.upper() if sched else "ETS"
        freq_tag = f" (freq={config.get('egs_update_freq', 10)})" if s == "EGS" else ""
        if mixing in (None, "none", ""):
            return f"{prefix}{s}{freq_tag} — No Mixing"
        return f"{prefix}{s}{freq_tag} + Mixing"
    return run_name


# ── Group rules for upload ─────────────────────────────────────────────────────
_GROUP_RULES = [
    ("_none_", "B", "No Augmentation"),
    ("_static_sgd", "B", "Static Aug"),
    ("_static_mixing_", "B", "Static + Mixing"),
    ("_random_", "B", "Random Aug"),
    ("_randaugment_", "B", "RandAugment"),
    ("_ets_mix_both_", "M", "ETS + Both Mixing"),
    ("_ets_nomix_", "M", "ETS No Mix"),
    ("_lps_mix_both_", "M", "LPS + Both Mixing"),
    ("_lps_nomix_", "M", "LPS No Mix"),
    ("_egs_mix_both_", "M", "EGS + Both Mixing"),
    ("_egs_nomix_", "M", "EGS No Mix"),
    ("_lps_", "A_LPS", "LPS Ablation"),
    ("_ets_", "A_ETS", "ETS Ablation"),
    ("_egs_", "A_EGS", "EGS Ablation"),
]


def _get_group_and_label(run_name: str):
    name = run_name.lower()
    for keyword, group, label in _GROUP_RULES:
        if keyword.lower() in name:
            return group, label
    return "Other", run_name


# ══════════════════════════════════════════════════════════════════════════════
class WandbTools:
    """Unified W&B management for curriculum augmentation thesis."""

    def __init__(self, project: str = WANDB_PROJECT):
        self.project = project
        self.api = wandb.Api()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _get_finished_runs(self):
        return [
            r
            for r in self.api.runs(self.project)
            if r.state == "finished"
            and (r.display_name or r.name) not in SKIP_DISPLAY_NAMES
        ]

    # ── 1. Upload local checkpoints ───────────────────────────────────────────
    def upload(self, run_names: list = None):
        """Upload local checkpoint histories to W&B."""
        import torch

        if run_names:
            files = [CHECKPOINT_DIR / f"{n}_history.pt" for n in run_names]
        else:
            files = sorted(CHECKPOINT_DIR.glob("*_history.pt"))
        files = [f for f in files if f.exists()]

        if not files:
            print("No history files found.")
            return

        print(f"Uploading {len(files)} run(s) to W&B project '{self.project}'...\n")

        for history_path in files:
            ckpt_path = Path(str(history_path).replace("_history.pt", "_best.pth"))
            ckpt = (
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ckpt_path.exists()
                else {}
            )
            history = torch.load(history_path, map_location="cpu", weights_only=False)

            if "train_loss" not in history and "history" in history:
                history = history["history"]

            n_epochs = len(history.get("train_loss", []))
            if n_epochs == 0:
                print(f"  Skipping {history_path.name} — empty")
                continue

            run_name = history_path.stem.replace("_history", "")
            cfg = dict(ckpt.get("cfg", {}))
            group, label = _get_group_and_label(run_name)

            name_lower = run_name.lower()
            if "_lps_" in name_lower:
                cfg["scheduler_type"] = "LPS"
            elif "_ets_" in name_lower:
                cfg["scheduler_type"] = "ETS"
            else:
                cfg["scheduler_type"] = "none"

            print(f"  {run_name}  ({n_epochs} ep)  [{group}]")

            wandb.init(
                project=self.project,
                name=run_name,
                config=cfg,
                group=group,
                tags=[group, label],
                reinit=True,
            )

            has_val = len(history.get("val_loss", [])) == n_epochs
            has_top5 = len(history.get("val_top5", [])) == n_epochs

            for epoch in range(n_epochs):
                log = {
                    "epoch": epoch + 1,
                    "train_loss": history["train_loss"][epoch],
                    "train_acc": history["train_acc"][epoch] * 100,
                }
                if has_val:
                    log["val_loss"] = history["val_loss"][epoch]
                    log["val_acc"] = history["val_acc"][epoch] * 100
                if has_top5:
                    log["val_top5"] = history["val_top5"][epoch] * 100
                wandb.log(log, step=epoch + 1)

            test_top1 = ckpt.get("test_top1", 0) * 100
            test_top5 = ckpt.get("test_top5", 0) * 100
            wandb.log({"test_top1": test_top1, "test_top5": test_top5}, step=n_epochs)

            for k, v in {
                "best_val_acc": ckpt.get("val_acc", 0) * 100,
                "test_top1": test_top1,
                "test_top5": test_top5,
                "best_epoch": ckpt.get("epoch", 0),
                "total_minutes": ckpt.get("total_minutes", 0),
            }.items():
                wandb.run.summary[k] = v

            wandb.finish()
            print(f"    test={test_top1:.2f}%  done ✓")

    # ── 2. Rename runs to clean display names ─────────────────────────────────
    def rename(self):
        """Set readable display names on all finished runs."""
        runs = self._get_finished_runs()
        print(f"Renaming {len(runs)} runs...")
        for run in runs:
            name = get_display_name(run.name, run.config)
            if run.display_name != name:
                run.display_name = name
                run.save()
                print(f"  {run.name[:50]:<50} → {name}")
        print("Done.")

    # ── 3. Comparison table ───────────────────────────────────────────────────
    def table(self):
        """Build/update the comparison table in W&B."""
        PAPER_BASELINES = [
            [
                "AutoAugment",
                "WideResNet-28-10",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "82.90",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
            [
                "RandAugment",
                "WideResNet-28-10",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "83.30",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
            [
                "Fast AutoAugment",
                "WideResNet-28-10",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "82.70",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
            [
                "TrivialAugment",
                "WideResNet-28-10",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "84.30",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
            [
                "Adversarial AutoAug",
                "WideResNet-28-10",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "85.80",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
            [
                "MADAug",
                "ResNet-50",
                "CIFAR-100",
                "—",
                "—",
                "—",
                "65.61",
                "—",
                "—",
                None,
                "—",
                "—",
                "Paper Baseline",
            ],
        ]

        BASELINE_AUGS = {"none", "static", "static_mixing", "random", "randaugment"}

        runs = self._get_finished_runs()
        rows = []

        from models.registry import MODEL_DISPLAY_NAMES

        for run in runs:
            display = run.display_name or run.name
            config = run.config
            summary = run.summary
            aug = config.get("augmentation", "")
            mixing = config.get("mix_mode", "—")
            model = MODEL_DISPLAY_NAMES.get(
                config.get("model", ""), config.get("model", "—")
            )
            dataset = (
                config.get("dataset", "cifar100")
                .upper()
                .replace("CIFAR100", "CIFAR-100")
            )

            if aug == "tiered_curriculum":
                schedule = (config.get("tier_schedule") or "ets").upper()
                mix_show = "—" if mixing in (None, "none") else "Yes"
                exp_type = "Core Run" if mixing not in (None, "none") else "Ablation"
            else:
                schedule = "—"
                mix_show = "Yes" if "mixing" in aug else "—"
                exp_type = "Baseline" if aug in BASELINE_AUGS else "Other"

            val_acc = summary.get("best_val_acc")
            test_top1 = summary.get("test_top1")
            test_top5 = summary.get("test_top5")
            best_ep = summary.get("best_epoch")
            total_min = summary.get("total_minutes")
            date_s = config.get("run_ts") or (
                run.created_at[:10] if run.created_at else "—"
            )

            val_s = f"{val_acc:.2f}" if val_acc is not None else "—"
            t1_s = f"{test_top1:.2f}" if test_top1 is not None else "—"
            t5_s = f"{test_top5:.2f}" if test_top5 is not None else "—"
            gap_s = (
                f"{abs(float(val_s) - float(t1_s)):.2f}"
                if val_s != "—" and t1_s != "—"
                else "—"
            )
            time_s = f"{total_min:.0f} min" if total_min is not None else "—"

            rows.append(
                [
                    display,
                    model,
                    dataset,
                    schedule,
                    mix_show,
                    val_s,
                    t1_s,
                    t5_s,
                    gap_s,
                    best_ep,
                    time_s,
                    date_s,
                    exp_type,
                ]
            )
            print(f"  {display:<25} test={t1_s}")

        rows.extend(PAPER_BASELINES)
        order = {"Baseline": 0, "Core Run": 1, "Ablation": 2, "Paper Baseline": 3}
        rows.sort(key=lambda r: order.get(r[-1], 9))

        columns = [
            "Experiment",
            "Model",
            "Dataset",
            "Schedule",
            "Mixing",
            "Val Top-1 (%)",
            "Test Top-1 (%)",
            "Test Top-5 (%)",
            "Val-Test Gap",
            "Best Epoch",
            "Time",
            "Date",
            "Type",
        ]

        run_id = (
            TABLE_RUN_ID_FILE.read_text().strip()
            if TABLE_RUN_ID_FILE.exists()
            else None
        )
        print(f"\n{'Resuming' if run_id else 'Creating'} table run...")

        with wandb.init(
            project=self.project,
            name="results-comparison-table",
            id=run_id,
            resume="allow",
        ) as wrun:
            if not TABLE_RUN_ID_FILE.exists():
                TABLE_RUN_ID_FILE.write_text(wrun.id)
            wrun.log({"Results Comparison": wandb.Table(columns=columns, data=rows)})
        print(f"Table updated — {len(rows)} rows.")

    # ── 4. Sync — run rename + table ─────────────────────────────────────────
    def sync(self):
        """Rename runs and update comparison table in one shot."""
        print("=" * 50)
        print("Step 1/2 — Renaming runs")
        print("=" * 50)
        self.rename()

        print("\n" + "=" * 50)
        print("Step 2/2 — Updating comparison table")
        print("=" * 50)
        self.table()

        print("\n✓ Sync complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="W&B tools for curriculum augmentation thesis"
    )
    parser.add_argument(
        "command",
        choices=["upload", "rename", "table", "sync"],
        help="Operation to run",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="(upload only) specific run names to upload",
    )
    parser.add_argument(
        "--project",
        default=WANDB_PROJECT,
        help=f"W&B project name (default: {WANDB_PROJECT})",
    )
    args = parser.parse_args()

    tools = WandbTools(project=args.project)

    if args.command == "upload":
        tools.upload(args.run)
    elif args.command == "rename":
        tools.rename()
    elif args.command == "table":
        tools.table()
    elif args.command == "sync":
        tools.sync()


if __name__ == "__main__":
    main()
