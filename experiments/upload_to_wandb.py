"""
experiments/upload_to_wandb.py
Replay completed experiment history files to Weights & Biases.

Usage:
    python experiments/upload_to_wandb.py                        # all runs
    python experiments/upload_to_wandb.py --run resnet50_tiered_lps_nomix_sgd_multistep_ep100_cifar100
    python experiments/upload_to_wandb.py --project my-project
"""

import argparse
from pathlib import Path
import torch
import wandb

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DEFAULT_PROJECT = "curriculum-augmentation"

# Maps run name keywords → experiment group tag
GROUP_RULES = [
    ("_none_",              "B",       "No Augmentation"),
    ("_static_sgd",         "B",       "Static Aug"),
    ("_static_mixing_",     "B",       "Static + Mixing"),
    ("_random_",            "B",       "Random Aug"),
    ("_randaugment_",       "B",       "RandAugment"),
    ("flip_crop",           "B_COMBO", "Manual Pipeline"),
    ("autoaugment",         "B_PAPER", "AutoAugment"),
    ("trivialaugment",      "B_PAPER", "TrivialAugment"),
    ("augmix",              "B_PAPER", "AugMix"),
    ("madaug",              "B_PAPER", "MADAug"),
    ("_ets_mix_both_",      "M",       "ETS + Both Mixing"),
    ("_ets_mix_cutmix_",    "M",       "ETS + CutMix"),
    ("_ets_mix_mixup_",     "M",       "ETS + MixUp"),
    ("_ets_nomix_",         "M",       "ETS No Mix"),
    ("_lps_mix_both_",      "M",       "LPS + Both Mixing"),
    ("_lps_mix_cutmix_",    "M",       "LPS + CutMix"),
    ("_lps_mix_mixup_",     "M",       "LPS + MixUp"),
    ("_lps_nomix_",         "M",       "LPS No Mix"),
    ("_lps_",               "A_LPS",   "LPS Ablation"),
    ("_ets_",               "A_ETS",   "ETS Ablation"),
    ("cl_strength",         "A_CL",    "CL Strength Ablation"),
]


def get_group_and_label(run_name: str):
    name = run_name.lower()
    for keyword, group, label in GROUP_RULES:
        if keyword.lower() in name:
            return group, label
    return "Other", run_name


def upload_run(history_path: Path, project: str):
    ckpt_path = Path(str(history_path).replace("_history.pt", "_best.pth"))

    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else {}
    history = torch.load(history_path, map_location="cpu", weights_only=False)

    # handle both formats: history stored standalone or inside checkpoint
    if "train_loss" not in history and "history" in history:
        history = history["history"]

    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        print(f"  Skipping {history_path.name} — empty history")
        return

    run_name        = history_path.stem.replace("_history", "")
    cfg             = ckpt.get("cfg", {})
    group, label    = get_group_and_label(run_name)

    print(f"  Uploading: {run_name}  ({n_epochs} epochs)  [{group}]")

    wandb.init(
        project   = project,
        name      = run_name,
        config    = cfg,
        group     = group,
        tags      = [group, label],
        reinit    = True,
    )

    has_val   = len(history.get("val_loss", [])) == n_epochs
    has_top5  = len(history.get("val_top5", [])) == n_epochs

    for epoch in range(n_epochs):
        log = {
            "epoch":      epoch + 1,
            "train_loss": history["train_loss"][epoch],
            "train_acc":  history["train_acc"][epoch] * 100,
        }
        if has_val:
            log["val_loss"] = history["val_loss"][epoch]
            log["val_acc"]  = history["val_acc"][epoch] * 100
        if has_top5:
            log["val_top5"] = history["val_top5"][epoch] * 100

        wandb.log(log, step=epoch + 1)

    # log final summary metrics
    summary = {
        "best_val_acc":  ckpt.get("val_acc", 0) * 100,
        "test_top1":     ckpt.get("test_top1", 0) * 100,
        "test_top5":     ckpt.get("test_top5", 0) * 100,
        "best_epoch":    ckpt.get("epoch", 0),
        "total_minutes": ckpt.get("total_minutes", 0),
    }
    for k, v in summary.items():
        wandb.run.summary[k] = v

    wandb.finish()
    print(f"    best_val={summary['best_val_acc']:.2f}%  "
          f"test={summary['test_top1']:.2f}%  done ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",     type=str, nargs="+", default=None,
                        help="One or more run names to upload (without _history.pt)")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT,
                        help=f"W&B project name (default: {DEFAULT_PROJECT})")
    parser.add_argument("--dir",     type=str, default=str(CHECKPOINT_DIR),
                        help="Checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = Path(args.dir)
    if args.run:
        history_files = [ckpt_dir / f"{name}_history.pt" for name in args.run]
    else:
        history_files = sorted(ckpt_dir.glob("*_history.pt"))

    history_files = [f for f in history_files if f.exists()]

    if not history_files:
        print("No history files found.")
        return

    print(f"Project : {args.project}")
    print(f"Runs    : {len(history_files)}\n")

    for history_path in history_files:
        upload_run(history_path, project=args.project)

    print(f"\nDone. View at: https://wandb.ai/home → project '{args.project}'")


if __name__ == "__main__":
    main()
