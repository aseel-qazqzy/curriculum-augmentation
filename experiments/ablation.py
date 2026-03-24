"""
experiments/ablation.py
Ablation study runner — systematically tests each component of your CL method.

Ablations to run:
    A1 — CL Schedule:   sigmoid vs linear vs cosine vs step
    A2 — CL Mode:       inverse vs direct vs normalized
    A3 — CL Blend:      pure epoch vs pure sample vs mixed
    A4 — Warmup:        no warmup vs 5 vs 10 epochs
    A5 — Scheduler:     MultiStepLR vs CosineAnnealingLR (already done in baselines)
    A6 — Label Smooth:  0.0 vs 0.1 (CIFAR-100 specific)

Usage:
    # Run a specific ablation group
    python experiments/ablation.py --group schedule
    python experiments/ablation.py --group mode
    python experiments/ablation.py --group blend

    # Run all ablations (takes a long time)
    python experiments/ablation.py --group all

    # Quick debug run
    python experiments/ablation.py --group schedule --debug
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.train import main as run_cl_experiment, DEFAULT_CONFIG


# ABLATION DEFINITIONS
# Base config — same as your main CL experiment
BASE_CFG = {
    **DEFAULT_CONFIG,
    "dataset":       "cifar10",
    "epochs":        150,
    "lr":            0.1,
    "optimizer":     "sgd",
    "scheduler":     "multistep",
    "cl_schedule":   "sigmoid",
    "cl_mode":       "inverse",
    "cl_blend":      0.7,
    "warmup_epochs": 5,
    "label_smoothing": 0.0,
    "use_wandb":     False,
    "seed":          42,
}

ABLATION_GROUPS = {

    # ── A1: Which epoch-level schedule works best? 
    "schedule": [
        {**BASE_CFG,
         "cl_schedule":   "sigmoid",
         "experiment_name": "ablation_cl_schedule_sigmoid"},
        {**BASE_CFG,
         "cl_schedule":   "linear",
         "experiment_name": "ablation_cl_schedule_linear"},
        {**BASE_CFG,
         "cl_schedule":   "cosine",
         "experiment_name": "ablation_cl_schedule_cosine"},
        {**BASE_CFG,
         "cl_schedule":   "step",
         "experiment_name": "ablation_cl_schedule_step"},
    ],

    # ── A2: How should loss map to difficulty? 
    "mode": [
        {**BASE_CFG,
         "cl_mode":       "inverse",
         "experiment_name": "ablation_cl_mode_inverse"},
        {**BASE_CFG,
         "cl_mode":       "direct",
         "experiment_name": "ablation_cl_mode_direct"},
        {**BASE_CFG,
         "cl_mode":       "normalized",
         "experiment_name": "ablation_cl_mode_normalized"},
    ],

    # ── A3: How much should epoch vs sample-level drive difficulty? ──
    "blend": [
        {**BASE_CFG,
         "cl_blend":      0.0,
         "experiment_name": "ablation_cl_blend_0.0_pure_sample"},
        {**BASE_CFG,
         "cl_blend":      0.3,
         "experiment_name": "ablation_cl_blend_0.3"},
        {**BASE_CFG,
         "cl_blend":      0.7,
         "experiment_name": "ablation_cl_blend_0.7"},
        {**BASE_CFG,
         "cl_blend":      1.0,
         "experiment_name": "ablation_cl_blend_1.0_pure_epoch"},
    ],

    # ── A4: Does warmup help? 
    "warmup": [
        {**BASE_CFG,
         "warmup_epochs": 0,
         "experiment_name": "ablation_warmup_0"},
        {**BASE_CFG,
         "warmup_epochs": 5,
         "experiment_name": "ablation_warmup_5"},
        {**BASE_CFG,
         "warmup_epochs": 10,
         "experiment_name": "ablation_warmup_10"},
    ],

    # ── A5: Label smoothing (CIFAR-100) 
    "smoothing": [
        {**BASE_CFG,
         "dataset":        "cifar100",
         "label_smoothing": 0.0,
         "experiment_name": "ablation_smoothing_0.0_cifar100"},
        {**BASE_CFG,
         "dataset":        "cifar100",
         "label_smoothing": 0.05,
         "experiment_name": "ablation_smoothing_0.05_cifar100"},
        {**BASE_CFG,
         "dataset":        "cifar100",
         "label_smoothing": 0.1,
         "experiment_name": "ablation_smoothing_0.1_cifar100"},
    ],
}

# Convenience: all groups in one list
ABLATION_GROUPS["all"] = [
    cfg
    for group_cfgs in ABLATION_GROUPS.values()
    for cfg in group_cfgs
]

# RUNNER
def run_ablation_group(group: str, debug: bool = False):
    if group not in ABLATION_GROUPS:
        raise ValueError(f"Unknown group '{group}'. "
                         f"Available: {list(ABLATION_GROUPS.keys())}")

    configs = ABLATION_GROUPS[group]
    results = []

    print(f"\n{'═'*60}")
    print(f"  ABLATION: {group.upper()}")
    print(f"  {len(configs)} experiments to run")
    print(f"{'═'*60}\n")

    for i, cfg in enumerate(configs):
        cfg = cfg.copy()
        if debug:
            cfg["debug"]  = True
            cfg["epochs"] = 2

        print(f"\n[{i+1}/{len(configs)}] {cfg['experiment_name']}")
        print(f"  cl_schedule={cfg.get('cl_schedule')}  "
              f"cl_mode={cfg.get('cl_mode')}  "
              f"cl_blend={cfg.get('cl_blend')}  "
              f"warmup={cfg.get('warmup_epochs')}")

        try:
            history, best_val, test_top1 = run_cl_experiment(cfg)
            results.append({
                "name":      cfg["experiment_name"],
                "best_val":  best_val * 100,
                "test_top1": test_top1 * 100,
                "cfg":       cfg,
            })
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "name":      cfg["experiment_name"],
                "best_val":  None,
                "test_top1": None,
                "cfg":       cfg,
            })

    print_ablation_summary(group, results)
    return results

# SUMMARY TABLE
def print_ablation_summary(group: str, results: list):
    print(f"\n{'═'*70}")
    print(f"  ABLATION RESULTS — {group.upper()}")
    print(f"{'═'*70}\n")

    print(f"  {'Experiment':<50} {'Best Val':>9} {'Test Top-1':>10}")
    print("  " + "─" * 72)

    best_val = max((r["best_val"] for r in results if r["best_val"]), default=0)

    for r in results:
        if r["best_val"] is None:
            print(f"  {r['name']:<50}  {'FAILED':>9}")
            continue
        marker = "  ★" if r["best_val"] >= best_val else ""
        print(f"  {r['name']:<50} {r['best_val']:>8.2f}%  "
              f"{r['test_top1']:>8.2f}%{marker}")

    print("  " + "─" * 72)
    print("\n  ★ = best in group\n")

    # Key finding
    if len(results) >= 2:
        sorted_r = sorted([r for r in results if r["best_val"]],
                          key=lambda x: x["best_val"], reverse=True)
        if len(sorted_r) >= 2:
            diff = sorted_r[0]["best_val"] - sorted_r[-1]["best_val"]
            print(f"  Range: {diff:.2f}% between best and worst setting")
            print(f"  Best:  {sorted_r[0]['name']}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--group", type=str, default="schedule",
                        choices=list(ABLATION_GROUPS.keys()),
                        help="Which ablation group to run")
    parser.add_argument("--debug", action="store_true",
                        help="Quick 2-epoch test run")
    args = parser.parse_args()

    run_ablation_group(args.group, debug=args.debug)