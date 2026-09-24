"""
experiments/madaug/controlled_baselines/protocol.py

Controlled-protocol rerun of Static Mixing / ETS / LPS / EGS for the MADAug comparison.
NEW runs; they never replace or mix with the original (frozen) baseline results.

Controlled protocol = the original baseline code (experiments/train_baseline.py, frozen) with
exactly three intended changes:
  1. fixed stratified 49k/1k CIFAR-100 split shared with MADAug (splits/cifar100_val1000.json)
  2. final-epoch reporting (no validation-based checkpoint selection, no early stopping)
  3. isolated outputs: results/controlled_baselines/, run names end in "_ctrl49k"
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.madaug.make_split import DEFAULT_OUT as CONTROLLED_SPLIT_FILE
from experiments.madaug.make_split import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLED_ROOT = PROJECT_ROOT / "results" / "controlled_baselines"
N_TRAIN_TOTAL = 50000
N_VAL = 1000


def _inside_controlled_root(path) -> bool:
    try:
        Path(path).resolve().relative_to(CONTROLLED_ROOT.resolve())
        return True
    except ValueError:
        return False


def apply_controlled_protocol(cfg: dict, args) -> dict:
    """Called from train_controlled.py __main__ after the original CLI overrides."""
    if cfg["dataset"] != "cifar100":
        raise ValueError("Controlled protocol is defined for CIFAR-100 only.")
    if getattr(args, "val_split", None) is not None:
        raise ValueError(
            "--val_split is not allowed: the controlled protocol uses the fixed 49k/1k split file."
        )
    if cfg.get("early_stopping_patience", 0):
        raise ValueError(
            "--early_stopping_patience is not allowed: controlled protocol reports the final epoch."
        )

    cfg["split_file"] = str(CONTROLLED_SPLIT_FILE)
    # Informational only (1,000 / 50,000). Keeps the original LPS guard (val_split != 0) satisfied;
    # the loaders never read it.
    cfg["val_split"] = N_VAL / N_TRAIN_TOTAL

    if getattr(args, "checkpoint_dir", None) is None:
        cfg["checkpoint_dir"] = str(CONTROLLED_ROOT / "checkpoints")
    if getattr(args, "log_dir", None) is None:
        cfg["log_dir"] = str(CONTROLLED_ROOT / "logs")
    for key in ("checkpoint_dir", "log_dir"):
        if not _inside_controlled_root(cfg[key]):
            raise ValueError(
                f"--{key} must be inside {CONTROLLED_ROOT} (got {cfg[key]})."
            )
    if cfg.get("resume") and not _inside_controlled_root(cfg["resume"]):
        raise ValueError(f"--resume must point inside {CONTROLLED_ROOT}.")
    return cfg


def metrics_path(cfg: dict) -> Path:
    return (
        Path(cfg["checkpoint_dir"]).parent
        / "metrics"
        / f"{cfg['experiment_name']}.json"
    )


def guard_existing_run(cfg: dict):
    """Refuse to overwrite a completed controlled run (unless resuming)."""
    m = metrics_path(cfg)
    if m.exists() and not cfg.get("resume"):
        if json.loads(m.read_text()).get("complete"):
            raise FileExistsError(
                f"Completed controlled run already exists: {m}. Refusing to overwrite."
            )


def method_label(cfg: dict) -> str:
    if cfg["augmentation"] == "tiered_curriculum":
        return cfg.get("tier_schedule", "ets")
    return cfg["augmentation"]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return None


def write_controlled_metrics(
    cfg, history, final_ckpt, test_top1, test_top5, total_time
):
    """Final-epoch metrics in the same schema as MADAug's metrics JSON."""
    train_idx, val_idx = load_split(cfg["split_file"])
    reported_epoch = final_ckpt.get("epoch")
    out = {
        "run_name": cfg["experiment_name"],
        "protocol": "controlled_49k1k_final_epoch",
        "method": method_label(cfg),
        "complete": True,
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "reported_epoch": reported_epoch,
        "reported_epoch_is_final": reported_epoch == cfg["epochs"],
        "final_epoch_top1": test_top1 * 100,
        "final_epoch_top1_error": (1 - test_top1) * 100,
        "final_epoch_top5": test_top5 * 100,
        "reported": "final epoch (no validation- or test-based selection)",
        "split": {
            "file": cfg["split_file"],
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "use_amp": bool(cfg.get("use_amp")),
        "total_minutes": total_time / 60,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cfg": {
            k: v
            for k, v in cfg.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
        "history": history,
    }
    p = metrics_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Controlled metrics: {p}")
    return out
