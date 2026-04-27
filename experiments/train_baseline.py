"""
train_baseline.py - baseline training, no curriculum
augmentation: none / static / random / randaugment
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tiny_imagenet_loaders,
    get_static_transforms,
    get_no_augmentation_transforms,
)
from models.registry import get_model
from experiments.utils import (
    set_seed,
    get_device,
    build_optimizer,
    build_scheduler,
    setup_logging,
)
from experiments.config import BASE_CONFIG
from training.trainer import train_one_epoch, evaluate


DEFAULT_CONFIG = {
    **BASE_CONFIG,
    "augmentation": "static",
    "ra_n": 2,
    "ra_m": 9,
    "experiment_name": "resnet18_static_aug_sgd_multistep",
}


def build_transforms(cfg: dict):
    aug = cfg["augmentation"]
    dataset = cfg["dataset"]

    if aug == "none":
        return get_no_augmentation_transforms(dataset)

    elif aug in ("static", "static_mixing"):
        from augmentations.policies import StaticAugmentation

        policy = StaticAugmentation(
            dataset=dataset, strength=cfg.get("fixed_strength", 0.7)
        )
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "random":
        from augmentations.policies import RandomAugmentation

        policy = RandomAugmentation(dataset=dataset)
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "randaugment":
        from augmentations.policies import RandAugmentPolicy

        policy = RandAugmentPolicy(dataset=dataset, N=cfg["ra_n"], M=cfg["ra_m"])
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "tiered_curriculum":
        schedule = cfg.get("tier_schedule", "ets")
        if schedule == "egs":
            raise NotImplementedError(
                f"tier_schedule='{schedule}' is not yet implemented. "
                f"LPS and EGS are planned for June. Use --tier_schedule ets."
            )

        from augmentations.policies import ThreeTierCurriculumAugmentation

        def _resolve_tier(val, frac, epochs):
            if val is None:
                return int(epochs * frac)
            return int(val * epochs) if 0.0 < val < 1.0 else int(val)

        epochs = cfg["epochs"]
        policy = ThreeTierCurriculumAugmentation(
            dataset=dataset,
            t1=_resolve_tier(cfg.get("tier_t1"), 0.33, epochs),
            t2=_resolve_tier(cfg.get("tier_t2"), 0.66, epochs),
            strength=cfg.get("fixed_strength", 0.7),
        )
        return policy.get_train_transform(), policy.get_val_transform()

    else:
        raise ValueError(
            f"Unknown augmentation '{aug}'. "
            f"Choose from: none, static, static_mixing, random, randaugment, tiered_curriculum"
        )


def main(cfg: dict):
    dataset = cfg["dataset"]
    if cfg["experiment_name"] == DEFAULT_CONFIG["experiment_name"]:
        aug = cfg["augmentation"]
        if aug == "tiered_curriculum":
            schedule = cfg.get("tier_schedule", "ets")
            mix_mode = cfg.get("mix_mode", "both")
            mix_tag = "nomix" if mix_mode == "none" else f"mix_{mix_mode}"
            cfg["experiment_name"] = (
                f"{cfg['model']}_tiered_{schedule}_{mix_tag}"
                f"_{cfg['optimizer']}_{cfg['scheduler']}"
            )
        else:
            cfg["experiment_name"] = (
                f"{cfg['model']}_{aug}_{cfg['optimizer']}_{cfg['scheduler']}"
            )

    epochs = cfg["epochs"]
    if f"_ep{epochs}" not in cfg["experiment_name"]:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_ep{epochs}"

    if not cfg["experiment_name"].endswith(f"_{dataset}"):
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{dataset}"

    tee = setup_logging(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    print(f"\n{'=' * 60}")
    print(f"  Experiment  : {cfg['experiment_name']}")
    print(f"  Dataset     : {cfg['dataset']}")
    print(f"  Model       : {cfg['model']}")
    print(
        f"  Augmentation: {cfg['augmentation']}"
        + (
            f"  (N={cfg['ra_n']}, M={cfg['ra_m']})"
            if cfg["augmentation"] == "randaugment"
            else ""
        )
    )
    if cfg["augmentation"] == "tiered_curriculum":
        from augmentations.policies import (
            _TIER_STRENGTH_FRACS,
            _TIER_N_OPS,
            _STRENGTH_RAMP_EPOCHS,
        )

        def _resolve_tier(val, frac, epochs):
            if val is None:
                return int(epochs * frac)
            return int(val * epochs) if 0.0 < val < 1.0 else int(val)

        t1 = _resolve_tier(cfg.get("tier_t1"), 0.33, cfg["epochs"])
        t2 = _resolve_tier(cfg.get("tier_t2"), 0.66, cfg["epochs"])
        ceil = cfg.get("fixed_strength", 0.7)
        s1 = ceil * _TIER_STRENGTH_FRACS[1]
        s2 = ceil * _TIER_STRENGTH_FRACS[2]
        s3 = ceil * _TIER_STRENGTH_FRACS[3]
        is_lps = cfg.get("tier_schedule") == "lps"
        t1_str = "loss-guided" if is_lps else f"ep   1-{t1:2d}"
        t2_str = "loss-guided" if is_lps else f"ep {t1 + 1:2d}-{t2:2d}"
        t3_str = "loss-guided" if is_lps else f"ep {t2 + 1:2d}-end"
        print(
            f"  Tier 1 ({t1_str}): flip, crop, translate_x/y"
            f"  |  sample {_TIER_N_OPS[1]}/4  |  strength {s1:.2f}"
        )
        print(
            f"  Tier 2 ({t2_str}): +color_jitter, rotation, shear, auto_contrast, equalize, sharpness"
            f"  |  sample {_TIER_N_OPS[2]}/10  |  strength {s2:.2f} (ramp {_STRENGTH_RAMP_EPOCHS} ep)"
        )
        print(
            f"  Tier 3 ({t3_str}): +grayscale, cutout, contrast, brightness"
            f"  |  sample {_TIER_N_OPS[3]}/14  |  strength {s3:.2f} (ramp {_STRENGTH_RAMP_EPOCHS} ep)"
        )
        mix_mode = cfg.get("mix_mode", "both")
        if mix_mode != "none":
            ramp_str = "ramp enabled" if cfg.get("mix_ramp", False) else "ramp disabled"
            print(
                f"  Mixing      : {mix_mode}  alpha={cfg.get('mix_alpha', 1.0)}  "
                f"p={cfg.get('mix_prob', 0.5)}  (Tier 3 only | {ramp_str})"
            )
        if cfg.get("tier_schedule") == "lps":
            print(
                f"  LPS         : tau={cfg['lps_tau']}  window={cfg['lps_window']}  "
                f"min_epochs_per_tier={cfg['lps_min_epochs']}"
            )
    if cfg["augmentation"] == "static_mixing":
        mix_mode = cfg.get("mix_mode", "both")
        print(
            f"  Ops         : all 7 from epoch 1  |  strength {cfg.get('fixed_strength', 0.7)}"
        )
        print(
            f"  Mixing      : {mix_mode}  alpha={cfg.get('mix_alpha', 1.0)}  "
            f"p={cfg.get('mix_prob', 0.5)}  (from epoch 1)"
        )
    print(f"  Epochs      : {cfg['epochs']} | Batch: {cfg['batch_size']}")

    if cfg.get("use_wandb"):
        import wandb
        from experiments.upload_to_wandb import get_group_and_label

        _group, _label = get_group_and_label(cfg["experiment_name"])
        _name_lower = cfg["experiment_name"].lower()
        _wandb_cfg = dict(cfg)
        if "_lps_" in _name_lower:
            _wandb_cfg["scheduler_type"] = "LPS"
        elif "_ets_" in _name_lower:
            _wandb_cfg["scheduler_type"] = "ETS"
        else:
            _wandb_cfg["scheduler_type"] = "none"
        wandb.init(
            project="curriculum-augmentation",
            name=cfg["experiment_name"],
            config=_wandb_cfg,
            group=_group,
            tags=[_group, _label],
        )

    train_transform, val_transform = build_transforms(cfg)

    # LPS manages tier transitions via set_tier(); prevent the transform from
    # falling back to ETS epoch-boundaries before the first set_tier() call.
    if cfg.get("tier_schedule") == "lps" and hasattr(train_transform, "set_tier"):
        train_transform.set_tier(1)

    if cfg["dataset"] == "cifar100":
        loader_fn = get_cifar100_loaders
    elif cfg["dataset"] == "tiny_imagenet":
        loader_fn = get_tiny_imagenet_loaders
    else:
        loader_fn = get_cifar10_loaders
    train_loader, val_loader, test_loader = loader_fn(
        root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        val_split=cfg["val_split"],
        train_transform=train_transform,
        test_transform=val_transform,
        debug=cfg.get("debug", False),
    )

    num_classes = {"cifar100": 100, "tiny_imagenet": 200}.get(cfg["dataset"], 10)
    model = get_model(cfg["model"], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer, _ = build_optimizer(model, cfg)
    scheduler, milestones = build_scheduler(optimizer, cfg)

    # Loss CL
    lps_scheduler = None
    if cfg.get("tier_schedule") == "lps":
        from training.losses import LossPlateauScheduler

        lps_scheduler = LossPlateauScheduler(
            tau=cfg["lps_tau"],
            window=cfg["lps_window"],
            min_epochs_per_tier=cfg["lps_min_epochs"],
            higher_is_better=False,  # tracks val_loss; lower is better
        )

    use_amp = cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  AMP         : enabled (float16)")

    mixer = None
    if (
        cfg["augmentation"] in ("tiered_curriculum", "static_mixing")
        and cfg.get("mix_mode", "both") != "none"
    ):
        from augmentations.mixing import BatchMixer

        mixer = BatchMixer(
            mode=cfg.get("mix_mode", "both"),
            alpha=cfg.get("mix_alpha", 1.0),
            p=cfg.get("mix_prob", 0.5),
        )

    print(f"{'=' * 60}\n")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(
        cfg["checkpoint_dir"], f"{cfg['experiment_name']}_best.pth"
    )
    history_path = os.path.join(
        cfg["checkpoint_dir"], f"{cfg['experiment_name']}_history.pt"
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5": [],
    }
    best_val_acc = -1.0
    start_epoch = 1
    start_time = time.time()
    es_patience = cfg.get("early_stopping_patience", 0)
    es_counter = 0

    resume_path = cfg.get("resume")
    if resume_path:
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler:
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                scheduler.last_epoch = ckpt["epoch"]
        history = ckpt["history"]
        best_val_acc = ckpt["val_acc"]
        start_epoch = ckpt["epoch"] + 1
        print(
            f"  Resuming from epoch {start_epoch} | best val acc so far: {best_val_acc * 100:.2f}%\n"
        )

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if hasattr(train_transform, "set_epoch"):
            train_transform.set_epoch(epoch)

        if cfg["augmentation"] == "static_mixing":
            active_mixer = mixer  # always on from epoch 1
        elif mixer is not None and hasattr(train_transform, "mix_scale"):
            scale = train_transform.mix_scale()  # 0→1 at Tier 3 boundary
            if scale <= 0.0:
                active_mixer = None
            elif cfg.get(
                "mix_ramp", False
            ):  # ramp enabled: gradually increase p and alpha
                mixer.p = cfg.get("mix_prob", 0.5) * scale
                mixer.alpha = 0.2 + (cfg.get("mix_alpha", 1.0) - 0.2) * scale
                active_mixer = mixer
            else:  # ramp disabled: full strength from Tier 3 ep1
                mixer.p = cfg.get("mix_prob", 0.5)
                mixer.alpha = cfg.get("mix_alpha", 1.0)
                active_mixer = mixer
        else:
            active_mixer = None
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            mixer=active_mixer,
        )

        use_val = val_loader is not None
        if use_val:
            val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc, val_top5 = 0.0, 0.0, 0.0

        if lps_scheduler is not None:
            # Track val_loss: less sensitive to LR drops than train loss (val loss can
            # increase after LR drops due to overfitting), keeps LPS conceptually loss-based.
            lps_scheduler.update(val_loss)
            if lps_scheduler.should_advance():
                new_tier = lps_scheduler.advance(epoch)
                train_transform.set_tier(new_tier)
                print(f"  LPS: advancing to Tier {new_tier} at epoch {epoch}")

        if scheduler:
            scheduler.step()
            # When learning rate drops reset loss history
            if lps_scheduler is not None and epoch in milestones:
                lps_scheduler.notify_lr_drop()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)
        torch.save(history, history_path)

        if cfg.get("use_wandb"):
            import wandb

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc * 100,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if use_val:
                log_dict.update(
                    {
                        "val_loss": val_loss,
                        "val_acc": val_acc * 100,
                        "val_top5": val_top5 * 100,
                    }
                )
            wandb.log(log_dict, step=epoch)

        if epoch % cfg["log_every"] == 0 or epoch == 1 or cfg.get("debug"):
            elapsed = time.time() - start_time
            tier_str = (
                f" | {train_transform.tier_label()}"
                if hasattr(train_transform, "tier_label")
                else ""
            )
            if active_mixer is not None:
                tier_str += f" + mix:{cfg.get('mix_mode', 'both')}"
            elif mixer is not None:
                tier_str += " + mix:pending"
            if use_val:
                print(
                    f"Epoch [{epoch:>3}/{cfg['epochs']}] "
                    f"Train: {train_loss:.4f} / {train_acc * 100:.2f}% | "
                    f"Val: {val_loss:.4f} / {val_acc * 100:.2f}% | "
                    f"Top-5: {val_top5 * 100:.2f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}{tier_str} | "
                    f"Time: {elapsed:.0f}s"
                )
            else:
                print(
                    f"Epoch [{epoch:>3}/{cfg['epochs']}] "
                    f"Train: {train_loss:.4f} / {train_acc * 100:.2f}% | "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}{tier_str} | "
                    f"Time: {elapsed:.0f}s"
                )

        if use_val:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                es_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()
                        if scheduler
                        else None,
                        "val_acc": val_acc,
                        "history": history,
                        "cfg": {**cfg, "milestones": milestones},
                    },
                    ckpt_path,
                )
                print(f"Best saved (epoch={epoch}, val_acc={val_acc * 100:.2f}%)")
            else:
                es_counter += 1
                if es_patience > 0 and es_counter >= es_patience:
                    print(
                        f"Early stopping at epoch {epoch} — no improvement for {es_patience} epochs."
                    )
                    break
        else:
            # Full-train mode: save at last epoch only
            if epoch == cfg["epochs"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()
                        if scheduler
                        else None,
                        "val_acc": 0.0,
                        "history": history,
                        "cfg": {**cfg, "milestones": milestones},
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved (epoch={epoch}, full-train mode)")

    if Path(ckpt_path).exists():
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        best_ckpt = {"epoch": 0, "val_acc": 0.0, "history": history, "cfg": cfg}

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion, device)
    total_time = time.time() - start_time

    best_ckpt["test_top1"] = test_top1
    best_ckpt["test_top5"] = test_top5
    best_ckpt["total_minutes"] = total_time / 60
    torch.save(best_ckpt, ckpt_path)

    last_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    last_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
    last_val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
    last_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    best_epoch = best_ckpt.get("epoch", "—")
    full_train_mode = val_loader is None

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  FINAL RESULTS: {cfg['experiment_name']}")
    mode_str = (
        "full-train (val_split=0.0)"
        if full_train_mode
        else f"dev (val_split={cfg['val_split']})"
    )
    print(f"  Mode: {mode_str}")
    print(sep)
    print(f"  {'Metric':<22}  {'Value':>10}")
    print(f"  {'─' * 22}  {'─' * 10}")
    print(f"  {'Train Loss (last ep)':<22}  {last_train_loss:>10.4f}")
    print(f"  {'Train Acc  (last ep)':<22}  {last_train_acc * 100:>9.2f}%")
    if not full_train_mode:
        print(f"  {'Val Loss   (last ep)':<22}  {last_val_loss:>10.4f}")
        print(f"  {'Val Acc    (last ep)':<22}  {last_val_acc * 100:>9.2f}%")
    print(f"  {'─' * 22}  {'─' * 10}")
    if not full_train_mode:
        print(
            f"  {'Best Val Top-1':<22}  {best_val_acc * 100:>9.2f}%  (epoch {best_epoch})"
        )
    print(f"  {'Test Top-1':<22}  {test_top1 * 100:>9.2f}%")
    print(f"  {'Test Top-5':<22}  {test_top5 * 100:>9.2f}%")
    if not full_train_mode:
        print(f"  {'Val–Test Gap':<22}  {abs(best_val_acc - test_top1) * 100:>9.2f}%")
    print(f"  {'─' * 22}  {'─' * 10}")
    print(f"  {'Total Time':<22}  {total_time / 60:>9.1f} min")
    print(sep)
    if lps_scheduler is not None and lps_scheduler.tier_change_log:
        print(f"  LPS Tier transitions: {lps_scheduler.tier_change_log}")

    if cfg.get("use_wandb"):
        import wandb

        wandb.log(
            {"test_top1": test_top1 * 100, "test_top5": test_top5 * 100},
            step=cfg["epochs"],
        )
        wandb.run.summary["test_top1"] = test_top1 * 100
        wandb.run.summary["test_top5"] = test_top5 * 100
        wandb.run.summary["best_val_acc"] = best_val_acc * 100
        wandb.run.summary["total_minutes"] = total_time / 60
        wandb.finish()

    tee.close()
    return history, best_val_acc, test_top1


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training — CIFAR-10/100")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cifar10", "cifar100", "tiny_imagenet"],
    )
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
            "baseline_cnn",
        ],
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        choices=[
            "none",
            "static",
            "static_mixing",
            "random",
            "randaugment",
            "tiered_curriculum",
        ],
    )
    parser.add_argument("--ra_n", type=int, default=None)
    parser.add_argument("--ra_m", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam"])
    parser.add_argument(
        "--scheduler", type=str, default=None, choices=["multistep", "cosine", "none"]
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="2 epochs on 512 samples — quick smoke test",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--val_split",
        type=float,
        default=None,
        help="Validation fraction: 0.1=dev mode (45k/5k), 0.0=full-train mode (50k)",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument(
        "--fixed_strength",
        type=float,
        default=None,
        help="Augmentation op strength 0.0–1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--tier_schedule",
        type=str,
        default=None,
        choices=["ets", "lps", "egs"],
        help="Scheduling signal for tiered_curriculum: "
        "ets=Epoch-Threshold, lps=Loss-Plateau, egs=Entropy-Guided (default: ets)",
    )
    parser.add_argument(
        "--tier_t1",
        type=float,
        default=None,
        help="ETS: tier 1 boundary — int≥1 = absolute epoch, float 0–1 = fraction of epochs (default: 0.33)",
    )
    parser.add_argument(
        "--tier_t2",
        type=float,
        default=None,
        help="ETS: tier 2 boundary — int≥1 = absolute epoch, float 0–1 = fraction of epochs (default: 0.66)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision (CUDA only)",
    )
    parser.add_argument(
        "--mix_mode",
        type=str,
        default=None,
        choices=["cutmix", "mixup", "both", "none"],
        help="Batch mixing in Tier 3 (default: both)",
    )
    parser.add_argument(
        "--mix_alpha",
        type=float,
        default=None,
        help="Beta(alpha, alpha) for mix ratio (default: 1.0)",
    )
    parser.add_argument(
        "--mix_prob",
        type=float,
        default=None,
        help="Probability of mixing a batch in Tier 3 (default: 0.5)",
    )
    parser.add_argument(
        "--mix_ramp",
        action="store_true",
        default=False,
        help="Gradually ramp mixing p and alpha over first 5 epochs of Tier 3 (default: disabled)",
    )
    parser.add_argument(
        "--lps_tau",
        type=float,
        default=None,
        help="LPS: plateau threshold — min relative improvement to stay in tier (default: 0.02)",
    )
    parser.add_argument(
        "--lps_window",
        type=int,
        default=None,
        help="LPS: epochs lookback window for plateau detection (default: 5)",
    )
    parser.add_argument(
        "--lps_min_epochs",
        type=int,
        default=None,
        help="LPS: minimum epochs per tier before advancing (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    for key in [
        "dataset",
        "model",
        "augmentation",
        "ra_n",
        "ra_m",
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "scheduler",
        "experiment_name",
        "checkpoint_dir",
        "resume",
        "val_split",
        "early_stopping_patience",
        "fixed_strength",
        "tier_schedule",
        "tier_t1",
        "tier_t2",
        "mix_mode",
        "mix_alpha",
        "mix_prob",
        "mix_ramp",
        "lps_tau",
        "lps_window",
        "lps_min_epochs",
        "seed",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if args.debug:
        cfg["debug"] = True
        cfg["epochs"] = 2
    if args.use_wandb:
        cfg["use_wandb"] = True
    if args.use_amp:
        cfg["use_amp"] = True

    main(cfg)
