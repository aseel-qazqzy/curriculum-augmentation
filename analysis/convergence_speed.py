import torch, sys
from pathlib import Path

sys.path.insert(0, ".")

CKPT = Path("results/cluster/checkpoints")
thresholds = [0.70, 0.75, 0.80]
runs = {
    "Static": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS (both)": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS (cutmix)": "ets_cutmix_19op_100ep_s42_ep100_cifar100_s42_p19",
    "ETS (mixup)": "ets_mixup_19op_100ep_s42_ep100_cifar100_s42_p19",
    "ETS (nomix)": "ets_nomix_19op_100ep_s42_ep100_cifar100_s42_p19",
    "ETS (t2only)": "wideresnet_tiered_ets_t2_only_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS (2-tier)": "wideresnet_tiered_ets_two_tier_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19",
}

# load all available histories once
data = {}
for name, stem in runs.items():
    path = CKPT / f"{stem}_history.pt"
    if path.exists():
        data[name] = torch.load(path, map_location="cpu", weights_only=False)

# --- Table 1: Convergence Speed ---
print("\n=== Convergence Speed (first epoch val_acc >= threshold) ===")
header = f"{'Method':<16}  " + "  ".join(f"{int(t * 100)}%@ep" for t in thresholds)
print(header)
print("-" * len(header))
for name in runs:
    if name not in data:
        print(f"{name:<16}  (file not found)")
        continue
    accs = data[name]["val_acc"]
    epochs = [
        next((i + 1 for i, a in enumerate(accs) if a >= t), None) for t in thresholds
    ]
    print(f"{name:<16}  " + "  ".join(f"{e:>6}" if e else f"{'—':>6}" for e in epochs))

# --- Table 2: Overfitting Gap ---
print("\n=== Overfitting Gap (last epoch) ===")
print(f"{'Method':<16}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  {'BestVal':>8}")
print("-" * 52)
for name in runs:
    if name not in data:
        print(f"{name:<16}  (file not found)")
        continue
    train = data[name]["train_acc"]
    val = data[name]["val_acc"]
    best_val = max(val)
    gap = (train[-1] - val[-1]) * 100
    print(
        f"{name:<16}  {train[-1] * 100:>6.2f}%  {val[-1] * 100:>6.2f}%  {gap:>6.2f}pp  {best_val * 100:>7.2f}%"
    )
