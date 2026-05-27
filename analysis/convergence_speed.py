import torch, sys
from pathlib import Path

sys.path.insert(0, ".")

CKPT = Path("checkpoints")
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

header = f"{'Method':<16}  " + "  ".join(f"{int(t * 100)}%@ep" for t in thresholds)
print(header)
print("-" * len(header))

for name, stem in runs.items():
    path = CKPT / f"{stem}_history.pt"
    if not path.exists():
        print(f"{name:<16}  (file not found)")
        continue
    h = torch.load(path, map_location="cpu", weights_only=False)
    accs = h["val_acc"]
    epochs = [
        next((i + 1 for i, a in enumerate(accs) if a >= t), None) for t in thresholds
    ]
    print(f"{name:<16}  " + "  ".join(f"{e:>6}" if e else f"{'—':>6}" for e in epochs))
