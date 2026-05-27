import torch, sys
from pathlib import Path

sys.path.insert(0, '.')

CKPT = Path("checkpoints")
thresholds = [0.70, 0.75, 0.80]
runs = {
    "Static": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19"
}
print(f"{'Method':<12}", "  ".join(f"{int(t*100)}%@ep" for t in thresholds))
for name, stem in runs.items():
    h = torch.load(CKPT / f"{stem}_history.pt", map_location="cpu", weights_only=False)
    accs = h["val_acc"]
    epochs = [next((i+1 for i,a in enumerate(accs) if a >= t), None) for t in thresholds]
    print(f"{name:<12}", "  ".join(str(e) if e else "—" for e in epochs))
