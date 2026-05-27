# Expected Calibration Error
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, ".")

from data.datasets import get_cifar100_loaders
from models.registry import get_model

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BINS = 10

methods = {
    "Static": "wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "ETS": "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "LPS": "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19_best.pth",
    "EGS": "egs_v2_100ep_s42_ep100_cifar100_s42_p19_best.pth",
}


def compute_ece(confidences, correct, n_bins=N_BINS):
    ece = 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    for i in range(n_bins):
        in_bin = [
            j for j, c in enumerate(confidences) if bin_edges[i] <= c < bin_edges[i + 1]
        ]
        if not in_bin:
            continue
        avg_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        avg_acc = sum(correct[j] for j in in_bin) / len(in_bin)
        weight = len(in_bin) / len(confidences)
        ece += weight * abs(avg_conf - avg_acc)
    return ece


_, _, test_loader = get_cifar100_loaders(val_split=0.1, batch_size=128)

print(f"{'Method':<12}  {'ECE':>6}  {'Acc':>6}")
print("-" * 28)

for name, ckpt_file in methods.items():
    ckpt_path = CKPT / ckpt_file
    if not ckpt_path.exists():
        print(f"{name:<12}  (file not found)")
        continue

    model = get_model("wideresnet", num_classes=100).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_confidences, all_correct = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            probs = F.softmax(model(images), dim=1)
            confidence, predicted = probs.max(dim=1)
            all_confidences.extend(confidence.cpu().tolist())
            all_correct.extend((predicted == labels).cpu().tolist())

    ece = compute_ece(all_confidences, all_correct)
    acc = sum(all_correct) / len(all_correct) * 100
    print(f"{name:<12}  {ece:>6.4f}  {acc:>5.2f}%")
