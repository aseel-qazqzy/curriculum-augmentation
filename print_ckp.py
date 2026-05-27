import torch, sys
sys.path.insert(0, '.')
from data.datasets import get_cifar100_loaders
from models.registry import get_model
from training.trainer import evaluate

ckpt_path = "checkpoints/ets_mixup_19op_200ep_s3407_full_dataset_ep200_cifar100_s3407_p19_best.pth"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg  = ckpt.get("config", {})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = get_model("wideresnet", num_classes=100).to(device)
model.load_state_dict(ckpt["model_state_dict"])

_, _, test_loader = get_cifar100_loaders(batch_size=128, val_split=0.0, num_workers=4)

criterion = torch.nn.CrossEntropyLoss()
loss, top1, top5 = evaluate(model, test_loader,criterion, device)
print(f"\n  Test Top-1 : {top1:.2f}%")
print(f"  Test Top-5 : {top5:.2f}%")
print(f"  Test Loss  : {loss:.4f}")

