#!/bin/bash
# Plot commands for all completed runs
# Run from project root with venv active:
#   source venv/bin/activate && bash plot_commands.sh

cd "$(dirname "$0")"

# ── ETS mixing ablation — what type of mixing helps? ──────────────────────────
python analysis/plot_curves.py --mode single \
  --name "ets_nomix_19op_100ep_s42_ep100_cifar100_s42_p19" \
  --label "ETS no-mix" \
  --title "WideResNet · ETS · CIFAR-100"

python analysis/plot_curves.py --mode single \
  --name "ets_cutmix_19op_100ep_s42_ep100_cifar100_s42_p19" \
  --label "ETS + CutMix only" \
  --title "WideResNet · ETS · CIFAR-100"

python analysis/plot_curves.py --mode single \
  --name "ets_mixup_19op_100ep_s42_ep100_cifar100_s42_p19" \
  --label "ETS + MixUp only" \
  --title "WideResNet · ETS · CIFAR-100"

# ── EGS (best: v2, 80.01% ± 0.27%) ───────────────────────────────────────────
python analysis/plot_curves.py --mode single \
  --name "egs_v2_100ep_s42_ep100_cifar100_s42_p19" \
  --label "EGS" \
  --title "WideResNet · EGS · CIFAR-100"

# ── EGS multi-seed — proves statistical stability ─────────────────────────────
python analysis/plot_curves.py --mode single \
  --name "egs__v2_100ep_wideresnet_s123_ep100_cifar100_s123_p19" \
  --label "EGS (s123)" \
  --title "WideResNet · EGS · CIFAR-100"

python analysis/plot_curves.py --mode single \
  --name "egs_v2_100ep_wideresnet_s456_ep100_cifar100_s456_p19" \
  --label "EGS (s456)" \
  --title "WideResNet · EGS · CIFAR-100"

# ── ResNet-50 — backbone generalisation ──────────────────────────────────────
python analysis/plot_curves.py --mode single \
  --name "ets_resnet50_19op_100ep_s42_ep100_cifar100_s42_p19" \
  --label "ETS (ResNet-50)" \
  --title "ResNet-50 · ETS · CIFAR-100"

python analysis/plot_curves.py --mode single \
  --name "egs_v2_resnet50_19op_100ep_s42_ep100_cifar100_s42_p19" \
  --label "EGS (ResNet-50)" \
  --title "ResNet-50 · EGS · CIFAR-100"

# ── Tiny-ImageNet — dataset generalisation ────────────────────────────────────
python analysis/plot_curves.py --mode single \
  --name "ets_wrn_tinyimagenet_19op_100ep_s42_ep100_tiny_imagenet_s42_p19" \
  --label "ETS (Tiny-ImageNet)" \
  --title "WideResNet · ETS · Tiny-ImageNet"


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED — multi-run comparison figures
# ══════════════════════════════════════════════════════════════════════════════

# ── fig10: EGS multi-seed confidence bands (mean ± 1σ across 3 seeds) ────────
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from analysis.plot_curves import *
seeds_data = [
    load_history("egs_v2_100ep_s42_ep100_cifar100_s42_p19"),
    load_history("egs__v2_100ep_wideresnet_s123_ep100_cifar100_s123_p19"),
    load_history("egs_v2_100ep_wideresnet_s456_ep100_cifar100_s456_p19"),
]
method_seeds = {
    "EGS": {"histories": [h for h in seeds_data if h is not None], "color": PALETTE["cl"]}
}
if method_seeds["EGS"]["histories"]:
    fig10_multiseed(method_seeds, title="WideResNet · EGS · CIFAR-100 · 3 seeds",
                    fname="fig10_egs_multiseed.png")
else:
    print("  fig10: no EGS histories found, skipping.")
EOF


# ── fig14 + fig1: Mixing ablation — no-mix vs CutMix vs MixUp ────────────────
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from analysis.plot_curves import *
runs = build_runs([
    ("ETS no-mix",   "ets_nomix_19op_100ep_s42_ep100_cifar100_s42_p19",  PALETTE["static"]),
    ("ETS + CutMix", "ets_cutmix_19op_100ep_s42_ep100_cifar100_s42_p19", PALETTE["cosine"]),
    ("ETS + MixUp",  "ets_mixup_19op_100ep_s42_ep100_cifar100_s42_p19",  PALETTE["adam"]),
], CHECKPOINT_DIR)
fig14_ablation_components(runs, fname="fig14_mixing_ablation.png")
fig1_val_comparison(runs, title="Mixing Ablation · WideResNet · ETS · CIFAR-100",
                    fname="fig1_mixing_ablation.png")
print_analysis(runs)
EOF


# ── fig15 + fig1 + fig7 + fig16: ETS vs EGS scheduling comparison ────────────
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from analysis.plot_curves import *
runs = build_runs([
    ("ETS no-mix",  "ets_nomix_19op_100ep_s42_ep100_cifar100_s42_p19",         PALETTE["static"]),
    ("ETS + CutMix","ets_cutmix_19op_100ep_s42_ep100_cifar100_s42_p19",        PALETTE["cosine"]),
    ("ETS + MixUp", "ets_mixup_19op_100ep_s42_ep100_cifar100_s42_p19",         PALETTE["adam"]),
    ("EGS",         "egs_v2_100ep_s42_ep100_cifar100_s42_p19",                 PALETTE["cl"]),
], CHECKPOINT_DIR)
fig15_scheduling_comparison(runs, fname="fig15_ets_vs_egs.png")
fig1_val_comparison(runs, title="ETS vs EGS · WideResNet · CIFAR-100",
                    fname="fig1_scheduling_comparison.png")
fig7_gap_over_epochs(runs, fname="fig7_scheduling_gap.png")
fig16_convergence_speed(runs, fname="fig16_scheduling_convergence.png")
print_analysis(runs)
EOF


# ── fig1 + fig5: Backbone comparison — WideResNet vs ResNet-50 ───────────────
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from analysis.plot_curves import *
runs = build_runs([
    ("ETS (WideResNet)", "ets_nomix_19op_100ep_s42_ep100_cifar100_s42_p19",        PALETTE["cl"]),
    ("ETS (ResNet-50)",  "ets_resnet50_19op_100ep_s42_ep100_cifar100_s42_p19",     PALETTE["static"]),
    ("EGS (WideResNet)", "egs_v2_100ep_s42_ep100_cifar100_s42_p19",               PALETTE["adam"]),
    ("EGS (ResNet-50)",  "egs_v2_resnet50_19op_100ep_s42_ep100_cifar100_s42_p19", PALETTE["cosine"]),
], CHECKPOINT_DIR)
fig1_val_comparison(runs, title="Backbone Comparison · CIFAR-100 · ETS vs EGS",
                    fname="fig1_backbone_comparison.png")
fig5_summary(runs, fname="fig5_backbone_comparison.png")
print_analysis(runs)
EOF


# ── fig12 + fig9: Tiny-ImageNet tier transition zoom ─────────────────────────
python - << 'EOF'
import sys; sys.path.insert(0, '.')
from analysis.plot_curves import *
h = load_history("ets_wrn_tinyimagenet_19op_100ep_s42_ep100_tiny_imagenet_s42_p19")
if h:
    fig12_tier_zoom(h, tier_epochs=[20, 45], window=12,
                    title="WideResNet · ETS · Tiny-ImageNet · 100 epochs",
                    fname="fig12_tier_zoom_tinyimagenet.png")
    fig9_single_run(h, label="ETS (Tiny-ImageNet)", tier_epochs=[20, 45],
                    title="WideResNet · ETS · Tiny-ImageNet · 100 epochs",
                    fname="fig9_single_run_tinyimagenet.png")
else:
    print("  Tiny-ImageNet history not found, skipping.")
EOF
