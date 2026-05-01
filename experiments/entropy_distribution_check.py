import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compute_entropy import entropy_to_difficulty

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Path to .npy entropy file")
args = parser.parse_args()

entropy_path = Path(args.file)
data = np.load(entropy_path)
log_C = data[0]
scores = data[1:]

# derive label from filename for plot title and output
label = entropy_path.stem  # e.g. entropy_cifar100_wideresnet

print(f"N samples       : {len(scores):,}")
print(f"log(C)          : {log_C:.4f}")
print(f"Range           : [{scores.min():.4f}, {scores.max():.4f}]")
print(
    f"Mean entropy    : {scores.mean():.4f}  ({scores.mean() / log_C * 100:.1f}% of max)"
)
print(f"NaN / Inf       : {np.isnan(scores).sum()} / {np.isinf(scores).sum()}")
print(f"Zeros (unfilled): {(scores == 0.0).sum()}")

plt.figure(figsize=(8, 4))
plt.hist(scores, bins=100, color="steelblue", edgecolor="none")
plt.axvline(
    log_C, color="red", linestyle="--", label=f"max entropy (log C = {log_C:.2f})"
)
plt.axvline(
    scores.mean(), color="orange", linestyle="--", label=f"mean = {scores.mean():.2f}"
)
plt.xlabel("Entropy H(x)")
plt.ylabel("Count")
plt.title(f"Per-sample entropy distribution — {label}")
plt.legend()
plt.tight_layout()
out_png = Path("analysis") / f"{label}.png"
plt.savefig(out_png)
print(f"Plot saved → {out_png}")

difficulty = entropy_to_difficulty(scores, log_C)
print(f"\nDifficulty range: [{difficulty.min():.4f}, {difficulty.max():.4f}]")
print(
    f"  Tier 1 aug (light)  — uncertain samples (high entropy): {(difficulty < 0.33).sum():,}"
)
print(
    f"  Tier 2 aug (medium) — medium  samples                 : {((difficulty >= 0.33) & (difficulty < 0.66)).sum():,}"
)
print(
    f"  Tier 3 aug (heavy)  — confident samples (low entropy) : {(difficulty > 0.66).sum():,}"
)
