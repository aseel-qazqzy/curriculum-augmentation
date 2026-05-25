"""
analysis/perclass_plots.py

Per-class accuracy visualisations:
  Fig 1 — Hardest classes: all 5 methods side-by-side (bottom-15 by Static acc)
  Fig 2 — Class-level gains: Static → ETS / LPS / EGS v2 (top-15 gains each)
  Fig 3 — Full 100-class heatmap sorted by ETS accuracy

Run:
    python analysis/perclass_plots.py

Outputs saved to results/figs/
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

DATA = {
    "No Aug": [
        89,
        91,
        64,
        56,
        58,
        76,
        81,
        68,
        87,
        84,
        52,
        49,
        72,
        59,
        57,
        85,
        73,
        88,
        60,
        67,
        89,
        88,
        69,
        88,
        82,
        57,
        65,
        71,
        83,
        72,
        71,
        80,
        64,
        61,
        80,
        54,
        82,
        74,
        67,
        87,
        68,
        84,
        78,
        83,
        57,
        63,
        54,
        63,
        95,
        87,
        56,
        66,
        69,
        95,
        86,
        45,
        93,
        81,
        88,
        62,
        88,
        71,
        87,
        70,
        51,
        51,
        74,
        58,
        96,
        82,
        79,
        78,
        43,
        61,
        53,
        84,
        90,
        70,
        60,
        81,
        67,
        78,
        90,
        67,
        69,
        86,
        74,
        86,
        78,
        87,
        77,
        81,
        60,
        59,
        94,
        70,
        57,
        82,
        51,
        73,
    ],
    "Static": [
        94,
        91,
        61,
        57,
        64,
        79,
        82,
        76,
        93,
        86,
        64,
        56,
        86,
        76,
        84,
        87,
        79,
        90,
        77,
        75,
        95,
        91,
        85,
        81,
        82,
        74,
        83,
        64,
        86,
        79,
        77,
        76,
        72,
        59,
        81,
        42,
        84,
        77,
        79,
        90,
        76,
        90,
        73,
        91,
        59,
        65,
        66,
        58,
        96,
        91,
        54,
        87,
        75,
        93,
        82,
        44,
        91,
        80,
        97,
        71,
        86,
        69,
        78,
        73,
        63,
        77,
        82,
        66,
        98,
        87,
        79,
        81,
        46,
        68,
        59,
        92,
        90,
        73,
        72,
        87,
        58,
        78,
        95,
        73,
        79,
        91,
        84,
        90,
        90,
        88,
        80,
        84,
        73,
        72,
        89,
        76,
        70,
        86,
        64,
        80,
    ],
    "ETS": [
        96,
        91,
        70,
        72,
        76,
        85,
        85,
        79,
        96,
        87,
        57,
        54,
        86,
        73,
        76,
        85,
        79,
        91,
        78,
        73,
        91,
        94,
        88,
        83,
        81,
        67,
        82,
        83,
        88,
        78,
        78,
        86,
        78,
        70,
        83,
        54,
        89,
        82,
        85,
        94,
        76,
        90,
        91,
        91,
        72,
        62,
        63,
        65,
        97,
        92,
        67,
        83,
        71,
        94,
        88,
        59,
        92,
        86,
        97,
        71,
        93,
        73,
        82,
        82,
        67,
        71,
        93,
        76,
        94,
        83,
        88,
        85,
        56,
        72,
        67,
        96,
        96,
        79,
        74,
        89,
        82,
        79,
        94,
        82,
        78,
        93,
        83,
        94,
        87,
        94,
        90,
        90,
        76,
        77,
        93,
        80,
        76,
        92,
        70,
        79,
    ],
    "LPS": [
        95,
        95,
        75,
        71,
        77,
        85,
        86,
        76,
        95,
        90,
        57,
        55,
        82,
        77,
        81,
        87,
        79,
        92,
        80,
        70,
        93,
        92,
        87,
        86,
        90,
        74,
        78,
        85,
        88,
        77,
        80,
        84,
        80,
        73,
        85,
        54,
        88,
        80,
        87,
        93,
        69,
        94,
        84,
        87,
        68,
        70,
        62,
        67,
        98,
        93,
        72,
        83,
        71,
        95,
        89,
        52,
        92,
        88,
        95,
        73,
        92,
        78,
        86,
        85,
        68,
        79,
        90,
        70,
        97,
        84,
        83,
        88,
        55,
        69,
        64,
        95,
        95,
        87,
        71,
        92,
        78,
        85,
        95,
        78,
        80,
        90,
        82,
        89,
        87,
        93,
        82,
        91,
        71,
        71,
        93,
        81,
        70,
        90,
        60,
        81,
    ],
    "EGS v2": [
        92,
        92,
        70,
        66,
        63,
        82,
        81,
        82,
        98,
        93,
        59,
        63,
        88,
        81,
        79,
        81,
        76,
        83,
        77,
        76,
        90,
        91,
        82,
        80,
        84,
        78,
        84,
        66,
        90,
        82,
        77,
        82,
        77,
        72,
        89,
        48,
        84,
        78,
        82,
        92,
        74,
        92,
        76,
        89,
        73,
        67,
        63,
        66,
        96,
        92,
        70,
        90,
        66,
        93,
        81,
        70,
        91,
        84,
        95,
        76,
        84,
        78,
        80,
        72,
        67,
        74,
        85,
        67,
        98,
        81,
        89,
        81,
        54,
        64,
        65,
        92,
        89,
        72,
        70,
        89,
        82,
        80,
        95,
        75,
        75,
        89,
        83,
        84,
        89,
        92,
        86,
        88,
        72,
        76,
        95,
        79,
        67,
        86,
        64,
        81,
    ],
}

METHODS = list(DATA.keys())
ACCS = {m: np.array(DATA[m], dtype=float) for m in METHODS}

COLORS = {
    "No Aug": "#999999",
    "Static": "#D55E00",
    "ETS": "#0072B2",
    "LPS": "#009E73",
    "EGS v2": "#E69F00",
}

SAVE_DIR = Path("results/figs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── Fig 1: Hardest classes ────────────────────────────────────────────────────


def fig_hardest(n=15):
    static_vals = ACCS["Static"]
    order = sorted(range(100), key=lambda i: static_vals[i])[:n]
    labels = [CLASSES[i] for i in order]

    x = np.arange(len(labels))
    width = 0.15
    offsets = (
        np.linspace(-(len(METHODS) - 1) / 2, (len(METHODS) - 1) / 2, len(METHODS))
        * width
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for m, off in zip(METHODS, offsets):
        vals = [ACCS[m][i] for i in order]
        ax.bar(
            x + off,
            vals,
            width,
            label=m,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.set_title(
        f"Bottom-{n} Hardest Classes (ranked by Static Mixing accuracy)\n"
        "WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    for suffix, dpi in [("", 150), ("_hd", 300)]:
        p = SAVE_DIR / f"perclass_hardest{suffix}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SAVE_DIR}/perclass_hardest.png")


# ── Fig 2: Gain chart ─────────────────────────────────────────────────────────


def fig_gains(n=12):
    targets = [("ETS", "#0072B2"), ("LPS", "#009E73"), ("EGS v2", "#E69F00")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(
        "Per-Class Accuracy Gain vs Static Mixing (top gains only)\n"
        "WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (method, color) in zip(axes, targets):
        gains = ACCS[method] - ACCS["Static"]
        order = sorted(range(100), key=lambda i: gains[i], reverse=True)[:n]
        labels = [CLASSES[i] for i in order]
        vals = [gains[i] for i in order]

        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=color, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Accuracy gain over Static Mixing (pp)", fontsize=9)
        ax.set_title(f"Static → {method}", fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"+{v:.0f}pp",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    for suffix, dpi in [("", 150), ("_hd", 300)]:
        p = SAVE_DIR / f"perclass_gains{suffix}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SAVE_DIR}/perclass_gains.png")


# ── Fig 3: Full heatmap ───────────────────────────────────────────────────────


def fig_heatmap():
    order = sorted(range(100), key=lambda i: ACCS["ETS"][i])
    labels = [CLASSES[i] for i in order]
    matrix = np.array([[ACCS[m][i] for m in METHODS] for i in order])  # (100, 5)

    fig, ax = plt.subplots(figsize=(7, 20))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_yticks(range(100))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # annotate each cell
    for row in range(100):
        for col in range(len(METHODS)):
            v = matrix[row, col]
            color = "white" if v < 60 else "black"
            ax.text(
                col, row, f"{v:.0f}", ha="center", va="center", fontsize=5, color=color
            )

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.4, pad=0.01)
    ax.set_title(
        "Per-Class Accuracy Heatmap\n"
        "WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42\n"
        "(sorted by ETS accuracy, low → high)",
        fontsize=9,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    for suffix, dpi in [("", 150), ("_hd", 300)]:
        p = SAVE_DIR / f"perclass_heatmap{suffix}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SAVE_DIR}/perclass_heatmap.png")


# ── Fig 4: Regression chart ───────────────────────────────────────────────────


def fig_regressions(n=10):
    targets = [("ETS", "#0072B2"), ("LPS", "#009E73"), ("EGS v2", "#E69F00")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Per-Class Accuracy Regressions vs Static Mixing (classes that got worse)\n"
        "WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (method, color) in zip(axes, targets):
        gains = ACCS[method] - ACCS["Static"]
        order = sorted(range(100), key=lambda i: gains[i])[:n]
        labels = [CLASSES[i] for i in order]
        vals = [gains[i] for i in order]

        y = np.arange(len(labels))
        bars = ax.barh(
            y, vals, color=color, edgecolor="white", linewidth=0.5, alpha=0.75
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Accuracy change vs Static Mixing (pp)", fontsize=9)
        ax.set_title(f"Static → {method}", fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        min_v = min(vals)
        ax.set_xlim(min_v - 0.5, 0.8)

        for bar, v in zip(bars, vals):
            # place label in the middle of the bar
            ax.text(
                v / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}pp",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()
    for suffix, dpi in [("", 150), ("_hd", 300)]:
        p = SAVE_DIR / f"perclass_regressions{suffix}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {SAVE_DIR}/perclass_regressions.png")


if __name__ == "__main__":
    print("Generating per-class accuracy plots...")
    fig_hardest()
    fig_gains()
    fig_regressions()
    fig_heatmap()
    print("Done.")
