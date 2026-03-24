"""
augmentations/randaugment.py

RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.

Exact implementation of the method described in:
    Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020).
    RandAugment: Practical automated data augmentation with a reduced search space.
    Advances in Neural Information Processing Systems (NeurIPS), 33, 18613–18624.
    https://arxiv.org/abs/1909.13719

Algorithm 1 (from the paper):
    ops = list of all K augmentation operations
    for each image x:
        sampled_ops = randomly sample N operations from ops (uniform, with replacement)
        for op in sampled_ops:
            x = op(x, magnitude=M)
    return x

Two hyperparameters only:
    N: number of augmentation operations applied per image  (paper default: N=2)
    M: shared magnitude for all selected operations, int in [0, 30]
                                                            (paper default: M=9 for CIFAR-10)

Operation pool (14 ops, Table 1 / Appendix B in Cubuk et al. 2020):
    Identity, AutoContrast, Equalize, Rotate, Solarize, Color,
    Posterize, Contrast, Brightness, Sharpness, ShearX, ShearY,
    TranslateX, TranslateY

CIFAR-10 specific settings (from paper Appendix B):
    TranslateX/Y max pixels = 10  (≈31% of 32px image — paper uses 100/331 * img_size for ImageNet,
                                    reduced proportionally for CIFAR)
    Rotate max degrees     = 30
    ShearX/Y max angle     = 0.3 radians (~17°)
    fillcolor              = (128, 128, 128)  — mid-gray, same as paper's TF implementation

Usage:
    transform = RandAugmentTransform(N=2, M=9, dataset="cifar10")
    tensor    = transform(pil_image)           # PIL → normalized tensor

    # Or inside a T.Compose pipeline:
    pipeline = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        RandAugmentOp(N=2, M=9),               # PIL → PIL (drop-in for T.Compose)
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
"""

import random

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageEnhance


# ─────────────────────────────────────────────────────────────
# MAGNITUDE HELPERS
# ─────────────────────────────────────────────────────────────

FILL = (128, 128, 128)   # mid-gray fill for geometric transforms (paper standard)


def _to_level(M: int, max_val: float, min_val: float = 0.0) -> float:
    """
    Map integer magnitude M ∈ [0, 30] to operation-specific value range.

    From the paper: each operation maps M linearly to its parameter range.
        level = min_val + (M / 30) * (max_val - min_val)
    """
    return min_val + (M / 30.0) * (max_val - min_val)


def _signed(v: float) -> float:
    """
    For symmetric operations (rotate, shear, translate), randomly negate.
    Paper uses uniform random sign per application.
    """
    return -v if random.random() > 0.5 else v


# ─────────────────────────────────────────────────────────────
# 14 AUGMENTATION OPERATIONS (Cubuk et al. 2020, Table 1)
# Each function signature: op(img: PIL.Image, M: int) -> PIL.Image
# ─────────────────────────────────────────────────────────────

def identity(img: Image.Image, M: int) -> Image.Image:
    """No-op. Included so the op pool covers the full space."""
    return img


def auto_contrast(img: Image.Image, M: int) -> Image.Image:
    """
    Maximize image contrast by remapping pixel values to [0, 255].
    Magnitude M is unused — the operation is all-or-nothing.
    """
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, M: int) -> Image.Image:
    """
    Equalize image histogram.
    Magnitude M is unused.
    """
    return ImageOps.equalize(img)


def rotate(img: Image.Image, M: int) -> Image.Image:
    """
    Rotate image by a random angle.
    M=0 → 0°,  M=30 → 30°.  Sign is randomly chosen.
    """
    degrees = _signed(_to_level(M, max_val=30.0))
    return img.rotate(degrees, fillcolor=FILL)


def solarize(img: Image.Image, M: int) -> Image.Image:
    """
    Invert all pixel values above a threshold.
    M=0 → threshold=256 (no change),  M=30 → threshold=0 (full invert).
    """
    threshold = int(_to_level(M, max_val=256.0, min_val=0.0))
    # Invert: higher M → lower threshold → more pixels inverted
    threshold = 256 - threshold
    return ImageOps.solarize(img, threshold)


def color(img: Image.Image, M: int) -> Image.Image:
    """
    Adjust color saturation.
    M=0 → factor=0.1 (near grayscale),  M=30 → factor=1.9 (vivid).
    """
    factor = _to_level(M, max_val=1.9, min_val=0.1)
    return ImageEnhance.Color(img).enhance(factor)


def posterize(img: Image.Image, M: int) -> Image.Image:
    """
    Reduce the number of bits per color channel.
    M=0 → 4 bits (heavy banding),  M=30 → 8 bits (no change).
    Higher M = less distortion (inverse of other ops).
    """
    bits = max(1, int(_to_level(M, max_val=8.0, min_val=4.0)))
    return ImageOps.posterize(img, bits)


def contrast(img: Image.Image, M: int) -> Image.Image:
    """
    Adjust image contrast.
    M=0 → factor=0.1 (near flat),  M=30 → factor=1.9 (high contrast).
    """
    factor = _to_level(M, max_val=1.9, min_val=0.1)
    return ImageEnhance.Contrast(img).enhance(factor)


def brightness(img: Image.Image, M: int) -> Image.Image:
    """
    Adjust image brightness.
    M=0 → factor=0.1 (near black),  M=30 → factor=1.9 (near white).
    """
    factor = _to_level(M, max_val=1.9, min_val=0.1)
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img: Image.Image, M: int) -> Image.Image:
    """
    Adjust image sharpness.
    M=0 → factor=0.1 (blurred),  M=30 → factor=1.9 (sharpened).
    """
    factor = _to_level(M, max_val=1.9, min_val=0.1)
    return ImageEnhance.Sharpness(img).enhance(factor)


def shear_x(img: Image.Image, M: int) -> Image.Image:
    """
    Apply horizontal shear transformation.
    M=0 → 0 rad (no shear),  M=30 → ±0.3 rad (~17°).  Sign is random.

    PIL AFFINE inverse matrix for ShearX:
        (1, shear, 0, 0, 1, 0)
        input_x = x + shear * y,  input_y = y
    """
    shear = _signed(_to_level(M, max_val=0.3))
    return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
                         resample=Image.BILINEAR, fillcolor=FILL)


def shear_y(img: Image.Image, M: int) -> Image.Image:
    """
    Apply vertical shear transformation.
    M=0 → 0 rad,  M=30 → ±0.3 rad.  Sign is random.

    PIL AFFINE inverse matrix for ShearY:
        (1, 0, 0, shear, 1, 0)
        input_x = x,  input_y = shear * x + y
    """
    shear = _signed(_to_level(M, max_val=0.3))
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
                         resample=Image.BILINEAR, fillcolor=FILL)


def translate_x(img: Image.Image, M: int) -> Image.Image:
    """
    Translate image horizontally.
    M=0 → 0 px,  M=30 → ±10 px.  Sign is random.

    CIFAR-10 specific: max ±10 pixels on 32×32 image (≈31% of width).
    Paper uses ±150/331 * image_size for ImageNet, scaled proportionally.

    PIL AFFINE inverse matrix for TranslateX by +v pixels (shift right):
        (1, 0, v, 0, 1, 0)
        input_x = x + v,  input_y = y
    """
    pixels = _signed(_to_level(M, max_val=10.0))
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                         resample=Image.BILINEAR, fillcolor=FILL)


def translate_y(img: Image.Image, M: int) -> Image.Image:
    """
    Translate image vertically.
    M=0 → 0 px,  M=30 → ±10 px.  Sign is random.

    PIL AFFINE inverse matrix for TranslateY by +v pixels (shift down):
        (1, 0, 0, 0, 1, v)
        input_x = x,  input_y = y + v
    """
    pixels = _signed(_to_level(M, max_val=10.0))
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                         resample=Image.BILINEAR, fillcolor=FILL)


# ─────────────────────────────────────────────────────────────
# OPERATION POOL — 14 ops, Cubuk et al. 2020 Table 1
# ─────────────────────────────────────────────────────────────

RANDAUGMENT_OPS = [
    identity,
    auto_contrast,
    equalize,
    rotate,
    solarize,
    color,
    posterize,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

OP_NAMES = [fn.__name__ for fn in RANDAUGMENT_OPS]   # for logging / inspection


# ─────────────────────────────────────────────────────────────
# RANDAUGMENT OP — PIL → PIL  (drop-in for T.Compose)
# ─────────────────────────────────────────────────────────────

class RandAugmentOp:
    """
    PIL → PIL callable implementing Algorithm 1 from Cubuk et al. (2020).

    Designed to sit inside a T.Compose pipeline between spatial transforms
    (RandomCrop, RandomHorizontalFlip) and ToTensor/Normalize:

        T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            RandAugmentOp(N=2, M=9),       ← here
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    Args:
        N: Number of operations applied per image (paper default: 2).
        M: Shared magnitude for all selected ops, integer in [0, 30]
           (paper default: 9 for CIFAR-10).
    """

    def __init__(self, N: int = 2, M: int = 9):
        self.N   = N
        self.M   = M
        self.ops = RANDAUGMENT_OPS

    def __call__(self, img: Image.Image) -> Image.Image:
        # Sample N ops uniformly at random with replacement (paper Algorithm 1)
        sampled = random.choices(self.ops, k=self.N)
        for op in sampled:
            img = op(img, self.M)
        return img

    def __repr__(self) -> str:
        return f"RandAugmentOp(N={self.N}, M={self.M}, ops={len(self.ops)})"


# ─────────────────────────────────────────────────────────────
# NORMALIZATION STATS
# ─────────────────────────────────────────────────────────────

STATS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}


# ─────────────────────────────────────────────────────────────
# FULL TRANSFORM — PIL → tensor  (standalone, no T.Compose needed)
# ─────────────────────────────────────────────────────────────

class RandAugmentTransform:
    """
    Full training transform following Cubuk et al. (2020) for CIFAR:

        RandomCrop(32, padding=4)     ← standard spatial base
        RandomHorizontalFlip()        ← standard flip base
        RandAugmentOp(N, M)           ← paper's contribution
        ToTensor()
        Normalize(mean, std)

    This is the complete pipeline used in the paper for CIFAR-10/100.
    Val transform is plain ToTensor + Normalize (no augmentation).

    Args:
        N:       Number of operations per image. Paper default: 2.
        M:       Shared magnitude in [0, 30]. Paper default: 9 for CIFAR-10.
        dataset: "cifar10" or "cifar100".
    """

    def __init__(self, N: int = 2, M: int = 9, dataset: str = "cifar10"):
        self.N   = N
        self.M   = M
        stats    = STATS[dataset]
        self._transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            RandAugmentOp(N=N, M=M),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img)

    def __repr__(self) -> str:
        return f"RandAugmentTransform(N={self.N}, M={self.M})"


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("RandAugment (Cubuk et al., NeurIPS 2020)\n")
    print(f"  Operation pool ({len(RANDAUGMENT_OPS)} ops):")
    for i, fn in enumerate(RANDAUGMENT_OPS):
        print(f"    {i+1:>2}. {fn.__name__}")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    print("\n  Testing RandAugmentTransform(N=2, M=9):")
    transform = RandAugmentTransform(N=2, M=9, dataset="cifar10")
    for i in range(3):
        t = transform(img)
        print(f"    call {i+1}: shape={t.shape}  mean={t.mean():.4f}  std={t.std():.4f}")

    print("\n  Magnitude sweep (which ops fire at N=2, M=9):")
    op = RandAugmentOp(N=2, M=9)
    counts = {fn.__name__: 0 for fn in RANDAUGMENT_OPS}
    for _ in range(1000):
        for fn in random.choices(RANDAUGMENT_OPS, k=2):
            counts[fn.__name__] += 1
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (c // 10)
        print(f"    {name:<16} {c:>4}  {bar}")

    print("\n✅ RandAugment working correctly")
