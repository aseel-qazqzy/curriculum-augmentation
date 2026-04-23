"""augmentations/primitives.py — individual augmentation operations."""

import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random

def random_flip(img: Image.Image, strength: float = 0.1) -> Image.Image:
    p = 0.5 * strength
    return TF.hflip(img) if random.random() < p else img

def random_crop(img: Image.Image, strength: float = 0.1) -> Image.Image:
    size = img.size[0]
    padding = max(1, int(4 * strength))
    return T.RandomCrop(size, padding=padding)(img)

def color_jitter(img: Image.Image, strength: float = 0.5) -> Image.Image:
    b = 0.4 * strength
    c = 0.4 * strength
    s = 0.4 * strength
    h = 0.1 * strength
    return T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)(img)

def random_rotation(img: Image.Image, strength: float = 0.5) -> Image.Image:
    max_angle = 15 * strength
    angle     = random.uniform(-max_angle, max_angle)
    return TF.rotate(img, angle, fill=0)

def random_shear(img: Image.Image, strength: float = 0.5) -> Image.Image:
    max_shear = 10 * strength
    return T.RandomAffine(degrees=0, shear=max_shear)(img)

def cutout(img: Image.Image, strength: float = 1.0) -> Image.Image:
    img_tensor = TF.to_tensor(img)
    _, h, w    = img_tensor.shape
    cutout_size = int(w * 0.5 * strength)
    if cutout_size < 1:
        return img
    x = random.randint(0, w - cutout_size)
    y = random.randint(0, h - cutout_size)
    img_tensor[:, y:y+cutout_size, x:x+cutout_size] = 0.0
    return TF.to_pil_image(img_tensor)

def random_grayscale(img: Image.Image, strength: float = 1.0) -> Image.Image:
    p = 0.2 * strength
    return T.RandomGrayscale(p=p)(img)

def solarize(img: Image.Image, strength: float = 1.0) -> Image.Image:
    threshold = int(256 - (128 * strength))
    return ImageOps.solarize(img, threshold)

def posterize(img: Image.Image, strength: float = 1.0) -> Image.Image:
    bits = max(1, int(8 - (4 * strength)))
    return ImageOps.posterize(img, bits)

def gaussian_blur(img: Image.Image, strength: float = 0.5) -> Image.Image:
    kernel_size = max(3, int(3 + 4 * strength))
    if kernel_size % 2 == 0:   # kernel must be odd
        kernel_size += 1
    sigma = 0.1 + 1.9 * strength
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)


# ── New ops (match RandAugment paper op pool) ─────────────────────────────────

def translate_x(img: Image.Image, strength: float = 0.5) -> Image.Image:
    max_px    = int(img.size[0] * 0.33 * strength)
    pixels    = random.randint(-max_px, max_px) if max_px > 0 else 0
    return TF.affine(img, angle=0, translate=(pixels, 0), scale=1.0, shear=0, fill=128)

def translate_y(img: Image.Image, strength: float = 0.5) -> Image.Image:
    max_px    = int(img.size[1] * 0.33 * strength)
    pixels    = random.randint(-max_px, max_px) if max_px > 0 else 0
    return TF.affine(img, angle=0, translate=(0, pixels), scale=1.0, shear=0, fill=128)

def equalize(img: Image.Image, strength: float = 1.0) -> Image.Image:
    return ImageOps.equalize(img) if random.random() < 0.5 * strength else img

def auto_contrast(img: Image.Image, strength: float = 1.0) -> Image.Image:
    return ImageOps.autocontrast(img) if random.random() < 0.5 * strength else img

def sharpness(img: Image.Image, strength: float = 0.5) -> Image.Image:
    factor = 1.0 + random.uniform(-0.8 * strength, 0.8 * strength)
    return ImageEnhance.Sharpness(img).enhance(max(0.1, factor))

def enhance_contrast(img: Image.Image, strength: float = 0.5) -> Image.Image:
    factor = 1.0 + random.uniform(-0.8 * strength, 0.8 * strength)
    return ImageEnhance.Contrast(img).enhance(max(0.1, factor))

def enhance_brightness(img: Image.Image, strength: float = 0.5) -> Image.Image:
    factor = 1.0 + random.uniform(-0.8 * strength, 0.8 * strength)
    return ImageEnhance.Brightness(img).enhance(max(0.1, factor))


AUGMENTATION_REGISTRY = {
    # Tier 1 — geometry only, semantics fully preserved
    "flip":          (random_flip,        1,   0.0),
    "crop":          (random_crop,        1,   0.0),
    "translate_x":   (translate_x,        1,   0.0),
    "translate_y":   (translate_y,        1,   0.0),
    # Tier 2 — colour / texture distortion, structure intact
    "auto_contrast": (auto_contrast,      2,   0.20),
    "equalize":      (equalize,           2,   0.22),
    "sharpness":     (sharpness,          2,   0.30),
    "color_jitter":  (color_jitter,       2,   0.35),
    "rotation":      (random_rotation,    2,   0.40),
    "shear":         (random_shear,       2,   0.45),
    # Tier 3 — information removal / aggressive distortion
    "grayscale":     (random_grayscale,   3,   0.55),
    "contrast":      (enhance_contrast,   3,   0.60),
    "brightness":    (enhance_brightness, 3,   0.65),
    "blur":          (gaussian_blur,      3,   0.68),
    "cutout":        (cutout,             3,   0.72),
    "solarize":      (solarize,           3,   0.82),
    "posterize":     (posterize,          3,   0.88),
}


def get_active_augmentations(difficulty: float) -> list:
    active = []
    for name, (fn, level, threshold) in AUGMENTATION_REGISTRY.items():
        if difficulty >= threshold:
            strength = min(1.0, (difficulty - threshold) / max(0.01, 1.0 - threshold))
            active.append((name, fn, strength))
    return active


def apply_augmentations(img: Image.Image, difficulty: float) -> Image.Image:
    active = get_active_augmentations(difficulty)
    for name, fn, strength in active:
        img = fn(img, strength)
    return img


if __name__ == "__main__":
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    print("Testing augmentation primitives...")
    for diff in [0.0, 0.25, 0.5, 0.75, 1.0]:
        active = get_active_augmentations(diff)
        names  = [n for n, _, _ in active]
        print(f"  difficulty={diff:.2f} → active: {names}")

    print("\nDone.")
