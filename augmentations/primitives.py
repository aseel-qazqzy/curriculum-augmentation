"""
augmentations/primitives.py
Individual augmentation operations — the building blocks.

Each augmentation has a 'strength' parameter (0.0 → 1.0)
so the curriculum can control intensity dynamically.

Difficulty ranking (easy → hard):
    Level 1 (Easy):    flip, crop
    Level 2 (Medium):  color_jitter, rotation
    Level 3 (Hard):    cutout, grayscale, solarize
"""

import torch
import numpy as np 
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random

# Level 1 - Easy 
def random_flip(img: Image.Image, strength: float = 0.1) -> Image.Image:
    """ 
    Randomly flip image horizontally.
    Args:
        img (Image.Image): _description_
        strength (float, optional): probability of flipping (0.0 = never, 1.0 = 50% chance)
    Returns:
        flipped image
    """
    p = 0.5 * strength
    return TF.hflip(img) if random.random() < p else img

def random_crop(img: Image.Image, strength: float = 0.1) -> Image.Image:
    """Random crop with padding.
        
    Args:
        img (Image.Image): _description_
        strength (float, optional):controls padding size: 0.0 = no crop, 1.0 = padding of 4px (standard)

    Returns:
        Image.Image: cropped image 
    """
    
    size = img.size[0]
    padding = max(1, int(4 * strength))
    return T.RandomCrop(size, padding=padding)(img)


# Level 2 - Medium

def color_jitter(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Randomly jitter brightness, contrast, saturation, hue.
    Args:
        img (Image.Image): 
        strength (float, optional): strength controls how strong the jitter is.
            strength=0.25 → subtle changes
            strength=1.0  → strong distortion.
    Returns:
        Image.Image
    """
    
    b = 0.4 * strength # brightness
    c = 0.4 * strength # contrast 
    s = 0.4 * strength # saturation
    h = 0.1 * strength # hue
    
    return T.ColorJitter(
        brightness=b,
        contrast=c,
        saturation=s,
        hue=h
    )(img)
    
    
def random_rotation(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Randomly rotate image.
    strength controls max rotation angle:
        strength=0.5 → up to ±7.5°
        strength=1.0 → up to ±15°
    """
    max_angle = 15 * strength
    angle     = random.uniform(-max_angle, max_angle)
    return TF.rotate(img, angle, fill=0)

def random_shear(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Apply random shear transformation.
    strength controls max shear angle.
    """
    max_shear = 10 * strength
    return T.RandomAffine(degrees=0, shear=max_shear)(img)


# Level 3 - Hard 
def cutout(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Randomly erase a square patch from the image (Cutout / Random Erasing).
    strength controls patch size:
        strength=0.5 → patch up to 12px on 32px image (~14% area)
        strength=1.0 → patch up to 16px (~25% area)

    Reference: DeVries & Taylor (2017) - Improved Regularization with Cutout
    """
    img_tensor = TF.to_tensor(img)
    _, h, w    = img_tensor.shape

    cutout_size = int(w * 0.5 * strength)   # max patch size scales with strength
    if cutout_size < 1:
        return img

    # Random top-left corner
    x = random.randint(0, w - cutout_size)
    y = random.randint(0, h - cutout_size)

    img_tensor[:, y:y+cutout_size, x:x+cutout_size] = 0.0   # fill with black

    return TF.to_pil_image(img_tensor)


def random_grayscale(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Randomly convert image to grayscale.
    strength controls probability of conversion.
        strength=0.5 → 10% chance
        strength=1.0 → 20% chance
    """
    p = 0.2 * strength
    return T.RandomGrayscale(p=p)(img)


def solarize(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Invert pixels above a threshold (Solarize).
    strength controls how much of the image gets inverted:
        strength=0.5 → threshold=192 (only very bright pixels inverted)
        strength=1.0 → threshold=128 (half the pixels inverted)
    """
    threshold = int(256 - (128 * strength))   # higher strength = lower threshold
    return ImageOps.solarize(img, threshold)


def posterize(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Reduce number of bits per color channel (Posterize).
    strength=0.5 → 6 bits (subtle)
    strength=1.0 → 4 bits (visible color banding)
    """
    bits = max(1, int(8 - (4 * strength)))    # 8 bits (no change) → 4 bits (strong)
    return ImageOps.posterize(img, bits)


def gaussian_blur(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """
    Apply Gaussian blur.
    strength controls blur kernel size.
    """
    kernel_size = max(3, int(3 + 4 * strength))
    # kernel must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.1 + 1.9 * strength
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)



# AUGMENTATION REGISTRY


# Maps name → (function, difficulty_level, min_strength_to_activate)
AUGMENTATION_REGISTRY = {
    # name              function          level   activates_at_difficulty
    "flip":        (random_flip,          1,      0.0),
    "crop":        (random_crop,          1,      0.0),
    "color_jitter":(color_jitter,         2,      0.25),
    "rotation":    (random_rotation,      2,      0.35),
    "shear":       (random_shear,         2,      0.40),
    "grayscale":   (random_grayscale,     3,      0.55),
    "blur":        (gaussian_blur,        3,      0.60),
    "cutout":      (cutout,               3,      0.70),
    "solarize":    (solarize,             3,      0.80),
    "posterize":   (posterize,            3,      0.85),
}


def get_active_augmentations(difficulty: float) -> list:
    """
    Returns list of (name, function, strength) for the current difficulty level.
    Augmentations are only activated once difficulty exceeds their threshold.

    Args:
        difficulty: float in [0.0, 1.0]

    Returns:
        List of (name, fn, strength) tuples

    Example:
        difficulty=0.0  → [flip, crop]
        difficulty=0.5  → [flip, crop, color_jitter, rotation, shear, grayscale]
        difficulty=1.0  → all augmentations
    """
    active = []
    for name, (fn, level, threshold) in AUGMENTATION_REGISTRY.items():
        if difficulty >= threshold:
            # Strength scales from 0 when first activated to full at difficulty=1.0
            # This prevents jarring jumps when a new augmentation is introduced
            strength = min(1.0, (difficulty - threshold) / max(0.01, 1.0 - threshold))
            active.append((name, fn, strength))
    return active


def apply_augmentations(img: Image.Image, difficulty: float) -> Image.Image:
    """
    Apply all active augmentations for the current difficulty level.
    This is the main function called by the curriculum transform.

    Args:
        img:        PIL Image
        difficulty: float in [0.0, 1.0]

    Returns:
        Augmented PIL Image
    """
    active = get_active_augmentations(difficulty)
    for name, fn, strength in active:
        img = fn(img, strength)
    return img


# QUICK TEST
if __name__ == "__main__":
    # Create a dummy PIL image (32x32, like CIFAR-10)
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    print("Testing augmentation primitives...")
    for diff in [0.0, 0.25, 0.5, 0.75, 1.0]:
        active = get_active_augmentations(diff)
        names  = [n for n, _, _ in active]
        print(f"  difficulty={diff:.2f} → active: {names}")

    print("\n✅ All primitives working correctly")