"""augmentations/clip_scorer.py — CLIP-based augmentation difficulty scorer.

Scores how semantically different an augmented image is from its original
using a frozen CLIP ViT-B/32 image encoder.  Higher score = harder augmentation.
"""

import torch
import torch.nn.functional as F


class CLIPDifficultyScorer:
    """Returns per-image difficulty as 1 - cosine_similarity of CLIP embeddings.

    Both original and augmented images are passed as (B, 3, H, W) float tensors
    in [0, 1].  They are upsampled to 224×224, CLIP-normalised, encoded with a
    frozen ViT-B/32, and the per-image cosine distance is returned.

    Score = 0  → augmented image is semantically identical to original.
    Score → 2  → augmented image is semantically opposite (rare in practice).
    Typical range for CIFAR-100 augmentations: [0.0, 0.5].

    CLIP is kept fully frozen — no gradients are computed through it.

    Args:
        model_name : open_clip model identifier (default "ViT-B-32")
        pretrained : open_clip pretrained weights tag (default "openai")
        device     : torch.device; auto-detected if None
    """

    # CLIP ViT-B/32 normalisation constants
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: torch.device | None = None,
    ):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip is required: pip install open-clip-torch"
            ) from e

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model — only the visual encoder is used
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = model.to(device).eval()

        # Freeze every parameter
        for p in self._model.parameters():
            p.requires_grad_(False)

        # Pre-build normalisation tensors (broadcastable to (B, 3, H, W))
        self._mean = torch.tensor(self._MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self._STD, device=device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Upsample to 224×224 and apply CLIP normalisation.

        Args:
            images: (B, 3, H, W) float tensor in [0, 1]
        Returns:
            (B, 3, 224, 224) normalised tensor ready for CLIP
        """
        if images.shape[-2] != 224 or images.shape[-1] != 224:
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
        return (images - self._mean) / self._std

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised CLIP image embeddings of shape (B, D)."""
        feats = self._model.encode_image(self._preprocess(images))
        return F.normalize(feats.float(), dim=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, original: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor:
        """Compute per-image augmentation difficulty scores.

        Args:
            original : (B, 3, H, W) float tensor in [0, 1] — unaugmented images
            augmented: (B, 3, H, W) float tensor in [0, 1] — augmented images

        Returns:
            scores: (B,) float tensor — 1 - cosine_similarity per image.
                    Higher score = augmentation changed semantics more.
        """
        orig_feats = self._encode(original.to(self.device))
        aug_feats = self._encode(augmented.to(self.device))

        # Dot product of unit vectors = cosine similarity
        cos_sim = (orig_feats * aug_feats).sum(dim=-1)  # (B,)
        return 1.0 - cos_sim

    @torch.no_grad()
    def score_batch_mean(
        self, original: torch.Tensor, augmented: torch.Tensor
    ) -> float:
        """Convenience wrapper — returns mean difficulty score over the batch."""
        return float(self(original, augmented).mean())
