"""
Reusable corruption utilities that operate directly on raw inputs before
feeding anything into the base model. This enables consistent modality-drop
simulation across different backbones / evaluation scripts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFilter


@dataclass
class ImageCorruptionConfig:
    occlusion_prob: float = 0.5
    occlusion_ratio: float = 0.25  # max area ratio to occlude
    blur_prob: float = 0.5
    blur_radius: float = 3.0


class ImageCorruptor:
    """
    Applies spatial occlusion + blur on raw PIL images, mimicking missing
    visual modality prior to encoding.
    """

    def __init__(self, config: ImageCorruptionConfig | None = None) -> None:
        self.config = config or ImageCorruptionConfig()

    def __call__(self, image: Image.Image) -> Image.Image:
        corrupted = image.copy()
        if random.random() < self.config.occlusion_prob:
            corrupted = self._apply_occlusion(corrupted)
        if random.random() < self.config.blur_prob:
            corrupted = corrupted.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
        return corrupted

    def _apply_occlusion(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        max_w = int(width * self.config.occlusion_ratio)
        max_h = int(height * self.config.occlusion_ratio)
        occ_w = random.randint(max(1, max_w // 2), max_w)
        occ_h = random.randint(max(1, max_h // 2), max_h)
        x0 = random.randint(0, max(0, width - occ_w))
        y0 = random.randint(0, max(0, height - occ_h))
        draw.rectangle([x0, y0, x0 + occ_w, y0 + occ_h], fill=(0, 0, 0))
        return image


@dataclass
class PseudoTextCorruptionConfig:
    drop_prob: float = 0.3


class PseudoTextCorruptor:
    """
    Drops a fraction of pseudo-text entries before retrieval.
    """

    def __init__(self, config: PseudoTextCorruptionConfig | None = None) -> None:
        self.config = config or PseudoTextCorruptionConfig()

    def __call__(self, pseudo_text: Iterable[str]) -> List[str]:
        entries = [p for p in pseudo_text if p]
        if not entries:
            return entries
        kept = [p for p in entries if random.random() > self.config.drop_prob]
        if not kept:
            kept = [entries[0]]
        return kept
