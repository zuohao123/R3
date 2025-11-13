"""
Light-weight feature builder that maps raw PMC samples into tensor inputs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from PIL import Image
    from PIL import ImageDraw, ImageFilter
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


class SimpleFeatureBuilder:
    """
    Generates deterministic question/vision token tensors by hashing textual inputs.
    This keeps the training loop functional without requiring heavy external
    tokenizers or image encoders. Users can swap in a custom builder that loads
    real model features as long as it exposes the same `build` API.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        question_length: int = 32,
        vision_tokens: int = 16,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.hidden_size = hidden_size
        self.question_length = question_length
        self.vision_tokens = vision_tokens
        self.dtype = dtype

    def build(self, sample: Dict, corruption_report: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        question_tokens = self._encode_question(sample.get("question", ""))
        image_key = sample.get("image_path") or sample.get("id", "unknown")
        vision_tokens = self._encode_vision(image_key, corruption_report)
        return {
            "question_tokens": question_tokens,
            "vision_tokens": vision_tokens,
        }

    def _encode_question(self, text: str) -> torch.Tensor:
        tokens = text.lower().split()
        features = torch.zeros(self.question_length, self.hidden_size, dtype=self.dtype)
        for idx, token in enumerate(tokens[: self.question_length]):
            features[idx] = self._hash_to_vector(token)
        return features

    def _encode_vision(self, key: str, corruption_report: Optional[Dict]) -> torch.Tensor:
        tokens = self._load_image_embedding(key, corruption_report)
        if tokens is None:
            tokens = self._generate_hashed_tokens(str(key))

        if corruption_report:
            severity = corruption_report.get("overall_uncertainty", 0.0)
            if severity > 0:
                noise_gen = torch.Generator()
                noise_gen.manual_seed(self._hash_to_seed(f"{key}_noise"))
                noise = torch.randn(
                    tokens.shape,
                    generator=noise_gen,
                    dtype=tokens.dtype,
                    device=tokens.device,
                ) * severity
                tokens = tokens + noise
        return tokens

    def _load_image_embedding(self, key: str, corruption_report: Optional[Dict]) -> Optional[torch.Tensor]:
        if not key or Image is None or np is None:
            return None
        key_str = str(key)
        if key_str.startswith(("http://", "https://")):
            return None
        path = Path(key_str)
        if not path.exists():
            return None
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return None
        image = self._apply_vision_corruption(
            image,
            corruption_report,
            key_str,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.size == 0:
            return None
        flat = array.reshape(-1)
        needed = self.vision_tokens * self.hidden_size
        if flat.size < needed:
            reps = (needed + flat.size - 1) // flat.size
            flat = np.tile(flat, reps)
        flat = flat[:needed]
        tensor = torch.from_numpy(flat.copy()).to(self.dtype)
        return tensor.view(self.vision_tokens, self.hidden_size)

    def _generate_hashed_tokens(self, key: str) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self._hash_to_seed(key))
        return torch.randn(
            self.vision_tokens,
            self.hidden_size,
            generator=generator,
            dtype=self.dtype,
        )

    def _hash_to_vector(self, token: str) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self._hash_to_seed(token))
        return torch.randn(self.hidden_size, generator=generator, dtype=self.dtype)

    @staticmethod
    def _hash_to_seed(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def _apply_vision_corruption(
        self,
        image: "Image.Image",
        corruption_report: Optional[Dict],
        key: str,
    ) -> "Image.Image":
        if not corruption_report:
            return image
        vision_meta = corruption_report.get("modalities", {}).get("vision")
        if not vision_meta:
            return image
        img = image.copy()
        seed = self._hash_to_seed(f"{key}_vision")
        rng = torch.Generator()
        rng.manual_seed(seed)

        blur_severity = vision_meta.get("severity", 0.0) if vision_meta.get("type") == "blur" else vision_meta.get("blur")
        if blur_severity:
            radius = max(0.5, float(blur_severity) * 5.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        occlusion = vision_meta.get("occlusion")
        if occlusion:
            img = self._apply_occlusion(img, float(occlusion), seed)

        crop = vision_meta.get("crop")
        if crop:
            img = self._apply_crop(img, float(crop), seed)
        return img

    def _apply_occlusion(self, image: "Image.Image", severity: float, seed: int) -> "Image.Image":
        severity = max(0.05, min(severity, 0.95))
        width, height = image.size
        occ_w = max(1, int(width * severity))
        occ_h = max(1, int(height * severity * 0.6))

        gen = torch.Generator()
        gen.manual_seed(seed)
        pos_x = int(torch.rand((), generator=gen).item() * max(1, width - occ_w))
        pos_y = int(torch.rand((), generator=gen).item() * max(1, height - occ_h))

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [pos_x, pos_y, pos_x + occ_w, pos_y + occ_h],
            fill=(0, 0, 0),
        )
        return overlay

    def _apply_crop(self, image: "Image.Image", severity: float, seed: int) -> "Image.Image":
        severity = max(0.05, min(severity, 0.9))
        width, height = image.size
        crop_w = int(width * (1.0 - severity * 0.5))
        crop_h = int(height * (1.0 - severity * 0.5))

        gen = torch.Generator()
        gen.manual_seed(seed + 1)
        start_x = int(torch.rand((), generator=gen).item() * max(1, width - crop_w))
        start_y = int(torch.rand((), generator=gen).item() * max(1, height - crop_h))

        cropped = image.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        return cropped.resize((width, height), Image.BILINEAR)
