"""
Light-weight feature builder that maps raw PMC samples into tensor inputs.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

import torch


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
        generator = torch.Generator()
        generator.manual_seed(self._hash_to_seed(key))
        tokens = torch.randn(
            self.vision_tokens,
            self.hidden_size,
            generator=generator,
            dtype=self.dtype,
        )
        if corruption_report:
            severity = corruption_report.get("overall_uncertainty", 0.0)
            if severity > 0:
                noise_gen = torch.Generator()
                noise_gen.manual_seed(self._hash_to_seed(f"{key}_noise"))
                noise = torch.randn_like(tokens, generator=noise_gen) * severity
                tokens = tokens + noise
        return tokens

    def _hash_to_vector(self, token: str) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self._hash_to_seed(token))
        return torch.randn(self.hidden_size, generator=generator, dtype=self.dtype)

    @staticmethod
    def _hash_to_seed(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)
