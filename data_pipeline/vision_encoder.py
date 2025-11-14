"""
Lightweight vision encoder that converts image paths into fixed-length embeddings.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


class VisionEncoder:
    """
    Wraps a CLIP vision backbone to produce deterministic vision token sequences
    that can be consumed by the reconstruction module without depending on the
    original Qwen vision tower.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cpu",
        cache_size: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def encode(self, image_path: str, vision_tokens: int, hidden_size: int) -> torch.Tensor:
        key = f"{image_path}:{vision_tokens}:{hidden_size}"
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.clone()

        image = self._load_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state.squeeze(0).to(torch.float32)
        feats = self._resize_sequence(feats, vision_tokens)
        feats = self._match_hidden_size(feats, hidden_size)
        feats = feats.cpu()
        self._maybe_cache(key, feats)
        return feats.clone()

    def _load_image(self, path: str) -> Image.Image:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(resolved).convert("RGB")

    def _resize_sequence(self, feats: torch.Tensor, target_tokens: int) -> torch.Tensor:
        seq_len = feats.shape[0]
        if seq_len >= target_tokens:
            return feats[:target_tokens]
        pad = target_tokens - seq_len
        pad_tensor = feats.new_zeros(pad, feats.shape[1])
        return torch.cat([feats, pad_tensor], dim=0)

    def _match_hidden_size(self, feats: torch.Tensor, hidden_size: int) -> torch.Tensor:
        current_dim = feats.shape[1]
        if current_dim == hidden_size:
            return feats
        if current_dim > hidden_size:
            return feats[:, :hidden_size]
        repeat = (hidden_size + current_dim - 1) // current_dim
        expanded = feats.repeat(1, repeat)
        return expanded[:, :hidden_size]

    def _maybe_cache(self, key: str, value: torch.Tensor) -> None:
        if self.cache_size == 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value
