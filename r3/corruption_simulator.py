"""
Uncertainty-aware corruption simulator used in the R^3 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class CorruptionModuleConfig:
    hidden_size: int = 4096
    enable: bool = True
    image_dropout: float = 0.15
    text_dropout: float = 0.15
    noise_scale: float = 0.05


class UncertaintyAwareCorruptionSimulator(nn.Module):
    """
    Estimates token-wise confidence and optionally injects structured degradations.
    """

    def __init__(self, config: CorruptionModuleConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.img_conf_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.txt_conf_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        vision_feats: torch.Tensor,
        text_feats: torch.Tensor,
        apply_corruption: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_feats: (batch, num_img_tokens, dim)
            text_feats: (batch, num_txt_tokens, dim)
        """
        img_conf = torch.sigmoid(self.img_conf_head(vision_feats)).squeeze(-1)
        txt_conf = torch.sigmoid(self.txt_conf_head(text_feats)).squeeze(-1)

        if not (self.training and apply_corruption and self.config.enable):
            return vision_feats, text_feats, img_conf, txt_conf

        vision_feats = self._apply_image_corruption(vision_feats, img_conf)
        text_feats = self._apply_text_corruption(text_feats, txt_conf)
        return vision_feats, text_feats, img_conf, txt_conf

    def _apply_image_corruption(self, feats: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        keep_prob = conf.unsqueeze(-1)
        dropout_mask = torch.bernoulli(
            keep_prob.clamp(min=1.0 - self.config.image_dropout)
        )
        noise = torch.randn_like(feats) * self.config.noise_scale * (1 - keep_prob)
        return feats * dropout_mask + noise

    def _apply_text_corruption(self, feats: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        keep_prob = conf.unsqueeze(-1)
        mask = torch.bernoulli(
            keep_prob.clamp(min=1.0 - self.config.text_dropout)
        )
        noise = torch.randn_like(feats) * self.config.noise_scale * (1 - keep_prob)
        return feats * mask + noise
