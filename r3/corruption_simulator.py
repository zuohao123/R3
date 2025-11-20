"""
Uncertainty-aware corruption simulator used in the R^3 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CorruptionModuleConfig:
    hidden_size: int = 4096
    enable: bool = True
    image_dropout: float = 0.15
    text_dropout: float = 0.15
    noise_scale: float = 0.05
    block_dropout: float = 0.15   # contiguous occlusion over image tokens
    blur_strength: float = 0.5    # blend factor for blurred features when masked


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

        # 模型级腐蚀已取消；仅输出置信度用于后续模块
        return vision_feats, text_feats, img_conf, txt_conf

    # 保留占位函数以兼容旧调用，但不再做任何腐蚀
    def _apply_image_corruption(self, feats: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return feats

    def _apply_text_corruption(self, feats: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return feats
