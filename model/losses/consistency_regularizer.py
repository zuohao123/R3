"""
Consistency regularizer aligning corrupted and full-modality reasoning.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class ConsistencyRegularizer:
    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    def __call__(self, preds_full: torch.Tensor, preds_corrupted: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(preds_full, preds_corrupted)
        return loss * self.weight
