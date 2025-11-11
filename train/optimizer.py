"""
Optimizer helpers for adapter-based fine-tuning.
"""
from __future__ import annotations

from typing import Iterable

import torch


def build_optimizer(model: torch.nn.Module, lr_backbone: float, lr_adapter: float) -> torch.optim.Optimizer:
    backbone_params = []
    adapter_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            adapter_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": adapter_params, "lr": lr_adapter},
    ]
    return torch.optim.AdamW(param_groups)
