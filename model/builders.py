"""
Factory helpers to instantiate training-ready R^3 models.
"""
from __future__ import annotations

from typing import Dict

from model.backbones.qwen3_vl_adapter import Qwen3VLAdapter
from model.backbones.simple_decoder import SimpleDecoder
from model.modules.adaptive_gate import AdaptiveGateController
from model.modules.tri_path_enhancer import TriPathEnhancer


def build_r3_model(model_cfg: Dict | None = None) -> Qwen3VLAdapter:
    model_cfg = model_cfg or {}
    hidden_size = model_cfg.get("hidden_size", 256)
    enhancer = TriPathEnhancer(
        hidden_size=hidden_size,
        prefix_length=model_cfg.get("prefix_length", 32),
        memory_tokens=model_cfg.get("memory_tokens", 16),
        imputation_tokens=model_cfg.get("imputation_tokens", 8),
    )
    gate_controller = AdaptiveGateController(default_value=model_cfg.get("gate_default", 0.7))
    backbone = SimpleDecoder(hidden_size=hidden_size, answer_vocab=model_cfg.get("vocab_size", 8192))
    return Qwen3VLAdapter(backbone=backbone, enhancer=enhancer, gate_controller=gate_controller)
