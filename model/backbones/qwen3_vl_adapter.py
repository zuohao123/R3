"""
Wrapper around Qwen3-VL that exposes adapter hooks for tri-path enhancement.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


class Qwen3VLAdapter(torch.nn.Module):
    def __init__(self, backbone, enhancer, gate_controller) -> None:
        super().__init__()
        self.backbone = backbone
        self.enhancer = enhancer
        self.gate_controller = gate_controller

    def forward(
        self,
        question_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        pseudo_text: List[List[str]],
        retrieval: List[List[Dict]],
        corruption_report: List[Dict],
        question: Optional[List[str]] = None,
        image_path: Optional[List[str]] = None,
        answer: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        gate_values = self.gate_controller(corruption_report, retrieval)
        enhanced_text, enhanced_memory, imputation = self.enhancer(
            question_tokens, pseudo_text, retrieval, gate_values
        )
        outputs = self.backbone(
            question_tokens=enhanced_text,
            vision_tokens=vision_tokens,
            memory_cache=enhanced_memory,
            imputation_tokens=imputation,
            answer=answer,
            pseudo_text=pseudo_text,
            retrieval=retrieval,
            corruption_report=corruption_report,
            question=question,
            image_path=image_path,
            **kwargs,
        )
        outputs["gate_values"] = gate_values
        return outputs
