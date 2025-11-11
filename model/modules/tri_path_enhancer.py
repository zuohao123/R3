"""
Tri-path enhancement module for selective reconstruction.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class TriPathEnhancer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prefix_length: int = 32,
        memory_tokens: int = 16,
        imputation_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_encoder = torch.nn.Linear(hidden_size, hidden_size)
        self.memory_adapter = torch.nn.Linear(hidden_size, hidden_size)
        self.imputation_embeddings = torch.nn.Parameter(
            torch.randn(imputation_tokens, hidden_size)
        )
        self.prefix_length = prefix_length
        self.memory_tokens = memory_tokens

    def forward(
        self,
        question_tokens: torch.Tensor,
        pseudo_text: List[str],
        retrieval: List[Dict],
        gates: Dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_prefix = self._build_prefix(question_tokens, pseudo_text, gates["text"])
        memory_bank = self._build_memory(question_tokens, retrieval, gates["memory"])
        imputation = self._build_imputation(gates["imputation"])
        enhanced_question = torch.cat([text_prefix, question_tokens], dim=1)
        return enhanced_question, memory_bank, imputation

    def _build_prefix(
        self,
        question_tokens: torch.Tensor,
        pseudo_text: List[str],
        gate_value: float,
    ) -> torch.Tensor:
        pooled = question_tokens[:, : self.prefix_length, :]
        encoded = self.prefix_encoder(pooled) * gate_value
        return encoded

    def _build_memory(
        self,
        question_tokens: torch.Tensor,
        retrieval: List[Dict],
        gate_value: float,
    ) -> torch.Tensor:
        pooled = question_tokens[:, : self.memory_tokens, :]
        adapted = self.memory_adapter(pooled) * gate_value
        return adapted

    def _build_imputation(self, gate_value: float) -> torch.Tensor:
        return self.imputation_embeddings.unsqueeze(0) * gate_value
