"""
Selective reconstruction and reasoning modules for R^3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class ReconstructionModuleConfig:
    hidden_size: int = 4096
    prefix_length: int = 32
    memory_tokens: int = 32
    imputation_tokens: int = 16
    enable_prefix: bool = True
    enable_memory: bool = True
    enable_imputation: bool = True


class PrefixEncoder(nn.Module):
    def __init__(self, config: ReconstructionModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

    def forward(self, evidence_embeddings: torch.Tensor) -> torch.Tensor:
        if evidence_embeddings.numel() == 0:
            return evidence_embeddings.new_zeros(
                evidence_embeddings.size(0),
                0,
                self.config.hidden_size,
            )
        pooled = evidence_embeddings.mean(dim=2)
        output, _ = self.gru(pooled)
        return output[:, : self.config.prefix_length, :]


class EvidenceMemory(nn.Module):
    def __init__(self, config: ReconstructionModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size,
            num_heads=8,
            batch_first=True,
        )

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if memory.numel() == 0:
            return hidden
        attn_out, _ = self.cross_attn(hidden, memory, memory)
        return hidden + gate.view(-1, 1, 1) * attn_out


class LatentImputationTokens(nn.Module):
    def __init__(self, config: ReconstructionModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.token_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_conf: torch.Tensor,
        evidence_summary: torch.Tensor,
    ) -> torch.Tensor:
        low_conf_mask = (1.0 - text_conf).unsqueeze(-1)
        weighted = (text_embeddings * low_conf_mask).sum(dim=1)
        fused = torch.cat([weighted, evidence_summary], dim=-1)
        tokens = self.token_proj(fused).unsqueeze(1)
        tokens = tokens.repeat(1, self.config.imputation_tokens, 1)
        return tokens


class AdaptiveGatingController(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, img_conf: torch.Tensor, txt_conf: torch.Tensor, evidence_scores: torch.Tensor) -> torch.Tensor:
        summary = torch.stack(
            [
                1.0 - img_conf.mean(dim=1),
                1.0 - txt_conf.mean(dim=1),
                evidence_scores.mean(dim=1) if evidence_scores.numel() else torch.zeros_like(img_conf.mean(dim=1)),
                evidence_scores.max(dim=1).values if evidence_scores.numel() else torch.zeros_like(img_conf.mean(dim=1)),
            ],
            dim=1,
        )
        gates = torch.sigmoid(self.proj(summary))
        return gates  # [batch, 2]


class SelectiveReconstruction(nn.Module):
    """
    Combines textual prefix, evidence memory, and imputation tokens.
    """

    def __init__(self, config: ReconstructionModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.prefix = PrefixEncoder(config)
        self.memory = EvidenceMemory(config)
        self.imputation = LatentImputationTokens(config)
        self.gating = AdaptiveGatingController(config.hidden_size)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_attention: torch.Tensor,
        vision_embeddings: torch.Tensor,
        retrieval: Dict,
        img_conf: torch.Tensor,
        txt_conf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        evidence_embeddings = retrieval.get("embeddings")
        evidence_scores = retrieval.get("scores", torch.zeros(text_embeddings.size(0), 1, device=text_embeddings.device))
        gates = self.gating(img_conf, txt_conf, evidence_scores)

        prefix_tokens = self._build_prefix(evidence_embeddings) if self.config.enable_prefix else text_embeddings.new_zeros(text_embeddings.size(0), 0, text_embeddings.size(-1))
        imputation_tokens = self._build_imputation(text_embeddings, txt_conf, evidence_embeddings, gates) if self.config.enable_imputation else text_embeddings.new_zeros(text_embeddings.size(0), 0, text_embeddings.size(-1))
        augmented_inputs = torch.cat([prefix_tokens, text_embeddings, imputation_tokens], dim=1)
        augmented_attention = torch.cat(
            [
                torch.ones(text_embeddings.size(0), prefix_tokens.size(1), device=text_embeddings.device, dtype=text_attention.dtype),
                text_attention,
                torch.ones(text_embeddings.size(0), imputation_tokens.size(1), device=text_embeddings.device, dtype=text_attention.dtype),
            ],
            dim=1,
        )

        if self.config.enable_memory and evidence_embeddings.numel():
            augmented_inputs = self.memory(augmented_inputs, evidence_embeddings.squeeze(2), gates[:, 0])

        return {
            "inputs_embeds": augmented_inputs,
            "attention_mask": augmented_attention,
            "gates": gates,
        }

    def _build_prefix(self, evidence_embeddings: torch.Tensor) -> torch.Tensor:
        if evidence_embeddings.numel() == 0:
            return evidence_embeddings.new_zeros(evidence_embeddings.size(0), 0, evidence_embeddings.size(-1))
        return self.prefix(evidence_embeddings.squeeze(2))

    def _build_imputation(
        self,
        text_embeddings: torch.Tensor,
        txt_conf: torch.Tensor,
        evidence_embeddings: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_embeddings.numel() == 0:
            evidence_summary = torch.zeros(text_embeddings.size(0), text_embeddings.size(-1), device=text_embeddings.device)
        else:
            evidence_summary = evidence_embeddings.squeeze(2).mean(dim=1)
        tokens = self.imputation(text_embeddings, txt_conf, evidence_summary)
        return gates[:, 1].view(-1, 1, 1) * tokens
