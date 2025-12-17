"""
Selective reconstruction and reasoning modules for R^3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Avoid cuDNN flatten_parameters issues on some GPUs by disabling the flatten hook.
        self.gru.flatten_parameters = lambda: None

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
        # Keep module weights in fp32; cast activations to fp32 for internal math,
        # then cast outputs back to the original dtype for the base model.
        base_dtype = text_embeddings.dtype
        device = text_embeddings.device
        evidence_embeddings = retrieval.get("embeddings")
        if evidence_embeddings is None:
            evidence_embeddings = torch.zeros(
                text_embeddings.size(0),
                0,
                1,
                text_embeddings.size(-1),
                device=device,
                dtype=torch.float32,
            )
        else:
            evidence_embeddings = evidence_embeddings.to(device=device).float()
        evidence_scores = retrieval.get(
            "scores",
            torch.zeros(text_embeddings.size(0), 1, device=device, dtype=torch.float32),
        )
        evidence_scores = evidence_scores.to(device=device).float()

        # Align batch for conf and evidence_scores
        b = text_embeddings.size(0)
        img_conf = self._align_batch(img_conf.to(device=device).float(), b)
        txt_conf = self._align_batch(txt_conf.to(device=device).float(), b)
        evidence_scores = self._align_batch(evidence_scores, b)

        if evidence_embeddings.numel() == 0:
            gates = torch.zeros(text_embeddings.size(0), 2, device=device, dtype=torch.float32)
        else:
            gates = self.gating(img_conf, txt_conf, evidence_scores)

        prefix_tokens = (
            self._build_prefix(evidence_embeddings)
            if (self.config.enable_prefix and evidence_embeddings.numel() != 0)
            else torch.zeros(text_embeddings.size(0), 0, text_embeddings.size(-1), device=device, dtype=torch.float32)
        )
        # If there is no evidence (e.g., retrieval disabled), do NOT append dummy imputation tokens.
        # Otherwise, the positional layout shifts even though gates drive them to ~0.
        imputation_tokens = (
            self._build_imputation(text_embeddings.to(device=device).float(), txt_conf, evidence_embeddings, gates)
            if (self.config.enable_imputation and evidence_embeddings.numel() != 0)
            else torch.zeros(text_embeddings.size(0), 0, text_embeddings.size(-1), device=device, dtype=torch.float32)
        )
        augmented_inputs = torch.cat([prefix_tokens, text_embeddings.to(device=device).float(), imputation_tokens], dim=1)
        augmented_attention = torch.cat(
            [
                torch.ones(text_embeddings.size(0), prefix_tokens.size(1), device=device, dtype=text_attention.dtype),
                text_attention,
                torch.ones(text_embeddings.size(0), imputation_tokens.size(1), device=device, dtype=text_attention.dtype),
            ],
            dim=1,
        )

        if self.config.enable_memory and evidence_embeddings.numel():
            augmented_inputs = self.memory(augmented_inputs, evidence_embeddings.squeeze(2), gates[:, 0])

        return {
            "inputs_embeds": augmented_inputs.to(dtype=base_dtype),
            "attention_mask": augmented_attention,
            "gates": gates.to(dtype=base_dtype),
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
            evidence_summary = torch.zeros(
                text_embeddings.size(0),
                text_embeddings.size(-1),
                device=text_embeddings.device,
                dtype=text_embeddings.dtype,
            )
        else:
            evidence_summary = evidence_embeddings.squeeze(2).mean(dim=1)
        tokens = self.imputation(text_embeddings, txt_conf, evidence_summary)
        return gates[:, 1].view(-1, 1, 1) * tokens

    @staticmethod
    def _align_batch(t: torch.Tensor, target_b: int) -> torch.Tensor:
        """
        Ensure tensor has batch=target_b. If batch=1, expand; otherwise truncate to min.
        """
        if t.size(0) == target_b:
            return t
        if t.size(0) == 1:
            return t.expand(target_b, *([-1] * (t.dim() - 1)))
        # truncate to smallest to avoid shape mismatch
        min_b = min(target_b, t.size(0))
        return t[:min_b]


class TriPathReasoner(nn.Module):
    """
    Lightweight Transformer encoder (2 layers, 8 heads) to refine fused tokens
    after SelectiveReconstruction. Keeps dimensionality intact.
    设计目的：在 Prefix/Memory/Imputation 融合后，再做一次上下文建模与抑噪。
    """

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Build key padding mask: True for pads
        base_dtype = inputs_embeds.dtype
        key_padding = None
        if attention_mask is not None:
            key_padding = attention_mask == 0
        refined = self.encoder(inputs_embeds.float(), src_key_padding_mask=key_padding)
        refined = self.layer_norm(refined)
        return refined.to(dtype=base_dtype)
