"""
Tri-path enhancement module for selective reconstruction.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

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
        pseudo_text: Sequence[Sequence[str]],
        retrieval: Sequence[Sequence[Dict]],
        gates: Dict[str, Sequence[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_gate = self._gate_tensor(gates.get("text"), question_tokens)
        memory_gate = self._gate_tensor(gates.get("memory"), question_tokens)
        imputation_gate = self._gate_tensor(gates.get("imputation"), question_tokens)

        text_prefix = self._build_prefix(question_tokens, pseudo_text, retrieval, text_gate)
        memory_bank = self._build_memory(question_tokens, retrieval, memory_gate)
        imputation = self._build_imputation(imputation_gate)
        enhanced_question = torch.cat([text_prefix, question_tokens], dim=1)
        return enhanced_question, memory_bank, imputation

    def _build_prefix(
        self,
        question_tokens: torch.Tensor,
        pseudo_text: Sequence[Sequence[str]],
        retrieval: Sequence[Sequence[Dict]],
        gate: torch.Tensor,
    ) -> torch.Tensor:
        question_slice = question_tokens[:, : self.prefix_length, :]
        pseudo_segments = self._encode_segments(pseudo_text, question_tokens, self.prefix_length)
        retrieval_segments = self._encode_segments(
            [[ " ".join(doc.get("pseudo_text", [])) for doc in docs ] for docs in retrieval],
            question_tokens,
            self.prefix_length,
        )
        fused = (question_slice + pseudo_segments + retrieval_segments) / 3.0
        encoded = self.prefix_encoder(fused)
        return encoded * gate

    def _build_memory(
        self,
        question_tokens: torch.Tensor,
        retrieval: Sequence[Sequence[Dict]],
        gate: torch.Tensor,
    ) -> torch.Tensor:
        question_slice = question_tokens[:, : self.memory_tokens, :]
        retrieval_segments = self._encode_segments(
            [[ " ".join(doc.get("pseudo_text", [])) for doc in docs ] for docs in retrieval],
            question_tokens,
            self.memory_tokens,
        )
        fused = (question_slice + retrieval_segments) / 2.0
        adapted = self.memory_adapter(fused)
        return adapted * gate

    def _build_imputation(self, gate: torch.Tensor) -> torch.Tensor:
        batch = gate.shape[0]
        tokens = self.imputation_embeddings.unsqueeze(0).to(gate.device)
        tokens = tokens.expand(batch, -1, -1)
        return tokens * gate

    def _encode_segments(
        self,
        segments: Sequence[Sequence[str]],
        reference: torch.Tensor,
        max_tokens: int,
    ) -> torch.Tensor:
        batch, _, hidden_size = reference.shape
        device = reference.device
        dtype = reference.dtype
        encoded = torch.zeros(batch, max_tokens, hidden_size, device=device, dtype=dtype)
        for idx, group in enumerate(segments):
            if idx >= batch:
                break
            tokens = []
            for segment in group or []:
                tokens.extend(segment.lower().split())
            for pos, token in enumerate(tokens[:max_tokens]):
                encoded[idx, pos] = self._hash_to_vector(token, dtype=dtype, device=device)
        return encoded

    def _gate_tensor(self, values: Sequence[float] | float | None, reference: torch.Tensor) -> torch.Tensor:
        batch = reference.shape[0]
        if values is None:
            gate_vals = torch.ones(batch, dtype=reference.dtype, device=reference.device)
        elif isinstance(values, Sequence):
            gate_vals = torch.tensor(values, dtype=reference.dtype, device=reference.device)
            if gate_vals.numel() != batch:
                gate_vals = gate_vals.repeat(batch // gate_vals.numel() + 1)[:batch]
        else:
            gate_vals = torch.full((batch,), float(values), dtype=reference.dtype, device=reference.device)
        gate = gate_vals.view(batch, 1, 1).clamp(0.05, 1.5)
        return gate

    def _hash_to_vector(self, token: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self._hash_to_seed(token))
        vec = torch.randn(self.hidden_size, generator=generator, dtype=dtype)
        return vec.to(device)

    @staticmethod
    def _hash_to_seed(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)
