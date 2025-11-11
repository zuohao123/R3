"""
Minimal decoder that mimics Qwen-style reasoning with lightweight components.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

import torch
import torch.nn.functional as F


class SimpleDecoder(torch.nn.Module):
    """
    Provides a stand-in backbone for environments where the actual Qwen3-VL
    weights are unavailable. It consumes the enhanced question tokens,
    projected visual tokens, and adapter memories, and produces logits over a
    synthetic vocabulary.
    """

    def __init__(self, hidden_size: int = 256, answer_vocab: int = 8192) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.answer_vocab = answer_vocab
        self.text_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.vision_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.memory_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.imputation_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, answer_vocab)

    def forward(
        self,
        question_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        memory_cache: torch.Tensor,
        imputation_tokens: torch.Tensor,
        answer: List[str] | None = None,
        **_,
    ) -> Dict:
        fused_question = self.text_proj(question_tokens).mean(dim=1)
        fused_vision = self.vision_proj(vision_tokens).mean(dim=1)
        fused_memory = self.memory_proj(memory_cache).mean(dim=1)
        fused_imputation = self.imputation_proj(imputation_tokens).mean(dim=1)
        fused = fused_question + fused_vision + fused_memory + fused_imputation
        logits = self.classifier(self.norm(fused))

        outputs = {"logits": logits}
        if answer is not None:
            targets = self._encode_answers(answer, logits.device)
            outputs["loss"] = F.cross_entropy(logits, targets)
        else:
            outputs["loss"] = torch.zeros(1, device=logits.device)
        outputs["predictions"] = [f"token_{idx}" for idx in logits.argmax(dim=-1).tolist()]
        return outputs

    def _encode_answers(self, answers: List[str], device: torch.device) -> torch.Tensor:
        ids = [self._hash_to_vocab_id(ans) for ans in answers]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _hash_to_vocab_id(self, text: str) -> int:
        digest = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.answer_vocab
