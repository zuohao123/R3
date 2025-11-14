"""
Pseudo-text guided adaptive retrieval for the R^3 architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class RetrievalModuleConfig:
    hidden_size: int = 4096
    top_k: int = 3
    enable: bool = True


class PseudoTextRetrievalModule(nn.Module):
    """
    Lightweight retrieval built on pseudo-text tokens.
    """

    def __init__(self, config: RetrievalModuleConfig, embedding_layer: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.evidence_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scorer = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        question_embeddings: torch.Tensor,
        pseudo_text: Sequence[Sequence[str]],
        img_conf: torch.Tensor,
        txt_conf: torch.Tensor,
    ) -> Dict[str, torch.Tensor | List[List[str]]]:
        if not self.config.enable:
            batch_size = question_embeddings.size(0)
            return {
                "texts": [[] for _ in range(batch_size)],
                "embeddings": torch.zeros(batch_size, 0, 1, self.config.hidden_size, device=question_embeddings.device),
                "scores": torch.zeros(batch_size, 0, device=question_embeddings.device),
            }

        query = self._build_query(question_embeddings, txt_conf)  # (b, d)
        evidence_embeddings, evidence_texts = self._encode_evidence(pseudo_text, question_embeddings.device)
        scores = self._score(query, evidence_embeddings, img_conf, txt_conf)
        topk_embeddings, topk_texts, topk_scores = self._select_topk(evidence_embeddings, evidence_texts, scores)
        return {
            "texts": topk_texts,
            "embeddings": topk_embeddings,
            "scores": topk_scores,
        }

    def _build_query(self, question_embeddings: torch.Tensor, txt_conf: torch.Tensor) -> torch.Tensor:
        weights = (1.0 - txt_conf).unsqueeze(-1)
        pooled = (question_embeddings * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return self.query_proj(pooled)

    def _encode_evidence(
        self,
        pseudo_text: Sequence[Sequence[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        embeddings = []
        texts: List[List[str]] = []
        for entries in pseudo_text:
            encoded_entries = []
            stored_texts = []
            for text in entries:
                if not text:
                    continue
                tokens = torch.tensor(
                    [hash(text) % self.embedding_layer.num_embeddings],
                    device=device,
                )
                vec = self.embedding_layer(tokens)
                encoded_entries.append(vec.mean(dim=0))
                stored_texts.append(text)
            if not encoded_entries:
                encoded_entries = [torch.zeros(self.config.hidden_size, device=device)]
                stored_texts.append("")
            embeddings.append(torch.stack(encoded_entries))
            texts.append(stored_texts)
        max_len = max(e.size(0) for e in embeddings)
        padded = []
        for emb in embeddings:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=device)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        stacked = torch.stack(padded)  # (b, evidences, d)
        return stacked.unsqueeze(2), texts  # reshape to (b, evidences, 1, d) for downstream

    def _score(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        img_conf: torch.Tensor,
        txt_conf: torch.Tensor,
    ) -> torch.Tensor:
        proj = self.evidence_proj(evidence).squeeze(2)  # (b, evidences, d)
        query = query.unsqueeze(1)
        logits = torch.cosine_similarity(query, proj, dim=-1)
        noise_gate = 1.0 - torch.mean(torch.cat([img_conf, txt_conf], dim=1), dim=1, keepdim=True)
        return logits * noise_gate

    def _select_topk(
        self,
        embeddings: torch.Tensor,
        texts: List[List[str]],
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[str]], torch.Tensor]:
        k = min(self.config.top_k, embeddings.size(1))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        batch_embeddings = []
        batch_texts: List[List[str]] = []
        for b in range(embeddings.size(0)):
            indices = topk_indices[b]
            batch_embeddings.append(embeddings[b, indices])
            batch_texts.append([texts[b][idx] if idx < len(texts[b]) else "" for idx in indices])
        stacked_embeddings = torch.stack(batch_embeddings)
        return stacked_embeddings, batch_texts, topk_scores
