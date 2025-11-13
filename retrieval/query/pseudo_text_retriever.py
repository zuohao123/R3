"""
Retriever that works on pseudo-text queries built from corrupted samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RetrievalConfig:
    top_k: int = 5
    noise_threshold: float = 0.6


class PseudoTextRetriever:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.index: List[Dict] = []

    def load_index(self, corpus: List[Dict]) -> None:
        self.index = corpus

    def query(self, question: str, pseudo_text: List[str], noise_score: float) -> List[Dict]:
        if not self.index:
            return []
        query_tokens = " ".join([question] + pseudo_text).lower().split()
        scored: List[Tuple[float, Dict]] = []
        for doc in self.index:
            doc_tokens = " ".join(doc.get("pseudo_text", [])).lower().split()
            if not doc_tokens:
                continue
            overlap = len(set(query_tokens) & set(doc_tokens))
            if overlap == 0:
                continue
            normalized = overlap / max(len(doc_tokens), 1)
            scored.append((normalized, doc))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)

        dynamic_top_k = self.config.top_k
        if noise_score > self.config.noise_threshold:
            reduction = min(noise_score, 0.95)
            dynamic_top_k = max(1, int(self.config.top_k * (1.0 - reduction)))

        top_docs: List[Dict] = []
        for score, doc in scored[: dynamic_top_k]:
            doc_copy = dict(doc)
            doc_copy["retrieval_score"] = score
            top_docs.append(doc_copy)
        return top_docs
