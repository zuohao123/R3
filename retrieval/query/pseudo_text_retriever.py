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
            doc_tokens = " ".join(doc["pseudo_text"]).lower().split()
            overlap = len(set(query_tokens) & set(doc_tokens))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored[: self.config.top_k] if score > 0]
        if noise_score > self.config.noise_threshold:
            top_docs = top_docs[: max(1, len(top_docs) // 2)]
        return top_docs
