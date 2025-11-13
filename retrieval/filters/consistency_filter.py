"""
Filters retrieved evidence that contradicts observed modalities.
"""
from __future__ import annotations

from typing import Dict, List


class ConsistencyFilter:
    def __init__(self, contradiction_token: str = "[CONTRADICT]") -> None:
        self.contradiction_token = contradiction_token

    def __call__(self, retrieved: List[Dict], pseudo_text: List[str]) -> List[Dict]:
        pseudo_set = set(" ".join(pseudo_text).split())
        filtered = []
        for doc in retrieved:
            if self.contradiction_token in doc.get("pseudo_text", []):
                continue
            doc_tokens = set(" ".join(doc.get("pseudo_text", [])).split())
            overlap = len(doc_tokens & pseudo_set)
            score = doc.get("retrieval_score", 0.0)
            if overlap == 0 or score <= 0:
                continue
            filtered.append(doc)
        return filtered
