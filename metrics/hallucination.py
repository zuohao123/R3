"""
Utilities to measure hallucination rate by comparing answers with evidence.
"""
from __future__ import annotations

from typing import Iterable, List


def compute_hallucination_rate(predictions: List[str], evidences: List[List[str]]) -> float:
    hallucinations = 0
    for pred, docs in zip(predictions, evidences):
        flat_evidence = " ".join(sum(docs, [])).lower()
        if pred.lower() not in flat_evidence:
            hallucinations += 1
    return hallucinations / max(len(predictions), 1)
