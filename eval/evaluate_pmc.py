"""
Evaluation entrypoint for partially corrupted benchmarks.
"""
from __future__ import annotations

from typing import Dict

import torch


def evaluate(model: torch.nn.Module, dataloader, retriever) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    hallucinations = 0

    with torch.no_grad():
        for batch in dataloader:
            retrieved = retriever(batch)
            outputs = model(**batch, retrieval=retrieved)
            predictions = outputs["predictions"]
            total += len(predictions)
            correct += sum(p == gt for p, gt in zip(predictions, batch["answer"]))
            hallucinations += sum(pred not in doc.get("pseudo_text", []) for pred, doc in zip(predictions, retrieved))

    accuracy = correct / max(total, 1)
    hallucination_rate = hallucinations / max(total, 1)
    return {"accuracy": accuracy, "hallucination_rate": hallucination_rate}
