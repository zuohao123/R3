"""
Full-modality evaluation script for sanity checking.
"""
from __future__ import annotations

from typing import Dict

import torch


def evaluate_full(model: torch.nn.Module, dataloader) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            predictions = outputs["predictions"]
            total += len(predictions)
            correct += sum(p == gt for p, gt in zip(predictions, batch["answer"]))
    return {"accuracy": correct / max(total, 1)}
