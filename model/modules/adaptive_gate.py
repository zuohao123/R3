"""
Adaptive gating controller that modulates each reconstruction path.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


class AdaptiveGateController:
    def __init__(self, default_value: float = 0.7) -> None:
        self.default_value = default_value

    def __call__(self, corruption_report: Dict | List[Dict], retrieval: List | List[List[Dict]]) -> Dict[str, float]:
        if isinstance(corruption_report, Sequence) and corruption_report and isinstance(
            corruption_report[0], dict
        ):
            uncertainties = [report.get("overall_uncertainty", 0.0) for report in corruption_report]
            uncertainty = sum(uncertainties) / max(len(uncertainties), 1)
        else:
            uncertainty = corruption_report.get("overall_uncertainty", 0.0) if isinstance(corruption_report, dict) else 0.0

        if isinstance(retrieval, Sequence) and retrieval and isinstance(retrieval[0], list):
            retrieved = [1.0 if docs else 0.0 for docs in retrieval]
            retrieval_quality = sum(retrieved) / max(len(retrieved), 1)
        else:
            retrieval_quality = 1.0 if retrieval else 0.0
        return {
            "text": max(self.default_value - uncertainty * 0.5, 0.1),
            "memory": max(self.default_value * retrieval_quality, 0.1),
            "imputation": max(self.default_value + uncertainty * 0.3, 0.1),
        }
