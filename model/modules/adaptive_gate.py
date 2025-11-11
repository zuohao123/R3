"""
Adaptive gating controller that modulates each reconstruction path.
"""
from __future__ import annotations

from typing import Dict, List


class AdaptiveGateController:
    def __init__(self, default_value: float = 0.7) -> None:
        self.default_value = default_value

    def __call__(self, corruption_report: Dict, retrieval: List[Dict]) -> Dict[str, float]:
        uncertainty = corruption_report.get("overall_uncertainty", 0.0)
        retrieval_quality = 1.0 if retrieval else 0.0
        return {
            "text": max(self.default_value - uncertainty * 0.5, 0.1),
            "memory": max(self.default_value * retrieval_quality, 0.1),
            "imputation": max(self.default_value + uncertainty * 0.3, 0.1),
        }
