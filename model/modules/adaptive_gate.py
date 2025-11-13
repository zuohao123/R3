"""
Adaptive gating controller that modulates each reconstruction path.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


class AdaptiveGateController:
    def __init__(self, default_value: float = 0.7) -> None:
        self.default_value = default_value

    def __call__(self, corruption_report: Sequence[Dict], retrieval: Sequence[Sequence[Dict]]) -> Dict[str, List[float]]:
        reports = self._ensure_sequence(corruption_report)
        retrieved = self._ensure_sequence(retrieval)
        batch = max(len(reports), len(retrieved))

        text_gates: List[float] = []
        memory_gates: List[float] = []
        imputation_gates: List[float] = []

        for idx in range(batch):
            report = reports[idx] if idx < len(reports) else {}
            docs = retrieved[idx] if idx < len(retrieved) else []

            uncertainty = report.get("overall_uncertainty", 0.0) if isinstance(report, dict) else 0.0
            retrieval_quality = self._retrieval_quality(docs)

            text_gate = self._clamp(self.default_value - 0.4 * uncertainty + 0.2 * retrieval_quality)
            memory_gate = self._clamp(self.default_value * (0.5 + retrieval_quality))
            imputation_gate = self._clamp(self.default_value + 0.5 * uncertainty - 0.2 * retrieval_quality)

            text_gates.append(text_gate)
            memory_gates.append(memory_gate)
            imputation_gates.append(imputation_gate)

        return {
            "text": text_gates,
            "memory": memory_gates,
            "imputation": imputation_gates,
        }

    def _ensure_sequence(self, value):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
        return [value]

    def _retrieval_quality(self, docs: Sequence[Dict]) -> float:
        if not docs:
            return 0.0
        scored = [doc.get("retrieval_score", 0.0) for doc in docs]
        if not scored:
            return 0.0
        return max(sum(scored) / len(scored), 0.05)

    def _clamp(self, value: float) -> float:
        return max(0.05, min(value, 1.5))
