"""
Implements structured corruption operators for Partial Modality Corruption (PMC).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CorruptionConfig:
    blur_prob: float = 0.3
    occlusion_prob: float = 0.3
    crop_prob: float = 0.2
    ocr_noise_prob: float = 0.4
    max_severity: float = 1.0


class CorruptionSimulator:
    def __init__(self, config: CorruptionConfig) -> None:
        self.config = config

    def _sample_severity(self) -> float:
        severity = random.random() * self.config.max_severity
        return severity

    def simulate(self, sample: Dict) -> Tuple[Dict, Dict]:
        """
        Returns a tuple of (corrupted_sample, corruption_report).
        """
        corruption_report = {"modalities": {}, "overall_uncertainty": 0.0}
        corrupted = sample.copy()
        severity_scores = []

        if random.random() < self.config.blur_prob:
            severity = self._sample_severity()
            corruption_report["modalities"]["vision"] = {
                "type": "blur",
                "severity": severity,
            }
            severity_scores.append(severity)

        if random.random() < self.config.occlusion_prob:
            severity = self._sample_severity()
            corruption_report["modalities"].setdefault("vision", {})
            corruption_report["modalities"]["vision"]["occlusion"] = severity
            severity_scores.append(severity)

        if random.random() < self.config.crop_prob:
            severity = self._sample_severity()
            corruption_report["modalities"].setdefault("vision", {})
            corruption_report["modalities"]["vision"]["crop"] = severity
            severity_scores.append(severity)

        if random.random() < self.config.ocr_noise_prob:
            severity = self._sample_severity()
            corruption_report["modalities"]["text"] = {
                "type": "ocr_noise",
                "severity": severity,
            }
            severity_scores.append(severity)
            extra = corrupted.get("extra", {})
            noisy_tokens = []
            for token in extra.get("ocr_tokens", []):
                if random.random() < 0.5:
                    noisy_tokens.append("<UNK>")
                else:
                    noisy_tokens.append(token)
            extra["ocr_tokens"] = noisy_tokens
            corrupted["extra"] = extra

        corruption_report["overall_uncertainty"] = (
            sum(severity_scores) / len(severity_scores) if severity_scores else 0.0
        )
        return corrupted, corruption_report
