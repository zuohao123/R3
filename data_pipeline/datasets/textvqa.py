"""
TextVQA dataset reader that exposes metadata required by the PMC pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .base_dataset import BasePMCDataset


class TextVQADataset(BasePMCDataset):
    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"textvqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        return annotations

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        image_path = self.root / "images" / sample_meta["image"]
        return {
            "question": sample_meta["question"],
            "answer": sample_meta.get("answer"),
            "image_path": image_path.as_posix(),
            "extra": {"ocr_tokens": sample_meta.get("ocr_tokens", [])},
        }
