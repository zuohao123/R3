"""
DocVQA dataset helper that provides document-specific metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .base_dataset import BasePMCDataset


class DocVQADataset(BasePMCDataset):
    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"docvqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        return annotations

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        image_meta = sample_meta["image"]
        image_rel = Path(str(image_meta))
        if image_rel.is_absolute():
            image_path = image_rel
        else:
            parts = list(image_rel.parts)
            # Some annotations already include the "documents/" prefix, and some even duplicate it.
            while len(parts) > 1 and parts[0] == parts[1] and parts[0] == "documents":
                parts.pop(0)
            image_rel = Path(*parts)
            image_path = self.root / image_rel if "documents" in image_rel.parts else (self.root / "documents" / image_rel)
        return {
            "question": sample_meta["question"],
            "answer": sample_meta.get("answer"),
            "image_path": image_path.as_posix(),
            "extra": {
                "ocr_tokens": sample_meta.get("ocr_tokens", []),
                "layout": sample_meta.get("layout"),
            },
        }
