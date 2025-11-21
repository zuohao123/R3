"""
InfoVQA dataset adapter for R^3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .base_dataset import BasePMCDataset


class InfoVQADataset(BasePMCDataset):
    """
    Expects infovqa_{split}.json with fields:
    [
      {
        "id": "...",
        "question": "...",
        "answer": "...",
        "image": "path or url",
        "ocr_tokens": [...],
        "captions": [...]
      }
    ]
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.image_root = root / "images"
        super().__init__(root, split)

    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"infovqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("InfoVQA annotation must be a list.")
        normalized: List[Dict] = []
        for idx, entry in enumerate(payload):
            sample = dict(entry)
            sid = entry.get("id") or f"{self.split}_{idx}"
            sample["id"] = sid
            normalized.append(sample)
        return normalized

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        image_candidate = sample_meta.get("image") or sample_meta.get("image_path") or sample_meta.get("image_id")
        image_path = self._resolve_image_path(str(image_candidate or sample_meta["id"]))
        extra = {
            "ocr_tokens": self._normalize_ocr_tokens(sample_meta.get("ocr_tokens") or []),
            "captions": sample_meta.get("captions", []),
            "context_evidence": sample_meta.get("context_evidence", []),
            "metadata": {
                "dataset": "InfoVQA",
                "split": self.split,
            },
        }
        return {
            "question": sample_meta["question"],
            "answer": sample_meta.get("answer"),
            "image_path": image_path,
            "extra": extra,
        }

    def _resolve_image_path(self, identifier: str) -> str:
        if not identifier:
            return identifier
        path_obj = Path(identifier)
        candidates = [path_obj.name]
        if not path_obj.suffix:
            candidates.extend([f"{path_obj.name}.png", f"{path_obj.name}.jpg"])
        for cand in candidates:
            resolved = self.image_root / cand
            if resolved.exists():
                return resolved.as_posix()
        return identifier

    def _normalize_ocr_tokens(self, tokens: List) -> List[Dict]:
        normalized: List[Dict] = []
        for token in tokens:
            if isinstance(token, str):
                normalized.append({"text": token, "bbox": [0, 0, 0, 0], "conf": 1.0, "src": "ocr"})
            elif isinstance(token, dict):
                normalized.append(
                    {
                        "text": token.get("text", ""),
                        "bbox": token.get("bbox", [0, 0, 0, 0]),
                        "conf": token.get("conf", 1.0),
                        "src": token.get("src", "ocr"),
                    }
                )
        return normalized
