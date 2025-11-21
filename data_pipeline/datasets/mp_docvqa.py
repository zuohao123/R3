"""
MP-DocVQA dataset adapter with Page-as-Evidence support.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base_dataset import BasePMCDataset


class MPDocVQADataset(BasePMCDataset):
    """
    Expects mp_docvqa_{split}.json with fields:
    [
      {
        "id": "docid_page",
        "doc_id": "...",
        "page": 0,
        "question": "...",
        "answer": "...",
        "image": "path/to/page.png",
        "ocr_tokens": [...],
        "captions": [...]
      }, ...
    ]
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.image_root = root / "images"
        super().__init__(root, split)

    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"mp_docvqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("MP-DocVQA annotation must be a list of page entries.")
        # Group by document for cross-page context
        by_doc: Dict[str, List[Dict]] = {}
        for entry in payload:
            doc_id = str(entry.get("doc_id") or entry.get("document_id") or "")
            if not doc_id:
                continue
            by_doc.setdefault(doc_id, []).append(entry)
        for doc_entries in by_doc.values():
            doc_entries.sort(key=lambda x: x.get("page", 0))
        normalized: List[Dict] = []
        for entries in by_doc.values():
            for idx, entry in enumerate(entries):
                sample = dict(entry)
                sample_id = (
                    entry.get("id")
                    or f"{entry.get('doc_id', 'doc')}_{entry.get('page', idx)}"
                )
                sample["id"] = sample_id
                sample["_page_index"] = idx
                sample["_doc_entries"] = entries
                normalized.append(sample)
        return normalized

    def __getitem__(self, idx: int) -> Dict:
        sample_meta = self.samples[idx]
        raw_item = self._load_raw_item(sample_meta)
        # Attach neighbor page OCR/Caption as context_evidence (Page-as-Evidence)
        page_idx = sample_meta.get("_page_index", 0)
        doc_entries = sample_meta.get("_doc_entries", [])
        context: List[str] = []
        for neighbor in [page_idx - 1, page_idx + 1]:
            if 0 <= neighbor < len(doc_entries):
                ctx_entry = doc_entries[neighbor]
                context.extend(self._extract_text(ctx_entry.get("ocr_tokens", [])))
                context.extend([c for c in ctx_entry.get("captions", []) if c])
        raw_item["extra"].setdefault("context_evidence", context)
        return {
            "id": sample_meta["id"],
            "question": raw_item["question"],
            "answer": raw_item.get("answer"),
            "image_path": raw_item.get("image_path"),
            "extra": raw_item.get("extra", {}),
        }

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        image_candidate = sample_meta.get("image") or sample_meta.get("image_path")
        image_path = self._resolve_image_path(str(image_candidate))
        extra = {
            "ocr_tokens": self._normalize_ocr_tokens(sample_meta.get("ocr_tokens") or []),
            "captions": sample_meta.get("captions", []),
            "context_evidence": sample_meta.get("context_evidence", []),
            "metadata": {
                "dataset": "MP-DocVQA",
                "split": self.split,
                "doc_id": sample_meta.get("doc_id"),
                "page": sample_meta.get("page"),
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
            candidates.extend([f"{path_obj.name}.png", f"{path_obj.name}.jpg", f"{path_obj.name}.jpeg"])
        for candidate in candidates:
            resolved = self.image_root / candidate
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

    @staticmethod
    def _extract_text(tokens: List) -> List[str]:
        out: List[str] = []
        for token in tokens:
            if isinstance(token, str):
                out.append(token)
            elif isinstance(token, dict):
                if token.get("text"):
                    out.append(token["text"])
        return out
