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
        "questionId": 337,
        "question": "what is the date mentioned in this letter?",
        "doc_id": "xnbl0037",
        "page_ids": ["xnbl0037_p0", "xnbl0037_p1"],
        "answers": ["1/8/93"],
        "answer_page_idx": 0,
        "data_split": "train"
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
            raise ValueError("MP-DocVQA annotation must be a list of question entries.")

        # Pattern A: question-level entries with page_ids (官方格式)
        if any(isinstance(e.get("page_ids"), list) for e in payload):
            normalized: List[Dict] = []
            for entry in payload:
                question_id = entry.get("questionId")
                question = entry["question"]
                doc_id = entry.get("doc_id") or entry.get("docId") or entry.get("id") or question_id
                page_ids = entry.get("page_ids", [])
                answers = entry.get("answers", [])
                answer_page_idx = entry.get("answer_page_idx", 0)

                for page_idx, page_id in enumerate(page_ids):
                    sample = {
                        "id": f"{question_id}_{page_id}",
                        "questionId": question_id,
                        "question": question,
                        "doc_id": doc_id,
                        "page_id": page_id,
                        "page_idx": page_idx,
                        "answer": answers[0] if answers else "",
                        "all_answers": answers,
                        "answer_page_idx": answer_page_idx,
                        "is_answer_page": (page_idx == answer_page_idx),
                        "total_pages": len(page_ids),
                        "all_page_ids": page_ids,
                        "data_split": entry.get("data_split", self.split),
                    }
                    normalized.append(sample)
            return normalized

        # Pattern B: 已展开的每页记录，包含 image/page_id/question/answer
        normalized_flat: List[Dict] = []
        for idx, entry in enumerate(payload):
            sid = entry.get("id") or f"{self.split}_{idx}"
            page_id = entry.get("page_id") or entry.get("image") or entry.get("image_id") or sid
            question = entry.get("question", "")
            answer = entry.get("answer", entry.get("answers", [""])[0] if isinstance(entry.get("answers"), list) else "")
            normalized_flat.append(
                {
                    "id": str(sid),
                    "questionId": entry.get("questionId"),
                    "question": question,
                    "doc_id": entry.get("doc_id") or entry.get("docId"),
                    "page_id": page_id,
                    "page_idx": entry.get("page_idx", 0),
                    "answer": answer,
                    "all_answers": entry.get("answers", [answer] if answer else []),
                    "answer_page_idx": entry.get("answer_page_idx", 0),
                    "is_answer_page": entry.get("is_answer_page", True),
                    "total_pages": entry.get("total_pages", 1),
                    "all_page_ids": entry.get("all_page_ids", [page_id]),
                    "data_split": entry.get("data_split", self.split),
                    "ocr_tokens": entry.get("ocr_tokens", []),
                    "captions": entry.get("captions", []),
                    "context_evidence": entry.get("context_evidence", []),
                }
            )
        return normalized_flat

    def __getitem__(self, idx: int) -> Dict:
        sample_meta = self.samples[idx]
        raw_item = self._load_raw_item(sample_meta)
        
        # Attach neighbor page context as context_evidence (Page-as-Evidence)
        context: List[str] = []
        current_page_idx = sample_meta.get("page_idx", 0)
        all_page_ids = sample_meta.get("all_page_ids", [])
        
        # Add context from neighboring pages
        for neighbor_offset in [-1, 1]:
            neighbor_idx = current_page_idx + neighbor_offset
            if 0 <= neighbor_idx < len(all_page_ids):
                neighbor_page_id = all_page_ids[neighbor_idx]
                # Try to load OCR/captions for neighbor page
                neighbor_image_path = self._resolve_image_path(neighbor_page_id)
                if neighbor_image_path:
                    # Add page identifier as context
                    context.append(f"[PAGE {neighbor_idx}] {neighbor_page_id}")
                    # Note: In a real implementation, you might want to load
                    # actual OCR tokens for neighbor pages here
        
        raw_item["extra"].setdefault("context_evidence", context)
        return {
            "id": sample_meta["id"],
            "question": raw_item["question"],
            "answer": raw_item.get("answer"),
            "image_path": raw_item.get("image_path"),
            "extra": raw_item.get("extra", {}),
        }

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        # Use page_id as the image identifier
        page_id = sample_meta.get("page_id") or sample_meta.get("image") or sample_meta.get("image_path") or ""
        image_path = self._resolve_image_path(page_id)
        
        extra = {
            "ocr_tokens": self._normalize_ocr_tokens(sample_meta.get("ocr_tokens") or []),
            "captions": sample_meta.get("captions", []),
            "context_evidence": sample_meta.get("context_evidence", []),
            "metadata": {
                "dataset": "MP-DocVQA",
                "split": self.split,
                "questionId": sample_meta.get("questionId"),
                "doc_id": sample_meta.get("doc_id"),
                "page_id": sample_meta.get("page_id"),
                "page_idx": sample_meta.get("page_idx"),
                "answer_page_idx": sample_meta.get("answer_page_idx"),
                "is_answer_page": sample_meta.get("is_answer_page", False),
                "total_pages": sample_meta.get("total_pages", 1),
                "all_answers": sample_meta.get("all_answers", []),
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

        # Accept already-resolved relative paths
        rel_candidate = self.root / identifier
        if rel_candidate.exists():
            return rel_candidate.as_posix()

        # For MP-DocVQA, page_id format is typically "docid_p0", "docid_p1", etc.
        path_obj = Path(identifier)
        candidates = [
            path_obj.name,
            f"{path_obj.name}.png",
            f"{path_obj.name}.jpg",
            f"{path_obj.name}.jpeg",
        ]

        # Also try without the page suffix for some datasets
        if "_p" in identifier:
            base_name = identifier.split("_p")[0]
            page_num = identifier.split("_p")[1] if "_p" in identifier else "0"
            candidates.extend(
                [
                    f"{base_name}_page_{page_num}.png",
                    f"{base_name}_page_{page_num}.jpg",
                    f"{base_name}_{page_num}.png",
                    f"{base_name}_{page_num}.jpg",
                ]
            )

        for candidate in candidates:
            resolved = self.image_root / candidate
            if resolved.exists():
                return resolved.as_posix()

        return str(self.image_root / path_obj.name)

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
