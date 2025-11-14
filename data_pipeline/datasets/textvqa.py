"""
TextVQA dataset reader that exposes metadata required by the PMC pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base_dataset import BasePMCDataset


class TextVQADataset(BasePMCDataset):
    def __init__(self, root: Path, split: str = "train") -> None:
        super().__init__(root, split)
        self.image_root = self._discover_image_root()

    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"textvqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        if isinstance(annotations, dict):
            # Official TextVQA releases wrap samples inside a `data` list along
            # with extra metadata. Older snapshots might already be a list, so
            # we normalize both cases here.
            entries = annotations.get("data") or annotations.get("annotations")
            if entries is None:
                raise ValueError(
                    "Unexpected TextVQA annotation format. Expected `data` or "
                    "`annotations` field when JSON is a dict."
                )
        elif isinstance(annotations, list):
            entries = annotations
        else:
            raise ValueError(f"Unsupported annotation payload type: {type(annotations).__name__}")

        normalized: List[Dict] = []
        # todo
        entries = entries[:2]
        for idx, raw in enumerate(entries):
            sample = dict(raw)
            sample_id = (
                raw.get("id")
                or raw.get("question_id")
                or f"{self.split}_{idx}"
            )
            sample["id"] = sample_id
            if "image" not in sample:
                sample["image"] = raw.get("image_id") or raw.get("image") or str(sample_id)
            if "answer" not in sample:
                answers = raw.get("answers")
                if isinstance(answers, list) and answers:
                    first = answers[0]
                    sample["answer"] = first.get("answer") if isinstance(first, dict) else first
                elif isinstance(answers, dict) and "answer" in answers:
                    sample["answer"] = answers["answer"]
            normalized.append(sample)
        # todo 数据样例
        print(normalized)
        return normalized

    def _discover_image_root(self) -> Optional[Path]:
        candidate_dirs = [
            self.root / "images",
            self.root.parent / "textvqa_image",
        ]
        for directory in candidate_dirs:
            if directory.exists():
                return directory
        return None

    def _resolve_image_path(self, identifier: str) -> str:
        if not identifier:
            return identifier
        if identifier.startswith(("http://", "https://")):
            return identifier
        if self.image_root is None:
            return identifier

        path_obj = Path(identifier)
        candidates = [path_obj.name]
        if not path_obj.suffix:
            candidates.extend(
                [
                    f"{path_obj.name}.jpg",
                    f"{path_obj.name}.png",
                    f"{path_obj.name}.jpeg",
                ]
            )

        for candidate in candidates:
            resolved = self.image_root / candidate
            if resolved.exists():
                return resolved.as_posix()
        return identifier

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        image_candidate = sample_meta.get("image") or sample_meta.get("image_id")
        # Fall back to Flickr URLs when available so that each sample still
        # produces a deterministic vision key even if the local file does
        # not exist on disk.
        image_candidate = (
            image_candidate
            or sample_meta.get("flickr_original_url")
            or sample_meta.get("flickr_300k_url")
            or str(sample_meta["id"])
        )
        image_path = self._resolve_image_path(str(image_candidate))
        extra = {
            # 统一附带 OCR / Caption / Table 字段，方便后续伪文本构建
            "ocr_tokens": self._normalize_ocr_tokens(sample_meta.get("ocr_tokens") or []),
            "captions": sample_meta.get("captions", []),
            "tables": sample_meta.get("tables", []),
            "metadata": {
                "dataset": "TextVQA",
                "split": self.split,
            },
        }
        return {
            "question": sample_meta["question"],
            "answer": sample_meta.get("answer"),
            "image_path": image_path,
            "extra": extra,
        }

    def _normalize_ocr_tokens(self, tokens: List) -> List[Dict]:
        # 不同数据版本的 OCR 格式不一致，这里统一为 dict 形式，便于下游处理
        normalized: List[Dict] = []
        for token in tokens:
            if isinstance(token, str):
                normalized.append(
                    {
                        "text": token,
                        "bbox": [0, 0, 0, 0],
                        "conf": 1.0,
                        "src": "ocr",
                    }
                )
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
