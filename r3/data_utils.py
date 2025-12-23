"""
Shared dataset utilities for training/evaluation.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image
from torch.utils.data import Dataset

from data_pipeline.datasets import BasePMCDataset
from data_pipeline.corruptions import (
    ImageCorruptionConfig,
    ImageCorruptor,
    PseudoTextCorruptionConfig,
    PseudoTextCorruptor,
)
from r3.retrieval_module import PseudoTextBuilder


class R3Dataset(Dataset):
    def __init__(
        self,
        base_dataset: BasePMCDataset,
        vision_tokens: int,
        hidden_size: int,
        apply_corruption: bool = True,
        pseudo_builder: Optional[PseudoTextBuilder] = None,
        pseudo_text_drop_prob: float = 0.3,
        pseudo_corpus: Optional[Dict[str, List[str]]] = None,
        image_corruptor: Optional[ImageCorruptor] = None,
        pseudo_text_corruptor: Optional[PseudoTextCorruptor] = None,
        pseudo_text_max_items: Optional[int] = None,
        pseudo_text_max_chars: Optional[int] = None,
        pseudo_text_chunk_tokens: Optional[int] = 32,
        corruption_prob: float = 1.0,
    ) -> None:
        self.base = base_dataset
        self.vision_tokens = vision_tokens
        self.hidden_size = hidden_size
        self.apply_corruption = apply_corruption
        self.pseudo_builder = pseudo_builder or PseudoTextBuilder()
        self.pseudo_corpus = pseudo_corpus or {}
        self.image_corruptor = image_corruptor or ImageCorruptor(ImageCorruptionConfig())
        self.pseudo_text_max_items = pseudo_text_max_items
        self.pseudo_text_max_chars = pseudo_text_max_chars
        self.pseudo_text_chunk_tokens = pseudo_text_chunk_tokens
        self.corruption_prob = float(corruption_prob)
        if pseudo_text_corruptor:
            self.pseudo_text_corruptor = pseudo_text_corruptor
        else:
            self.pseudo_text_corruptor = PseudoTextCorruptor(
                PseudoTextCorruptionConfig(drop_prob=pseudo_text_drop_prob)
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        sample = copy.deepcopy(self.base[idx])
        answer = sample.get("answer")
        if answer is None:
            answer = "UNKNOWN"
        elif isinstance(answer, (list, tuple)):
            answer = answer[0] if answer else "UNKNOWN"
        elif not isinstance(answer, str):
            answer = str(answer)
        pseudo_text = self._inline_pseudo_text(sample)
        # Prefer offline pseudo-text corpus if provided
        if sample.get("id") in self.pseudo_corpus:
            pseudo_text = self.pseudo_corpus[sample["id"]]
        pseudo_text = self._coalesce_pseudo_text(pseudo_text, self.pseudo_text_chunk_tokens)
        pseudo_text = self._truncate_pseudo_text(pseudo_text)
        if not pseudo_text and self.pseudo_builder:
            pseudo_text = self.pseudo_builder.build(sample)
        pseudo_text = self._coalesce_pseudo_text(pseudo_text, self.pseudo_text_chunk_tokens)
        pseudo_text = self._truncate_pseudo_text(pseudo_text)
        do_corrupt = self.apply_corruption and (random.random() < self.corruption_prob)
        corrupted_pseudo = self.pseudo_text_corruptor(pseudo_text) if do_corrupt else pseudo_text
        image = self._load_image(sample.get("image_path"))
        clean_image = image.copy() if image else None
        corrupted_image = self.image_corruptor(image) if (image and do_corrupt) else clean_image
        clean = {
            "question": sample["question"],
            "labels": answer,
            "pseudo_text": pseudo_text,
            "image_path": sample.get("image_path"),
            "image": clean_image,
            "vision_tokens": self.vision_tokens,
            "hidden_size": self.hidden_size,
        }
        corrupted_branch = {
            "question": sample["question"],  # 问题保持不变，仅模拟伪文本/视觉缺失
            "labels": answer,
            "pseudo_text": corrupted_pseudo,
            "image_path": sample.get("image_path"),
            "image": corrupted_image,
            "vision_tokens": self.vision_tokens,
            "hidden_size": self.hidden_size,
        }
        return {"id": sample["id"], "clean": clean, "corrupted": corrupted_branch}

    def _truncate_pseudo_text(self, entries: List[str]) -> List[str]:
        if not entries:
            return entries
        result = entries
        if self.pseudo_text_max_items is not None:
            result = result[: max(0, int(self.pseudo_text_max_items))]
        if self.pseudo_text_max_chars is not None:
            max_chars = int(self.pseudo_text_max_chars)
            if max_chars > 0:
                result = [str(t)[:max_chars] for t in result]
        return result

    @staticmethod
    def _should_coalesce(entries: List[str]) -> bool:
        if not entries or len(entries) < 20:
            return False
        word_counts = []
        for entry in entries:
            text = str(entry).strip()
            if not text:
                continue
            word_counts.append(len(text.split()))
        if not word_counts:
            return False
        avg_words = sum(word_counts) / len(word_counts)
        short_ratio = sum(1 for c in word_counts if c <= 2) / len(word_counts)
        return avg_words <= 2.0 and short_ratio >= 0.6

    @staticmethod
    def _normalize_entry(text: str) -> str:
        if text is None:
            return ""
        cleaned = " ".join(str(text).strip().split())
        if not cleaned:
            return ""
        alnum = sum(ch.isalnum() for ch in cleaned)
        if alnum == 0 and len(cleaned) <= 3:
            return ""
        if len(cleaned) >= 6:
            uniq = set(cleaned.lower())
            if len(uniq) <= 2 and (alnum / len(cleaned)) < 0.6:
                return ""
        return cleaned

    @classmethod
    def _coalesce_pseudo_text(cls, entries: List[str], chunk_tokens: Optional[int]) -> List[str]:
        if not chunk_tokens or chunk_tokens <= 0:
            return entries
        normalized = []
        for entry in entries:
            cleaned = cls._normalize_entry(entry)
            if cleaned:
                normalized.append(cleaned)
        if not normalized:
            return entries
        if not cls._should_coalesce(normalized):
            return normalized
        tokens: List[str] = []
        for entry in normalized:
            tokens.extend(entry.split())
        if not tokens:
            return normalized
        chunks: List[str] = []
        for idx in range(0, len(tokens), chunk_tokens):
            chunk = " ".join(tokens[idx : idx + chunk_tokens]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _inline_pseudo_text(sample: Dict) -> List[str]:
        entries: List[str] = []
        extra = sample.get("extra", {}) or {}
        for ctx in extra.get("context_evidence", []):
            if ctx:
                entries.append(str(ctx))
        for token in extra.get("ocr_tokens", []):
            if isinstance(token, dict):
                span = token.get("text", "")
            else:
                span = str(token)
            if span:
                entries.append(span)
        for caption in extra.get("captions", []):
            if caption:
                entries.append(caption)
        return entries

    @staticmethod
    def _load_image(path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        project_root = Path(__file__).resolve().parent.parent
        candidates = []
        try:
            candidates.append(Path(path))
        except Exception:
            return None
        dup_fix = {
            "documents/documents": "documents",
            "charts/charts": "charts",
            "images/images": "images",
            "pics/pics": "pics",
        }
        for dup, fix in dup_fix.items():
            if dup in path:
                try:
                    candidates.append(Path(path.replace(dup, fix, 1)))
                except Exception:
                    pass
        expanded: List[Path] = []
        for cand in candidates:
            expanded.append(cand)
            if not cand.is_absolute():
                expanded.append(project_root / cand)
        for cand in expanded:
            if cand.exists():
                try:
                    return Image.open(cand).convert("RGB")
                except Exception:
                    continue
        return None


def collate_fn(batch: List[Dict]) -> Dict:
    ids = [item["id"] for item in batch]
    clean = {
        "question": [item["clean"]["question"] for item in batch],
        "labels": [item["clean"]["labels"] for item in batch],
        "pseudo_text": [item["clean"]["pseudo_text"] for item in batch],
        "image_path": [item["clean"].get("image_path") for item in batch],
        "images": [item["clean"].get("image") for item in batch],
        "vision_tokens": batch[0]["clean"].get("vision_tokens"),
        "hidden_size": batch[0]["clean"].get("hidden_size"),
    }
    corrupted = {
        "question": [item["corrupted"]["question"] for item in batch],
        "labels": [item["corrupted"]["labels"] for item in batch],
        "pseudo_text": [item["corrupted"]["pseudo_text"] for item in batch],
        "image_path": [item["corrupted"].get("image_path") for item in batch],
        "images": [item["corrupted"].get("image") for item in batch],
        "vision_tokens": batch[0]["corrupted"].get("vision_tokens"),
        "hidden_size": batch[0]["corrupted"].get("hidden_size"),
    }
    return {"ids": ids, "clean": clean, "corrupted": corrupted}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pseudo_corpus(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Pseudo-text corpus not found: {path}")
    records: Dict[str, List[str]] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            pseudo = obj.get("pseudo_text", [])
            if doc_id:
                records[str(doc_id)] = pseudo
    return records
