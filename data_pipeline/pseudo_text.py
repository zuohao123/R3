"""
Utility helpers for constructing and caching pseudo-text corpora.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from r3.retrieval_module import PseudoTextBuilder


def build_corpus(
    dataset,
    builder: Optional[PseudoTextBuilder] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Iterates over a dataset (BasePMCDataset compatible) and materializes
    structured pseudo-text entries that can be fed into HybridRetriever.
    """
    builder = builder or PseudoTextBuilder()
    corpus: List[Dict] = []
    total = len(dataset)
    upper = min(total, limit) if limit else total
    for idx in range(upper):
        sample = dataset[idx]
        # 通过 PseudoTextBuilder 将 OCR/Caption/Tables 转为统一格式
        entries = builder.build(sample)
        if not entries:
            continue
        corpus.append(
            {
                "doc_id": sample["id"],
                "pseudo_text": entries,
                "metadata": {
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "image_path": sample.get("image_path", ""),
                    "split_index": idx,
                },
            }
        )
    return corpus


def save_corpus(corpus: Sequence[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 以 JSONL 方式保存，方便增量 append 与快速读取
    with path.open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> List[Dict]:
    path = Path(path)
    corpus: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            corpus.append(json.loads(line))
    return corpus
