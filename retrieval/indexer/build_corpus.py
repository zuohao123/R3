"""
Utility to textualize multimodal corpora and persist embeddings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def build_corpus(samples: Iterable[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            entry = {
                "id": sample["id"],
                "pseudo_text": sample["pseudo_text"],
                "answer": sample.get("answer"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return corpus_path
