"""
Base dataset definition shared across PMC benchmarks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BasePMCDataset:
    """
    Implements a light-weight interface to expose both full and corrupted samples.
    Subclasses are expected to override `_load_raw_item` to parse dataset-specific
    annotations.
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.samples = self._build_index()

    def _build_index(self) -> List[Dict]:
        raise NotImplementedError

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_meta = self.samples[idx]
        raw_item = self._load_raw_item(sample_meta)
        return {
            "id": sample_meta["id"],
            "question": raw_item["question"],
            "answer": raw_item.get("answer"),
            "image_path": raw_item.get("image_path"),
            "extra": raw_item.get("extra", {}),
        }

    @staticmethod
    def collate(items: List[Dict]) -> Dict:
        batch = {k: [item[k] for item in items] for k in items[0]}
        return batch
