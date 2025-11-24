"""
Dataset registry and helpers for PMC benchmarks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from .base_dataset import BasePMCDataset
from .chartqa import ChartQADataset
from .docvqa import DocVQADataset
from .infovqa import InfoVQADataset
from .mp_docvqa import MPDocVQADataset
from .mtvqa import MTVQADataset
from .slidevqa import SlideVQADataset
from .textvqa import TextVQADataset

DATASET_REGISTRY: Dict[str, Type[BasePMCDataset]] = {
    "textvqa": TextVQADataset,
    "mp_docvqa": MPDocVQADataset,
    "infovqa": InfoVQADataset,
    "chartqa": ChartQADataset,
    "docvqa": DocVQADataset,
    "slidevqa": SlideVQADataset,
    "mtvqa": MTVQADataset,
}


def create_dataset(dataset_type: str, root: Path, split: str) -> BasePMCDataset:
    key = dataset_type.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return DATASET_REGISTRY[key](root, split=split)


def detect_dataset_type(root: Path) -> str:
    """
    Auto-detect dataset type from directory structure/filenames.
    """
    root = Path(root)
    markers = [
        ("textvqa", [f"textvqa_{s}.json" for s in ("train", "val", "test")]),
        ("mp_docvqa", [f"mp_docvqa_{s}.json" for s in ("train", "val", "test")]),
        ("infovqa", [f"infovqa_{s}.json" for s in ("train", "val", "test")]),
        ("chartqa", [f"chartqa_{s}.json" for s in ("train", "val", "test")]),
        ("docvqa", [f"docvqa_{s}.json" for s in ("train", "val", "test")]),
        ("slidevqa", [f"slidevqa_{s}.json" for s in ("train", "val", "test")]),
        ("mtvqa", [f"mtvqa_{s}.json" for s in ("train", "val", "test")]),
    ]
    for dataset_key, files in markers:
        for fname in files:
            if (root / fname).exists():
                return dataset_key
    name = root.name.lower()
    for dataset_key in DATASET_REGISTRY:
        if dataset_key.replace("_", "") in name or dataset_key in name:
            return dataset_key
    return "textvqa"


__all__ = [
    "BasePMCDataset",
    "ChartQADataset",
    "DocVQADataset",
    "InfoVQADataset",
    "MPDocVQADataset",
    "MTVQADataset",
    "SlideVQADataset",
    "TextVQADataset",
    "DATASET_REGISTRY",
    "create_dataset",
    "detect_dataset_type",
]
