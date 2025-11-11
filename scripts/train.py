"""
Command-line training entrypoint for the R^3 pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Type

import torch
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.corruption.simulator import CorruptionConfig
from data_pipeline.dataloaders.pmc_datamodule import PMCDatamodule
from data_pipeline.datasets.chartqa import ChartQADataset
from data_pipeline.datasets.docvqa import DocVQADataset
from data_pipeline.datasets.textvqa import TextVQADataset
from data_pipeline.features import SimpleFeatureBuilder
from model.builders import build_r3_model
from retrieval.query.pseudo_text_retriever import PseudoTextRetriever, RetrievalConfig
from train.trainer import Trainer

DATASET_REGISTRY: Dict[str, Type] = {
    "textvqa": TextVQADataset,
    "chartqa": ChartQADataset,
    "docvqa": DocVQADataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R^3 on PMC benchmarks.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config.")
    parser.add_argument("--index", type=Path, default=None, help="Optional retrieval corpus (JSONL).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset_factory(name: str, root: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{name}'. Options: {list(DATASET_REGISTRY.keys())}")
    dataset_cls = DATASET_REGISTRY[name]
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    def factory(split: str = "train"):
        return dataset_cls(root=root_path, split=split)

    return factory


def load_retrieval_index(path: Path | None) -> list[Dict]:
    if path is None or not path.exists():
        return []
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    dataset_factory = build_dataset_factory(dataset_cfg.get("name", "textvqa"), dataset_cfg.get("root", "."))

    corruption_cfg = CorruptionConfig(**cfg.get("corruption", {}))
    model_cfg = cfg.get("model", {})
    feature_builder = SimpleFeatureBuilder(
        hidden_size=model_cfg.get("hidden_size", 256),
        question_length=model_cfg.get("question_length", 32),
        vision_tokens=model_cfg.get("vision_tokens", 16),
    )

    datamodule = PMCDatamodule(
        dataset_factory=dataset_factory,
        batch_size=args.batch_size or dataset_cfg.get("batch_size", 4),
        num_workers=args.num_workers or dataset_cfg.get("num_workers", 2),
        corruption_config=corruption_cfg,
        feature_builder=feature_builder,
    )

    retriever_cfg = RetrievalConfig(**cfg.get("retrieval", {}))
    retriever = PseudoTextRetriever(retriever_cfg)
    corpus = load_retrieval_index(args.index)
    if corpus:
        retriever.load_index(corpus)
        print(f"[Retrieval] Loaded {len(corpus)} pseudo-text entries from {args.index}")
    else:
        print("[Retrieval] No index provided, proceeding with empty corpus.")

    model = build_r3_model(model_cfg)
    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        retriever=retriever,
        device=args.device,
        log_interval=cfg.get("training", {}).get("log_interval", 20),
    )

    training_cfg = cfg.get("training", {})
    epochs = args.epochs or training_cfg.get("epochs", 1)
    print(f"[Training] Starting for {epochs} epoch(s) on device {args.device}")
    trainer.fit(
        epochs=epochs,
        optimizer_cfg=training_cfg,
        train_split=dataset_cfg.get("split", "train"),
    )


if __name__ == "__main__":
    main()
