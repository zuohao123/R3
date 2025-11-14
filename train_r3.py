"""
Training entrypoint for the R^3 multimodal reasoning system.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import logging
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data_pipeline.datasets.textvqa import TextVQADataset
from data_pipeline.vision_encoder import VisionEncoder
from r3.r3_model import R3Model, R3ModelConfig


class R3Dataset(Dataset):
    def __init__(
        self,
        base_dataset: TextVQADataset,
        vision_encoder: Optional[VisionEncoder],
        vision_tokens: int,
        hidden_size: int,
        apply_corruption: bool = True,
    ) -> None:
        self.base = base_dataset
        self.vision_encoder = vision_encoder
        self.vision_tokens = vision_tokens
        self.hidden_size = hidden_size
        self.apply_corruption = apply_corruption

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        sample = copy.deepcopy(self.base[idx])
        corrupted_question = self._corrupt_question(sample["question"]) if self.apply_corruption else sample["question"]
        pseudo_text = self._inline_pseudo_text(sample)
        vision_feats = self._vision_features(sample.get("image_path"))
        clean = {
            "question": sample["question"],
            "vision_embeddings": vision_feats,
            "labels": sample.get("answer", "UNKNOWN"),
            "pseudo_text": pseudo_text,
        }
        corrupted_branch = {
            "question": corrupted_question,
            "vision_embeddings": vision_feats.clone(),
            "labels": sample.get("answer", "UNKNOWN"),
            "pseudo_text": pseudo_text,
        }
        return {"id": sample["id"], "clean": clean, "corrupted": corrupted_branch}

    def _vision_stub(self, image_path: Optional[str]) -> torch.Tensor:
        seed_source = image_path or "unknown"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8]
        generator = torch.Generator()
        generator.manual_seed(int(digest, 16))
        return torch.randn(self.vision_tokens, self.hidden_size, generator=generator)

    def _vision_features(self, image_path: Optional[str]) -> torch.Tensor:
        if self.vision_encoder and image_path:
            try:
                feats = self.vision_encoder.encode(image_path, self.vision_tokens, self.hidden_size)
                if feats.dim() == 3:
                    feats = feats.squeeze(0)
                return feats
            except Exception:
                pass
        return self._vision_stub(image_path)

    @staticmethod
    def _inline_pseudo_text(sample: Dict) -> List[str]:
        entries: List[str] = []
        extra = sample.get("extra", {}) or {}
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
    def _corrupt_question(question: str) -> str:
        tokens = question.split()
        if len(tokens) <= 2:
            return question
        drop = max(1, len(tokens) // 4)
        for _ in range(drop):
            if tokens:
                tokens.pop(torch.randint(0, len(tokens), (1,)).item())
        return " ".join(tokens)


def collate_fn(batch: List[Dict]) -> Dict:
    ids = [item["id"] for item in batch]
    clean = {
        "question": [item["clean"]["question"] for item in batch],
        "vision_embeddings": torch.stack([item["clean"]["vision_embeddings"] for item in batch], dim=0),
        "labels": [item["clean"]["labels"] for item in batch],
        "pseudo_text": [item["clean"]["pseudo_text"] for item in batch],
    }
    corrupted = {
        "question": [item["corrupted"]["question"] for item in batch],
        "vision_embeddings": torch.stack([item["corrupted"]["vision_embeddings"] for item in batch], dim=0),
        "labels": [item["corrupted"]["labels"] for item in batch],
        "pseudo_text": [item["corrupted"]["pseudo_text"] for item in batch],
    }
    return {"ids": ids, "clean": clean, "corrupted": corrupted}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R^3 on PMC benchmarks.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/r3"))
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Loading config from %s", args.config)
    cfg = load_yaml(args.config)  # 读取 YAML 配置
    dataset_section = cfg.get("dataset", {})
    dataset_root = Path(dataset_section["root"])
    split = dataset_section.get("split", "train")
    base_dataset = TextVQADataset(dataset_root, split=split)  # 构造基础数据集
    logging.info("Dataset initialized: %s split=%s size=%d", dataset_root, split, len(base_dataset))
    vision_cfg = cfg.get("vision", {})
    vision_encoder = None
    if vision_cfg.get("encoder"):
        vision_encoder = VisionEncoder(
            model_name=vision_cfg.get("encoder", "openai/clip-vit-large-patch14"),
            device=vision_cfg.get("device", "cpu"),
            cache_size=vision_cfg.get("cache_size", 256),
        )
        logging.info(
            "Vision encoder enabled: %s (device=%s)",
            vision_cfg.get("encoder"),
            vision_cfg.get("device", "cpu"),
        )
    else:
        logging.info("Vision encoder disabled; falling back to deterministic stubs.")

    train_dataset = R3Dataset(
        base_dataset,
        vision_encoder,
        vision_tokens=cfg["model"].get("vision_tokens", 16),
        hidden_size=cfg["model"].get("hidden_size", 4096),
        apply_corruption=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_section.get("batch_size", 2),
        shuffle=True,
        num_workers=dataset_section.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    logging.info(
        "DataLoader prepared (batch_size=%d, num_workers=%d)",
        dataset_section.get("batch_size", 2),
        dataset_section.get("num_workers", 0),
    )

    model_section = cfg.get("model", {})
    training_section = cfg.get("training", {})
    model_cfg = R3ModelConfig(
        model_name=model_section.get("name", "Qwen/Qwen3-VL-8B-Instruct"),
        lora_rank=model_section.get("lora_rank", 32),
        lora_alpha=model_section.get("lora_alpha", 16),
        hidden_size=model_section.get("hidden_size", 4096),
        provider=model_section.get("provider", "huggingface"),
        token=model_section.get("token"),
        cache_dir=model_section.get("cache_dir"),
        revision=model_section.get("revision"),
        local_files_only=model_section.get("local_files_only", False),
        enable_corruption=model_section.get("enable_corruption", True),
        enable_retrieval=model_section.get("enable_retrieval", True),
        enable_prefix=model_section.get("enable_prefix", True),
        enable_memory=model_section.get("enable_memory", True),
        enable_imputation=model_section.get("enable_imputation", True),
        enable_consistency=model_section.get("enable_consistency", True),
        lambda_consistency=model_section.get("lambda_consistency", 0.3),
        top_k=model_section.get("top_k", 3),
    )
    model = R3Model(model_cfg).to(args.device)
    logging.info("Model initialized on %s with backbone %s", args.device, model_cfg.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_section.get("learning_rate", 2e-4), weight_decay=training_section.get("weight_decay", 0.05))

    model.train()
    total_epochs = training_section.get("epochs", 1)
    for epoch in range(total_epochs):
        logging.info("Starting epoch %d/%d", epoch + 1, total_epochs)
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            outputs["loss"].backward()
            optimizer.step()
            if step % training_section.get("log_interval", 10) == 0:
                print(
                    f"[Epoch {epoch}] step {step} "
                    f"loss: {outputs['loss'].item():.4f} "
                    f"task: {outputs['task_loss'].item():.4f} "
                    f"consistency: {outputs['consistency_loss'].item():.4f}"
                )
        logging.info(
            "Epoch %d finished. last_loss=%.4f task=%.4f",
            epoch + 1,
            outputs["loss"].item(),
            outputs["task_loss"].item(),
        )
    # todo 先注释掉
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # model.save_checkpoint(args.output_dir)  # 保存 LoRA Adapter + Tokenizer + Reconstructor
    logging.warning("Checkpoint saving is currently disabled (todo). Intended path: %s", args.output_dir)


if __name__ == "__main__":
    main()
