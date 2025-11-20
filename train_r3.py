"""
Training entrypoint for the R^3 multimodal reasoning system.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional

import logging
import json
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from data_pipeline.datasets.textvqa import TextVQADataset
from data_pipeline.corruptions import (
    ImageCorruptionConfig,
    ImageCorruptor,
    PseudoTextCorruptionConfig,
    PseudoTextCorruptor,
)
from r3.r3_model import R3Model, R3ModelConfig
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from r3.retrieval_module import PseudoTextBuilder


class R3Dataset(Dataset):
    def __init__(
        self,
        base_dataset: TextVQADataset,
        vision_tokens: int,
        hidden_size: int,
        apply_corruption: bool = True,
        pseudo_builder: Optional[PseudoTextBuilder] = None,
        pseudo_text_drop_prob: float = 0.3,
        pseudo_corpus: Optional[Dict[str, List[str]]] = None,
        image_corruptor: Optional[ImageCorruptor] = None,
        pseudo_text_corruptor: Optional[PseudoTextCorruptor] = None,
    ) -> None:
        self.base = base_dataset
        self.vision_tokens = vision_tokens
        self.hidden_size = hidden_size
        self.apply_corruption = apply_corruption
        self.pseudo_builder = pseudo_builder or PseudoTextBuilder()
        self.pseudo_corpus = pseudo_corpus or {}
        self.image_corruptor = image_corruptor or ImageCorruptor(ImageCorruptionConfig())
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
        pseudo_text = self._inline_pseudo_text(sample)
        # Prefer offline pseudo-text corpus if provided
        if sample.get("id") in self.pseudo_corpus:
            pseudo_text = self.pseudo_corpus[sample["id"]]
        if not pseudo_text and self.pseudo_builder:
            pseudo_text = self.pseudo_builder.build(sample)
        corrupted_pseudo = (
            self.pseudo_text_corruptor(pseudo_text) if self.apply_corruption else pseudo_text
        )
        image = self._load_image(sample.get("image_path"))
        clean_image = image.copy() if image else None
        corrupted_image = self.image_corruptor(image) if (image and self.apply_corruption) else clean_image
        clean = {
            "question": sample["question"],
            "labels": sample.get("answer", "UNKNOWN"),
            "pseudo_text": pseudo_text,
            "image_path": sample.get("image_path"),
            "image": clean_image,
            "vision_tokens": self.vision_tokens,
            "hidden_size": self.hidden_size,
        }
        corrupted_branch = {
            "question": sample["question"],  # 问题保持不变，仅模拟伪文本/视觉缺失
            "labels": sample.get("answer", "UNKNOWN"),
            "pseudo_text": corrupted_pseudo,
            "image_path": sample.get("image_path"),
            "image": corrupted_image,
            "vision_tokens": self.vision_tokens,
            "hidden_size": self.hidden_size,
        }
        return {"id": sample["id"], "clean": clean, "corrupted": corrupted_branch}

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
    def _load_image(path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
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


class CurriculumScheduler(TrainerCallback):
    """
    Epoch-wise curriculum that ramps up corruption + consistency weight.
    """

    def __init__(self, warmup_drop: float = 0.1, hard_drop: float = 0.4) -> None:
        super().__init__()
        self.warmup_drop = warmup_drop
        self.hard_drop = hard_drop

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        epoch = int(state.epoch or 0)
        if epoch <= 1:
            dropout = self.warmup_drop
            lambda_c = 0.0
        else:
            dropout = self.hard_drop
            lambda_c = 1.0

        if hasattr(model, "simulator"):
            model.simulator.config.image_dropout = dropout
            model.simulator.config.text_dropout = dropout
        if hasattr(model, "config"):
            model.config.lambda_consistency = lambda_c


class R3Trainer(Trainer):
    """
    Custom Trainer that computes dual-branch loss inline.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device
        tokenizer = model.base_vlm.tokenizer
        max_length = getattr(model.config, "max_seq_length", 1024)

        clean_split = inputs["clean"]
        corrupted_split = inputs["corrupted"]

        clean_tokens, clean_pseudo = self._tokenize_branch(tokenizer, clean_split, max_length, device)
        corrupted_tokens, corrupted_pseudo = self._tokenize_branch(tokenizer, corrupted_split, max_length, device)
        clean_vision = self._get_vision_embeddings(model, clean_split, device)
        corrupted_vision = self._get_vision_embeddings(model, corrupted_split, device)

        with torch.no_grad():
            teacher_out = model(
                input_ids=clean_tokens["input_ids"],
                attention_mask=clean_tokens["attention_mask"],
                pixel_values=clean_vision,
                labels=clean_tokens["labels"],
                pseudo_text=clean_pseudo,
                is_clean_branch=True,
            )

        student_out = model(
            input_ids=corrupted_tokens["input_ids"],
            attention_mask=corrupted_tokens["attention_mask"],
            pixel_values=corrupted_vision,
            labels=corrupted_tokens["labels"],
            pseudo_text=corrupted_pseudo,
            is_clean_branch=False,
        )

        loss_task = student_out["loss"]
        loss_consistency = F.mse_loss(
            teacher_out["pooled_hidden"].detach(),
            student_out["pooled_hidden"],
        )
        lambda_c = getattr(model.config, "lambda_consistency", 0.0)
        total_loss = loss_task + lambda_c * loss_consistency

        outputs = {
            "task_loss": loss_task.detach(),
            "consistency_loss": loss_consistency.detach(),
            "retrieval": student_out.get("retrieval"),
        }
        if return_outputs:
            return total_loss, outputs
        return total_loss

    @staticmethod
    def _tokenize_branch(tokenizer, split: Dict, max_length: int, device: torch.device):
        questions = split["question"]
        labels_text = split.get("labels", [""] * len(questions))
        pseudo_text = split.get("pseudo_text", [[] for _ in questions])

        prompts = [
            f"{R3Trainer._format_pseudo_text(pseudo)}\nQuestion: {q}\nAnswer:" if R3Trainer._format_pseudo_text(pseudo) else f"Question: {q}\nAnswer:"
            for q, pseudo in zip(questions, pseudo_text)
        ]
        prompt_tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        full_text = [f"{p} {label}".strip() for p, label in zip(prompts, labels_text)]
        text_tokens = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}

        labels = text_tokens["input_ids"].clone()
        labels[text_tokens["attention_mask"] == 0] = -100
        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)
        for idx, length in enumerate(prompt_lengths):
            labels[idx, : length.item()] = -100
        text_tokens["labels"] = labels
        return text_tokens, pseudo_text

    @staticmethod
    def _format_pseudo_text(pseudo: List[str]) -> str:
        return "\n".join([p for p in pseudo if p])

    @staticmethod
    def _get_vision_embeddings(model, split: Dict, device: torch.device) -> torch.Tensor:
        """
        Prefer base vision tower; fallback to precomputed embeddings if provided.
        """
        images = split.get("images")
        if images:
            processed = [img if img is not None else Image.new("RGB", (224, 224), color="black") for img in images]
            return model.base_vlm.encode_images(
                images=processed,
                vision_tokens=split.get("vision_tokens", 16),
                hidden_size=split.get("hidden_size", model.config.hidden_size),
                device=device,
            )
        image_paths = split.get("image_path")
        if image_paths:
            processed = [p if p else torch.zeros(3, 224, 224) for p in image_paths]
            return model.base_vlm.encode_images(
                images=processed,
                vision_tokens=split.get("vision_tokens", 16),
                hidden_size=split.get("hidden_size", model.config.hidden_size),
                device=device,
            )
        raise ValueError("No vision input found for this batch.")


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
    pseudo_corpus = load_pseudo_corpus(dataset_section.get("pseudo_corpus"))
    base_dataset = TextVQADataset(dataset_root, split=split)  # 构造基础数据集
    logging.info("Dataset initialized: %s split=%s size=%d", dataset_root, split, len(base_dataset))

    train_dataset = R3Dataset(
        base_dataset,
        vision_tokens=cfg["model"].get("vision_tokens", 16),
        hidden_size=cfg["model"].get("hidden_size", 4096),
        apply_corruption=True,
        pseudo_builder=PseudoTextBuilder(),
        pseudo_corpus=pseudo_corpus,
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
    model = R3Model(model_cfg)
    logging.info("Model initialized with backbone %s", model_cfg.model_name)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=training_section.get("epochs", 1),
        per_device_train_batch_size=dataset_section.get("batch_size", 2),
        learning_rate=training_section.get("learning_rate", 2e-4),
        weight_decay=training_section.get("weight_decay", 0.05),
        logging_steps=training_section.get("log_interval", 10),
        warmup_ratio=training_section.get("warmup_ratio", 0.05),
        report_to="none",
        remove_unused_columns=False,
        bf16=model_cfg.bf16 and torch.cuda.is_available(),
    )

    trainer = R3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[CurriculumScheduler()],
    )
    trainer.train()
    logging.info("Training finished. Checkpoints at %s", args.output_dir)


if __name__ == "__main__":
    main()
