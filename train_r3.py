"""
Training entrypoint for the R^3 multimodal reasoning system.
"""
from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path
from typing import Dict, List, Optional

import logging
import json
import random
import os
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):  # type: ignore[attr-defined]
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

from data_pipeline.datasets import BasePMCDataset, create_dataset, detect_dataset_type
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
import torch.distributed as dist
from accelerate import Accelerator

# Patch accelerate unwrap_model signature for older versions (no keep_torch_compile)
if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
    _orig_unwrap = Accelerator.unwrap_model

    def _compat_unwrap(self, model, keep_fp32_wrapper=False, **kwargs):
        return _orig_unwrap(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = _compat_unwrap

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


class MultiTaskDataset(Dataset):
    """
    Simple random-mixing multi-task dataset.
    Each __getitem__ ignores idx and samples one of the sub-datasets by weight.
    """

    def __init__(self, datasets: List[R3Dataset], weights: Optional[List[float]] = None) -> None:
        self.datasets = datasets
        if weights is None:
            weights = [1.0] * len(datasets)
        total = sum(weights)
        self.probs = [w / total for w in weights]
        self.lengths = [len(ds) for ds in datasets]

    def __len__(self) -> int:
        # approximate length: sum of sub-datasets
        return sum(self.lengths)

    def __getitem__(self, idx: int) -> Dict:
        choice = random.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]
        ds = self.datasets[choice]
        ridx = random.randint(0, len(ds) - 1)
        return ds[ridx]


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


class LossLogger(TrainerCallback):
    """
    Logs loss metrics to python logger every logging step on rank0.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Only log from rank0 to avoid duplication
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank > 0:
            return
        msg = []
        if "loss" in logs:
            msg.append(f"loss={logs['loss']:.4f}")
        if "task_loss" in logs:
            msg.append(f"task_loss={float(logs['task_loss']):.4f}")
        if "consistency_loss" in logs:
            msg.append(f"consistency_loss={float(logs['consistency_loss']):.4f}")
        if "learning_rate" in logs:
            msg.append(f"lr={logs['learning_rate']:.6f}")
        if "epoch" in logs:
            msg.append(f"epoch={logs['epoch']:.4f}")
        if msg:
            self.logger.info(" | ".join(msg))


class R3Trainer(Trainer):
    """
    Custom Trainer that computes dual-branch loss inline.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Unwrap DDP for attribute access while keeping forward on the wrapped model.
        base_model = model.module if hasattr(model, "module") else model
        device = next(base_model.parameters()).device
        tokenizer = base_model.base_vlm.tokenizer
        max_length = getattr(base_model.config, "max_seq_length", 1024)

        clean_split = inputs["clean"]
        corrupted_split = inputs["corrupted"]

        clean_tokens, clean_pseudo = self._tokenize_branch(tokenizer, clean_split, max_length, device)
        corrupted_tokens, corrupted_pseudo = self._tokenize_branch(tokenizer, corrupted_split, max_length, device)
        clean_vision = self._get_vision_embeddings(base_model, clean_split, device)
        corrupted_vision = self._get_vision_embeddings(base_model, corrupted_split, device)

        with torch.no_grad():
            teacher_out = model(
                input_ids=clean_tokens["input_ids"],
                attention_mask=clean_tokens["attention_mask"],
                pixel_values=clean_vision,
                labels=None,
                pseudo_text=clean_pseudo,
                is_clean_branch=True,
            )

        student_out = model(
            input_ids=corrupted_tokens["input_ids"],
            attention_mask=corrupted_tokens["attention_mask"],
            pixel_values=corrupted_vision,
            labels=None,
            pseudo_text=corrupted_pseudo,
            is_clean_branch=False,
        )
        # Ensure loss inputs are float32 to avoid Half/Float mismatch in backward.
        vision_tokens = corrupted_vision.size(1)
        student_logits = student_out["logits"].float()
        loss_task = self._causal_ce(student_logits, corrupted_tokens["labels"], vision_tokens)
        loss_consistency = F.mse_loss(
            teacher_out["pooled_hidden"].detach().float(),
            student_out["pooled_hidden"].float(),
        )
        lambda_c = getattr(base_model.config, "lambda_consistency", 0.0)
        # Ensure loss is in float32 to avoid backward dtype issues under mixed precision.
        loss_task = loss_task.float()
        loss_consistency = loss_consistency.float()
        total_loss = (loss_task + lambda_c * loss_consistency).float()

        outputs = {
            "task_loss": loss_task.detach(),
            "consistency_loss": loss_consistency.detach(),
            "retrieval": student_out.get("retrieval"),
        }
        if return_outputs:
            return total_loss, outputs
        return total_loss

    @staticmethod
    def _causal_ce(logits: torch.Tensor, labels: torch.Tensor, vision_tokens: int = 0) -> torch.Tensor:
        """
        Manual causal LM loss with label shift, ignoring -100.
        """
        labels = labels.to(logits.device)
        if vision_tokens > 0:
            labels = F.pad(labels, (0, vision_tokens), value=-100)
        # Align lengths if still mismatched (truncate or pad labels).
        seq_len = logits.size(1)
        if labels.size(1) < seq_len:
            pad = seq_len - labels.size(1)
            labels = F.pad(labels, (0, pad), value=-100)
        elif labels.size(1) > seq_len:
            labels = labels[:, :seq_len]
        # Shift: predict token t using logits at t-1.
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

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
    parser.add_argument("--log_file", type=Path, default=None, help="Optional path to save training logs.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max training steps (overrides epochs).")
    parser.add_argument("--quick_eval_every", type=int, default=None, help="If set, run a 1-sample quick eval every N logging events on rank0.")
    parser.add_argument("--log_interval", type=int, default=None, help="Override training.log_interval for logging_steps.")
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

    # 运行示例（中文说明）:
    # 单卡训练:
    # python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
    #
    # 多卡训练 (torchrun 自动启用 DDP):
    # torchrun --nproc_per_node=4 train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file, mode="w"))
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )
    logging.info("Loading config from %s", args.config)
    cfg = load_yaml(args.config)  # 读取 YAML 配置
    dataset_section = cfg.get("dataset", {})
    pseudo_corpus = load_pseudo_corpus(dataset_section.get("pseudo_corpus"))
    logging.info("Config loaded. Preparing datasets...")

    def build_single_dataset(section: Dict) -> R3Dataset:
        ds_root = Path(section["root"])
        ds_split = section.get("split", "train")
        ds_type = section.get("type", "textvqa")
        if ds_type == "auto":
            ds_type = detect_dataset_type(ds_root)
        base_ds = create_dataset(ds_type, ds_root, split=ds_split)
        logging.info(
            "Dataset initialized: type=%s root=%s split=%s size=%d",
            ds_type,
            ds_root,
            ds_split,
            len(base_ds),
        )
        # Corruption configs
        apply_corr = section.get("apply_corruption", True)
        pt_drop = section.get("pseudo_text_drop_prob", 0.3)
        img_corr_cfg = section.get("image_corruption", {})
        image_corruptor = ImageCorruptor(
            ImageCorruptionConfig(
                occlusion_prob=img_corr_cfg.get("occlusion_prob", 0.5),
                occlusion_ratio=img_corr_cfg.get("occlusion_ratio", 0.25),
                blur_prob=img_corr_cfg.get("blur_prob", 0.5),
                blur_radius=img_corr_cfg.get("blur_radius", 3.0),
            )
        )
        pseudo_text_corruptor = PseudoTextCorruptor(
            PseudoTextCorruptionConfig(drop_prob=pt_drop)
        )
        return R3Dataset(
            base_ds,
            vision_tokens=cfg["model"].get("vision_tokens", 16),
            hidden_size=cfg["model"].get("hidden_size", 4096),
            apply_corruption=apply_corr,
            pseudo_builder=PseudoTextBuilder(),
            pseudo_corpus=pseudo_corpus,
            image_corruptor=image_corruptor,
            pseudo_text_corruptor=pseudo_text_corruptor,
            pseudo_text_drop_prob=pt_drop,
        )

    multi_cfg = dataset_section.get("multi")
    if multi_cfg:
        datasets: List[R3Dataset] = []
        weights: List[float] = []
        for entry in multi_cfg:
            datasets.append(build_single_dataset(entry))
            weights.append(entry.get("weight", 1.0))
        train_dataset = MultiTaskDataset(datasets, weights=weights)
        logging.info("Multi-task dataset initialized with %d subsets", len(datasets))
    else:
        train_dataset = build_single_dataset(dataset_section)
        logging.info("Single dataset initialized.")
    # Cache one clean sample for quick eval
    quick_eval_sample = collate_fn([train_dataset[0]]) if len(train_dataset) > 0 else None

    model_section = cfg.get("model", {})
    training_section = cfg.get("training", {})
    if args.log_interval is not None:
        training_section["log_interval"] = args.log_interval
    model_cfg = R3ModelConfig(
        model_name=model_section.get("name", "Qwen/Qwen3-VL-8B-Instruct"),
        lora_rank=model_section.get("lora_rank", 32),
        lora_alpha=model_section.get("lora_alpha", 16),
        hidden_size=model_section.get("hidden_size", 4096),
        bf16=model_section.get("bf16", True),
        load_in_4bit=model_section.get("load_in_4bit", False),
        load_in_8bit=model_section.get("load_in_8bit", False),
        device_map=model_section.get("device_map"),
        low_cpu_mem_usage=model_section.get("low_cpu_mem_usage", True),
        gradient_checkpointing=model_section.get("gradient_checkpointing", True),
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
        retrieval_cache_path=model_section.get("retrieval_cache_path"),
        retrieval_corpus_path=model_section.get("retrieval_corpus_path"),
    )
    logging.info("Loading model: %s (provider=%s)", model_cfg.model_name, model_cfg.provider)
    model = R3Model(model_cfg)
    logging.info("Model initialized with backbone %s", model_cfg.model_name)
    # Mark as model-parallel if hf_device_map exists to prevent Trainer from moving to a single GPU
    try:
        if hasattr(model, "base_vlm") and hasattr(model.base_vlm.model, "hf_device_map"):
            model.is_model_parallel = True
            model.is_parallelizable = True
            model.model_parallel = True
            model.hf_device_map = model.base_vlm.model.hf_device_map
    except Exception:
        pass
    if hasattr(model, "base_vlm") and hasattr(model.base_vlm, "model") and hasattr(model.base_vlm.model, "config"):
        model.base_vlm.model.config.use_cache = False
        if hasattr(model.base_vlm.model, "gradient_checkpointing_disable"):
            model.base_vlm.model.gradient_checkpointing_disable()

    ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=training_section.get("epochs", 1),
        max_steps=args.max_steps if args.max_steps is not None else training_section.get("max_steps"),
        per_device_train_batch_size=dataset_section.get("batch_size", 2),
        gradient_accumulation_steps=training_section.get("grad_accum_steps", 1),
        learning_rate=training_section.get("learning_rate", 2e-4),
        weight_decay=training_section.get("weight_decay", 0.05),
        logging_steps=training_section.get("log_interval", 10),
        warmup_ratio=training_section.get("warmup_ratio", 0.05),
        report_to=training_section.get("report_to", "tensorboard"),
        logging_dir=str(training_section.get("logging_dir", args.output_dir / "logs")),
        remove_unused_columns=False,
        bf16=model_cfg.bf16 and torch.cuda.is_available(),
        fp16=training_section.get("fp16", False) and torch.cuda.is_available(),
        gradient_checkpointing=training_section.get("grad_checkpoint", False),
        dataloader_num_workers=training_section.get("num_workers", 0),
        ddp_find_unused_parameters=False if ddp else None,
        ddp_backend="nccl" if ddp else None,
        save_safetensors=False,  # avoid safetensors shared-memory save errors due to shared embeddings
        # place_model_on_device not available in this transformers version; we rely on model.hf_device_map
    )

    # Quick eval callback: runs on rank0 every N logging events if enabled
    class QuickEvalCallback(TrainerCallback):
        def __init__(self, sample: Optional[Dict], every: Optional[int], tokenizer, processor, logger):
            self.sample = sample
            self.every = every
            self.tokenizer = tokenizer
            self.processor = processor
            self.logger = logger

        def on_log(self, args, state, control, **kwargs):
            if self.sample is None or self.every is None:
                return
            if state.global_step == 0 or state.global_step % self.every != 0:
                return
            # rank check
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
            if local_rank not in (-1, 0):
                return
            model = kwargs.get("model", None)
            if model is None:
                return
            model_was_training = model.training
            model.eval()
            try:
                clean = self.sample["clean"]
                q = clean["question"][0]
                img = clean["images"][0] if clean["images"][0] is not None else None
                img_path = None
                if img is None:
                    paths = clean.get("image_path")
                    if isinstance(paths, list) and paths and paths[0]:
                        from PIL import Image
                        # Try common locations to resolve relative path issues.
                        candidates = [Path(paths[0])]
                        if "documents/documents" in paths[0]:
                            candidates.append(Path(paths[0].replace("documents/documents", "documents", 1)))
                        if dataset_section.get("root"):
                            root = Path(dataset_section["root"])
                            candidates.append(root / paths[0])
                            candidates.append(root / Path(paths[0]).name)
                            # also try fixing duplicated segment under root
                            if "documents/documents" in paths[0]:
                                candidates.append(root / paths[0].replace("documents/documents", "documents", 1))
                        loaded = False
                        for c in candidates:
                            if c.exists():
                                img_path = str(c)
                                img = Image.open(c).convert("RGB")
                                loaded = True
                                break
                        if not loaded:
                            raise FileNotFoundError(f"Image not found for quick_eval: {paths[0]}, tried {candidates}")
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": q + "\nPlease answer with the short answer only."}],
                    }
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    gen = model.base_vlm.model.generate(  # type: ignore
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1,
                        temperature=0.0,
                        top_p=1.0,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen_ids = gen[0][input_len:]
                pred = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                tgt = clean.get("labels", [""])[0] if isinstance(clean.get("labels"), list) else ""
                self.logger.info(f"[quick_eval step={state.global_step}] q={q} | pred={pred} | tgt={tgt} | img={img_path}")
            except Exception as e:
                self.logger.warning(f"[quick_eval] failed: {e}")
            finally:
                if model_was_training:
                    model.train()

    callbacks = [CurriculumScheduler(), LossLogger(logging.getLogger())]
    # build quick eval callback
    if args.quick_eval_every is not None and quick_eval_sample is not None and hasattr(model, "base_vlm"):
        callbacks.append(
            QuickEvalCallback(
                sample=quick_eval_sample,
                every=args.quick_eval_every,
                tokenizer=model.base_vlm.tokenizer,
                processor=model.base_vlm.processor,
                logger=logging.getLogger(),
            )
        )

    trainer = R3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )
    logging.info("Starting training for %s epochs", training_args.num_train_epochs)
    trainer.train()
    logging.info("Training finished. Checkpoints at %s", args.output_dir)


if __name__ == "__main__":
    main()
