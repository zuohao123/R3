"""
Training entrypoint for the R^3 multimodal reasoning system.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, List, Optional

import logging
import random
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# Torch/Transformers compatibility:
# - Some transformers builds call `torch.is_autocast_enabled(device_type)`, but older torch only supports
#   `torch.is_autocast_enabled()` with no arguments.
try:  # pragma: no cover
    _orig_is_autocast_enabled = torch.is_autocast_enabled
    try:
        _orig_is_autocast_enabled("cuda")
    except TypeError:
        def _compat_is_autocast_enabled(device_type=None):  # type: ignore[override]
            return _orig_is_autocast_enabled()
        torch.is_autocast_enabled = _compat_is_autocast_enabled  # type: ignore[assignment]
except Exception:
    pass
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
from r3.data_utils import R3Dataset, collate_fn, load_yaml, load_pseudo_corpus
import torch.distributed as dist
from accelerate import Accelerator

# Patch accelerate unwrap_model signature for older versions (no keep_torch_compile)
if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
    _orig_unwrap = Accelerator.unwrap_model

    def _compat_unwrap(self, model, keep_fp32_wrapper=False, **kwargs):
        return _orig_unwrap(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = _compat_unwrap

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
        if "lambda_consistency" in logs:
            msg.append(f"lambda_c={float(logs['lambda_consistency']):.4f}")
        if "labels_valid" in logs:
            msg.append(f"labels_valid={int(float(logs['labels_valid']))}")
        if "prompt_len" in logs:
            msg.append(f"prompt_len={float(logs['prompt_len']):.1f}")
        if "pseudo_avg_items" in logs:
            msg.append(f"pseudo_avg_items={float(logs['pseudo_avg_items']):.2f}")
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

    def __init__(
        self,
        *args,
        lr_lora_mult: float = 0.5,
        lr_r3_mult: float = 1.0,
        use_chat_template: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_lora_mult = lr_lora_mult
        self.lr_r3_mult = lr_r3_mult
        self.use_chat_template = use_chat_template
        self._last_metrics: Dict[str, float] = {}

    def log(self, logs: Dict[str, float]) -> None:
        # Inject latest per-step metrics captured in compute_loss.
        if getattr(self, "_last_metrics", None):
            for key, val in self._last_metrics.items():
                if key not in logs:
                    logs[key] = val
        super().log(logs)

    def create_optimizer(self):
        """
        Use separate LR for LoRA weights vs. R³ modules.
        - R³ modules (simulator/retrieval/reconstruction/reasoner): lr = base_lr * lr_r3_mult
        - LoRA weights under base_vlm.model: lr = base_lr * lr_lora_mult
        """
        if self.optimizer is not None:
            return self.optimizer

        base_lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)
        model = self.model.module if hasattr(self.model, "module") else self.model

        lora_params = []
        r3_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("base_vlm.model"):
                lora_params.append(param)
            else:
                r3_params.append(param)

        param_groups = []
        if r3_params:
            param_groups.append({"params": r3_params, "lr": base_lr * float(self.lr_r3_mult), "weight_decay": wd})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": base_lr * float(self.lr_lora_mult), "weight_decay": wd})
        if not param_groups:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": wd}]

        self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        # Rank0-only summary: confirms which parts of R³/LoRA are trainable and their scale.
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        except Exception:
            local_rank = -1
        if local_rank in (-1, 0):
            r3_numel = sum(p.numel() for p in r3_params)
            lora_numel = sum(p.numel() for p in lora_params)
            logging.info(
                "Trainable params: R3=%d (lr_mult=%.3f) | LoRA+backbone-adapter=%d (lr_mult=%.3f)",
                r3_numel,
                float(self.lr_r3_mult),
                lora_numel,
                float(self.lr_lora_mult),
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Unwrap DDP for attribute access while keeping forward on the wrapped model.
        base_model = model.module if hasattr(model, "module") else model
        try:
            device = base_model.base_vlm.model.get_input_embeddings().weight.device
        except Exception:
            device = next(base_model.parameters()).device
        tokenizer = base_model.base_vlm.tokenizer
        max_length = getattr(base_model.config, "max_seq_length", 1024)

        corrupted_split = inputs["corrupted"]

        use_pseudo_text = bool(getattr(base_model.config, "enable_retrieval", False))
        corrupted_tokens, corrupted_pseudo = self._tokenize_branch(
            tokenizer,
            corrupted_split,
            max_length,
            device,
            use_chat_template=self.use_chat_template,
            use_pseudo_text=use_pseudo_text,
        )
        corrupted_vision = self._get_vision_embeddings(base_model, corrupted_split, device)

        enable_consistency = bool(getattr(base_model.config, "enable_consistency", False))
        lambda_c = float(getattr(base_model.config, "lambda_consistency", 0.0)) if enable_consistency else 0.0
        # Consistency warmup schedule: start after N steps, optionally ramp up over R steps.
        if lambda_c > 0.0:
            try:
                start_step = int(getattr(base_model.config, "consistency_start_step", 0) or 0)
                ramp_steps = int(getattr(base_model.config, "consistency_ramp_steps", 0) or 0)
                global_step = int(getattr(self.state, "global_step", 0))
                if global_step < start_step:
                    lambda_c = 0.0
                elif ramp_steps > 0:
                    scale = min(1.0, float(global_step - start_step) / float(ramp_steps))
                    lambda_c *= scale
            except Exception:
                pass

        teacher_out = None
        if lambda_c > 0.0:
            clean_split = inputs["clean"]
            clean_tokens, clean_pseudo = self._tokenize_branch(
                tokenizer,
                clean_split,
                max_length,
                device,
                use_chat_template=self.use_chat_template,
                use_pseudo_text=use_pseudo_text,
            )
            clean_vision = self._get_vision_embeddings(base_model, clean_split, device)
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
        vision_tokens = int(corrupted_vision.size(1))
        # Align labels to the combined sequence layout based on the *actual* retrieval output:
        # [prefix_tokens] + [text tokens] + [imputation tokens] + [vision tokens]
        prefix_tokens = 0
        imputation_tokens = 0
        try:
            retrieval = student_out.get("retrieval") if isinstance(student_out, dict) else None
            evidence_emb = retrieval.get("embeddings") if isinstance(retrieval, dict) else None
            evidence_count = 0
            if torch.is_tensor(evidence_emb) and evidence_emb.numel() > 0:
                evidence_count = int(evidence_emb.size(1))
            if evidence_count > 0:
                if bool(getattr(base_model.config, "enable_prefix", True)):
                    prefix_len_cfg = int(getattr(getattr(base_model, "reconstruction", None).config, "prefix_length", 0))
                    prefix_tokens = min(prefix_len_cfg, evidence_count) if prefix_len_cfg > 0 else 0
                if bool(getattr(base_model.config, "enable_imputation", True)):
                    imputation_tokens = int(
                        getattr(getattr(base_model, "reconstruction", None).config, "imputation_tokens", 0)
                    )
        except Exception:
            prefix_tokens = 0
            imputation_tokens = 0
        student_logits = student_out["logits"].float()
        loss_task = self._causal_ce(
            student_logits,
            corrupted_tokens["labels"],
            prefix_tokens=prefix_tokens,
            imputation_tokens=imputation_tokens,
            vision_tokens=vision_tokens,
            sample_ids=inputs.get("ids"),
        )
        if teacher_out is None:
            loss_consistency = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        else:
            loss_consistency = F.mse_loss(
                teacher_out["pooled_hidden"].detach().float(),
                student_out["pooled_hidden"].float(),
            )
        # Ensure loss is in float32 to avoid backward dtype issues under mixed precision.
        loss_task = loss_task.float()
        loss_consistency = loss_consistency.float()
        total_loss = (loss_task + lambda_c * loss_consistency).float()
        # Track extra metrics for logging.
        try:
            pseudo_count = (
                sum(len(p) for p in corrupted_pseudo) / max(1, len(corrupted_pseudo))
                if isinstance(corrupted_pseudo, list)
                else 0.0
            )
        except Exception:
            pseudo_count = 0.0
        valid_labels = int((corrupted_tokens["labels"] != -100).sum().item())
        prompt_len = 0.0
        try:
            prompt_lengths = corrupted_tokens.get("prompt_length")
            if torch.is_tensor(prompt_lengths) and prompt_lengths.numel() > 0:
                prompt_len = float(prompt_lengths.float().mean().item())
        except Exception:
            prompt_len = 0.0
        self._last_metrics = {
            "task_loss": float(loss_task.detach().float().cpu()),
            "consistency_loss": float(loss_consistency.detach().float().cpu()),
            "lambda_consistency": float(lambda_c),
            "labels_valid": float(valid_labels),
            "prompt_len": float(prompt_len),
            "pseudo_avg_items": float(pseudo_count),
        }
        # Trainer expects loss on args.device (cuda:0 for model-parallel). Move explicitly to avoid device mismatch.
        try:
            target_device = getattr(self.args, "device", None)
            if target_device is None:
                target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                target_device = torch.device(target_device)
        except Exception:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_loss = total_loss.to(target_device)
        loss_task = loss_task.to(target_device)
        loss_consistency = loss_consistency.to(target_device)

        outputs = {
            "task_loss": loss_task.detach(),
            "consistency_loss": loss_consistency.detach(),
            "retrieval": student_out.get("retrieval"),
        }
        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _save(self, output_dir: str | os.PathLike, state_dict=None):
        """
        Override HF Trainer saving to avoid `safetensors` failures when the model contains
        shared tensors (e.g., retrieval module reusing the backbone embedding layer).

        We always save a standard `pytorch_model.bin` via `torch.save`, which supports shared storage.
        """
        if not self.is_world_process_zero():
            return
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        if state_dict is None:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        # Save a lightweight config snapshot for reproducibility.
        try:
            cfg = getattr(model_to_save, "config", None)
            if cfg is not None and hasattr(cfg, "to_json_string"):
                with open(os.path.join(output_dir, "r3_config.json"), "w", encoding="utf-8") as f:
                    f.write(cfg.to_json_string())
        except Exception:
            pass

    @staticmethod
    def _causal_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        prefix_tokens: int = 0,
        imputation_tokens: int = 0,
        vision_tokens: int = 0,
        sample_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Manual causal LM loss with label shift, ignoring -100.
        """
        labels = labels.to(logits.device)
        if prefix_tokens > 0:
            labels = F.pad(labels, (prefix_tokens, 0), value=-100)
        tail = int(imputation_tokens) + int(vision_tokens)
        if tail > 0:
            labels = F.pad(labels, (0, tail), value=-100)
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
        valid = (shift_labels != -100).sum().item()
        if valid == 0:
            # All tokens are masked; this batch提供不了有效监督，记录一次 warning。
            sid = sample_ids if sample_ids is not None else []
            logging.warning("Causal CE skipped: all labels are -100 (sample_ids=%s)", sid)
            return logits.new_tensor(0.0)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    @staticmethod
    def _tokenize_branch(
        tokenizer,
        split: Dict,
        max_length: int,
        device: torch.device,
        use_chat_template: bool = False,
        use_pseudo_text: bool = True,
    ):
        def build_prompt(question: str, pseudo_entries: List[str]) -> str:
            pseudo_block = R3Trainer._format_pseudo_text(pseudo_entries)
            user_content = f"{pseudo_block}\nQuestion: {question}".strip() if pseudo_block else f"Question: {question}"
            user_content = user_content + "\nPlease answer with the short answer only."
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": user_content}]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return f"{user_content}\nAnswer:"

        def prompt_len(prompt: str) -> int:
            return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        def trim_pseudo_to_budget(question: str, pseudo_entries: List[str]) -> List[str]:
            if not pseudo_entries:
                return []
            # Keep some room for the short answer; allow at least 2 tokens.
            min_answer_tokens = 32
            max_prompt_len = max_length - min_answer_tokens
            base_prompt = build_prompt(question, [])
            base_len = prompt_len(base_prompt)
            if base_len >= max_prompt_len:
                return []
            trimmed: List[str] = []
            for entry in pseudo_entries:
                if not entry:
                    continue
                candidate = trimmed + [entry]
                if prompt_len(build_prompt(question, candidate)) <= max_prompt_len:
                    trimmed = candidate
                    continue
                # If the next entry overflows, truncate it to fit the remaining budget.
                remaining = max_prompt_len - prompt_len(build_prompt(question, trimmed))
                if remaining <= 0:
                    break
                entry_ids = tokenizer(entry, add_special_tokens=False)["input_ids"]
                if len(entry_ids) > remaining:
                    entry_ids = entry_ids[:remaining]
                entry_text = tokenizer.decode(entry_ids, skip_special_tokens=True).strip()
                if entry_text:
                    trimmed.append(entry_text)
                break
            return trimmed

        questions = split["question"]
        labels_text = split.get("labels", [""] * len(questions))
        # 防止空答案导致全 -100：遇到空字符串时替换为占位符。
        sane_labels = []
        for lbl in labels_text:
            if isinstance(lbl, str) and lbl.strip() == "":
                sane_labels.append("UNKNOWN")
            else:
                sane_labels.append(lbl)
        labels_text = sane_labels
        raw_pseudo = split.get("pseudo_text", [[] for _ in questions]) if use_pseudo_text else [[] for _ in questions]
        prompt_pseudo = []
        for q, pseudo in zip(questions, raw_pseudo):
            prompt_pseudo.append(trim_pseudo_to_budget(q, pseudo) if use_pseudo_text else [])

        prompts = []
        for q, pseudo in zip(questions, prompt_pseudo):
            prompts.append(build_prompt(q, pseudo))
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
        text_tokens["prompt_length"] = prompt_lengths.to(device)
        for idx, length in enumerate(prompt_lengths):
            labels[idx, : length.item()] = -100
        # If any sample has all labels masked (prompt truncated out the answer),
        # rebuild prompts without pseudo-text for those samples to recover supervision.
        if (labels != -100).sum(dim=1).eq(0).any():
            fixed_pseudo = []
            for idx, entries in enumerate(prompt_pseudo):
                if (labels[idx] != -100).any():
                    fixed_pseudo.append(entries)
                else:
                    fixed_pseudo.append([])  # drop pseudo-text to shorten prompt
            prompts = []
            for q, pseudo in zip(questions, fixed_pseudo):
                prompts.append(build_prompt(q, pseudo))
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
            text_tokens["prompt_length"] = prompt_lengths.to(device)
            for idx, length in enumerate(prompt_lengths):
                labels[idx, : length.item()] = -100
        text_tokens["labels"] = labels
        # Return the full pseudo text for retrieval (do not over-truncate evidence).
        return text_tokens, raw_pseudo

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
    parser.add_argument(
        "--quick_eval_every",
        type=int,
        default=None,
        help="If set, run a 1-sample quick eval every N optimizer steps on rank0.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="If set, run a periodic score eval every N steps (rank0 only). Defaults to save_steps when eval_samples>0.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=0,
        help="Number of samples for periodic score eval (0=disabled).",
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for periodic score eval generation.",
    )
    parser.add_argument("--log_interval", type=int, default=None, help="Override training.log_interval for logging_steps.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Optional path to a HuggingFace Trainer checkpoint dir (e.g., checkpoint-1000) to resume from.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=Path,
        default=None,
        help="Optional path to a checkpoint dir/file to initialize model weights from (does NOT resume optimizer/scheduler).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Optional checkpoint save interval (overrides TrainingArguments default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    # 运行示例（中文说明）:
    # 单卡训练:
    # python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
    #
    # 多卡训练 (torchrun 自动启用 DDP):
    # torchrun --nproc_per_node=4 train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    log_handlers = [logging.StreamHandler()]
    # In DDP, only rank0 writes train.log to avoid interleaved / duplicated logs.
    if args.log_file and local_rank in (-1, 0):
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
    pseudo_corpus_path = dataset_section.get("pseudo_corpus")
    pseudo_corpus = load_pseudo_corpus(pseudo_corpus_path)
    if pseudo_corpus:
        logging.info("Loaded pseudo-text corpus entries: %d", len(pseudo_corpus))
        # Fail fast if corpus exists but only contains fallback anchors like [Q]/[ID].
        has_real_text = False
        for entries in pseudo_corpus.values():
            for text in entries:
                stripped = str(text).strip()
                if not stripped:
                    continue
                if not (stripped.startswith("[Q]") or stripped.startswith("[ID]")):
                    has_real_text = True
                    break
            if has_real_text:
                break
        if not has_real_text:
            raise ValueError(
                "Pseudo-text corpus only contains [Q]/[ID] fallback entries. "
                "Please regenerate OCR/Caption pseudo-text before training."
            )
    else:
        if pseudo_corpus_path:
            raise ValueError(
                f"Pseudo-text corpus is empty or missing: {pseudo_corpus_path}. "
                "Training is stopped to avoid fallback-only retrieval."
            )
        logging.warning("Pseudo-text corpus is empty; fallback pseudo-text will be used.")
    logging.info("Config loaded. Preparing datasets...")

    def build_single_dataset(section: Dict) -> R3Dataset:
        ds_root = Path(section["root"])
        if not ds_root.is_absolute():
            ds_root = project_root / ds_root
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
        pt_max_items = section.get("pseudo_text_max_items")
        pt_max_chars = section.get("pseudo_text_max_chars")
        corr_prob = section.get("corruption_prob", 1.0)
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
            pseudo_text_max_items=pt_max_items,
            pseudo_text_max_chars=pt_max_chars,
            pseudo_text_chunk_tokens=entry.get(
                "pseudo_text_chunk_tokens",
                dataset_section.get("pseudo_text_chunk_tokens", 32),
            ),
            corruption_prob=corr_prob,
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

    eval_dataset = None
    if args.eval_samples and int(args.eval_samples) > 0:
        eval_split = dataset_section.get("eval_split", "val")
        if multi_cfg:
            eval_datasets: List[R3Dataset] = []
            eval_weights: List[float] = []
            for entry in multi_cfg:
                eval_entry = dict(entry)
                eval_entry["split"] = eval_split
                eval_datasets.append(build_single_dataset(eval_entry))
                eval_weights.append(eval_entry.get("weight", 1.0))
            eval_dataset = MultiTaskDataset(eval_datasets, weights=eval_weights)
        else:
            eval_entry = dict(dataset_section)
            eval_entry["split"] = eval_split
            eval_dataset = build_single_dataset(eval_entry)

    model_section = cfg.get("model", {})
    training_section = cfg.get("training", {})
    if args.log_interval is not None:
        training_section["log_interval"] = args.log_interval
    model_cfg = R3ModelConfig(
        model_name=model_section.get("name", "Qwen/Qwen3-VL-8B-Instruct"),
        lora_rank=model_section.get("lora_rank", 32),
        lora_alpha=model_section.get("lora_alpha", 16),
        hidden_size=model_section.get("hidden_size", 4096),
        max_seq_length=model_section.get("max_seq_length", 1024),
        bf16=model_section.get("bf16", True),
        dtype=model_section.get("dtype", "auto"),
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
        use_pseudo_query=model_section.get("use_pseudo_query", True),
        pseudo_query_weight=model_section.get("pseudo_query_weight", 0.6),
        retrieval_cache_path=model_section.get("retrieval_cache_path"),
        retrieval_corpus_path=model_section.get("retrieval_corpus_path"),
        retrieval_max_evidence_tokens=model_section.get("retrieval_max_evidence_tokens", 128),
        retrieval_chunk_tokens=model_section.get("retrieval_chunk_tokens", 32),
        consistency_start_step=model_section.get("consistency_start_step", 0),
        consistency_ramp_steps=model_section.get("consistency_ramp_steps", 0),
    )
    logging.info(
        "Stage config: apply_corruption=%s | R3(corr=%s retr=%s prefix=%s mem=%s imp=%s cons=%s λ=%.3f top_k=%d) | "
        "dtype=%s fp16_amp=%s | vision_tokens=%s | per_device_bs=%s grad_accum=%s",
        str(bool(any((entry or {}).get("apply_corruption", True) for entry in (multi_cfg or [dataset_section])))),
        str(bool(model_cfg.enable_corruption)),
        str(bool(model_cfg.enable_retrieval)),
        str(bool(model_cfg.enable_prefix)),
        str(bool(model_cfg.enable_memory)),
        str(bool(model_cfg.enable_imputation)),
        str(bool(model_cfg.enable_consistency)),
        float(model_cfg.lambda_consistency),
        int(model_cfg.top_k),
        str(model_cfg.dtype),
        str(bool(training_section.get("fp16", False))),
        str(cfg["model"].get("vision_tokens", 16)),
        str(dataset_section.get("batch_size", 1)),
        str(training_section.get("grad_accum_steps", 1)),
    )
    logging.info("Loading model: %s (provider=%s)", model_cfg.model_name, model_cfg.provider)
    model = R3Model(model_cfg)
    logging.info("Model initialized with backbone %s", model_cfg.model_name)
    # Optionally initialize weights from a previous run without resuming optimizer/scheduler.
    if args.init_from_checkpoint is not None:
        init_path = args.init_from_checkpoint
        ckpt_file = init_path
        if init_path.is_dir():
            # Prefer our trainer override output.
            candidate_bin = init_path / "pytorch_model.bin"
            candidate_safe = init_path / "model.safetensors"
            if candidate_bin.exists():
                ckpt_file = candidate_bin
            elif candidate_safe.exists():
                ckpt_file = candidate_safe
            else:
                logging.warning("init_from_checkpoint=%s has no pytorch_model.bin/model.safetensors; skip.", init_path)
                ckpt_file = None
        if ckpt_file is not None and ckpt_file.exists():
            try:
                if str(ckpt_file).endswith(".safetensors"):
                    from safetensors.torch import load_file  # type: ignore

                    state = load_file(str(ckpt_file))
                else:
                    state = torch.load(ckpt_file, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = model.load_state_dict(state, strict=False)
                logging.info(
                    "Initialized model weights from %s (missing=%d unexpected=%d)",
                    ckpt_file,
                    len(missing),
                    len(unexpected),
                )
            except Exception as exc:
                logging.warning("Failed to init_from_checkpoint=%s: %s", ckpt_file, exc)
    # Mark as model-parallel only when the backbone is truly sharded across >1 device.
    # (In DDP, we may pass device_map={"": local_rank} which is single-device and should NOT disable DDP wrapping.)
    try:
        if hasattr(model, "base_vlm") and hasattr(model.base_vlm.model, "hf_device_map"):
            dmap = getattr(model.base_vlm.model, "hf_device_map", {}) or {}
            devices = set(dmap.values())
            if len(devices) > 1:
                model.is_model_parallel = True
                model.is_parallelizable = True
                model.model_parallel = True
                model.hf_device_map = dmap
    except Exception:
        pass
    if hasattr(model, "base_vlm") and hasattr(model.base_vlm, "model") and hasattr(model.base_vlm.model, "config"):
        model.base_vlm.model.config.use_cache = False

    ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    # Build TrainingArguments in a version-tolerant way (older transformers may not support some kwargs).
    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=training_section.get("epochs", 1),
        max_steps=args.max_steps if args.max_steps is not None else training_section.get("max_steps"),
        per_device_train_batch_size=dataset_section.get("batch_size", 2),
        gradient_accumulation_steps=training_section.get("grad_accum_steps", 1),
        learning_rate=training_section.get("learning_rate", 2e-4),
        weight_decay=training_section.get("weight_decay", 0.05),
        logging_steps=training_section.get("log_interval", 10),
        warmup_ratio=training_section.get("warmup_ratio", 0.05),
        lr_scheduler_type=training_section.get("lr_scheduler_type", "linear"),
        report_to=training_section.get("report_to", "tensorboard"),
        logging_dir=str(training_section.get("logging_dir", args.output_dir / "logs")),
        remove_unused_columns=False,
        bf16=model_cfg.bf16 and torch.cuda.is_available(),
        fp16=training_section.get("fp16", False) and torch.cuda.is_available(),
        gradient_checkpointing=training_section.get("grad_checkpoint", False),
        dataloader_num_workers=training_section.get("num_workers", 0),
        ddp_find_unused_parameters=False if ddp else None,
        ddp_backend="nccl" if ddp else None,
        # avoid safetensors shared-memory save errors due to shared embeddings (only if supported)
        save_safetensors=False,
    )
    if args.save_steps is not None:
        training_kwargs["save_steps"] = args.save_steps
    sig = inspect.signature(TrainingArguments.__init__)
    for k in list(training_kwargs.keys()):
        if k not in sig.parameters:
            training_kwargs.pop(k)
    training_args = TrainingArguments(**training_kwargs)

    # Quick eval callback: runs on rank0 every N logging events if enabled
    class QuickEvalCallback(TrainerCallback):
        def __init__(self, dataset: Dataset, every: Optional[int], tokenizer, processor, logger):
            self.dataset = dataset
            self.every = every
            self.tokenizer = tokenizer
            self.processor = processor
            self.logger = logger
            self._last_step: int = -1
            self._last_id: Optional[str] = None

        @staticmethod
        def _summarize_entries(entries: List[str], max_items: int = 5, max_chars: int = 160) -> List[str]:
            summary: List[str] = []
            for entry in entries[:max_items]:
                text = str(entry).replace("\n", " ").strip()
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."
                summary.append(text)
            return summary

        def _sample(self) -> Dict:
            # MultiTaskDataset ignores idx and samples randomly; this works for both cases.
            if len(self.dataset) <= 0:
                raise RuntimeError("Empty dataset; cannot quick-eval.")
            for _ in range(10):
                idx = random.randint(0, len(self.dataset) - 1)
                item = self.dataset[idx]
                sid = str(item.get("id", ""))
                if sid and sid != self._last_id:
                    self._last_id = sid
                    item["_quick_idx"] = idx
                    return item
            item = self.dataset[random.randint(0, len(self.dataset) - 1)]
            item["_quick_idx"] = -1
            return item

        def on_step_end(self, args, state, control, **kwargs):
            if self.every is None:
                return
            if state.global_step == 0 or state.global_step % self.every != 0:
                return
            if state.global_step == self._last_step:
                return
            self._last_step = int(state.global_step)
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
                item = self._sample()
                clean = item["clean"]
                corrupted = item["corrupted"]
                # Prefer corrupted branch if corruption is enabled and actually changed the image object.
                branch = corrupted if (corrupted.get("image") is not None and corrupted.get("image") is not clean.get("image")) else clean
                q = branch["question"]
                tgt = branch.get("labels", "")
                img = branch.get("image")
                img_path = branch.get("image_path")
                pseudo_entries = branch.get("pseudo_text", []) or []
                pseudo_summary = self._summarize_entries(pseudo_entries)
                if img is None and img_path:
                    img = R3Dataset._load_image(img_path)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": q + "\nPlease answer with the short answer only."}],
                    }
                ]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"Question: {q}\nAnswer:"
                # Prefer embedding device under model-parallel `device_map="auto"`.
                base_model = model.module if hasattr(model, "module") else model
                try:
                    dev = base_model.base_vlm.model.get_input_embeddings().weight.device
                except Exception:
                    dev = next(base_model.parameters()).device
                inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to(dev)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    gen = base_model.base_vlm.model.generate(  # type: ignore
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen_ids = gen[0][input_len:]
                pred = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                sid = str(item.get("id", ""))
                idx = item.get("_quick_idx", -1)
                self.logger.info(
                    f"[quick_eval step={state.global_step}] idx={idx} id={sid} | q={q} | pred={pred} | tgt={tgt} | img={img_path}"
                )
                # Log retrieval inputs/outputs for quick inspection when enabled.
                try:
                    if bool(getattr(base_model.config, "enable_retrieval", False)) and hasattr(base_model, "retrieval"):
                        # Build a short question-only prompt for retrieval query embeddings.
                        if hasattr(self.tokenizer, "apply_chat_template"):
                            q_prompt = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": f"Question: {q}"}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        else:
                            q_prompt = f"Question: {q}\nAnswer:"
                        max_q_len = min(int(getattr(base_model.config, "max_seq_length", 1024)), 512)
                        q_tokens = self.tokenizer(
                            q_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_q_len,
                        )["input_ids"].to(dev)
                        q_embeds = base_model.base_vlm.model.get_input_embeddings()(q_tokens)
                        txt_conf = torch.zeros(q_embeds.size()[:2], device=dev)
                        img_conf = torch.zeros((q_embeds.size(0), 1), device=dev)
                        retrieval = base_model.retrieval(q_embeds, [pseudo_entries], img_conf, txt_conf)
                        top_texts = retrieval.get("texts", [[]])[0]
                        top_scores = retrieval.get("scores", None)
                        if torch.is_tensor(top_scores):
                            scores = top_scores[0].detach().cpu().tolist()
                            retrieved = [
                                f"{s:.3f}:{t}" for s, t in zip(scores, self._summarize_entries(top_texts))
                            ]
                        else:
                            retrieved = self._summarize_entries(top_texts)
                        self.logger.info(f"[quick_eval step={state.global_step}] pseudo_text={pseudo_summary}")
                        self.logger.info(f"[quick_eval step={state.global_step}] retrieved={retrieved}")
                    else:
                        self.logger.info(f"[quick_eval step={state.global_step}] pseudo_text={pseudo_summary}")
                except Exception as exc:
                    self.logger.warning(f"[quick_eval] retrieval inspect failed: {exc}")
            except Exception as e:
                self.logger.warning(f"[quick_eval] failed: {e}")
            finally:
                if model_was_training:
                    model.train()

    class PeriodicEvalCallback(TrainerCallback):
        def __init__(
            self,
            dataset: Dataset,
            every: int,
            samples: int,
            tokenizer,
            processor,
            logger,
            use_chat_template: bool,
            max_new_tokens: int = 32,
        ):
            self.dataset = dataset
            self.every = every
            self.samples = samples
            self.tokenizer = tokenizer
            self.processor = processor
            self.logger = logger
            self.use_chat_template = use_chat_template
            self.max_new_tokens = max_new_tokens
            self._last_step: int = -1

        def on_step_end(self, args, state, control, **kwargs):
            if self.samples <= 0 or self.every <= 0:
                return
            if state.global_step == 0 or state.global_step % self.every != 0:
                return
            if state.global_step == self._last_step:
                return
            self._last_step = int(state.global_step)
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
            if local_rank not in (-1, 0):
                return
            model = kwargs.get("model", None)
            if model is None:
                return
            model_was_training = model.training
            model.eval()
            try:
                import importlib
                eval_mod = importlib.import_module("evaluate_r3")
                base_model = model.module if hasattr(model, "module") else model
                metrics = eval_mod.score_model_on_dataset(
                    base_model,
                    self.processor,
                    self.tokenizer,
                    self.dataset,
                    samples=int(self.samples),
                    use_chat_template=self.use_chat_template,
                    max_new_tokens=int(self.max_new_tokens),
                )
                self.logger.info(
                    "[periodic_eval step=%d] samples=%d acc=%.4f anls=%.4f avg_prompt_len=%.1f",
                    state.global_step,
                    int(metrics.get("samples", 0)),
                    float(metrics.get("accuracy", 0.0)),
                    float(metrics.get("anls", 0.0)),
                    float(metrics.get("avg_prompt_len", 0.0)),
                )
            except Exception as e:
                self.logger.warning(f"[periodic_eval] failed: {e}")
            finally:
                if model_was_training:
                    model.train()

    callbacks = [CurriculumScheduler(), LossLogger(logging.getLogger())]
    # build quick eval callback
    if args.quick_eval_every is not None and hasattr(model, "base_vlm"):
        callbacks.append(
            QuickEvalCallback(
                dataset=train_dataset,
                every=args.quick_eval_every,
                tokenizer=model.base_vlm.tokenizer,
                processor=model.base_vlm.processor,
                logger=logging.getLogger(),
            )
        )
    # build periodic score eval callback
    eval_every = args.eval_every
    if eval_every is None and args.eval_samples and args.save_steps is not None:
        eval_every = args.save_steps
    if eval_dataset is not None and eval_every is not None and eval_every > 0:
        callbacks.append(
            PeriodicEvalCallback(
                dataset=eval_dataset,
                every=int(eval_every),
                samples=int(args.eval_samples),
                tokenizer=model.base_vlm.tokenizer,
                processor=model.base_vlm.processor,
                logger=logging.getLogger(),
                use_chat_template=training_section.get("use_chat_template", False),
                max_new_tokens=int(args.eval_max_new_tokens),
            )
        )

    trainer = R3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        lr_lora_mult=float(training_section.get("lr_lora_mult", 0.5)),
        lr_r3_mult=float(training_section.get("lr_r3_mult", 1.0)),
        use_chat_template=bool(training_section.get("use_chat_template", False)),
    )
    logging.info("Starting training for %s epochs", training_args.num_train_epochs)
    if args.resume_from_checkpoint is not None:
        logging.info("Resuming from checkpoint: %s", args.resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint))
    else:
        trainer.train()
    logging.info("Training finished. Checkpoints at %s", args.output_dir)


if __name__ == "__main__":
    main()
