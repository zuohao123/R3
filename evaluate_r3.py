"""
Evaluation script for R^3 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
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
# Older torch (<2.3) may miss torch.compiler.is_compiling which new Qwen processors call.
if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False
    torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
elif not hasattr(torch.compiler, "is_compiling"):  # type: ignore[attr-defined]
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import os
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq

from data_pipeline.datasets import DATASET_REGISTRY, create_dataset, detect_dataset_type
from data_pipeline.corruptions import (
    ImageCorruptionConfig,
    ImageCorruptor,
    PseudoTextCorruptionConfig,
    PseudoTextCorruptor,
)
from r3.r3_model import R3Model, R3ModelConfig
from r3.data_utils import R3Dataset, collate_fn, load_yaml, load_pseudo_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R^3 checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to finetuned checkpoint file/dir; if omitted, use base backbone.")
    parser.add_argument("--ckpt_dir", type=Path, default=None, help="Alias for --checkpoint when pointing to a directory.")
    parser.add_argument("--split", type=str, default=None, help="Override dataset split for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for quick smoke tests.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional JSONL to dump predictions.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", *sorted(DATASET_REGISTRY.keys())],
    )
    parser.add_argument("--apply_corruption", action="store_true", help="Apply pre-encoding modality drops (Image/Pseudo-text).")
    parser.add_argument("--disable_retrieval", action="store_true", help="Disable retrieval module at eval time.")
    parser.add_argument("--disable_consistency", action="store_true", help="Disable consistency (always off in eval).")
    parser.add_argument("--disable_corruption", action="store_true", help="Disable corruption module at eval time.")
    parser.add_argument("--use_chat_template", action="store_true", help="Format prompts with Qwen chat template for base model eval.")
    parser.add_argument("--log_interval", type=int, default=0, help="Print interim metrics every N batches (0=off).")
    parser.add_argument("--log_samples", type=int, default=0, help="When logging, also print up to K (id, pred, target) pairs from that batch (0=off).")
    parser.add_argument("--native_eval", action="store_true", help="Bypass R3 wrapper; use native Qwen3-VL forward with official chat template.")
    parser.add_argument("--errors", type=Path, default=None, help="Optional JSONL to dump only mispredicted samples.")
    return parser.parse_args()


def levenshtein_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, char_a in enumerate(a, start=1):
        curr[0] = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[-1]


def anls(pred: str, target: str, threshold: float = 0.5) -> float:
    pred_norm = normalize_basic(pred)
    target_norm = normalize_basic(target)
    if not target_norm:
        return 1.0 if not pred_norm else 0.0
    distance = levenshtein_distance(pred_norm, target_norm)
    score = 1.0 - distance / max(len(pred_norm), len(target_norm))
    return score if score >= threshold else 0.0


def decode_predictions(logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> List[str]:
    """
    Decode predictions aligned with causal LM shift.
    logits[t] predicts token at t+1, so we shift both logits and labels by 1.
    """
    if logits.size(1) <= 1:
        return ["" for _ in range(logits.size(0))]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    pred_ids = shift_logits.argmax(dim=-1)
    predictions: List[str] = []
    for row_pred, row_label in zip(pred_ids, shift_labels):
        mask = row_label != -100
        if mask.size(0) < row_pred.size(0):
            pad = torch.zeros(row_pred.size(0) - mask.size(0), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, pad.bool()], dim=0)
        elif mask.size(0) > row_pred.size(0):
            mask = mask[: row_pred.size(0)]
        if mask.sum() == 0:
            predictions.append("")
            continue
        ids = row_pred[mask]
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        predictions.append(text)
    return predictions


def align_labels_for_r3(
    labels: torch.Tensor,
    prefix_tokens: int,
    imputation_tokens: int,
    vision_tokens: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Pad/trim labels to match the combined sequence layout:
    [prefix] + [text] + [imputation] + [vision]
    """
    labels = labels.clone()
    if prefix_tokens > 0:
        labels = F.pad(labels, (prefix_tokens, 0), value=-100)
    tail = int(imputation_tokens) + int(vision_tokens)
    if tail > 0:
        labels = F.pad(labels, (0, tail), value=-100)
    if labels.size(1) < seq_len:
        labels = F.pad(labels, (0, seq_len - labels.size(1)), value=-100)
    elif labels.size(1) > seq_len:
        labels = labels[:, :seq_len]
    return labels


def causal_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix_tokens: int = 0,
    imputation_tokens: int = 0,
    vision_tokens: int = 0,
) -> torch.Tensor:
    """
    Manual causal LM loss with label shift, ignoring -100.
    Mirrors train_r3.R3Trainer._causal_ce.
    """
    labels = labels.to(logits.device)
    if prefix_tokens > 0:
        labels = F.pad(labels, (prefix_tokens, 0), value=-100)
    tail = int(imputation_tokens) + int(vision_tokens)
    if tail > 0:
        labels = F.pad(labels, (0, tail), value=-100)
    seq_len = logits.size(1)
    if labels.size(1) < seq_len:
        labels = F.pad(labels, (0, seq_len - labels.size(1)), value=-100)
    elif labels.size(1) > seq_len:
        labels = labels[:, :seq_len]
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid = (shift_labels != -100).sum().item()
    if valid == 0:
        return logits.new_tensor(0.0)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def clean_generation_output(text: str) -> str:
    """
    Strip role markers like 'assistant'/'user' and keep the first non-empty line as answer.
    """
    t = text.strip()
    # Remove common role prefixes
    for prefix in ["assistant", "assistant:", "assistant\n", "assistant "]:
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
            break
    # Keep first non-empty line
    for line in t.splitlines():
        line = line.strip()
        if line:
            t = line
            break
    return t


def first_sentence(text: str) -> str:
    """
    Keep first non-empty line and cut at first sentence end if present.
    """
    t = text.strip()
    if not t:
        return t
    t = t.splitlines()[0].strip()
    for sep in [".", "!", "?"]:
        if sep in t:
            t = t.split(sep)[0].strip()
            break
    return t.strip(" \"'")


def normalize_basic(text: str) -> str:
    """
    Basic normalization for fair matching: lower, strip, collapse spaces, strip trailing punctuation/commas.
    """
    import re
    t = text.lower().strip()
    # remove markdown bold markers and quotes/backticks
    t = t.replace("**", " ").replace("`", " ").replace("“", " ").replace("”", " ")
    # keep digits/letters, replace other punct with space
    t = re.sub(r"[^\w\s.\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" ,.;:!?\"'")
    return t


def summarize_entries(entries: List[str], max_items: int = 5, max_chars: int = 320) -> List[str]:
    summary: List[str] = []
    for entry in entries[:max_items]:
        text = str(entry).replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        if text:
            summary.append(text)
    return summary


def _to_number(text: str) -> Optional[float]:
    """
    Try to parse a float from normalized text by stripping $, commas, and spaces.
    """
    try:
        cleaned = text.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except Exception:
        return None


def numbers_close(a: float, b: float) -> bool:
    """
    Consider numbers equal if rounding to 2 decimals matches, or relative diff < 0.5% or abs diff < 0.05.
    """
    if round(a, 2) == round(b, 2):
        return True
    diff = abs(a - b)
    if abs(b) > 1e-6 and diff / abs(b) < 0.005:
        return True
    return diff < 0.05


def is_correct(pred: str, target: str) -> bool:
    """
    Flexible exact match:
    1) normalized string equality
    2) numeric closeness (currency/decimal tolerance)
    """
    norm_pred = normalize_basic(pred)
    norm_tgt = normalize_basic(target)
    if norm_pred == norm_tgt:
        return True
    num_p = _to_number(norm_pred)
    num_t = _to_number(norm_tgt)
    if num_p is not None and num_t is not None:
        return numbers_close(num_p, num_t)
    return False


def best_span_match(pred: str, target: str) -> str:
    """
    For short answers (DocVQA/Chart/Info), pick a subspan of pred that best matches target
    under normalized Levenshtein, to reduce penalty from verbose generations.
    """
    pred_norm = normalize_basic(pred)
    tgt_norm = normalize_basic(target)
    if not pred_norm or not tgt_norm:
        return pred
    pred_tokens = pred_norm.split()
    tgt_tokens = tgt_norm.split()
    min_len = max(1, len(tgt_tokens) - 1)
    max_len = max(min_len, len(tgt_tokens) + 2)
    tgt_set = set(tgt_tokens)
    best = pred_norm
    best_score = -1.0
    for i in range(len(pred_tokens)):
        for j in range(i + min_len, min(len(pred_tokens), i + max_len) + 1):
            span = " ".join(pred_tokens[i:j])
            if not span:
                continue
            span_tokens = span.split()
            # Require at least one token overlap when possible
            if tgt_set and not (tgt_set & set(span_tokens)):
                continue
            score = 1.0 - levenshtein_distance(span, tgt_norm) / max(len(span), len(tgt_norm))
            if score > best_score:
                best_score = score
                best = span
    return best


def score_model_on_dataset(
    model,
    processor,
    tokenizer,
    dataset,
    samples: int = 300,
    use_chat_template: bool = False,
    max_new_tokens: int = 32,
) -> Dict[str, float]:
    """
    Lightweight evaluation for training-time periodic checks.
    Uses the in-memory model (no checkpoint reload).
    """
    model.eval()
    try:
        dev = model.base_vlm.model.get_input_embeddings().weight.device
    except Exception:
        dev = next(model.parameters()).device
    use_retrieval = bool(getattr(model.config, "enable_retrieval", False))
    use_corruption = bool(getattr(model.config, "enable_corruption", False))
    max_seq_length = int(getattr(model.config, "max_seq_length", 1024))

    total = 0
    correct = 0
    anls_sum = 0.0
    prompt_len_sum = 0
    for _ in range(min(samples, len(dataset))):
        item = dataset[random.randint(0, len(dataset) - 1)]
        clean = item["clean"]
        corrupted = item["corrupted"]
        branch = corrupted if use_corruption else clean
        q = branch["question"]
        tgt = branch.get("labels", "")
        img = branch.get("image")
        img_path = branch.get("image_path")
        pseudo_entries = branch.get("pseudo_text", []) or []
        if img is None and img_path:
            img = R3Dataset._load_image(img_path)

        evidence_entries = []
        if use_retrieval and hasattr(model, "retrieval"):
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                q_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"Question: {q}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                q_prompt = f"Question: {q}\nAnswer:"
            q_tokens = tokenizer(
                q_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(512, max_seq_length),
            )["input_ids"].to(dev)
            q_embeds = model.base_vlm.model.get_input_embeddings()(q_tokens)
            txt_conf = torch.zeros(q_embeds.size()[:2], device=dev)
            img_conf = torch.zeros((q_embeds.size(0), 1), device=dev)
            retrieval = model.retrieval(q_embeds, [pseudo_entries], img_conf, txt_conf)
            evidence_entries = retrieval.get("texts", [[]])[0]

        pseudo_block = "\n".join([p for p in evidence_entries if p]) if evidence_entries else ""
        user_content = f"{pseudo_block}\nQuestion: {q}".strip() if pseudo_block else f"Question: {q}"
        user_content = user_content + "\nPlease answer with the short answer only."
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{user_content}\nAnswer:"
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(dev)
        prompt_len_sum += int(inputs["input_ids"].shape[1])
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.base_vlm.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = gen[0][input_len:]
        pred_raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = best_span_match(pred_raw, tgt)
        total += 1
        if is_correct(pred, tgt):
            correct += 1
        anls_sum += anls(pred, tgt)
    accuracy = correct / max(1, total)
    anls_score = anls_sum / max(1, total)
    avg_prompt_len = prompt_len_sum / max(1, total)
    return {
        "samples": float(total),
        "accuracy": float(accuracy),
        "anls": float(anls_score),
        "avg_prompt_len": float(avg_prompt_len),
    }


def get_vision_embeddings(model, split: Dict, device: torch.device) -> torch.Tensor:
    """
    Mirror train-time vision embedding extraction.
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


def build_prompts(questions: List[str], pseudo_batch: List[List[str]], labels: List[str], tokenizer, use_chat_template: bool) -> List[str]:
    prompts = []
    pseudo_fmt = []
    for pseudo in pseudo_batch:
        pseudo_fmt.append("\n".join([p for p in pseudo if p]))
    for q, pseudo, lbl in zip(questions, pseudo_fmt, labels):
        user_content = f"{pseudo}\nQuestion: {q}".strip() if pseudo else f"Question: {q}"
        user_content = user_content + "\nPlease answer with the short answer only."
        if use_chat_template:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{user_content}\nAnswer:"
        full = f"{prompt} {lbl}".strip()
        prompts.append((prompt, full))
    return prompts


def tokenize_with_template(
    split: Dict,
    tokenizer,
    max_length: int,
    device: torch.device,
    use_chat_template: bool,
    use_pseudo_text: bool = True,
):
    def build_prompt(question: str, pseudo_entries: List[str]) -> str:
        pseudo_block = "\n".join([p for p in pseudo_entries if p])
        user_content = f"{pseudo_block}\nQuestion: {question}".strip() if pseudo_block else f"Question: {question}"
        user_content = user_content + "\nPlease answer with the short answer only."
        if use_chat_template:
            messages = [{"role": "user", "content": user_content}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{user_content}\nAnswer:"

    def prompt_len(prompt: str) -> int:
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def trim_pseudo_to_budget(question: str, pseudo_entries: List[str]) -> List[str]:
        if not pseudo_entries:
            return []
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
    raw_pseudo = split.get("pseudo_text", [[] for _ in questions]) if use_pseudo_text else [[] for _ in questions]

    prompt_pseudo = []
    for q, pseudo in zip(questions, raw_pseudo):
        prompt_pseudo.append(trim_pseudo_to_budget(q, pseudo) if use_pseudo_text else [])

    prompts = build_prompts(questions, prompt_pseudo, labels_text, tokenizer, use_chat_template)
    prompt_texts = [p for p, _ in prompts]
    full_texts = [f for _, f in prompts]

    prompt_tokens = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    text_tokens = tokenizer(
        full_texts,
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
    if (labels != -100).sum(dim=1).eq(0).any():
        fixed_pseudo = []
        for idx, entries in enumerate(prompt_pseudo):
            if (labels[idx] != -100).any():
                fixed_pseudo.append(entries)
            else:
                fixed_pseudo.append([])
        prompts = build_prompts(questions, fixed_pseudo, labels_text, tokenizer, use_chat_template)
        prompt_texts = [p for p, _ in prompts]
        full_texts = [f for _, f in prompts]
        prompt_tokens = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        text_tokens = tokenizer(
            full_texts,
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
    return text_tokens, raw_pseudo


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    training_cfg = cfg.get("training", {})
    use_chat_template = bool(args.use_chat_template or training_cfg.get("use_chat_template", False))
    split = args.split or eval_cfg.get("split") or dataset_cfg.get("eval_split", "val")
    apply_corruption = args.apply_corruption or eval_cfg.get("apply_corruption", False)
    if args.disable_corruption:
        apply_corruption = False

    # Resolve dataset root/type even when config uses multi-dataset setup.
    dataset_root = dataset_cfg.get("root")
    dataset_type = args.dataset_type
    if not dataset_root and "multi" in dataset_cfg:
        # Default to the first entry in multi for evaluation if no explicit root provided.
        first = dataset_cfg["multi"][0]
        dataset_root = first.get("root")
        if dataset_type == "auto":
            dataset_type = first.get("type", "auto")
    if dataset_root is None:
        raise ValueError("Dataset root is not specified; set dataset.root or choose an entry from dataset.multi.")
    dataset_root = Path(dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    dataset_root_rel = None
    try:
        dataset_root_rel = dataset_root.relative_to(project_root)
    except Exception:
        dataset_root_rel = None

    # Resolve dataset type
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        dataset_type = dataset_cfg.get("type", "auto")
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_root)
    pseudo_corpus = load_pseudo_corpus(dataset_cfg.get("pseudo_corpus"))
    if pseudo_corpus:
        print(f"[INFO] Loaded pseudo-text corpus entries: {len(pseudo_corpus)}")
    else:
        print("[WARN] Pseudo-text corpus is empty or missing; fallback pseudo-text will be used.")
    base_dataset = create_dataset(dataset_type, dataset_root, split=split)

    # Build corruption configs for eval (stage2/stage3 yaml include per-dataset settings under dataset.multi).
    pseudo_drop = dataset_cfg.get("pseudo_text_drop_prob", 0.3)
    image_corr_cfg = dataset_cfg.get("image_corruption", {}) or {}
    pseudo_text_max_items = dataset_cfg.get("pseudo_text_max_items")
    pseudo_text_max_chars = dataset_cfg.get("pseudo_text_max_chars")
    if "multi" in dataset_cfg:
        chosen = None
        if dataset_type != "auto":
            for entry in dataset_cfg["multi"]:
                if entry.get("type") == dataset_type:
                    chosen = entry
                    break
        if chosen is None:
            chosen = dataset_cfg["multi"][0]
        pseudo_drop = chosen.get("pseudo_text_drop_prob", pseudo_drop)
        image_corr_cfg = chosen.get("image_corruption", image_corr_cfg) or image_corr_cfg
        pseudo_text_max_items = chosen.get("pseudo_text_max_items", pseudo_text_max_items)
        pseudo_text_max_chars = chosen.get("pseudo_text_max_chars", pseudo_text_max_chars)

    eval_image_corruptor = None
    eval_pseudo_corruptor = None
    if apply_corruption:
        eval_image_corruptor = ImageCorruptor(
            ImageCorruptionConfig(
                occlusion_prob=image_corr_cfg.get("occlusion_prob", 0.5),
                occlusion_ratio=image_corr_cfg.get("occlusion_ratio", 0.25),
                blur_prob=image_corr_cfg.get("blur_prob", 0.5),
                blur_radius=image_corr_cfg.get("blur_radius", 3.0),
            )
        )
        eval_pseudo_corruptor = PseudoTextCorruptor(PseudoTextCorruptionConfig(drop_prob=pseudo_drop))

    dataset = R3Dataset(
        base_dataset,
        vision_tokens=cfg["model"].get("vision_tokens", 16),
        hidden_size=cfg["model"].get("hidden_size", 4096),
        apply_corruption=apply_corruption,
        pseudo_corpus=pseudo_corpus,
        image_corruptor=eval_image_corruptor,
        pseudo_text_corruptor=eval_pseudo_corruptor,
        pseudo_text_max_items=pseudo_text_max_items,
        pseudo_text_max_chars=pseudo_text_max_chars,
        pseudo_text_chunk_tokens=dataset_cfg.get("pseudo_text_chunk_tokens", 32),
    )
    if args.limit:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))

    eval_bs = 1 if args.native_eval else eval_cfg.get("batch_size", dataset_cfg.get("batch_size", 2))
    dataloader = DataLoader(
        dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    model_section = cfg.get("model", {})
    native_model = None
    processor = None
    tokenizer = None
    model = None
    use_native_generate = False

    if args.native_eval:
        model_path = model_section.get("name", "Qwen/Qwen3-VL-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        native_model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        native_model.eval()
    else:
        model_cfg = R3ModelConfig(
            model_name=model_section.get("name", "Qwen/Qwen3-VL-8B-Instruct"),
            lora_rank=model_section.get("lora_rank", 32),
            lora_alpha=model_section.get("lora_alpha", 16),
            hidden_size=model_section.get("hidden_size", 4096),
            max_seq_length=model_section.get("max_seq_length", 1024),
            bf16=model_section.get("bf16", False),
            dtype=model_section.get("dtype", "auto"),
            load_in_4bit=model_section.get("load_in_4bit", False),
            load_in_8bit=model_section.get("load_in_8bit", False),
            device_map=model_section.get("device_map"),
            low_cpu_mem_usage=model_section.get("low_cpu_mem_usage", True),
            gradient_checkpointing=model_section.get("gradient_checkpointing", False),
            provider=model_section.get("provider", "huggingface"),
            token=model_section.get("token"),
            cache_dir=model_section.get("cache_dir"),
            revision=model_section.get("revision"),
            local_files_only=model_section.get("local_files_only", False),
            enable_corruption=False if args.disable_corruption else model_section.get("enable_corruption", True),
            enable_retrieval=False if args.disable_retrieval else model_section.get("enable_retrieval", True),
            enable_prefix=model_section.get("enable_prefix", True),
            enable_memory=model_section.get("enable_memory", True),
            enable_imputation=model_section.get("enable_imputation", True),
            enable_consistency=False if args.disable_consistency else model_section.get("enable_consistency", False),
            top_k=model_section.get("top_k", 3),
            use_pseudo_query=model_section.get("use_pseudo_query", True),
            pseudo_query_weight=model_section.get("pseudo_query_weight", 0.6),
            retrieval_cache_path=model_section.get("retrieval_cache_path"),
            retrieval_corpus_path=model_section.get("retrieval_corpus_path"),
            retrieval_max_evidence_tokens=model_section.get("retrieval_max_evidence_tokens", 128),
            retrieval_chunk_tokens=model_section.get("retrieval_chunk_tokens", 32),
        )
        model = R3Model(model_cfg)
        # If the backbone is sharded via `device_map`, a global `.to(device)` will try to
        # materialize everything on one GPU and can OOM. Only move when not model-parallel.
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
        if model_cfg.device_map is None:
            model = model.to(args.device)
        ckpt_arg = args.checkpoint or args.ckpt_dir
        if ckpt_arg:
            ckpt_path = ckpt_arg
            if ckpt_path.is_dir():
                # HuggingFace shard格式（带 index.json）直接跳过手动加载，使用 base 模型权重
                if (ckpt_path / "model.safetensors.index.json").exists():
                    print(f"[INFO] Found safetensors index in {ckpt_path}, skip manual load and use base weights.")
                    ckpt_path = None
                else:
                    candidate_bin = ckpt_path / "pytorch_model.bin"
                    candidate_safe = ckpt_path / "model.safetensors"
                    if candidate_bin.exists():
                        ckpt_path = candidate_bin
                    elif candidate_safe.exists():
                        ckpt_path = candidate_safe
                    else:
                        print(f"[WARN] Checkpoint dir {ckpt_path} has no pytorch_model.bin/model.safetensors, skip loading finetuned weights.")
                        ckpt_path = None
            if ckpt_path is not None and ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
            elif ckpt_path is not None:
                print(f"[WARN] Checkpoint path {ckpt_path} not found, skip loading finetuned weights.")
        model.eval()
        # For stage1-like configs (no corruption/retrieval), prefer native generate on base_vlm.
        try:
            retrieval_enabled = bool(getattr(model.config, "enable_retrieval", False)) and not args.disable_retrieval
            # When retrieval is disabled, the R3 wrapper reduces to the base VLM.
            # In that case, prefer generate-based eval (even under corruption) for fair comparison.
            use_native_generate = (not args.native_eval) and (not retrieval_enabled)
        except Exception:
            use_native_generate = False

    # Metric switches by dataset type
    ds_lower = str(dataset_type).lower()
    is_docvqa = "docvqa" in ds_lower
    is_infovqa = "infovqa" in ds_lower
    is_chartqa = "chartqa" in ds_lower

    total_loss = 0.0
    total_batches = 0
    nan_loss_batches = 0
    nan_logits_batches = 0
    correct = 0
    total = 0
    anls_sum = 0.0
    dump_rows: List[Dict] = []
    error_rows: List[Dict] = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="eval", total=len(dataloader))):
            if use_native_generate:
                split_data = batch["clean"]
                q = split_data["question"][0]
                img = split_data["images"][0] if split_data["images"][0] is not None else None
                img_path = split_data["image_path"][0] if isinstance(split_data.get("image_path"), list) else None
                if img is None and img_path:
                    from PIL import Image
                    cand = Path(img_path)
                    if not cand.is_absolute():
                        cand = project_root / cand
                    if cand.exists():
                        img = Image.open(cand).convert("RGB")
                        img_path = str(cand)
                if img is None:
                    continue
                base_model = model.module if hasattr(model, "module") else model
                tokenizer = base_model.base_vlm.tokenizer
                processor = base_model.base_vlm.processor
                if processor is None:
                    processor = AutoProcessor.from_pretrained(
                        getattr(base_model.config, "model_name", model_section.get("name")),
                        trust_remote_code=True,
                    )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q + "\nPlease answer with the short answer only."},
                        ],
                    }
                ]
                if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"Question: {q}\nAnswer:"
                try:
                    dev = base_model.base_vlm.model.get_input_embeddings().weight.device
                except Exception:
                    dev = next(base_model.base_vlm.model.parameters()).device
                inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(dev)
                input_len = inputs["input_ids"].shape[1]
                gen = base_model.base_vlm.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                pred_text = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True).strip()
                pred_text = first_sentence(clean_generation_output(pred_text))
                target = split_data["labels"][0]
                pred_for_score = best_span_match(pred_text, target)
                total += 1
                if is_correct(pred_for_score, target):
                    correct += 1
                if is_docvqa:
                    anls_sum += anls(pred_for_score, target)
                if args.predictions:
                    dump_rows.append(
                        {
                            "id": batch["ids"][0],
                            "image_path": img_path,
                            "prediction": pred_text,
                            "scored_prediction": pred_for_score,
                            "target": target,
                        }
                    )
                if args.errors and not is_correct(pred_for_score, target):
                    error_rows.append(
                        {
                            "id": batch["ids"][0],
                            "image_path": img_path,
                            "prediction": pred_text,
                            "scored_prediction": pred_for_score,
                            "target": target,
                        }
                    )
                if args.log_interval and (idx + 1) % args.log_interval == 0:
                    interim_acc = correct / max(1, total)
                    msg = f"[eval] step {idx+1}/{len(dataloader)} acc={interim_acc:.4f}"
                    if is_docvqa:
                        interim_anls = anls_sum / max(1, total)
                        msg += f" anls={interim_anls:.4f}"
                    print(msg)
                    if args.log_samples:
                        pseudo_entries = split_data.get("pseudo_text", [[]])[0] if isinstance(split_data.get("pseudo_text"), list) else []
                        pseudo_summary = summarize_entries(pseudo_entries)
                        print(
                            f"  id={batch['ids'][0]} | img={img_path} | pred_raw={pred_text} | pred_scored={pred_for_score} | target={target}"
                        )
                        print(f"  pseudo_text={pseudo_summary}")
                        print("  retrieved=[]")
                continue
            if args.native_eval:
                # Expect batch_size=1; use corrupted split when apply_corruption is enabled
                split_key = "corrupted" if apply_corruption else "clean"
                split_data = batch[split_key]
                q = split_data["question"][0]
                img = split_data["images"][0] if split_data["images"][0] is not None else None
                img_path = split_data["image_path"][0] if isinstance(split_data.get("image_path"), list) else None
                # Debug: show whether corrupted images are present
                if idx == 0 and args.log_samples:
                    print(f"[debug] split={split_key} img_type={type(img)} img_path={img_path}")
                loaded = img is not None
                candidates: List[Path] = []
                if not loaded and img_path:
                    from PIL import Image

                    cand = Path(img_path)
                    candidates = [cand]
                    # Fix common duplicated segments.
                    fixes = {
                        "documents/documents": "documents",
                        "charts/charts": "charts",
                        "images/images": "images",
                        "pics/pics": "pics",
                    }
                    for dup, fix in fixes.items():
                        if dup in str(cand):
                            candidates.append(Path(str(cand).replace(dup, fix, 1)))
                    # Expand relative paths under project_root for robustness.
                    expanded: List[Path] = []
                    for c in candidates:
                        expanded.append(c)
                        if not c.is_absolute():
                            expanded.append(project_root / c)
                    candidates = expanded

                    # If still relative and not already prefixed by dataset_root_rel, try under dataset_root.
                    if dataset_root:
                        extra: List[Path] = []
                        for c in candidates:
                            if c.is_absolute():
                                continue
                            if dataset_root_rel is not None and str(c).startswith(str(dataset_root_rel)):
                                continue
                            extra.append(dataset_root / c)
                            extra.append(dataset_root / c.name)
                        candidates.extend(extra)
                    for c in candidates:
                        if c.exists():
                            img = Image.open(c).convert("RGB")
                            img_path = str(c)
                            loaded = True
                            break
                    # Apply corruption on-the-fly when dataset did not provide an in-memory corrupted image.
                    if loaded and apply_corruption and split_key == "corrupted" and eval_image_corruptor is not None:
                        img = eval_image_corruptor(img)
                if not loaded:
                    print(
                        f"[WARN] image not found for id={batch['ids'][0]} path={img_path}, "
                        f"tried {[str(c) for c in candidates]}; skip sample"
                    )
                    continue
                # Encourage short, direct answers
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q + "\nPlease answer with the short answer only."},
                        ],
                    }
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                proc_inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(native_model.device)
                input_len = proc_inputs["input_ids"].shape[1]
                gen_out = native_model.generate(
                    **proc_inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Slice off the prompt tokens
                gen_ids = gen_out[0][input_len:]
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred_text = first_sentence(clean_generation_output(pred_text))
                target = batch["clean"]["labels"][0]
                pred_for_score = best_span_match(pred_text, target)
                raw_preds = [pred_text]
                total += 1
                if is_correct(pred_for_score, target):
                    correct += 1
                anls_sum += anls(pred_for_score, target)
                predictions = [pred_for_score]
                targets = [target]
                if args.predictions:
                    dump_rows.append(
                        {
                            "id": batch["ids"][0],
                            "prediction": pred_text,
                            "scored_prediction": pred_for_score,
                            "target": target,
                            "image_path": img_path,
                        }
                    )
                if not is_correct(pred_for_score, target) and args.errors is not None:
                    error_rows.append(
                        {
                            "id": batch["ids"][0],
                            "image_path": img_path,
                            "prediction": pred_text,
                            "scored_prediction": pred_for_score,
                            "target": target,
                        }
                    )
                total_batches += 1
                # interim logging
                if args.log_interval and (idx + 1) % args.log_interval == 0:
                    interim_acc = correct / max(1, total)
                    msg = f"[eval] step {idx+1}/{len(dataloader)} acc={interim_acc:.4f}"
                    if is_docvqa:
                        interim_anls = anls_sum / max(1, total)
                        msg += f" anls={interim_anls:.4f}"
                    print(msg)
                    if args.log_samples:
                        k = min(args.log_samples, len(predictions))
                        for j in range(k):
                            pseudo_entries = []
                            if isinstance(split_data.get("pseudo_text"), list) and j < len(split_data["pseudo_text"]):
                                pseudo_entries = split_data["pseudo_text"][j] or []
                            pseudo_summary = summarize_entries(pseudo_entries)
                            print(
                                f"  id={batch['ids'][j]} | img={batch['clean']['image_path'][j] if isinstance(batch['clean'].get('image_path'), list) else None} "
                                f"| pred_raw={raw_preds[j]} | pred_scored={predictions[j]} | target={targets[j]}"
                            )
                            print(f"  pseudo_text={pseudo_summary}")
                            print("  retrieved=[]")
                continue

            try:
                device = model.base_vlm.model.get_input_embeddings().weight.device
            except Exception:
                device = next(model.parameters()).device
            clean_split = batch["clean"]
            corrupted_split = batch["corrupted"]
            tokenizer = model.base_vlm.tokenizer
            max_len = getattr(model.config, "max_seq_length", 2048)

            use_pseudo_text = bool(getattr(model.config, "enable_retrieval", False)) and not args.disable_retrieval
            corrupted_tokens, corrupted_pseudo = tokenize_with_template(
                corrupted_split,
                tokenizer,
                max_len,
                device,
                use_chat_template=use_chat_template,
                use_pseudo_text=use_pseudo_text,
            )
            corrupted_vision = get_vision_embeddings(model, corrupted_split, device)
            student_out = model(
                input_ids=corrupted_tokens["input_ids"],
                attention_mask=corrupted_tokens["attention_mask"],
                pixel_values=corrupted_vision,
                labels=None,
                pseudo_text=corrupted_pseudo,
                is_clean_branch=False,
            )

            # Manual CE loss to avoid shape mismatch; mirror training logic.
            vision_tokens = corrupted_vision.size(1)
            prefix_tokens = 0
            imputation_tokens = 0
            try:
                retrieval = student_out.get("retrieval") if isinstance(student_out, dict) else None
                evidence_emb = retrieval.get("embeddings") if isinstance(retrieval, dict) else None
                evidence_count = 0
                if torch.is_tensor(evidence_emb) and evidence_emb.numel() > 0:
                    evidence_count = int(evidence_emb.size(1))
                if evidence_count > 0:
                    if bool(getattr(model.config, "enable_prefix", True)):
                        prefix_len_cfg = int(getattr(getattr(model, "reconstruction", None).config, "prefix_length", 0))
                        prefix_tokens = min(prefix_len_cfg, evidence_count) if prefix_len_cfg > 0 else 0
                    if bool(getattr(model.config, "enable_imputation", True)):
                        imputation_tokens = int(
                            getattr(getattr(model, "reconstruction", None).config, "imputation_tokens", 0)
                        )
            except Exception:
                prefix_tokens = 0
                imputation_tokens = 0
            logits = student_out["logits"]
            if not torch.isfinite(logits).all():
                nan_logits_batches += 1
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = causal_ce(
                logits.float(),
                corrupted_tokens["labels"],
                prefix_tokens=prefix_tokens,
                imputation_tokens=imputation_tokens,
                vision_tokens=int(vision_tokens),
            )
            if torch.isfinite(loss):
                total_loss += loss.item()
                total_batches += 1
            else:
                nan_loss_batches += 1

            aligned_labels = align_labels_for_r3(
                corrupted_tokens["labels"],
                prefix_tokens=prefix_tokens,
                imputation_tokens=imputation_tokens,
                vision_tokens=int(vision_tokens),
                seq_len=logits.size(1),
            )
            raw_predictions = decode_predictions(logits, aligned_labels, tokenizer)
            predictions = [first_sentence(clean_generation_output(p)) for p in raw_predictions]
            retrieval_texts = None
            retrieval_scores = None
            if isinstance(retrieval, dict):
                retrieval_texts = retrieval.get("texts")
                retrieval_scores = retrieval.get("scores")
            scored_preds = [best_span_match(p, t) for p, t in zip(predictions, corrupted_split["labels"])]
            targets = corrupted_split["labels"]
            for j, (sample_id, pred_raw, scored_pred, target) in enumerate(
                zip(batch["ids"], predictions, scored_preds, targets)
            ):
                total += 1
                if is_correct(scored_pred, target):
                    correct += 1
                anls_sum += anls(scored_pred, target)
                if args.predictions:
                    dump_rows.append(
                        {"id": sample_id, "prediction": pred_raw, "scored_prediction": scored_pred, "target": target}
                    )
                if not is_correct(scored_pred, target) and args.errors is not None:
                    err_img = None
                    if isinstance(corrupted_split.get("image_path"), list) and j < len(corrupted_split["image_path"]):
                        err_img = corrupted_split["image_path"][j]
                    error_rows.append(
                        {
                            "id": sample_id,
                            "image_path": err_img,
                            "prediction": pred_raw,
                            "scored_prediction": scored_pred,
                            "target": target,
                        }
                    )

            if args.log_interval and (idx + 1) % args.log_interval == 0:
                interim_acc = correct / max(1, total)
                msg = f"[eval] step {idx+1}/{len(dataloader)} acc={interim_acc:.4f}"
                if is_docvqa:
                    interim_anls = anls_sum / max(1, total)
                    msg += f" anls={interim_anls:.4f}"
                print(msg)
                if args.log_samples:
                    k = min(args.log_samples, len(predictions))
                    for j in range(k):
                        img_path = None
                        if isinstance(corrupted_split.get("image_path"), list) and j < len(corrupted_split["image_path"]):
                            img_path = corrupted_split["image_path"][j]
                        print(
                            f"  id={batch['ids'][j]} | img={img_path} | pred_raw={predictions[j]} | pred_scored={scored_preds[j]} | target={targets[j]}"
                        )
                        pseudo_entries = []
                        if isinstance(corrupted_split.get("pseudo_text"), list) and j < len(corrupted_split["pseudo_text"]):
                            pseudo_entries = corrupted_split["pseudo_text"][j] or []
                        pseudo_summary = summarize_entries(pseudo_entries)
                        retrieved = []
                        if isinstance(retrieval_texts, list) and j < len(retrieval_texts):
                            texts = retrieval_texts[j]
                            scores = None
                            if torch.is_tensor(retrieval_scores):
                                scores = retrieval_scores[j].detach().cpu().tolist()
                            for idx_r, text in enumerate(texts):
                                entry = str(text).replace("\n", " ").strip()
                                if not entry:
                                    continue
                                if scores is not None and idx_r < len(scores):
                                    entry = f"{scores[idx_r]:.3f}:{entry}"
                                retrieved.append(entry)
                        retrieved = summarize_entries(retrieved)
                        print(f"  pseudo_text={pseudo_summary}")
                        print(f"  retrieved={retrieved}")

    avg_loss = total_loss / max(1, total_batches)
    accuracy = correct / max(1, total)
    metrics = {
        "split": split,
        "samples": total,
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
    }
    if is_docvqa:
        anls_score = anls_sum / max(1, total)
        metrics["anls"] = round(anls_score, 4)
    if nan_loss_batches:
        metrics["nan_loss_batches"] = nan_loss_batches
    if nan_logits_batches:
        metrics["nan_logits_batches"] = nan_logits_batches
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.predictions and dump_rows:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {args.predictions}")

    if args.errors and error_rows:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with args.errors.open("w", encoding="utf-8") as f:
            for row in error_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Errors saved to {args.errors}")


if __name__ == "__main__":
    main()

# python evaluate_r3.py --config configs/default.yaml --dataset_type mp_docvqa \
#   --checkpoint path/to/ckpt.pt \
#   --apply_corruption \
#   --predictions preds.jsonl
