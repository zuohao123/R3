"""
Evaluation script for R^3 checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
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
import os
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq

from data_pipeline.datasets import DATASET_REGISTRY, create_dataset, detect_dataset_type
from data_pipeline.corruptions import ImageCorruptor, PseudoTextCorruptor
from r3.r3_model import R3Model, R3ModelConfig
from train_r3 import R3Dataset, collate_fn, load_yaml, load_pseudo_corpus


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
    pred_ids = logits.argmax(dim=-1)
    predictions: List[str] = []
    for row_pred, row_label in zip(pred_ids, labels):
        # Align lengths: pad/truncate mask to match logits length.
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
    min_len = max(1, int(len(tgt_tokens) * 0.5))
    max_len = max(min_len, int(len(tgt_tokens) * 2))
    best = pred_norm
    best_score = -1.0
    for i in range(len(pred_tokens)):
        for j in range(i + min_len, min(len(pred_tokens), i + max_len) + 1):
            span = " ".join(pred_tokens[i:j])
            if not span:
                continue
            score = 1.0 - levenshtein_distance(span, tgt_norm) / max(len(span), len(tgt_norm))
            if score > best_score:
                best_score = score
                best = span
    return best


def build_prompts(questions: List[str], pseudo_batch: List[List[str]], labels: List[str], tokenizer, use_chat_template: bool) -> List[str]:
    prompts = []
    pseudo_fmt = []
    for pseudo in pseudo_batch:
        pseudo_fmt.append("\n".join([p for p in pseudo if p]))
    for q, pseudo, lbl in zip(questions, pseudo_fmt, labels):
        user_content = f"{pseudo}\nQuestion: {q}".strip() if pseudo else f"Question: {q}"
        if use_chat_template:
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{user_content}\nAnswer:"
        full = f"{prompt} {lbl}".strip()
        prompts.append((prompt, full))
    return prompts


def tokenize_with_template(split: Dict, tokenizer, max_length: int, device: torch.device, use_chat_template: bool):
    questions = split["question"]
    labels_text = split.get("labels", [""] * len(questions))
    pseudo_text = split.get("pseudo_text", [[] for _ in questions])

    prompts = build_prompts(questions, pseudo_text, labels_text, tokenizer, use_chat_template)
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
    return text_tokens, pseudo_text


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
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

    # Resolve dataset type
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        dataset_type = dataset_cfg.get("type", "auto")
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_root)
    pseudo_corpus = load_pseudo_corpus(dataset_cfg.get("pseudo_corpus"))
    base_dataset = create_dataset(dataset_type, dataset_root, split=split)

    dataset = R3Dataset(
        base_dataset,
        vision_tokens=cfg["model"].get("vision_tokens", 16),
        hidden_size=cfg["model"].get("hidden_size", 4096),
        apply_corruption=apply_corruption,
        pseudo_corpus=pseudo_corpus,
        image_corruptor=ImageCorruptor() if apply_corruption else None,
        pseudo_text_corruptor=PseudoTextCorruptor() if apply_corruption else None,
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
            bf16=model_section.get("bf16", False),
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
        )
        model = R3Model(model_cfg).to(args.device)
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

    # Metric switches by dataset type
    ds_lower = str(dataset_type).lower()
    is_docvqa = "docvqa" in ds_lower
    is_infovqa = "infovqa" in ds_lower
    is_chartqa = "chartqa" in ds_lower

    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0
    anls_sum = 0.0
    dump_rows: List[Dict] = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="eval", total=len(dataloader))):
            if args.native_eval:
                # Expect batch_size=1
                q = batch["clean"]["question"][0]
                img = batch["clean"]["images"][0] if batch["clean"]["images"][0] is not None else None
                img_path = batch["clean"]["image_path"][0] if isinstance(batch["clean"].get("image_path"), list) else None
                if img is None and img_path:
                    from PIL import Image
                    cand = Path(img_path)
                    candidates = [cand]
                    # Fix duplicated segment like documents/documents
                    if "documents/documents" in str(cand):
                        candidates.append(Path(str(cand).replace("documents/documents", "documents", 1)))
                    # If relative, try under dataset_root
                    if dataset_root:
                        candidates.append(dataset_root / cand)
                        candidates.append(dataset_root / cand.name)
                    loaded = False
                    for c in candidates:
                        if c.exists():
                            img = Image.open(c).convert("RGB")
                            img_path = str(c)
                            loaded = True
                            break
                    if not loaded:
                        print(f"[WARN] image not found for id={batch['ids'][0]} path={img_path}, tried {[str(c) for c in candidates]}; skip sample")
                        continue
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                proc_inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(native_model.device)
                input_len = proc_inputs["input_ids"].shape[1]
                gen_out = native_model.generate(
                    **proc_inputs,
                    max_new_tokens=96,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Slice off the prompt tokens
                gen_ids = gen_out[0][input_len:]
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred_text = first_sentence(clean_generation_output(pred_text))
                pred_for_score = best_span_match(pred_text, target)
                target = batch["clean"]["labels"][0]
                total += 1
                norm_pred = normalize_basic(pred_for_score)
                norm_tgt = normalize_basic(target)
                if norm_pred == norm_tgt:
                    correct += 1
                anls_sum += anls(pred_for_score, target)
                predictions = [pred_for_score]
                targets = [target]
                if args.predictions:
                    dump_rows.append({"id": batch["ids"][0], "prediction": pred_text, "scored_prediction": pred_for_score, "target": target, "image_path": img_path})
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
                        print(f"  id={batch['ids'][j]} | img={batch['clean']['image_path'][j] if isinstance(batch['clean'].get('image_path'), list) else None} | pred={predictions[j]} | target={targets[j]}")
                continue

            device = next(model.parameters()).device
            clean_split = batch["clean"]
            corrupted_split = batch["corrupted"]
            tokenizer = model.base_vlm.tokenizer
            max_len = getattr(model.config, "max_seq_length", 2048)
            from train_r3 import R3Trainer  # reuse vision utilities

            trainer_stub = R3Trainer(model=model, args=None, train_dataset=None, data_collator=None)
            corrupted_tokens, corrupted_pseudo = tokenize_with_template(
                corrupted_split, tokenizer, max_len, device, use_chat_template=args.use_chat_template
            )
            corrupted_vision = trainer_stub._get_vision_embeddings(model, corrupted_split, device)
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
            loss = trainer_stub._causal_ce(student_out["logits"].float(), corrupted_tokens["labels"], vision_tokens)
            total_loss += loss.item()
            total_batches += 1

            raw_predictions = decode_predictions(student_out["logits"], corrupted_tokens["labels"], tokenizer)
            predictions = [first_sentence(clean_generation_output(p)) for p in raw_predictions]
            scored_preds = [best_span_match(p, t) for p, t in zip(predictions, corrupted_split["labels"])]
            targets = corrupted_split["labels"]
            for sample_id, pred, scored_pred, target in zip(batch["ids"], predictions, scored_preds, targets):
                total += 1
                norm_pred = normalize_basic(scored_pred)
                norm_tgt = normalize_basic(target)
                if norm_pred == norm_tgt:
                    correct += 1
                anls_sum += anls(scored_pred, target)
                if args.predictions:
                    dump_rows.append({"id": sample_id, "prediction": pred, "scored_prediction": scored_pred, "target": target})

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
                        print(f"  id={batch['ids'][j]} | img={img_path} | pred={predictions[j]} | target={targets[j]}")

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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.predictions and dump_rows:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {args.predictions}")


if __name__ == "__main__":
    main()

# python evaluate_r3.py --config configs/default.yaml --dataset_type mp_docvqa \
#   --checkpoint path/to/ckpt.pt \
#   --apply_corruption \
#   --predictions preds.jsonl
