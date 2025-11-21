"""
Evaluation script for R^3 checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from data_pipeline.datasets.textvqa import TextVQADataset
from data_pipeline.datasets.mp_docvqa import MPDocVQADataset
from data_pipeline.datasets.infovqa import InfoVQADataset
from data_pipeline.corruptions import ImageCorruptor, PseudoTextCorruptor
from r3.r3_model import R3Model, R3ModelConfig
from train_r3 import R3Dataset, collate_fn, load_yaml, load_pseudo_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R^3 checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to finetuned checkpoint; if omitted, use base backbone.")
    parser.add_argument("--split", type=str, default=None, help="Override dataset split for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for quick smoke tests.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional JSONL to dump predictions.")
    parser.add_argument("--dataset_type", type=str, default="textvqa", choices=["textvqa", "mp_docvqa", "infovqa"])
    parser.add_argument("--apply_corruption", action="store_true", help="Apply pre-encoding modality drops (Image/Pseudo-text).")
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
    pred_norm = pred.lower().strip()
    target_norm = target.lower().strip()
    if not target_norm:
        return 1.0 if not pred_norm else 0.0
    distance = levenshtein_distance(pred_norm, target_norm)
    score = 1.0 - distance / max(len(pred_norm), len(target_norm))
    return score if score >= threshold else 0.0


def decode_predictions(logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> List[str]:
    pred_ids = logits.argmax(dim=-1)
    predictions: List[str] = []
    for row_pred, row_label in zip(pred_ids, labels):
        mask = row_label != -100
        if mask.sum() == 0:
            predictions.append("")
            continue
        ids = row_pred[mask]
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        predictions.append(text)
    return predictions


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    split = args.split or eval_cfg.get("split") or dataset_cfg.get("eval_split", "val")
    apply_corruption = args.apply_corruption or eval_cfg.get("apply_corruption", False)

    dataset_root = Path(dataset_cfg["root"])
    pseudo_corpus = load_pseudo_corpus(dataset_cfg.get("pseudo_corpus"))
    if args.dataset_type == "mp_docvqa":
        base_dataset = MPDocVQADataset(dataset_root, split=split)
    elif args.dataset_type == "infovqa":
        base_dataset = InfoVQADataset(dataset_root, split=split)
    else:
        base_dataset = TextVQADataset(dataset_root, split=split)

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

    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg.get("batch_size", dataset_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    model_section = cfg.get("model", {})
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
        enable_consistency=False,
        top_k=model_section.get("top_k", 3),
    )
    model = R3Model(model_cfg).to(args.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()

    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0
    anls_sum = 0.0
    dump_rows: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            device = next(model.parameters()).device
            clean_split = batch["clean"]
            corrupted_split = batch["corrupted"]
            tokenizer = model.base_vlm.tokenizer
            max_len = getattr(model.config, "max_seq_length", 1024)
            from train_r3 import R3Trainer  # reuse tokenizer/vision utilities

            trainer_stub = R3Trainer(
                model=model,
                args=None,
                train_dataset=None,
                data_collator=None,
            )
            clean_tokens, clean_pseudo = trainer_stub._tokenize_branch(tokenizer, clean_split, max_len, device)
            corrupted_tokens, corrupted_pseudo = trainer_stub._tokenize_branch(tokenizer, corrupted_split, max_len, device)
            clean_vision = trainer_stub._get_vision_embeddings(model, clean_split, device)
            corrupted_vision = trainer_stub._get_vision_embeddings(model, corrupted_split, device)

            clean_out = model(
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

            loss = student_out["loss"] if student_out.get("loss") is not None else torch.tensor(0.0, device=device)
            total_loss += loss.item()
            total_batches += 1

            predictions = decode_predictions(student_out["logits"], corrupted_tokens["labels"], tokenizer)
            targets = corrupted_split["labels"]
            for sample_id, pred, target in zip(batch["ids"], predictions, targets):
                total += 1
                if normalize_text(pred) == normalize_text(target):
                    correct += 1
                anls_sum += anls(pred, target)
                if args.predictions:
                    dump_rows.append({"id": sample_id, "prediction": pred, "target": target})

    avg_loss = total_loss / max(1, total_batches)
    accuracy = correct / max(1, total)
    anls_score = anls_sum / max(1, total)

    metrics = {
        "split": split,
        "samples": total,
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "anls": round(anls_score, 4),
    }
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

