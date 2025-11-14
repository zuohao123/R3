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
from data_pipeline.vision_encoder import VisionEncoder
from r3.r3_model import R3Model, R3ModelConfig
from train_r3 import R3Dataset, collate_fn, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R^3 checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None, help="Override dataset split for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for quick smoke tests.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional JSONL to dump predictions.")
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
    apply_corruption = eval_cfg.get("apply_corruption", False)

    dataset_root = Path(dataset_cfg["root"])
    base_dataset = TextVQADataset(dataset_root, split=split)

    vision_cfg = cfg.get("vision", {})
    vision_encoder = None
    if vision_cfg.get("encoder"):
        vision_encoder = VisionEncoder(
            model_name=vision_cfg.get("encoder", "openai/clip-vit-large-patch14"),
            device=vision_cfg.get("device", "cpu"),
            cache_size=vision_cfg.get("cache_size", 256),
        )

    dataset = R3Dataset(
        base_dataset,
        vision_encoder,
        vision_tokens=cfg["model"].get("vision_tokens", 16),
        hidden_size=cfg["model"].get("hidden_size", 4096),
        apply_corruption=apply_corruption,
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
            outputs = model(batch)  # 推理阶段仍沿用训练时的三路输入装配
            total_loss += outputs["loss"].item()
            total_batches += 1
            predictions = decode_predictions(outputs["corrupted_logits"], outputs["corrupted_labels"], model.base_vlm.tokenizer)
            targets = batch["corrupted"]["labels"]
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
