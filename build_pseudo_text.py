"""
Utility script to generate pseudo-text corpora when datasets lack OCR/caption fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from data_pipeline.datasets.textvqa import TextVQADataset
from data_pipeline.datasets.mp_docvqa import MPDocVQADataset
from data_pipeline.datasets.infovqa import InfoVQADataset
from data_pipeline.pseudo_text import save_corpus
from r3.retrieval_module import PseudoTextBuilder

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


def run_ocr(image_path: str) -> List[Dict]:
    # 使用 pytesseract 获得最基础的 OCR token，保证没有 OCR 标注的样本也能构建伪文本
    if pytesseract is None:
        raise ImportError("pytesseract is required for OCR-based pseudo-text generation.")
    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    tokens: List[Dict] = []
    for idx, text in enumerate(data["text"]):
        span = text.strip()
        if not span:
            continue
        bbox = [
            int(data["left"][idx]),
            int(data["top"][idx]),
            int(data["left"][idx] + data["width"][idx]),
            int(data["top"][idx] + data["height"][idx]),
        ]
        conf = float(data["conf"][idx]) if data["conf"][idx] != "-1" else 0.5
        tokens.append(
            {
                "text": span,
                "bbox": bbox,
                "conf": conf / 100.0,
                "src": "ocr",
            }
        )
    return tokens


def build_captions(model_name: str):
    # 通过任意开源视觉语言模型补充描述型 caption，提升语料覆盖度
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    def infer(image_path: str) -> Optional[str]:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=64)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    return infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pseudo-text corpus for PMC datasets.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to dataset directory.")
    parser.add_argument("--dataset_type", type=str, default="textvqa", choices=["textvqa", "mp_docvqa", "infovqa"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--enable_ocr", action="store_true", help="Run pytesseract OCR when samples lack tokens.")
    parser.add_argument("--caption_model", type=str, default=None, help="Optional vision-language caption model.")
    parser.add_argument("--default_conf", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    if args.dataset_type == "mp_docvqa":
        dataset = MPDocVQADataset(root, split=args.split)
    elif args.dataset_type == "infovqa":
        dataset = InfoVQADataset(root, split=args.split)
    else:
        dataset = TextVQADataset(root, split=args.split)
    builder = PseudoTextBuilder(default_conf=args.default_conf)
    caption_fn = None
    if args.caption_model:
        caption_fn = build_captions(args.caption_model)
    artifacts = []
    upper = min(len(dataset), args.limit) if args.limit else len(dataset)
    for idx in range(upper):
        sample = dataset[idx]
        extra = sample.setdefault("extra", {})
        if args.enable_ocr and not extra.get("ocr_tokens"):
            if not sample.get("image_path"):
                raise ValueError("Image path missing; cannot run OCR.")
            extra["ocr_tokens"] = run_ocr(sample["image_path"])  # 缺失 OCR 时即时推理
        if caption_fn:
            caption = caption_fn(sample.get("image_path", ""))
            if caption:
                extra.setdefault("captions", []).append(caption)  # 记录多条 caption 以丰富语料
        entries = builder.build(sample)
        artifacts.append(
            {
                "doc_id": sample["id"],
                "pseudo_text": entries,
                "metadata": {
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "image_path": sample.get("image_path", ""),
                    "split_index": idx,
                },
            }
        )
    save_corpus(artifacts, args.output)
    print(f"Pseudo-text corpus saved to {args.output}")


if __name__ == "__main__":
    main()
