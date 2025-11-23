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
from data_pipeline.datasets.chartqa import ChartQADataset
from data_pipeline.datasets.docvqa import DocVQADataset
from data_pipeline.datasets.slidevqa import SlideVQADataset
from data_pipeline.pseudo_text import save_corpus
from r3.retrieval_module import PseudoTextBuilder, PseudoTextBuilderConfig

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
    parser.add_argument("--dataset_root", type=Path, help="Path to single dataset directory.")
    parser.add_argument("--dataset_roots", type=str, nargs="+", help="Paths to multiple dataset directories.")
    parser.add_argument("--dataset_type", type=str, default="auto", 
                       choices=["auto", "textvqa", "mp_docvqa", "infovqa", "chartqa", "docvqa", "slidevqa"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap per dataset.")
    parser.add_argument("--enable_ocr", action="store_true", help="Run pytesseract OCR when samples lack tokens.")
    parser.add_argument("--caption_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Optional vision-language caption model.")
    parser.add_argument("--default_conf", type=float, default=0.75)
    return parser.parse_args()


def detect_dataset_type(root: Path) -> str:
    """
    Auto-detect dataset type based on directory structure and annotation files.
    """
    root = Path(root)
    
    # Check for specific annotation files
    if (root / f"textvqa_train.json").exists() or (root / f"textvqa_val.json").exists():
        return "textvqa"
    elif (root / f"mp_docvqa_train.json").exists() or (root / f"mp_docvqa_val.json").exists():
        return "mp_docvqa"
    elif (root / f"infovqa_train.json").exists() or (root / f"infovqa_val.json").exists():
        return "infovqa"
    elif (root / f"chartqa_train.json").exists() or (root / f"chartqa_val.json").exists():
        return "chartqa"
    elif (root / f"docvqa_train.json").exists() or (root / f"docvqa_val.json").exists():
        return "docvqa"
    elif (root / f"slidevqa_train.json").exists() or (root / f"slidevqa_val.json").exists():
        return "slidevqa"
    
    # Fallback: check directory names
    root_name = root.name.lower()
    if "textvqa" in root_name:
        return "textvqa"
    elif "mp_docvqa" in root_name or "mpdocvqa" in root_name:
        return "mp_docvqa"
    elif "infovqa" in root_name:
        return "infovqa"
    elif "chartqa" in root_name:
        return "chartqa"
    elif "docvqa" in root_name:
        return "docvqa"
    elif "slidevqa" in root_name:
        return "slidevqa"
    
    # Default fallback
    print(f"Warning: Could not auto-detect dataset type for {root}, defaulting to textvqa")
    return "textvqa"


def create_dataset(root: Path, dataset_type: str, split: str):
    """
    Create dataset instance based on type.
    """
    if dataset_type == "mp_docvqa":
        return MPDocVQADataset(root, split=split)
    elif dataset_type == "infovqa":
        return InfoVQADataset(root, split=split)
    elif dataset_type == "chartqa":
        return ChartQADataset(root, split=split)
    elif dataset_type == "docvqa":
        return DocVQADataset(root, split=split)
    elif dataset_type == "slidevqa":
        return SlideVQADataset(root, split=split)
    else:  # textvqa or default
        return TextVQADataset(root, split=split)


def process_single_dataset(root: Path, dataset_type: str, split: str, builder: PseudoTextBuilder, 
                          caption_fn, enable_ocr: bool, limit: Optional[int] = None) -> List[Dict]:
    """
    Process a single dataset and return pseudo-text artifacts.
    """
    print(f"Processing dataset: {root} (type: {dataset_type}, split: {split})")
    
    try:
        dataset = create_dataset(root, dataset_type, split)
    except FileNotFoundError as e:
        print(f"Warning: Skipping {root} - {e}")
        return []
    
    artifacts = []
    upper = min(len(dataset), limit) if limit else len(dataset)
    
    for idx in range(upper):
        try:
            sample = dataset[idx]
            extra = sample.setdefault("extra", {})
            
            # Add OCR if enabled and missing
            if enable_ocr and not extra.get("ocr_tokens"):
                if sample.get("image_path"):
                    try:
                        extra["ocr_tokens"] = run_ocr(sample["image_path"])
                    except Exception as e:
                        print(f"Warning: OCR failed for {sample['id']}: {e}")
            
            # Add caption if model provided
            if caption_fn and sample.get("image_path"):
                try:
                    caption = caption_fn(sample["image_path"])
                    if caption:
                        extra.setdefault("captions", []).append(caption)
                except Exception as e:
                    print(f"Warning: Caption generation failed for {sample['id']}: {e}")
            
            # Build pseudo-text entries
            entries = builder.build(sample)
            if entries:
                artifacts.append({
                    "doc_id": sample["id"],
                    "pseudo_text": entries,
                    "metadata": {
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "image_path": sample.get("image_path", ""),
                        "split_index": idx,
                        "dataset_type": dataset_type,
                        "dataset_root": str(root),
                    },
                })
        except Exception as e:
            print(f"Warning: Failed to process sample {idx} from {root}: {e}")
            continue
    
    print(f"Processed {len(artifacts)} samples from {root}")
    return artifacts


def main() -> None:
    args = parse_args()
    
    # Determine dataset roots to process
    dataset_roots = []
    if args.dataset_root:
        dataset_roots.append(Path(args.dataset_root))
    if args.dataset_roots:
        dataset_roots.extend([Path(root) for root in args.dataset_roots])
    
    if not dataset_roots:
        raise ValueError("Must specify either --dataset_root or --dataset_roots")
    
    config = PseudoTextBuilderConfig(default_conf=args.default_conf)
    builder = PseudoTextBuilder(config=config)
    caption_fn = None
    if args.caption_model:
        caption_fn = build_captions(args.caption_model)
    
    all_artifacts = []
    
    for root in dataset_roots:
        if not root.exists():
            print(f"Warning: Dataset root {root} does not exist, skipping...")
            continue
        
        # Auto-detect dataset type if needed
        dataset_type = args.dataset_type
        if dataset_type == "auto":
            dataset_type = detect_dataset_type(root)
        
        # Process this dataset
        artifacts = process_single_dataset(
            root, dataset_type, args.split, builder, caption_fn, 
            args.enable_ocr, args.limit
        )
        all_artifacts.extend(artifacts)
    
    # Save combined corpus
    save_corpus(all_artifacts, args.output)
    print(f"Combined pseudo-text corpus with {len(all_artifacts)} entries saved to {args.output}")


if __name__ == "__main__":
    main()

# 使用示例:

# 1. 处理单个数据集 (自动检测类型)
# python build_pseudo_text.py \
#   --dataset_root /path/to/textvqa \
#   --split train \
#   --output ./artifacts/pseudo_text_train.jsonl \
#   --enable_ocr

# 2. 处理多个数据集 (批量处理)
# python build_pseudo_text.py \
#   --dataset_roots /path/to/textvqa /path/to/mp_docvqa /path/to/infovqa /path/to/chartqa \
#   --split train \
#   --output ./artifacts/pseudo_text_combined_train.jsonl \
#   --enable_ocr \
#   --caption_model Qwen/Qwen3-VL-8B-Instruct

# 3. 指定特定数据集类型
# python build_pseudo_text.py \
#   --dataset_root /path/to/custom_dataset \
#   --dataset_type mp_docvqa \
#   --split train \
#   --output ./artifacts/pseudo_text_custom_train.jsonl \
#   --enable_ocr