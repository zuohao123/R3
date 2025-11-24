"""
Utility script to generate pseudo-text corpora when datasets lack OCR/caption fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from data_pipeline.datasets import DATASET_REGISTRY, create_dataset, detect_dataset_type
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


def resolve_model_path(model_name: str, cache_dir: Optional[Path], provider: str, token: Optional[str]) -> str:
    """
    Resolve model weights location. If provider=modelscope, download via modelscope snapshot_download
    so weights stay in本地缓存; otherwise fallback to Hugging Face.
    """
    if provider.lower() == "modelscope":
        try:
            from modelscope import snapshot_download  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "provider=modelscope 但未安装 modelscope，请先 pip install modelscope。"
            ) from exc
        local_dir = snapshot_download(
            model_id=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            use_auth_token=token,
        )
        return local_dir
    return model_name


def build_captions(
    model_name: str,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    provider: str = "huggingface",
    ignore_mismatched_sizes: bool = True,
):
    model_path = resolve_model_path(model_name, cache_dir, provider, token)
    # 显式检查 config.model_type，提示用户是否加载到了正确的 Qwen 版本
    try:
        cfg_obj = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        expected = "qwen3_vl" if "qwen3" in model_name.lower() else "qwen2_vl"
        if getattr(cfg_obj, "model_type", None) != expected:
            print(
                f"警告: 期望 {expected}，但实际 config.model_type={getattr(cfg_obj, 'model_type', None)}，"
                " 可能是缓存/transformers 版本导致加载了错误的模型类。"
            )
            print(f"  使用的权重路径: {model_path}")
    except Exception as cfg_err:
        print(f"提示: 读取 config 失败（可忽略）: {cfg_err}")
    if "Qwen" in model_name and "VL" in model_name:
        print(f"正在加载 Qwen-VL 模型: {model_name}")
        try:
            import torchvision
        except ImportError:
            print("警告: torchvision 未安装，正在尝试安装...")
            import subprocess
            subprocess.check_call(["pip", "install", "torchvision"])
            import torchvision
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

        print(f"成功加载 Qwen-VL 模型: {model_name}")

        def qwen_infer(image_path: str) -> Optional[str]:
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe this image briefly."},
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = processor.process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=64)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return output_text[0].strip() if output_text else None
            except Exception as e:
                print(f"Qwen-VL 图像描述生成失败 {image_path}: {e}")
                return None

        return qwen_infer

    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        model.eval()
    except Exception as e:
        print(f"警告: 无法加载模型 {model_name}: {e}")
        return None

    def general_infer(image_path: str) -> Optional[str]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"图像描述生成失败 {image_path}: {e}")
            return None

    return general_infer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pseudo-text corpus for PMC datasets.")
    parser.add_argument("--dataset_root", type=Path, help="Path to single dataset directory.")
    parser.add_argument("--dataset_roots", type=str, nargs="+", help="Paths to multiple dataset directories.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", *sorted(DATASET_REGISTRY.keys())],
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap per dataset.")
    parser.add_argument("--enable_ocr", action="store_true", help="Run pytesseract OCR when samples lack tokens.")
    parser.add_argument(
        "--caption_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Optional vision-language caption model.",
    )
    parser.add_argument("--default_conf", type=float, default=0.75)
    parser.add_argument(
        "--model_cache_dir",
        type=Path,
        default=Path("./hf_cache"),
        help="Directory to download/store caption/backbone models.",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="Optional token for gated models.")
    parser.add_argument("--local_files_only", action="store_true", help="Force using only local cached models.")
    parser.add_argument(
        "--provider",
        type=str,
        default="huggingface",
        choices=["huggingface"],
        help="Download provider for caption model.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Pass ignore_mismatched_sizes=True when loading caption model (helps Qwen3 checkpoints).",
    )
    return parser.parse_args()


def process_single_dataset(root: Path, dataset_type: str, split: str, builder: PseudoTextBuilder, 
                          caption_fn, enable_ocr: bool, limit: Optional[int] = None) -> List[Dict]:
    """
    Process a single dataset and return pseudo-text artifacts.
    """
    print(f"Processing dataset: {root} (type: {dataset_type}, split: {split})")
    
    try:
        dataset = create_dataset(dataset_type, root, split)
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
            
            # 处理 MTVQA 视频内容
            if dataset_type == "mtvqa" and sample.get("video_path"):
                try:
                    from data_pipeline.video_utils import VideoProcessor, process_mtvqa_video
                    video_processor = VideoProcessor(max_frames=8)
                    sample = process_mtvqa_video(sample, video_processor, caption_fn)
                except Exception as e:
                    print(f"Warning: Video processing failed for {sample['id']}: {e}")
            
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
        try:
            caption_fn = build_captions(
                args.caption_model,
                cache_dir=args.model_cache_dir,
                token=args.hf_token,
                local_files_only=args.local_files_only,
                provider=args.provider,
                ignore_mismatched_sizes=args.ignore_mismatched_sizes or True,
            )
        except Exception as e:
            print(f"警告: 图像描述模型初始化失败: {e}")
            print("将跳过图像描述生成，仅使用OCR和现有标注")
            caption_fn = None
    
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
#   --dataset_roots /path/to/textvqa /path/to/mp_docvqa /path/to/infovqa /path/to/chartqa /path/to/mtvqa \
#   --split train \
#   --output ./artifacts/pseudo_text_combined_train.jsonl \
#   --enable_ocr \
#   --caption_model Qwen/Qwen2-VL-7B-Instruct

# 3. 指定特定数据集类型
# python build_pseudo_text.py \
#   --dataset_root /path/to/custom_dataset \
#   --dataset_type mp_docvqa \
#   --split train \
#   --output ./artifacts/pseudo_text_custom_train.jsonl \
#   --enable_ocr


# python build_pseudo_text.py \
#   --dataset_root data_pipeline/data/infovqa \
#   --dataset_type infovqa \
#   --split train \
#   --output artifacts/infovqa_pseudo_text_train.jsonl \
#   --enable_ocr \
#   --caption_model Qwen/Qwen3-VL-8B-Instruct \
#   --provider huggingface \
#   --model_cache_dir ./hf_cache \
#   --local_files_only \
#   --ignore_mismatched_sizes
