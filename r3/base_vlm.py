"""
Base VLM wrapper around Qwen3-VL style backbones.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from pathlib import Path
from types import MethodType
import torch.nn as nn

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseVLMConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 16
    bf16: bool = True
    # Load backbone weights dtype. Options: "auto" (bf16->bf16 else fp32), "fp16", "fp32", "bf16".
    # Note: V100 does not support bf16; for 8×V100 DDP we recommend fp16 here.
    dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: Optional[str | Dict] = None
    low_cpu_mem_usage: bool = True
    gradient_checkpointing: bool = False
    tokenizer_path: Optional[str] = None
    adapter_path: Optional[str] = None
    provider: str = "huggingface"  # or "modelscope"
    token: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    local_files_only: bool = False


class BaseVLM(torch.nn.Module):
    def __init__(self, config: BaseVLMConfig) -> None:
        super().__init__()
        self.config = config
        model_path = self._prepare_model_source()
        tokenizer_source = config.tokenizer_path or model_path
        hf_kwargs = self._build_hf_kwargs()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **hf_kwargs)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IMPUTE_V]", "[IMPUTE_T]"]})
        try:
            self.processor = AutoProcessor.from_pretrained(tokenizer_source, **hf_kwargs)
        except Exception:
            self.processor = None
        torch_dtype = self._resolve_torch_dtype(config)
        config_obj = AutoConfig.from_pretrained(model_path, **hf_kwargs)
        backbone = self._load_backbone(model_path, config_obj, torch_dtype, hf_kwargs)

        is_quantized = config.load_in_4bit or config.load_in_8bit

        # Enable gradient checkpointing before wrapping with LoRA to lower peak memory.
        if config.gradient_checkpointing and not is_quantized and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()

        if config.adapter_path:
            self.model = PeftModel.from_pretrained(backbone, config.adapter_path)
        else:
            self.model = self._apply_lora(backbone, is_quantized=is_quantized)

    def forward(self, inputs_embeds, attention_mask, labels):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
        )

    def encode_images(
        self,
        images: Sequence,
        vision_tokens: int,
        hidden_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Uses the base vision tower (e.g., Qwen-VL) to obtain vision embeddings.
        Falls back to the tokenizer device and dtype; returns (batch, vision_tokens, hidden_size).
        """
        if self.processor is None:
            raise RuntimeError("No processor available for image encoding; ensure base model provides vision processor.")
        pil_images = [self._to_image(img) for img in images]

        def _run_processor(pils: List[Image.Image]):
            # Qwen3-VL processor expects paired text; provide empty prompts to avoid NoneType errors.
            dummy_text = [""] * len(pils)
            return self.processor(images=pils, text=dummy_text, return_tensors="pt")

        inputs = _run_processor(pil_images)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = inputs.get("images")
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values for vision encoding.")
        # Cast pixels to the vision tower dtype (not self.model.dtype) to avoid accidental fp32
        # when LoRA/R³ params are fp32.
        vision_mod = self._locate_vision_module(self.model)
        try:
            vision_dtype = next(vision_mod.parameters()).dtype if vision_mod is not None else None
        except Exception:
            vision_dtype = None
        if vision_dtype is None:
            try:
                vision_dtype = self.model.get_input_embeddings().weight.dtype
            except Exception:
                vision_dtype = torch.float16
        pixel_values = pixel_values.to(device=device, dtype=vision_dtype).contiguous()

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None:
            grid_thw = inputs.get("grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(device=device).contiguous()

        try:
            vision_hidden = self._forward_vision(pixel_values, grid_thw=grid_thw)
        except RuntimeError as exc:
            msg = str(exc)
            # cuDNN sometimes throws INTERNAL_ERROR for large/odd-shaped 3D conv workloads or
            # transient workspace allocation failures. Retry with cudnn disabled (slower but stable),
            # then (optionally) with a safety resize. If all retries fail, return zeros so training can continue.
            if "CUDNN_STATUS_INTERNAL_ERROR" in msg or "cuDNN error" in msg:
                try:
                    size_str = ",".join([f"{im.size[0]}x{im.size[1]}" for im in pil_images[:3]])
                except Exception:
                    size_str = "unknown"
                logger.warning(
                    "Vision forward hit cuDNN error; retrying with cudnn disabled. err=%s | img_sizes=%s",
                    msg,
                    size_str,
                )
                try:
                    with torch.backends.cudnn.flags(enabled=False):
                        vision_hidden = self._forward_vision(pixel_values, grid_thw=grid_thw)
                except Exception:
                    # Downscale extremely large images then retry once with cudnn disabled.
                    max_pixels = 1024 * 1024  # keep reasonably high-res for documents
                    resized: List[Image.Image] = []
                    for im in pil_images:
                        w, h = im.size
                        if w * h <= max_pixels:
                            resized.append(im)
                            continue
                        scale = (max_pixels / float(w * h)) ** 0.5
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        resized.append(im.resize((new_w, new_h)))
                    try:
                        inputs2 = _run_processor(resized)
                        pv2 = inputs2.get("pixel_values")
                        if pv2 is None:
                            pv2 = inputs2.get("images")
                        if pv2 is None:
                            raise ValueError("Processor did not return pixel_values on retry.")
                        pv2 = pv2.to(device=device, dtype=vision_dtype).contiguous()
                        g2 = inputs2.get("image_grid_thw")
                        if g2 is None:
                            g2 = inputs2.get("grid_thw")
                        if g2 is not None:
                            g2 = g2.to(device=device).contiguous()
                        with torch.backends.cudnn.flags(enabled=False):
                            vision_hidden = self._forward_vision(pv2, grid_thw=g2)
                    except Exception as exc2:
                        try:
                            if isinstance(device, torch.device) and device.type == "cuda":
                                alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                                reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                                logger.error(
                                    "Vision encoding failed after retries; returning zero embeddings to keep training running. err=%s | cuda_alloc=%.2fGiB cuda_reserved=%.2fGiB | img_sizes=%s",
                                    exc2,
                                    alloc_gb,
                                    reserved_gb,
                                    size_str,
                                )
                                # Best-effort cleanup to reduce fragmentation for subsequent batches.
                                torch.cuda.empty_cache()
                            else:
                                logger.error(
                                    "Vision encoding failed after retries; returning zero embeddings to keep training running. err=%s",
                                    exc2,
                                )
                        except Exception:
                            logger.error(
                                "Vision encoding failed after retries; returning zero embeddings to keep training running. err=%s",
                                exc2,
                            )
                        return torch.zeros(
                            (len(pil_images), vision_tokens, hidden_size),
                            device=device,
                            dtype=torch.float16,
                        )
            else:
                raise
        # If the vision tower lives on a different device under `device_map="auto"`,
        # _forward_vision may have executed on that device; bring features back.
        if vision_hidden.device != device:
            vision_hidden = vision_hidden.to(device=device)
        vision_hidden = self._resize_tokens(vision_hidden, vision_tokens)
        vision_hidden = self._match_hidden(vision_hidden, hidden_size)
        return vision_hidden

    def _forward_vision(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        vision_fn = self._locate_vision_module(self.model)
        if vision_fn is None:
            raise RuntimeError("Backbone does not expose a vision module; cannot encode images.")
        # Under `device_map="auto"`, the vision tower may be on a different GPU than the text embeddings.
        # Move pixel_values/grid_thw to the vision tower device for the forward.
        try:
            vision_device = next(vision_fn.parameters()).device
        except Exception:
            vision_device = None
        if vision_device is not None:
            if pixel_values.device != vision_device:
                pixel_values = pixel_values.to(device=vision_device)
            if grid_thw is not None and grid_thw.device != vision_device:
                grid_thw = grid_thw.to(device=vision_device)
        try:
            outputs = vision_fn(pixel_values, grid_thw=grid_thw) if grid_thw is not None else vision_fn(pixel_values)
        except TypeError:
            outputs = vision_fn(pixel_values)
        if isinstance(outputs, torch.Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("Vision module did not return hidden states.")
        if hidden.dim() == 2:
            # Some towers return (tokens, dim) without batch; add batch dim.
            hidden = hidden.unsqueeze(0)
        return hidden

    def _locate_vision_module(self, model):
        """
        Robustly find the vision tower, even when wrapped by PEFT/DDP.
        """
        candidates = [model]
        for attr in ["base_model", "model"]:
            sub = getattr(model, attr, None)
            if sub is not None:
                candidates.append(sub)
        if hasattr(model, "get_base_model"):
            try:
                candidates.append(model.get_base_model())
            except Exception:
                pass

        for m in candidates:
            if hasattr(m, "vision_model"):
                return m.vision_model
            if hasattr(m, "vision_tower"):
                vt = m.vision_tower
                try:
                    return vt() if callable(vt) else vt
                except Exception:
                    return vt
            if hasattr(m, "get_vision_tower"):
                try:
                    return m.get_vision_tower()
                except Exception:
                    pass
            if hasattr(m, "visual"):
                vis = getattr(m, "visual")
                try:
                    return vis() if callable(vis) else vis
                except Exception:
                    return vis
            if hasattr(m, "vision_encoder"):
                ve = getattr(m, "vision_encoder")
                try:
                    return ve() if callable(ve) else ve
                except Exception:
                    return ve

        for m in candidates:
            for name, sub in m.named_children():
                if "vision" in name or "visual" in name:
                    return sub
        # Final fallback: scan all submodules for a vision-like class name.
        for sub in model.modules():
            cls_name = sub.__class__.__name__.lower()
            if "vision" in cls_name and hasattr(sub, "forward"):
                return sub
        return None

    @staticmethod
    def _resize_tokens(feats: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if feats.size(1) == target_tokens:
            return feats
        if feats.size(1) > target_tokens:
            return feats[:, :target_tokens, :]
        pad = target_tokens - feats.size(1)
        pad_tensor = feats.new_zeros(feats.size(0), pad, feats.size(2))
        return torch.cat([feats, pad_tensor], dim=1)

    @staticmethod
    def _match_hidden(feats: torch.Tensor, hidden_size: int) -> torch.Tensor:
        if feats.size(-1) == hidden_size:
            return feats
        if feats.size(-1) > hidden_size:
            return feats[..., :hidden_size]
        repeat = (hidden_size + feats.size(-1) - 1) // feats.size(-1)
        expanded = feats.repeat(1, 1, repeat)
        return expanded[..., :hidden_size]

    @staticmethod
    def _to_image(img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            # assume CHW float in [0,1] or [0,255]
            if img.dim() == 3:
                img = img.detach().cpu()
                if img.max() <= 1.0:
                    img = (img * 255).clamp(0, 255)
                array = img.permute(1, 2, 0).byte().numpy()
                return Image.fromarray(array)
            raise ValueError("Unsupported tensor shape for image conversion.")
        return Image.open(str(img)).convert("RGB")

    def _apply_lora(self, model, is_quantized: bool = False):
        model = self._ensure_input_embeddings(model)
        # For k-bit (4/8bit) loading, prepare the model so LoRA can update inputs.
        if is_quantized:
            model_type = getattr(getattr(model, "config", None), "model_type", "") or model.__class__.__name__.lower()
            skip_bnb_prepare = any(tag in str(model_type).lower() for tag in ["qwen3_vl", "qwen2_vl", "qwen2_5_vl"])
            if not skip_bnb_prepare:
                try:
                    model = prepare_model_for_kbit_training(
                        model,
                        use_gradient_checkpointing=self.config.gradient_checkpointing,
                    )
                except NotImplementedError:
                    skip_bnb_prepare = True
            if skip_bnb_prepare:
                # Manual fallback: only do the minimum needed for LoRA on k-bit models.
                if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                embeddings = model.get_input_embeddings()
                if embeddings is not None:
                    embeddings.requires_grad_(True)
        lora_targets = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "vision_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(self.tokenizer))
        # Keep all trainable params in fp32 so AMP/GradScaler won't hit
        # "Attempting to unscale FP16 gradients." when the backbone is fp16.
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        return model

    @staticmethod
    def _resolve_torch_dtype(config: BaseVLMConfig) -> torch.dtype:
        # For quantized loading, keep computation in float32 to avoid half/float mismatch.
        if config.load_in_4bit or config.load_in_8bit:
            return torch.float32
        dtype = (config.dtype or "auto").lower()
        if dtype in ("fp16", "float16", "half"):
            return torch.float16
        if dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype in ("fp32", "float32", "full"):
            return torch.float32
        # auto: keep previous behavior
        return torch.bfloat16 if config.bf16 else torch.float32

    def _ensure_input_embeddings(self, model):
        """
        Some Vision2Seq wrappers (e.g., Qwen3-VL) do not expose get/set_input_embeddings
        on the top module. This redirects to a text submodule (language_model/model/transformer/etc.).
        """
        # If already works, return fast.
        try:
            _ = model.get_input_embeddings()
            return model
        except Exception:
            pass

        def _find_text_module():
            candidates = [
                getattr(model, "language_model", None),
                getattr(model, "model", None),
                getattr(model, "text_model", None),
                getattr(model, "transformer", None),
                getattr(model, "base_model", None),
            ]
            for cand in candidates:
                if cand is None:
                    continue
                if hasattr(cand, "get_input_embeddings"):
                    try:
                        emb = cand.get_input_embeddings()
                        if emb is not None:
                            return cand
                    except Exception:
                        continue
            # last resort: scan children
            for _, sub in model.named_children():
                if hasattr(sub, "get_input_embeddings"):
                    try:
                        emb = sub.get_input_embeddings()
                        if emb is not None:
                            return sub
                    except Exception:
                        continue
            # brute force: first embedding module
            for _, sub in model.named_modules():
                if isinstance(sub, nn.Embedding):
                    return sub
            return None

        text_module = _find_text_module()
        if text_module is None:
            # As a last resort, create a dummy getter that returns None to bypass NotImplementedError.
            def _get_input_embeddings(_self):
                return None

            def _set_input_embeddings(_self, value):
                return None

            model.get_input_embeddings = MethodType(_get_input_embeddings, model)
            model.set_input_embeddings = MethodType(_set_input_embeddings, model)
            return model

        def _get_input_embeddings(_self):
            # If text_module is an embedding, return directly; else call its getter.
            if isinstance(text_module, nn.Embedding):
                return text_module
            return text_module.get_input_embeddings()

        def _set_input_embeddings(_self, value):
            if isinstance(text_module, nn.Embedding):
                return
            if hasattr(text_module, "set_input_embeddings"):
                text_module.set_input_embeddings(value)

        model.get_input_embeddings = MethodType(_get_input_embeddings, model)
        model.set_input_embeddings = MethodType(_set_input_embeddings, model)

        # Patch enable_input_require_grads to avoid calling a missing implementation.
        def _enable_input_require_grads(_self):
            embedding = _self.get_input_embeddings()
            if embedding is not None:
                embedding.requires_grad_(True)

        model.enable_input_require_grads = MethodType(_enable_input_require_grads, model)
        return model

    def _prepare_model_source(self) -> str:
        # If model_name is a local path, use it directly
        local_path = Path(self.config.model_name)
        if local_path.exists():
            return str(local_path)
        if self.config.provider.lower() == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "模型 provider 设置为 'modelscope'，但未安装 modelscope 库。请运行 `pip install modelscope`。"
                ) from exc
            local_dir = snapshot_download(
                model_id=self.config.model_name,
                cache_dir=self.config.cache_dir,
                revision=self.config.revision,
                token=self.config.token,
            )
            return local_dir
        return self.config.model_name

    def _build_hf_kwargs(self) -> Dict:
        kwargs = {"trust_remote_code": True}
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        kwargs["local_files_only"] = self.config.local_files_only
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        if self.config.token:
            kwargs["token"] = self.config.token
        return kwargs

    def _load_backbone(self, model_path, config_obj, torch_dtype, hf_kwargs):
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # Use float32 compute for stability during backward to avoid half/float mismatches.
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Respect torchrun local rank to avoid sharding the model across all GPUs in each process.
        device_map = self.config.device_map
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0 and device_map == "auto":
            device_map = {"": local_rank}

        common_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        if bnb_config is not None:
            common_kwargs["quantization_config"] = bnb_config

        multimodal_types = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
        if getattr(config_obj, "model_type", None) in multimodal_types:
            return AutoModelForVision2Seq.from_pretrained(
                model_path,
                **common_kwargs,
                **hf_kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            **common_kwargs,
            **hf_kwargs,
        )
