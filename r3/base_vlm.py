"""
Base VLM wrapper around Qwen3-VL style backbones.
"""
from __future__ import annotations

import os
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


@dataclass
class BaseVLMConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 16
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: Optional[str | Dict] = None
    low_cpu_mem_usage: bool = True
    gradient_checkpointing: bool = True
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
        torch_dtype = torch.bfloat16 if config.bf16 else torch.float16
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
        # Qwen3-VL processor expects paired text; provide empty prompts to avoid NoneType errors.
        dummy_text = [""] * len(pil_images)
        inputs = self.processor(images=pil_images, text=dummy_text, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = inputs.get("images")
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values for vision encoding.")
        pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)
        vision_hidden = self._forward_vision(pixel_values)
        vision_hidden = self._resize_tokens(vision_hidden, vision_tokens)
        vision_hidden = self._match_hidden(vision_hidden, hidden_size)
        return vision_hidden

    def _forward_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_fn = self._locate_vision_module(self.model)
        if vision_fn is None:
            raise RuntimeError("Backbone does not expose a vision module; cannot encode images.")
        outputs = vision_fn(pixel_values)
        if isinstance(outputs, torch.Tensor):
            return outputs
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("Vision module did not return hidden states.")
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

        for m in candidates:
            for name, sub in m.named_children():
                if "vision" in name:
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
        return model

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
                bnb_4bit_compute_dtype=torch.float16,
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
