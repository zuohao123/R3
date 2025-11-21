"""
Base VLM wrapper around Qwen3-VL style backbones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


@dataclass
class BaseVLMConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 16
    bf16: bool = True
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
        if config.adapter_path:
            self.model = PeftModel.from_pretrained(backbone, config.adapter_path)
        else:
            self.model = self._apply_lora(backbone)

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
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values") or inputs.get("images")
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values for vision encoding.")
        pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)
        vision_hidden = self._forward_vision(pixel_values)
        vision_hidden = self._resize_tokens(vision_hidden, vision_tokens)
        vision_hidden = self._match_hidden(vision_hidden, hidden_size)
        return vision_hidden

    def _forward_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_fn = None
        if hasattr(self.model, "vision_model"):
            vision_fn = self.model.vision_model
        elif hasattr(self.model, "vision_tower"):
            vision_fn = self.model.vision_tower
        elif hasattr(self.model, "get_vision_tower"):
            vision_fn = self.model.get_vision_tower()
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

    def _apply_lora(self, model):
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

    def _prepare_model_source(self) -> str:
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
                use_auth_token=self.config.token,
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
        multimodal_types = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
        if getattr(config_obj, "model_type", None) in multimodal_types:
            return AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                **hf_kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **hf_kwargs,
        )
