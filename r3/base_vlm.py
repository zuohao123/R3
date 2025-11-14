"""
Base VLM wrapper around Qwen3-VL style backbones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer


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

    def _apply_lora(self, model):
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "vision_proj"]
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
