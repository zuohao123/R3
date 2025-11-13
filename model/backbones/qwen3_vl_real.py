"""
Thin wrapper that loads real Qwen3-VL weights via HuggingFace transformers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "transformers>=4.39 is required to load Qwen3-VL weights. "
            "Please install it via `pip install transformers`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(value: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.replace("torch.", "")
        if hasattr(torch, normalized):
            return getattr(torch, normalized)
        raise ValueError(f"Unsupported torch dtype string: {value}")
    raise TypeError(f"Unsupported dtype specification: {value}")


@dataclass
class QwenLoadConfig:
    pretrained_path: str
    torch_dtype: Optional[str | torch.dtype] = "float16"
    device_map: Optional[str | Dict] = None
    device: Optional[str] = None
    trust_remote_code: bool = True


class Qwen3VLRealBackbone(torch.nn.Module):
    """
    Consumes enhanced question/vision/memory embeddings and forwards them
    through the actual Qwen3-VL language tower using the `inputs_embeds`
    interface so pretrained weights can be reused without re-tokenizing.
    """

    def __init__(
        self,
        load_cfg: QwenLoadConfig,
        input_hidden_size: int,
        model_hidden_size: int,
    ) -> None:
        super().__init__()
        if not load_cfg.pretrained_path:
            raise ValueError("`pretrained_path` must be provided for qwen3-vl backbone.")
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        dtype = _resolve_dtype(load_cfg.torch_dtype)

        load_kwargs: Dict = {"trust_remote_code": load_cfg.trust_remote_code}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if load_cfg.device_map is not None:
            load_kwargs["device_map"] = load_cfg.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            load_cfg.pretrained_path,
            **load_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_cfg.pretrained_path,
            trust_remote_code=load_cfg.trust_remote_code,
        )
        if load_cfg.device:
            self.model.to(load_cfg.device)

        self.model_hidden_size = model_hidden_size or self.model.config.hidden_size
        self.input_hidden_size = input_hidden_size

        if self.model_hidden_size != self.model.config.hidden_size:
            raise ValueError(
                f"Config hidden size ({model_hidden_size}) does not match "
                f"loaded model hidden size ({self.model.config.hidden_size})."
            )

        self.text_proj = torch.nn.Linear(input_hidden_size, self.model_hidden_size)
        self.vision_proj = torch.nn.Linear(input_hidden_size, self.model_hidden_size)
        self.memory_proj = torch.nn.Linear(input_hidden_size, self.model_hidden_size)
        self.imputation_proj = torch.nn.Linear(input_hidden_size, self.model_hidden_size)

    def forward(
        self,
        question_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        memory_cache: torch.Tensor,
        imputation_tokens: torch.Tensor,
        answer: Optional[List[str]] = None,
        **_,
    ) -> Dict:
        device = question_tokens.device
        text = self.text_proj(question_tokens)
        vision = self.vision_proj(vision_tokens)
        memory = self.memory_proj(memory_cache)
        imputation = self.imputation_proj(imputation_tokens)

        embeddings = torch.cat([text, vision, memory, imputation], dim=1)
        attention_mask = embeddings.new_ones(embeddings.shape[:2], dtype=torch.long)
        outputs = self.model(
            inputs_embeds=embeddings.to(self.model.dtype),
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :]  # use last position for classification-style loss
        predictions = self._decode_predictions(logits)

        result = {"logits": logits, "predictions": predictions}
        if answer is not None:
            targets = self._encode_answers(answer, device=logits.device)
            result["loss"] = F.cross_entropy(logits, targets)
        else:
            result["loss"] = torch.zeros(1, device=logits.device)
        return result

    def _encode_answers(self, answers: List[str], device: torch.device) -> torch.Tensor:
        target_ids = []
        eos_id = getattr(self.tokenizer, "eos_token_id", 0)
        for ans in answers:
            if not ans:
                target_ids.append(eos_id)
                continue
            tokens = self.tokenizer(ans, add_special_tokens=False).input_ids
            target_ids.append(tokens[0] if tokens else eos_id)
        return torch.tensor(target_ids, dtype=torch.long, device=device)

    def _decode_predictions(self, logits: torch.Tensor) -> List[str]:
        token_ids = logits.argmax(dim=-1).tolist()
        return self.tokenizer.batch_decode(
            [[token_id] for token_id in token_ids],
            skip_special_tokens=True,
        )
