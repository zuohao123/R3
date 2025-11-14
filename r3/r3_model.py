"""
Top-level R^3 model definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from r3.base_vlm import BaseVLM, BaseVLMConfig
from r3.corruption_simulator import CorruptionModuleConfig, UncertaintyAwareCorruptionSimulator
from r3.reconstructor import ReconstructionModuleConfig, SelectiveReconstruction
from r3.retrieval_module import PseudoTextRetrievalModule, RetrievalModuleConfig


@dataclass
class R3ModelConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 16
    hidden_size: int = 4096
    max_seq_length: int = 1024
    bf16: bool = True
    provider: str = "huggingface"
    token: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    local_files_only: bool = False
    top_k: int = 3
    enable_corruption: bool = True
    enable_retrieval: bool = True
    enable_prefix: bool = True
    enable_memory: bool = True
    enable_imputation: bool = True
    enable_consistency: bool = True
    lambda_consistency: float = 0.3


class R3Model(torch.nn.Module):
    def __init__(self, config: R3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.base_vlm = BaseVLM(
            BaseVLMConfig(
                model_name=config.model_name,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                bf16=config.bf16,
                provider=config.provider,
                token=config.token,
                cache_dir=config.cache_dir,
                revision=config.revision,
                local_files_only=config.local_files_only,
            )
        )
        self.simulator = UncertaintyAwareCorruptionSimulator(
            CorruptionModuleConfig(
                hidden_size=config.hidden_size,
                enable=config.enable_corruption,
            )
        )
        self.retrieval = PseudoTextRetrievalModule(
            RetrievalModuleConfig(
                hidden_size=config.hidden_size,
                top_k=config.top_k,
                enable=config.enable_retrieval,
            ),
            embedding_layer=self.base_vlm.model.get_input_embeddings(),
        )
        self.reconstruction = SelectiveReconstruction(
            ReconstructionModuleConfig(
                hidden_size=config.hidden_size,
                prefix_length=32,
                memory_tokens=32,
                imputation_tokens=16,
                enable_prefix=config.enable_prefix,
                enable_memory=config.enable_memory,
                enable_imputation=config.enable_imputation,
            )
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        corr_outputs = self._forward_branch(batch["corrupted"], apply_corruption=True)
        losses = {
            "task_loss": corr_outputs["loss"],
            "consistency_loss": torch.tensor(0.0, device=corr_outputs["loss"].device),
            "alignment_loss": torch.tensor(0.0, device=corr_outputs["loss"].device),
            "refuse_loss": torch.tensor(0.0, device=corr_outputs["loss"].device),
            "ops_loss": torch.tensor(0.0, device=corr_outputs["loss"].device),
        }
        total_loss = corr_outputs["loss"]
        if self.config.enable_consistency and "clean" in batch:
            clean_outputs = self._forward_branch(batch["clean"], apply_corruption=False)
            consistency = F.mse_loss(clean_outputs["hidden"], corr_outputs["hidden"].detach())
            losses["consistency_loss"] = consistency
            total_loss = total_loss + self.config.lambda_consistency * consistency

        losses["total"] = total_loss
        losses["loss"] = total_loss
        losses["corrupted_logits"] = corr_outputs["logits"]
        losses["corrupted_labels"] = corr_outputs["labels"]
        return losses

    def _forward_branch(self, split: Dict, apply_corruption: bool) -> Dict[str, torch.Tensor]:
        texts = split["question"]
        labels_text = split.get("labels", [""] * len(texts))
        prefix_prompts = split.get("pseudo_text", [[] for _ in texts])

        prompts = [
            f"{self._format_pseudo_text(pseudo)}\nQuestion: {q}\nAnswer:"
            if pseudo
            else f"Question: {q}\nAnswer:"
            for q, pseudo in zip(texts, prefix_prompts)
        ]

        tokenizer = self.base_vlm.tokenizer
        prompt_tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        full_texts = [
            f"{prompt} {label}" if label else prompt for prompt, label in zip(prompts, labels_text)
        ]
        text_tokens = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(split["vision_embeddings"].device)
        prompt_tokens = {k: v.to(split["vision_embeddings"].device) for k, v in prompt_tokens.items()}

        text_embeddings = self.base_vlm.model.get_input_embeddings()(text_tokens["input_ids"])
        labels = text_tokens["input_ids"].clone()
        labels[text_tokens["attention_mask"] == 0] = -100
        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)
        for idx, length in enumerate(prompt_lengths):
            labels[idx, : length.item()] = -100

        vision_embeddings = split["vision_embeddings"].to(text_embeddings.device).to(text_embeddings.dtype)
        vision_embeddings, text_embeddings, img_conf, txt_conf = self.simulator(
            vision_embeddings,
            text_embeddings,
            apply_corruption=apply_corruption,
        )
        retrieval = self.retrieval(
            text_embeddings,
            split.get("pseudo_text", [[] for _ in texts]),
            img_conf,
            txt_conf,
        )
        recon_out = self.reconstruction(
            text_embeddings,
            text_tokens["attention_mask"],
            vision_embeddings,
            retrieval,
            img_conf,
            txt_conf,
        )
        combined_inputs, combined_attention = self._merge_modalities(
            recon_out["inputs_embeds"],
            recon_out["attention_mask"],
            vision_embeddings,
        )

        outputs = self.base_vlm(
            inputs_embeds=combined_inputs,
            attention_mask=combined_attention,
            labels=labels,
        )
        hidden = outputs.hidden_states[-1].mean(dim=1) if outputs.hidden_states else outputs.logits.mean(dim=1)
        return {
            "loss": outputs.loss,
            "hidden": hidden,
            "logits": outputs.logits.detach(),
            "labels": labels.detach(),
        }

    def _merge_modalities(
        self,
        text_embeds: torch.Tensor,
        text_attention: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_mask = torch.ones(
            vision_embeds.size(0),
            vision_embeds.size(1),
            device=text_attention.device,
            dtype=text_attention.dtype,
        )
        inputs = torch.cat([text_embeds, vision_embeds], dim=1)
        attention = torch.cat([text_attention, vision_mask], dim=1)
        return inputs, attention

    def _format_pseudo_text(self, pseudo: List[str]) -> str:
        return "\n".join(filter(None, pseudo))
