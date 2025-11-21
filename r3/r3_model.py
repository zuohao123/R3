"""
Top-level R^3 model definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from r3.base_vlm import BaseVLM, BaseVLMConfig
from r3.corruption_simulator import CorruptionModuleConfig, UncertaintyAwareCorruptionSimulator
from r3.reconstructor import ReconstructionModuleConfig, SelectiveReconstruction, TriPathReasoner
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
    retrieval_cache_path: Optional[str] = None
    retrieval_corpus_path: Optional[str] = None


class R3Model(torch.nn.Module):
    """
    Clean-teacher / corrupted-student dual-branch wrapper around Qwen-VL.
    """

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
                cache_path=config.retrieval_cache_path,
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
        self.reasoner = TriPathReasoner(config.hidden_size)
        if config.retrieval_corpus_path:
            try:
                self.retrieval.ingest_corpus(config.retrieval_corpus_path)
            except Exception:
                pass

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pseudo_text: Optional[Sequence[Sequence[str]]] = None,
        is_clean_branch: bool = False,
    ) -> Dict[str, torch.Tensor | Dict]:
        """
        Args:
            input_ids: 文本 token 序列。
            pixel_values: 视觉嵌入（已编码为图像 token 序列，形状 b x num_img_tokens x dim）。
            attention_mask: 文本注意力掩码。
            labels: 文本监督标签（与 input_ids 对齐）。
            pseudo_text: OCR/Caption/上下文伪文本。
            is_clean_branch: True 为教师分支（不回传梯度、不走检索/重建），False 为学生分支（完整 R³ 流程）。
        """
        pseudo_text = pseudo_text or [[] for _ in range(input_ids.size(0))]
        text_embeddings = self.base_vlm.model.get_input_embeddings()(input_ids)
        vision_embeddings = pixel_values.to(text_embeddings.device).to(text_embeddings.dtype)

        if is_clean_branch:
            outputs = self._forward_clean(text_embeddings, vision_embeddings, attention_mask, labels)
        else:
            outputs = self._forward_corrupted(
                text_embeddings,
                vision_embeddings,
                attention_mask,
                labels,
                pseudo_text,
            )
        return outputs

    def _forward_clean(
        self,
        text_embeddings: torch.Tensor,
        vision_embeddings: torch.Tensor,
        text_attention: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor | None]:
        """
        教师分支：仅做前向，跳过模拟器/检索/重建，加速且不占显存。
        """
        with torch.no_grad():
            combined_inputs, combined_attention, padded_labels = self._merge_modalities(
                text_embeddings,
                text_attention,
                vision_embeddings,
                labels,
            )
            outputs = self.base_vlm(
                inputs_embeds=combined_inputs,
                attention_mask=combined_attention,
                labels=padded_labels,
            )
        pooled = self._pool_hidden(outputs.hidden_states)
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states[-1],
            "pooled_hidden": pooled,
            "img_conf": None,
            "txt_conf": None,
            "retrieval": None,
        }

    def _forward_corrupted(
        self,
        text_embeddings: torch.Tensor,
        vision_embeddings: torch.Tensor,
        text_attention: torch.Tensor,
        labels: Optional[torch.Tensor],
        pseudo_text: Sequence[Sequence[str]],
    ) -> Dict[str, torch.Tensor | Dict]:
        # 1) 模拟器：仅产出置信度，不改写 token
        vision_embeddings, text_embeddings, img_conf, txt_conf = self.simulator(
            vision_embeddings,
            text_embeddings,
            apply_corruption=True,
        )
        # 2) 检索：使用伪文本（或外部库）+ 置信度做噪声感知召回
        retrieval = self.retrieval(
            text_embeddings,
            pseudo_text,
            img_conf,
            txt_conf,
        )
        # 3) 重建：前缀/记忆/填补融合
        recon_out = self.reconstruction(
            text_embeddings,
            text_attention,
            vision_embeddings,
            retrieval,
            img_conf,
            txt_conf,
        )
        # 4) TriPathReasoner 再细化融合后的 token
        refined_text = self.reasoner(recon_out["inputs_embeds"], recon_out["attention_mask"])
        combined_inputs, combined_attention, padded_labels = self._merge_modalities(
            refined_text,
            recon_out["attention_mask"],
            vision_embeddings,
            labels,
        )
        outputs = self.base_vlm(
            inputs_embeds=combined_inputs,
            attention_mask=combined_attention,
            labels=padded_labels,
        )
        pooled = self._pool_hidden(outputs.hidden_states)
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states[-1],
            "pooled_hidden": pooled,
            "img_conf": img_conf,
            "txt_conf": txt_conf,
            "retrieval": retrieval,
        }

    def _merge_modalities(
        self,
        text_embeds: torch.Tensor,
        text_attention: torch.Tensor,
        vision_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        vision_mask = torch.ones(
            vision_embeds.size(0),
            vision_embeds.size(1),
            device=text_attention.device,
            dtype=text_attention.dtype,
        )
        inputs = torch.cat([text_embeds, vision_embeds], dim=1)
        attention = torch.cat([text_attention, vision_mask], dim=1)
        if labels is None:
            return inputs, attention, None
        vision_labels = labels.new_full((labels.size(0), vision_embeds.size(1)), -100)
        padded_labels = torch.cat([labels, vision_labels], dim=1)
        return inputs, attention, padded_labels

    @staticmethod
    def _pool_hidden(hidden_states: Optional[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("Hidden states are required for consistency training.")
        return hidden_states[-1].mean(dim=1)
