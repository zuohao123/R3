"""
High-level trainer coordinating corruption, retrieval, and reconstruction.
"""
from __future__ import annotations

from time import process_time_ns
from typing import Dict

import torch

from data_pipeline.dataloaders.pmc_datamodule import PMCDatamodule
from model.losses.consistency_regularizer import ConsistencyRegularizer
from retrieval.filters.consistency_filter import ConsistencyFilter
from retrieval.query.pseudo_text_retriever import PseudoTextRetriever
from train.optimizer import build_optimizer


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        datamodule: PMCDatamodule,
        retriever: PseudoTextRetriever,
        device: str = "cuda",
        log_interval: int = 20,
    ) -> None:
        self.model = model.to(device)
        self.datamodule = datamodule
        self.retriever = retriever
        self.device = device
        self.log_interval = log_interval
        self.filter = ConsistencyFilter()
        self.consistency_loss = ConsistencyRegularizer()

    def _prepare_model_inputs(self, sample: Dict) -> Dict:
        allowed_keys = {
            "question_tokens",
            "vision_tokens",
            "pseudo_text",
            "corruption_report",
            "answer",
            "question",
            "image_path",
        }
        return {key: sample[key] for key in allowed_keys if key in sample}

    def _move_to_device(self, batch: Dict) -> Dict:
        tensor_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                tensor_batch[key] = self._move_to_device(value)
            else:
                tensor_batch[key] = value
        return tensor_batch

    def fit(
        self,
        epochs: int,
        optimizer_cfg: Dict,
        train_split: str = "train",
        val_split: str = "val",
    ) -> None:
        optimizer = build_optimizer(
            self.model,
            lr_backbone=optimizer_cfg.get("lr_backbone", 1e-5),
            lr_adapter=optimizer_cfg.get("lr_adapter", 5e-5),
        )
        self.datamodule.setup(train_split)
        train_loader = self.datamodule.dataloader(shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                corrupted_raw = self._move_to_device(batch["corrupted_sample"])
                full_raw = self._move_to_device(batch["full_sample"])
                corrupted = self._prepare_model_inputs(corrupted_raw)
                full = self._prepare_model_inputs(full_raw)
                retrieval = self._retrieve(batch)
                outputs_corrupted = self.model(**corrupted, retrieval=retrieval)
                outputs_full = self.model(**full, retrieval=retrieval)

                qa_loss = outputs_corrupted["loss"]
                consistency = self.consistency_loss(
                    outputs_full["logits"], outputs_corrupted["logits"]
                )
                loss = qa_loss + consistency
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step % self.log_interval == 0:
                    print(f"[Epoch {epoch}] step {step} loss {loss.item():.4f}")

    def _retrieve(self, batch: Dict) -> Dict:
        retrieval_results = []
        for question, pseudo_text, report in zip(
            batch["question"], batch["pseudo_text"], batch["corruption_report"]
        ):
            raw_docs = self.retriever.query(
                question, pseudo_text, report["overall_uncertainty"]
            )
            filtered_docs = self.filter(raw_docs, pseudo_text)
            retrieval_results.append(filtered_docs)
        return retrieval_results
