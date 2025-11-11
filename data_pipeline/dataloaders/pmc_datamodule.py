"""
Lightning-like datamodule that prepares PMC batches.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.corruption.simulator import CorruptionConfig, CorruptionSimulator
from data_pipeline.features import SimpleFeatureBuilder
from data_pipeline.pseudo_text.tokenizer import PseudoTextTokenizer


class PMCDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        simulator: CorruptionSimulator,
        pseudo_tokenizer: PseudoTextTokenizer,
        feature_builder: Optional[SimpleFeatureBuilder] = None,
    ) -> None:
        self.dataset = dataset
        self.simulator = simulator
        self.pseudo_tokenizer = pseudo_tokenizer
        self.feature_builder = feature_builder or SimpleFeatureBuilder()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        corrupted, report = self.simulator.simulate(sample)
        pseudo_segments = self.pseudo_tokenizer.build_pseudo_text(corrupted)
        full_sample = {
            **sample,
            **self.feature_builder.build(sample),
            "pseudo_text": pseudo_segments,
            "corruption_report": {"modalities": {}, "overall_uncertainty": 0.0},
        }
        corrupted_sample = {
            **corrupted,
            **self.feature_builder.build(corrupted, corruption_report=report),
            "pseudo_text": pseudo_segments,
            "corruption_report": report,
        }
        return {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample.get("answer"),
            "full_sample": full_sample,
            "corrupted_sample": corrupted_sample,
            "pseudo_text": pseudo_segments,
            "corruption_report": report,
        }

    @staticmethod
    def _collate_samples(samples: List[Dict]) -> Dict:
        collated = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    @staticmethod
    def collate(items: List[Dict]) -> Dict:
        collated = {}
        for key in items[0]:
            values = [item[key] for item in items]
            if key in {"full_sample", "corrupted_sample"}:
                collated[key] = PMCDatasetWrapper._collate_samples(values)
            elif isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated


class PMCDatamodule:
    def __init__(
        self,
        dataset_factory: Callable,
        batch_size: int = 4,
        num_workers: int = 2,
        corruption_config: Optional[CorruptionConfig] = None,
        feature_builder: Optional[SimpleFeatureBuilder] = None,
    ) -> None:
        self.dataset_factory = dataset_factory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.corruption_config = corruption_config or CorruptionConfig()
        self.feature_builder = feature_builder or SimpleFeatureBuilder()

    def setup(self, split: str) -> None:
        base_dataset = self.dataset_factory(split=split)
        self.wrapper = PMCDatasetWrapper(
            base_dataset,
            CorruptionSimulator(self.corruption_config),
            PseudoTextTokenizer(),
            self.feature_builder,
        )

    def dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.wrapper,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PMCDatasetWrapper.collate,
        )
