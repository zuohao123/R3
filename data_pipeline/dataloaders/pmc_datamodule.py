"""
Lightning-like datamodule that prepares PMC batches.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from torch.utils.data import DataLoader, Dataset

from data_pipeline.corruption.simulator import CorruptionConfig, CorruptionSimulator
from data_pipeline.pseudo_text.tokenizer import PseudoTextTokenizer


class PMCDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        simulator: CorruptionSimulator,
        pseudo_tokenizer: PseudoTextTokenizer,
    ) -> None:
        self.dataset = dataset
        self.simulator = simulator
        self.pseudo_tokenizer = pseudo_tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        corrupted, report = self.simulator.simulate(sample)
        pseudo_segments = self.pseudo_tokenizer.build_pseudo_text(corrupted)
        return {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample.get("answer"),
            "full_sample": sample,
            "corrupted_sample": corrupted,
            "pseudo_text": pseudo_segments,
            "corruption_report": report,
        }


class PMCDatamodule:
    def __init__(
        self,
        dataset_factory,
        batch_size: int = 4,
        num_workers: int = 2,
        corruption_config: Optional[CorruptionConfig] = None,
    ) -> None:
        self.dataset_factory = dataset_factory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.corruption_config = corruption_config or CorruptionConfig()

    def setup(self, split: str) -> None:
        base_dataset = self.dataset_factory(split=split)
        self.wrapper = PMCDatasetWrapper(
            base_dataset,
            CorruptionSimulator(self.corruption_config),
            PseudoTextTokenizer(),
        )

    def dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.wrapper,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.wrapper.dataset.collate,
        )
