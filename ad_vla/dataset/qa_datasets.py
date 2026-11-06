"""Question-answering views of a dataset, independent of which dataset it is.

A scene dataset takes part by implementing ``qa_pairs_by_index``. It keeps loading its own samples
and images; these views only decide which sample and question each item is.
"""

from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from ad_vla.dataset.data_types import E2EDataSample, QuestionAnswerPair


class QAPairDataset(Dataset):
    """Validation view: one deterministic item per question, grouped by frame.

    Adjacent questions share a frame, so a small per-worker cache decodes its images once.
    """

    def __init__(self, scene_dataset: Dataset) -> None:
        self.scene_dataset = scene_dataset
        pairs_by_index = scene_dataset.qa_pairs_by_index(strict=True)
        self.items = [
            (sample_index, qa_pair)
            for sample_index, qa_pairs in pairs_by_index.items()
            for qa_pair in qa_pairs
        ]
        if not self.items:
            raise ValueError("The configured QA dataset contains no questions.")
        self._sample_cache: OrderedDict[int, E2EDataSample] = OrderedDict()
        self._sample_cache_size = 2

    def __len__(self) -> int:
        return len(self.items)

    def _load_sample(self, sample_index: int) -> E2EDataSample:
        """Return scene sample ``sample_index``, reusing it while adjacent questions need it."""
        if sample_index in self._sample_cache:
            self._sample_cache.move_to_end(sample_index)
            return self._sample_cache[sample_index]
        sample = self.scene_dataset[sample_index]
        self._sample_cache[sample_index] = sample
        if len(self._sample_cache) > self._sample_cache_size:
            self._sample_cache.popitem(last=False)
        return sample

    def __getitem__(self, index: int) -> tuple[E2EDataSample, QuestionAnswerPair]:
        sample_index, qa_pair = self.items[index]
        return self._load_sample(sample_index), qa_pair


class QATrainDataset(Dataset):
    """Training view: only the frames with questions, with their questions drawn per item.

    A frame's questions are split into ``questions_per_frame`` fixed groups, and each of its items
    draws one question from its own group on every access. So one epoch never repeats a question
    of a frame, every question stays reachable across epochs, and no epoch state is needed.
    Workers have their own random seeds. ``questions_per_frame=None`` gives every question its own
    item, so one epoch sees them all.
    """

    def __init__(
        self,
        scene_dataset: Dataset,
        questions_per_frame: int | None = 1,
        seed: int = 0,
    ) -> None:
        if questions_per_frame is not None and questions_per_frame < 1:
            raise ValueError("questions_per_frame must be at least one, or null.")
        self.scene_dataset = scene_dataset
        generator = torch.Generator().manual_seed(seed)
        self.items: list[tuple[int, tuple[QuestionAnswerPair, ...]]] = []
        pairs_by_index = scene_dataset.qa_pairs_by_index(strict=False)
        for sample_index in sorted(pairs_by_index):
            pairs = pairs_by_index[sample_index]
            groups = questions_per_frame or len(pairs)
            order = torch.randperm(len(pairs), generator=generator).tolist()
            for group in range(min(groups, len(pairs))):
                self.items.append(
                    (sample_index, tuple(pairs[i] for i in order[group::groups]))
                )
        if not self.items:
            raise ValueError("The configured QA dataset contains no questions.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> E2EDataSample:
        sample_index, pairs = self.items[index]
        pair = pairs[int(torch.randint(len(pairs), ()).item())]
        return self.scene_dataset[sample_index].model_copy(update={"qa_pair": pair})


def collate_qa_pairs(
    batch: list[tuple[E2EDataSample, QuestionAnswerPair]],
) -> list[tuple[E2EDataSample, QuestionAnswerPair]]:
    """Keep QA samples as Python objects for model-side prompt construction."""
    return batch
