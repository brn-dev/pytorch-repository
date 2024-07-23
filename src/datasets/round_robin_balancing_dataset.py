from typing import Callable, TypeVar

from torch.utils.data import Dataset, WeightedRandomSampler

SampleType = TypeVar('SampleType')

class RoundRobinBalancingDataset(Dataset[SampleType]):

    def __init__(
            self,
            base_set: Dataset,
            num_classes: int,
            class_selector: Callable[[SampleType], int],
            epoch_length: int,
    ):
        self.num_classes = num_classes
        self.epoch_length = epoch_length

        self.samples_by_class: dict[int, list[SampleType]] = {cls: [] for cls in range(num_classes)}
        self.last_index_by_class: dict[int, int] = {cls: 0 for cls in range(num_classes)}

        for sample in base_set:
            sample_class = class_selector(sample)
            self.samples_by_class[sample_class].append(sample)

        for cls in range(num_classes):
            assert len(self.samples_by_class[cls]) > 0, f'Class {cls} has no samples!'

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index: int) -> SampleType:
        if index >= self.epoch_length:
            raise IndexError

        cls = index % self.num_classes
        cls_samples = self.samples_by_class[cls]

        sample = cls_samples[self.last_index_by_class[cls]]
        self.last_index_by_class[cls] = (self.last_index_by_class[cls] + 1) % len(cls_samples)

        return sample

