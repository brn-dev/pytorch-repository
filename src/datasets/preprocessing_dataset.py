from typing import TypeVar, Callable

from torch.utils.data import Dataset

SampleType = TypeVar('SampleType')

class PreprocessingDataset(Dataset[SampleType]):

    def __init__(
            self,
            base_set: Dataset[SampleType],
            preprocessing: Callable[[SampleType], SampleType]
    ):
        self.base_set = base_set
        self.preprocessing = preprocessing

        if hasattr(self.base_set, '__len__'):
            self.__len__ = self.base_set.__len__

    def __getitem__(self, index: int):
        return self.preprocessing(self.base_set[index])


class CachedPreprocessingDataset(Dataset[SampleType]):

    def __init__(
            self,
            base_set: Dataset[SampleType],
            preprocessing: Callable[[SampleType], SampleType]
    ):
        self.preprocessed_samples = []

        for sample in base_set:
            self.preprocessed_samples.append(preprocessing(sample))

    def __len__(self):
        return len(self.preprocessed_samples)

    def __getitem__(self, index: int):
        return self.preprocessed_samples[index]
