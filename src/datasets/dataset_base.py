import sys
from abc import ABC, abstractmethod
from os import PathLike
from typing import Tuple, Union

import dill
import numpy as np
from torch.utils.data import Dataset, Subset


# noinspection PyMethodMayBeStatic
class DatasetBase(ABC, Dataset):

    @abstractmethod
    def __len__(self) -> int:
        pass

    def split(self, parts: tuple[int, int, int], random_seed=42) -> Tuple[Subset, Subset, Subset]:
        """
        Split the dataset into training-, validation- and test-set.

        If e.g. (5, 1, 1) is passed for the ratios, then the training set will consist of 5/7 the amount of samples,
        the test set and validation will have 1/7 the amount of samples each

        If the test set is not needed, pass 0 as the 3rd ratio.
        An empty subset will then be created and return for the test set.

        :param parts: The ratios with which the set should be split into subsets
        :param random_seed: Seed for splitting

        :return:
        A tuple consisting of 3 subsets: (training_set, validation_set, test_set)
        """
        parts_sum = sum(parts)

        n_samples = len(self)

        validation_set_start = int(n_samples / parts_sum) * parts[0]
        test_set_start = int(n_samples / parts_sum) * (parts[0] + parts[1])

        shuffled_indices = np.random.default_rng(random_seed).permutation(n_samples)

        training_set_indices = shuffled_indices[:validation_set_start]
        validation_set_indices = shuffled_indices[validation_set_start:test_set_start]
        test_set_indices = shuffled_indices[test_set_start:]

        print(f'Splitting dataset into subsets with size '
              f'{len(training_set_indices)}, {len(validation_set_indices)}, {len(test_set_indices)}')

        return (
            Subset(self, indices=training_set_indices),
            Subset(self, indices=validation_set_indices),
            Subset(self, indices=test_set_indices),
        )

    def __flush_and_print(self, msg: str) -> None:
        sys.stdout.flush()
        sys.stdout.write(f'\r{msg}')

    def __load_from_dill(self, path: Union[str, PathLike]) -> any:
        with open(path, 'rb') as f:
            return dill.load(f)

    def __dump_to_dill(self, path: Union[str, PathLike], value: any) -> None:
        with open(path, 'wb') as f:
            dill.dump(value, f)
