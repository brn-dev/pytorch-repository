import abc
from dataclasses import dataclass
from queue import Queue
from typing import Any, TypedDict, Optional, TypeVar, Generic, Callable

from torch import nn, optim

ModelInfo = TypeVar('ModelInfo', bound=dict[str, Any])
OtherModelInfo = TypeVar('OtherModelInfo', bound=dict[str, Any])

StateDict = dict[str, Any]


class ModelEntry(TypedDict, Generic[ModelInfo]):
    model_id: str
    parent_model_id: Optional[str]

    model_info: ModelInfo

    last_update_time: str


ModelEntryFilter = Callable[[ModelEntry[ModelInfo]], bool]
ModelEntryMap = Callable[[ModelEntry[ModelInfo]], ModelEntry[ModelInfo]]


class ModelDB(abc.ABC, Generic[ModelInfo]):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplemented

    @abc.abstractmethod
    def close(self):
        raise NotImplemented

    @abc.abstractmethod
    def save_state_dict(
            self,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo,
            model_state_dict: StateDict,
            optimizer_state_dict: Optional[StateDict],
    ) -> ModelEntry[ModelInfo]:
        raise NotImplemented

    def save_model_state_dict(
            self,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo,
            model: nn.Module,
            optimizer: Optional[optim.Optimizer],
    ) -> ModelEntry[ModelInfo]:
        return self.save_state_dict(
            model_id=model_id,
            parent_model_id=parent_model_id,
            model_info=model_info,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        )

    @abc.abstractmethod
    def load_state_dict(
            self,
            model_id: str,
            load_optimizer: bool = True
    ) -> tuple[StateDict, Optional[StateDict]]:
        raise NotImplemented

    def load_model_state_dict(
            self,
            model_id: str,
            model: nn.Module,
            optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        model_state_dict, optimizer_state_dict = self.load_state_dict(
            model_id,
            load_optimizer=optimizer is not None
        )

        model.load_state_dict(model_state_dict)

        if optimizer is not None and optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

    @abc.abstractmethod
    def all_entries(self) -> list[ModelEntry[ModelInfo]]:
        raise NotImplemented

    @abc.abstractmethod
    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfo]:
        raise NotImplemented

    @abc.abstractmethod
    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        raise NotImplemented

    @abc.abstractmethod
    def _save_entry(self, entry: ModelEntry[ModelInfo]):
        raise NotImplemented

    def filtered_entries(self, entry_filter: ModelEntryFilter):
        return filter(entry_filter, self.all_entries())

    def map_entries(
            self,
            entry_map: ModelEntryMap,
            entry_filter: ModelEntryFilter = lambda entry: True,
    ):
        for entry in self.filtered_entries(entry_filter):
            self._save_entry(entry_map(entry))

    def delete_entries(self, entry_filter: ModelEntryFilter, delete_state_dict: bool):
        for entry in self.filtered_entries(entry_filter):
            self.delete_entry(entry['model_id'], delete_state_dict=delete_state_dict)

    def copy_from(
            self,
            other_db: 'ModelDB[OtherModelInfo]',
            entry_filter: Callable[[ModelEntry[OtherModelInfo]], bool] = lambda entry: True,
            entry_map: Callable[
                    [ModelEntry[OtherModelInfo], StateDict, StateDict],
                    tuple[ModelEntry[ModelInfo], StateDict, StateDict]
                ] = lambda entry, state_dict: (entry, state_dict)
    ):
        for other_entry in other_db.filtered_entries(entry_filter):
            model_state_dict, optim_state_dict = other_db.load_state_dict(
                other_entry['model_id'], load_optimizer=True
            )

            entry, model_state_dict, optim_state_dict = entry_map(other_entry, model_state_dict, optim_state_dict)

            self.save_state_dict(
                model_id=entry['model_id'],
                parent_model_id=entry['parent_model_id'],
                model_info=entry['model_info'],
                model_state_dict=model_state_dict,
                optimizer_state_dict=optim_state_dict,
            )


