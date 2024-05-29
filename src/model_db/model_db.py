import abc
from typing import Any, TypedDict, Optional, TypeVar, Generic, Callable

from torch import nn

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
            state_dict: dict[str, Any],
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo
    ) -> ModelEntry[ModelInfo]:
        raise NotImplemented

    def save_model_state_dict(
            self,
            model: nn.Module,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo,
    ) -> ModelEntry[ModelInfo]:
        return self.save_state_dict(model.state_dict(), model_id, parent_model_id, model_info)

    @abc.abstractmethod
    def load_state_dict(self, model_id: str) -> tuple[dict[str, Any], ModelEntry[ModelInfo]]:
        raise NotImplemented

    def load_model_state_dict(self, model: nn.Module, model_id: str) -> ModelEntry[ModelInfo]:
        state_dict, model_entry = self.load_state_dict(model_id)
        model.load_state_dict(state_dict)
        return model_entry

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
            self.delete_entry(entry['model_info'], delete_state_dict=delete_state_dict)

    def copy_from(
            self,
            other_db: 'ModelDB[OtherModelInfo]',
            filter_: Callable[[ModelEntry[OtherModelInfo]], bool] = lambda entry: True,
            map_: Callable[[ModelEntry[OtherModelInfo], StateDict], tuple[ModelEntry[ModelInfo], StateDict]] =
                    lambda entry, state_dict: (entry, state_dict)
    ):
        for other_entry in other_db.all_entries():
            if not filter_(other_entry):
                continue

            state_dict, _ = other_db.load_state_dict(other_entry['model_id'])

            entry, state_dict = map_(other_entry, state_dict)

            self.save_state_dict(
                state_dict=state_dict,
                model_id=entry['model_id'],
                parent_model_id=entry['parent_model_id'],
                model_info=entry['model_info'],
            )

