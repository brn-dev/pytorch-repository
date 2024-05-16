import abc
from typing import Any, TypedDict, Optional, TypeVar, Generic

from torch import nn

ModelInfoType = TypeVar('ModelInfoType', bound=dict[str, Any])


class ModelEntry(TypedDict, Generic[ModelInfoType]):
    model_id: str
    parent_model_id: Optional[str]
    state_dict_path: str

    model_info: ModelInfoType

    last_update_time: str


class ModelDB(abc.ABC, Generic[ModelInfoType]):

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
            model_info: ModelInfoType
    ) -> ModelEntry[ModelInfoType]:
        raise NotImplemented

    def save_model_state_dict(
            self,
            model: nn.Module,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfoType,
    ) -> ModelEntry[ModelInfoType]:
        return self.save_state_dict(model.state_dict(), model_id, parent_model_id, model_info)

    @abc.abstractmethod
    def load_state_dict(self, model_id: str) -> tuple[dict[str, Any], ModelEntry[ModelInfoType]]:
        raise NotImplemented

    def load_model_state_dict(self, model: nn.Module, model_id: str) -> ModelEntry[ModelInfoType]:
        state_dict, model_entry = self.load_state_dict(model_id)
        model.load_state_dict(state_dict)
        return model_entry

    @abc.abstractmethod
    def all_entries(self) -> list[ModelEntry[ModelInfoType]]:
        raise NotImplemented

    @abc.abstractmethod
    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfoType]:
        raise NotImplemented

    @abc.abstractmethod
    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        raise NotImplemented
