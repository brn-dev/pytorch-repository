import abc
from typing import Any, Callable, TypedDict, Optional

from torch import nn


class ModelEntry(TypedDict):
    model_id: str
    parent_model_id: Optional[str]
    state_dict_path: str

    model_info: dict[str, Any]

    init_function: Optional[str]

    last_update_time: str


class ModelDB(abc.ABC):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abc.abstractmethod
    def close(self):
        raise NotImplemented

    @abc.abstractmethod
    def save_model_state_dict(
            self,
            model: nn.Module,
            model_id: str,
            parent_model_id: str,
            model_info: dict[str, Any],
            init_function: Optional[Callable[[], nn.Module] | str] = None,
    ) -> ModelEntry:
        raise NotImplemented

    @abc.abstractmethod
    def load_model_state_dict(self, model: nn.Module, model_id: str) -> ModelEntry:
        raise NotImplemented

    @abc.abstractmethod
    def all_entries(self) -> list[ModelEntry]:
        raise NotImplemented

    @abc.abstractmethod
    def fetch_entry(self, model_id: str) -> ModelEntry:
        raise NotImplemented

    @abc.abstractmethod
    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        raise NotImplemented
