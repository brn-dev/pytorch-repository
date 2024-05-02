from typing import Any, Callable, Optional

from torch import nn

from src.model_db.model_db import ModelDB, ModelEntry, ModelInfoType


class DummyModelDB(ModelDB[ModelInfoType]):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        pass

    def save_model_state_dict(
            self,
            model: nn.Module,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfoType,
            init_function: Optional[Callable[[], nn.Module] | str] = None,
    ) -> ModelEntry[ModelInfoType]:
        pass

    def load_model_state_dict(
            self,
            model: nn.Module,
            model_id: str
    ) -> ModelEntry[ModelInfoType]:
        raise NotImplementedError('Dummy ModelDB can not load models')

    def all_entries(self) -> list[ModelEntry[ModelInfoType]]:
        raise NotImplementedError('Dummy ModelDB fetch entries')

    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfoType]:
        raise NotImplementedError('Dummy ModelDB fetch entries')

    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        pass
