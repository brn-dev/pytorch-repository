from typing import Any, Optional

from src.model_db.model_db import ModelDB, ModelEntry, ModelInfo, StateDict


class DummyModelDB(ModelDB[ModelInfo]):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return 'DummyModelDB()'

    def __len__(self):
        return 0

    def close(self):
        pass

    def save_state_dict(
            self,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo,
            model_state_dict: StateDict,
            optimizer_state_dict: Optional[StateDict],
    ) -> ModelEntry[ModelInfo]:
        pass

    def load_state_dict(
            self,
            model_id: str,
            load_optimizer: bool = True
    ) -> tuple[StateDict, Optional[StateDict]]:
        raise NotImplementedError('Dummy ModelDB can not load models')

    def all_entries(self) -> list[ModelEntry[ModelInfo]]:
        raise NotImplementedError('Dummy ModelDB fetch entries')

    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfo]:
        raise NotImplementedError('Dummy ModelDB fetch entries')

    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        pass

    def _save_entry(self, entry: ModelEntry[ModelInfo]):
        pass
