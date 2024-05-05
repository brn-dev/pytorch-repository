import multiprocessing
from typing import Any

from torch import nn

from src.model_db.model_db import ModelDB, ModelEntry, ModelInfoType


class MultiprocessingSyncWrapper(ModelDB):

    def __init__(self, model_db: ModelDB, access_lock: Any | None = None):
        self.model_db = model_db

        self.access_lock = access_lock or multiprocessing.Lock()

    def __len__(self):
        return len(self.model_db)

    def __enter__(self):
        return self.model_db.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.model_db.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        with self.access_lock:
            self.model_db.close()

    def save_model_state_dict(
            self,
            model: nn.Module,
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfoType
    ) -> ModelEntry[ModelInfoType]:
        with self.access_lock:
            return self.model_db.save_model_state_dict(model, model_id, parent_model_id, model_info)

    def load_model_state_dict(self, model: nn.Module, model_id: str) -> ModelEntry[ModelInfoType]:
        with self.access_lock:
            return self.model_db.load_model_state_dict(model, model_id)

    def all_entries(self) -> list[ModelEntry[ModelInfoType]]:
        with self.access_lock:
            return self.model_db.all_entries()

    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfoType]:
        with self.access_lock:
            return self.fetch_entry(model_id)

    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        with self.access_lock:
            self.model_db.delete_entry(model_id, delete_state_dict)
