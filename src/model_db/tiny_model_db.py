import datetime
import os
from pathlib import Path
from typing import Any

import torch
from tinydb import TinyDB
from tinydb.queries import QueryLike, Query

from src.model_db.model_db import ModelDB, ModelEntry, ModelInfoType


class TinyModelDB(ModelDB[ModelInfoType]):

    def __init__(
            self,
            base_path: str,
            db_file_name: str = '_model_db.json',
            readonly: bool = False
    ):
        self.base_path = base_path
        Path(base_path).mkdir(parents=True, exist_ok=True)

        self.db_file_name = db_file_name

        self.readonly = readonly
        access_mode = 'r' if readonly else 'r+'
        self.db = TinyDB(os.path.join(self.base_path, self.db_file_name), access_mode=access_mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self.db)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.base_path = }, {self.db_file_name = })'

    def close(self):
        self.db.close()

    def save_state_dict(
            self,
            state_dict: dict[str, Any],
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfoType,
    ) -> ModelEntry[ModelInfoType]:

        entry: ModelEntry[ModelInfoType] = {
            'model_id': model_id,
            'parent_model_id': parent_model_id,
            'model_info': model_info,
            'last_update_time': str(datetime.datetime.now())
        }

        state_dict_file_path = self.get_state_dict_file_path(entry)
        torch.save(state_dict, state_dict_file_path)

        self.db.upsert(entry, cond=self.create_model_id_query(model_id))

        return entry

    def load_state_dict(self, model_id: str) -> tuple[dict[str, Any], ModelEntry[ModelInfoType]]:
        entry: ModelEntry[ModelInfoType] = self.fetch_entry(model_id)
        state_dict = torch.load(self.get_state_dict_file_path(entry))

        return state_dict, entry

    def all_entries(self) -> list[ModelEntry[ModelInfoType]]:
        return self.db.all()

    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfoType]:
        search_result = self.db.search(self.create_model_id_query(model_id))

        if len(search_result) == 0:
            raise ValueError(f'Model id {model_id} not found')

        return search_result[0]

    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        if delete_state_dict:
            entry = self.fetch_entry(model_id)
            os.remove(self.get_state_dict_file_path(entry))

        self.db.remove(cond=self.create_model_id_query(model_id))

    def get_state_dict_file_path(self, model_entry: ModelEntry[ModelInfoType]):
        return os.path.join(self.base_path, f'{model_entry["model_id"]}--state_dict.pth')

    @staticmethod
    def create_model_id_query(model_id: str) -> QueryLike:
        entry_query = Query()
        return entry_query.model_id == model_id

