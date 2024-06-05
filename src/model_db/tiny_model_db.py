import datetime
import os
from pathlib import Path
from typing import Any, Optional

import tinydb.table
import torch
from tinydb import TinyDB
from tinydb.queries import QueryLike, Query

from src.model_db.model_db import ModelDB, ModelEntry, ModelInfo, StateDict


class TinyModelDB(ModelDB[ModelInfo]):

    def __init__(
            self,
            base_path: str,
            db_file_name: str = '_model_db.json',
            readonly: bool = False,
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
            model_id: str,
            parent_model_id: str,
            model_info: ModelInfo,
            model_state_dict: StateDict,
            optimizer_state_dict: Optional[StateDict] = None,
    ) -> ModelEntry[ModelInfo]:

        entry: ModelEntry[ModelInfo] = {
            'model_id': model_id,
            'parent_model_id': parent_model_id,
            'model_info': model_info,
            'last_update_time': str(datetime.datetime.now())
        }

        model_state_dict_file_path = self.get_model_state_dict_file_path(model_id)
        torch.save(model_state_dict, model_state_dict_file_path)

        if optimizer_state_dict is not None:
            optimizer_state_dict_file_path = self.get_optimizer_state_dict_file_path(model_id)
            torch.save(optimizer_state_dict, optimizer_state_dict_file_path)

        self._save_entry(entry)

        return entry

    def load_state_dict(
            self,
            model_id: str,
            load_optimizer: bool = True
    ) -> tuple[StateDict, Optional[StateDict]]:
        model_state_dict = torch.load(self.get_model_state_dict_file_path(model_id))

        optimizer_state_dict_path = self.get_optimizer_state_dict_file_path(model_id)
        optimizer_state_dict: Optional[StateDict] = None
        if load_optimizer and Path(optimizer_state_dict_path).is_file():
            optimizer_state_dict = torch.load(optimizer_state_dict_path)

        return model_state_dict, optimizer_state_dict

    def all_entries(self) -> list[ModelEntry[ModelInfo]]:
        return self.db.all()

    def fetch_entry(self, model_id: str) -> ModelEntry[ModelInfo]:
        return self.db.get(doc_id=self.get_doc_id(model_id))

    def delete_entry(self, model_id: str, delete_state_dict: bool) -> None:
        if delete_state_dict:
            os.remove(self.get_model_state_dict_file_path(model_id))

            optim_state_dict_path = Path(self.get_optimizer_state_dict_file_path(model_id))
            if optim_state_dict_path.is_file():
                os.remove(optim_state_dict_path)

        removed_doc_ids = self.db.remove(doc_ids=[self.get_doc_id(model_id)])
        assert len(removed_doc_ids) == 1, removed_doc_ids

    def delete_optim_state_dict(self, model_id: str):
        os.remove(self.get_optimizer_state_dict_file_path(model_id))

    def get_model_state_dict_file_path(self, model_id: str):
        return os.path.join(self.base_path, f'{model_id}--state_dict.pth')

    def get_optimizer_state_dict_file_path(self, model_id: str):
        return os.path.join(self.base_path, f'{model_id}--optim_state_dict.pth')

    def _save_entry(self, entry: ModelEntry[ModelInfo]):
        upserted_ids = self.db.upsert(tinydb.table.Document(entry, doc_id=self.get_doc_id(entry['model_id'])))
        assert len(upserted_ids) == 1, upserted_ids

    @staticmethod
    def get_doc_id(model_id: str):
        return hash(model_id)

    @staticmethod
    def create_model_id_query(model_id: str) -> QueryLike:
        entry_query = Query()
        return entry_query.model_id == model_id


