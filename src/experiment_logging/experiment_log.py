import json
import os
from typing import TypedDict, Optional, Any

from src.hyper_parameters import HyperParameters

DEFAULT_CATEGORY_KEY = '__default'


class ModelDBReference(TypedDict):
    model_db: str
    model_id: str


ExperimentLogItem = dict[str, Any]


class ExperimentLog(TypedDict):

    experiment_id: str
    experiment_tags: list[str]

    start_time: str
    end_time: Optional[str]
    end_exception: Optional[str]

    model_db_reference: Optional[ModelDBReference]

    hyper_parameters: HyperParameters
    system_info: dict[str, Any]
    setup: dict[str, str]
    notes: list[str]

    logs_by_category: dict[str, list[ExperimentLogItem]]


def load_experiment_log(file_path: str) -> ExperimentLog:
    with open(file_path, 'r') as file:
        return json.load(file)


def save_experiment_log(
        file_path: str,
        log: ExperimentLog,
        indent: Optional[int] = None
):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'w') as file:
        json.dump(log, file, indent=indent if indent is not None else None)
