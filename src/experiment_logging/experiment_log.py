import json
import os
from typing import TypedDict, Optional, Any

from src.hyper_parameters import HyperParameters
from src.scientific_float_json_encoder import ScientificFloatJsonEncoder

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

    hyper_parameters: HyperParameters
    system_info: dict[str, Any]
    setup: dict[str, str]
    notes: list[str]
    model_db_references: list[ModelDBReference]

    logs_by_category: dict[str, list[ExperimentLogItem]]


def get_log_items(log: ExperimentLog, category: str = DEFAULT_CATEGORY_KEY):
    return log['logs_by_category'][category]


def load_experiment_log(file_path: str) -> ExperimentLog:
    with open(file_path, 'r') as file:
        return json.load(file)


def save_experiment_log(
        file_path: str,
        log: ExperimentLog,
        indent: Optional[int] = None,
        float_precision: Optional[int] = 4,
):
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)


    with open(file_path, 'w') as file:
        if float_precision is None:
            json.dump(
                obj=log,
                fp=file,
                indent=indent if indent is not None else None,
            )
        else:
            json.dump(
                obj=log,
                fp=file,
                indent=indent if indent is not None else None,
                cls=ScientificFloatJsonEncoder,
                decimal_precision=float_precision
            )
