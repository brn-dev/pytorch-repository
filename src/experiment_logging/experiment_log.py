from typing import TypedDict, Optional, Any

from src.hyper_parameters import HyperParameters
from src.model_db.model_db import ModelEntry


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

    model_db_reference: Optional[ModelDBReference]

    hyper_parameters: HyperParameters
    setup: dict[str, str]
    notes: list[str]

    logs_by_category: dict[str, list[ExperimentLogItem]]
