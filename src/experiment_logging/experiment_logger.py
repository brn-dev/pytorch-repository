import os.path
from contextlib import contextmanager
from typing import Optional, Any, Callable

from src.datetime import get_current_timestamp
from src.experiment_logging.experiment_log import ExperimentLog, DEFAULT_CATEGORY_KEY, ExperimentLogItem, \
    ModelDBReference, load_experiment_log, save_experiment_log
from src.hyper_parameters import HyperParameters
from src.id_generation import generate_timestamp_id
from src.summary_statistics import format_summary_statistics, is_summary_statistics
from src.system_info import get_system_info
from src.utils import default_fn, format_current_exception, remove_duplicates_keep_order


class ExperimentLogger:
    _experiment_log: Optional[ExperimentLog]
    experiment_log: ExperimentLog

    def __init__(
            self,
            log_folder_path: str,
            save_on_exit: bool = True,
            log_pretty: bool = False,
    ):
        self._experiment_log = None
        self.log_folder_path = log_folder_path
        self.save_on_exit = save_on_exit
        self.log_pretty = log_pretty

        self.current_items_by_category: dict[str, ExperimentLogItem] = {}

    @property
    def experiment_log(self):
        if self._experiment_log is None:
            raise RuntimeError('Start an experiment first!')
        return self._experiment_log

    @experiment_log.setter
    def experiment_log(self, log: ExperimentLog):
        self._experiment_log = log

    @property
    def experiment_file_path(self):
        return os.path.join(self.log_folder_path, f'{self.experiment_log["experiment_id"]}.json')

    def start_experiment_log(
            self,
            experiment_id: Optional[str] = None,
            experiment_tags: Optional[list[str]] = None,
            start_time: Optional[str] = None,
            hyper_parameters: Optional[HyperParameters] = None,
            system_info: Optional[dict[str, Any]] = None,
            setup: Optional[dict[str, str]] = None,
            notes: Optional[list[str]] = None,
            model_db_references: list[ModelDBReference] = None,
    ):
        experiment_log: ExperimentLog = {
            'experiment_id': default_fn(experiment_id, lambda: generate_timestamp_id()),
            'experiment_tags': default_fn(experiment_tags, lambda: []),
            'start_time': default_fn(start_time, lambda: get_current_timestamp()),
            'end_time': None,
            'end_exception': None,
            'hyper_parameters': default_fn(hyper_parameters, lambda: {}),
            'system_info': default_fn(system_info, get_system_info),
            'setup': default_fn(setup, lambda: {}),
            'notes': default_fn(notes, lambda: []),
            'model_db_references': model_db_references or [],
            'logs_by_category': {}
        }
        self.experiment_log = experiment_log

    def add_item(
            self,
            item: ExperimentLogItem,
            category: str = DEFAULT_CATEGORY_KEY,
            clear_none_valued_entries: bool = True
    ) -> ExperimentLogItem:
        if category not in self.experiment_log['logs_by_category']:
            self.experiment_log['logs_by_category'][category] = []

        item['__timestamp'] = get_current_timestamp()

        if clear_none_valued_entries:
            item = {
                k: v
                for k, v in item.items()
                if v is not None
            }

        self.experiment_log['logs_by_category'][category].append(item)

        return item

    def get_log_item(self, index: int | slice, category: str = DEFAULT_CATEGORY_KEY):
        return self.experiment_log['logs_by_category'][category][index]

    def get_latest_log_item(self, category: str = DEFAULT_CATEGORY_KEY):
        return self.get_log_item(-1, category=category)

    def item_start(self, category: str = DEFAULT_CATEGORY_KEY):
        self.current_items_by_category[category] = {}

    def item_log(self, key: str, value: Any, category: str = DEFAULT_CATEGORY_KEY):
        current_item = self.current_items_by_category[category]
        current_item[key] = value

    def item_log_many(self, values: dict[str, Any], category: str = DEFAULT_CATEGORY_KEY):
        current_item = self.current_items_by_category[category]
        current_item.update(values)

    def item_end(self, category: str = DEFAULT_CATEGORY_KEY) -> ExperimentLogItem:
        item = self.current_items_by_category[category]
        self.add_item(item, category=category)
        return item

    def load_experiment_log(self, file_path: str):
        self.experiment_log = load_experiment_log(file_path)

    def save_experiment_log(self, file_path: Optional[str] = None) -> ExperimentLog:
        if file_path is None:
            file_path = self.experiment_file_path

        save_experiment_log(file_path, self._experiment_log, 2 if self.log_pretty else None)

        print(f'saved experiment log {self.experiment_log["experiment_id"]} at {file_path}')

        return self._experiment_log

    def end_experiment_log(
            self,
            exception_str: Optional[str] = None,
            save_experiment: bool = True,
            file_path: Optional[str] = None,
            hyper_parameters_update: Optional[HyperParameters] = None,
            setup_update: Optional[dict[str, str]] = None,
            notes_update: Optional[list[str]] = None,
            model_db_references_update: list[ModelDBReference] = None,
    ):
        assert save_experiment or file_path is None, 'file_path must be None when save_experiment is False'

        self.experiment_log['end_time'] = get_current_timestamp()

        if exception_str:
            self.experiment_log['end_exception'] = exception_str
        if hyper_parameters_update:
            self.experiment_log['hyper_parameters'].update(hyper_parameters_update)
        if setup_update:
            self.experiment_log['setup'].update(setup_update)
        if notes_update:
            remove_duplicates_keep_order(self.experiment_log['notes'] + notes_update)
        if hyper_parameters_update:
            remove_duplicates_keep_order(self.experiment_log['model_db_references'] + model_db_references_update)

        if save_experiment:
            self.save_experiment_log(file_path)

    def format_current_log_item(
            self,
            category: str = DEFAULT_CATEGORY_KEY,
            mean_format: str | None = '.3f',
            std_format: str | None = '.3f',
            min_value_format: str | None = None,
            max_value_format: str | None = None,
            n_format: str | None = None,
    ):
        self.format_log_item(
            self.current_items_by_category[category],
            mean_format=mean_format,
            std_format=std_format,
            min_value_format=min_value_format,
            max_value_format=max_value_format,
            n_format=n_format,
        )

    @staticmethod
    def format_log_item(
            item: ExperimentLogItem,
            mean_format: str | None = '.3f',
            std_format: str | None = '.3f',
            min_value_format: str | None = None,
            max_value_format: str | None = None,
            n_format: str | None = None,
            float_format: str | None = '.3f',
            **format_by_name: str | dict[str, str],
    ) -> str:
        components: list[str] = []

        for name, value in item.items():
            if (fmt := format_by_name.get(name)) is not None:
                if is_summary_statistics(value):
                    formatted_value = format_summary_statistics(value, **fmt)
                else:
                    formatted_value = format(value, fmt)

            elif is_summary_statistics(value):
                formatted_value = format_summary_statistics(
                    value,
                    mean_format=mean_format,
                    std_format=std_format,
                    min_value_format=min_value_format,
                    max_value_format=max_value_format,
                    n_format=n_format
                )
            elif isinstance(value, float):
                formatted_value = format(value, float_format)
            else:
                formatted_value = str(value)

            components.append(f'{name} = {formatted_value}')

        return ', '.join(components)

@contextmanager
def log_experiment(
        experiment_logger: ExperimentLogger,
        experiment_id: Optional[str] = None,
        experiment_tags: Optional[list[str]] = None,
        start_time: Optional[str] = None,
        hyper_parameters: Optional[HyperParameters] = None,
        system_info: Optional[dict[str, Any]] = None,
        setup: Optional[dict[str, str]] = None,
        notes: Optional[list[str]] = None,
        model_db_references: list[ModelDBReference] = None,
        on_end: Callable[[ExperimentLog], dict[str, Any]] = lambda _: {},
):
    experiment_logger.start_experiment_log(
        experiment_id=experiment_id,
        experiment_tags=experiment_tags,
        start_time=start_time,
        hyper_parameters=hyper_parameters,
        system_info=system_info,
        setup=setup,
        notes=notes,
        model_db_references=model_db_references,
    )
    with end_experiment_on_exit(experiment_logger, on_end=on_end):
        yield experiment_logger

@contextmanager
def save_experiment_on_exit(experiment_logger: ExperimentLogger):
    try:
        yield experiment_logger
    except Exception as e:
        raise e
    finally:
        experiment_logger.save_experiment_log()

@contextmanager
def end_experiment_on_exit(
        experiment_logger: ExperimentLogger,
        on_end: Callable[[ExperimentLog], dict[str, Any]] = lambda _: {}
):
    exception_str: Optional[str] = None
    try:
        yield experiment_logger
    except Exception as e:
        exception_str = format_current_exception()
        raise e
    finally:
        end_kwargs = on_end(experiment_logger.experiment_log)
        experiment_logger.end_experiment_log(exception_str=exception_str, **end_kwargs)
