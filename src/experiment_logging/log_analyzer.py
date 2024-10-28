import copy
import glob
import itertools
import numbers
import os.path
from itertools import groupby
from typing import Callable, Any, Optional, Iterable

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from matplotlib.lines import Line2D

from src.experiment_logging.experiment_log import ExperimentLog, ExperimentLogItem, DEFAULT_CATEGORY_KEY, \
    load_experiment_log, get_log_items, JSON_EXTENSION, ZIPPED_JSON_EXTENSION
from src.hyper_parameters import TYPE_KEY, FQ_TYPE_KEY
from src.utils import dict_diff

DONT_SHOW = {'dont': 'show'}
SAME_COLOR = '___same_color___'

STABLE_BASELINES_TAG = 'stable_baselines3'


class LogAnalyzer:

    def __init__(self):
        self.logs: list[ExperimentLog] = []

    def load_log(self, file_path: str):
        self.logs.append(load_experiment_log(file_path))

    def load_log_folder(self, folder_path: str, log_filter: Callable[[ExperimentLog], bool] = lambda log: True):
        json_files = glob.glob(os.path.join(folder_path, f'*{JSON_EXTENSION}'))
        zipped_json_files = glob.glob(os.path.join(folder_path, f'*{ZIPPED_JSON_EXTENSION}'))
        for file in (json_files + zipped_json_files):
            log = load_experiment_log(file)
            if log_filter(log):
                self.logs.append(log)

    def load_sb3_folder(
            self,
            folder_path: str,
            filter_file: Callable[[str], bool] = lambda file_name: True,
            rename_item_key: dict[str, str] = None,
            parse_item_value: dict[str, Callable[[Any], Any]] = None,
            filter_item: Callable[[dict[str, Any]], bool] = lambda item: True,
    ):
        files = glob.glob(os.path.join(folder_path, '*.txt'))
        for file in files:
            if not filter_file(file):
                continue

            log_items = []
            log: ExperimentLog = {
                'experiment_id': file,
                'experiment_tags': [STABLE_BASELINES_TAG],
                'start_time': '',
                'end_time': '',
                'end_exception': None,
                'hyper_parameters': {},
                'system_info': {},
                'setup': {},
                'notes': [],
                'model_db_references': None,
                'logs_by_category': {DEFAULT_CATEGORY_KEY: log_items},
            }
            with open(file, 'r') as f:
                in_item = False
                current_item = {}
                for line in f:
                    line = line.strip()
                    if line.startswith('---'):
                        # Toggle the in_item flag
                        if in_item:
                            # We're ending an item
                            if filter_item(current_item):
                                log_items.append(current_item)
                            current_item = {}
                        in_item = not in_item
                    elif in_item:
                        # We're inside an item
                        if line.startswith('|'):
                            # Remove leading and trailing '|', then split by '|'
                            parts = [part.strip() for part in line.strip('|').split('|')]
                            if len(parts) >= 2:
                                key = parts[0]
                                value = parts[1]
                                if rename_item_key is not None and key in rename_item_key:
                                    key = rename_item_key[key]
                                if parse_item_value is not None and key in parse_item_value:
                                    value = parse_item_value[key](value)
                                current_item[key] = value
            self.logs.append(log)

    def find_hyper_parameter_diffs(
            self,
            reference_log_id: str = None,
            comparison_ids: Optional[set[str]] = None,
            collapse_to_none_dicts: bool = True
    ) -> list[tuple[str, dict]]:
        def find_and_replace_to_none_dicts(elem: tuple[Any, Any] | dict | list):
            if isinstance(elem, tuple) and len(elem) == 2 and elem[1] is None and isinstance(elem[0], dict):
                from_dict = elem[0]
                collapsed_dict = {}

                if TYPE_KEY in from_dict:
                    collapsed_dict[TYPE_KEY] = from_dict[TYPE_KEY]
                if FQ_TYPE_KEY in from_dict:
                    collapsed_dict[FQ_TYPE_KEY] = from_dict[FQ_TYPE_KEY]

                collapsed_dict['...'] = '...'

                return collapsed_dict, None
            if isinstance(elem, dict):
                return {
                    k: find_and_replace_to_none_dicts(v)
                    for k, v in elem.items()
                }
            if isinstance(elem, list):
                return [
                    find_and_replace_to_none_dicts(v)
                    for v in elem
                ]
            return elem

        if reference_log_id is None:
            reference_log = self.logs[0]
        else:
            filtered_logs = [log for log in self.logs if log['experiment_id'] == reference_log_id]
            if len(filtered_logs) != 1:
                raise ValueError(f'Reference log with id {reference_log_id} not found!')
            reference_log = filtered_logs[0]

        differences: list[tuple[str, dict]] = []
        for log in self.logs:
            log_id = log['experiment_id']
            if log_id == reference_log['experiment_id']:
                continue
            if comparison_ids is not None and log_id not in comparison_ids:
                continue

            diff = dict_diff(reference_log['hyper_parameters'], log['hyper_parameters'])

            if collapse_to_none_dicts:
                diff = find_and_replace_to_none_dicts(diff)

            differences.append((log_id, diff))

        return differences

    def plot_logs_grouped(
            self,
            get_x: Callable[[ExperimentLogItem], float],
            get_y: Callable[[ExperimentLogItem], float],
            get_log_group: Callable[[ExperimentLog], str],
            ax: matplotlib.axes.Axes,
            filter_log: Callable[[ExperimentLog, str], bool] = lambda log: True,
            category: str = DEFAULT_CATEGORY_KEY,
            plot_individual: bool = False,
            mean_plot_kwargs: dict[str, Any] = None,
            fill_plot_kwargs: dict[str, Any] = None,
            individual_plot_kwargs: dict[str, Any] = None,
            latest_individual_plot_kwargs: dict[str, Any] = None,
            latest_n_individual_plots: int = 1
    ):
        mean_plot_kwargs = mean_plot_kwargs or {}
        fill_plot_kwargs = fill_plot_kwargs or {'alpha': 0.2}
        individual_plot_kwargs = individual_plot_kwargs or {'linewidth': 0.5}

        logs_by_group: dict[str, list[ExperimentLog]] = {}
        for log in self.logs:
            group = get_log_group(log)

            if not filter_log(log, group):
                continue

            if group in logs_by_group:
                logs_by_group[group].append(log)
            else:
                logs_by_group[group] = [log]

        filtered_logs = [l for logs in logs_by_group.values() for l in logs]

        latest_logs = sorted(filtered_logs, key=lambda l: l['start_time'], reverse=True)[:latest_n_individual_plots]

        for group, group_logs in logs_by_group.items():
            points: dict[float, list[float]] = {}

            for log in group_logs:
                for item in log['logs_by_category'][category]:
                    x = get_x(item)
                    y = get_y(item)

                    if x in points:
                        points[x].append(y)
                    else:
                        points[x] = [y]

            points = dict(sorted(points.items()))

            mins: list[float] = []
            maxs: list[float] = []
            means: list[float] = []

            for x, ys in points.items():
                mins.append(min(ys))
                maxs.append(max(ys))
                means.append(np.mean(ys))

            xs = list(points.keys())

            label = f'({len(group_logs)}x) {group}'
            prev_color = None

            if mean_plot_kwargs is not DONT_SHOW:
                prev_color = self._plot(
                    x=xs,
                    y=means,
                    ax=ax,
                    label=label,
                    **mean_plot_kwargs
                )[0].get_color()
            if fill_plot_kwargs is not DONT_SHOW:
                prev_color = ax.fill_between(xs, mins, maxs, **fill_plot_kwargs).get_facecolor()[0]

            if plot_individual:
                if latest_individual_plot_kwargs is None:
                    self._plot_logs(
                        logs=group_logs,
                        get_x=get_x,
                        get_y=get_y,
                        ax=ax,
                        get_label=lambda log_: f'{group_logs.index(log_) + 1}. {label}',
                        category=category,
                        color=prev_color if prev_color is not None else SAME_COLOR,
                        **individual_plot_kwargs
                    )[0].get_color()
                else:
                    group_non_latest_logs = [l for l in group_logs if l not in latest_logs]
                    lines = self._plot_logs(
                        logs=group_non_latest_logs,
                        get_x=get_x,
                        get_y=get_y,
                        ax=ax,
                        get_label=lambda log_: f'{group_logs.index(log_) + 1}. {label}',
                        category=category,
                        color=prev_color if prev_color is not None else SAME_COLOR,
                        **individual_plot_kwargs
                    )
                    if len(lines) > 0:
                        prev_color = lines[0].get_color()

                    group_latest_logs = [l for l in group_logs if l in latest_logs]
                    if len(group_latest_logs) > 0:
                        self._plot_logs(
                            logs=group_latest_logs,
                            get_x=get_x,
                            get_y=get_y,
                            ax=ax,
                            get_label=lambda log_: f'{group_logs.index(log_) + 1}. {label}',
                            category=category,
                            color=prev_color if prev_color is not None else SAME_COLOR,
                            **latest_individual_plot_kwargs
                        )

    def plot_logs(
            self,
            get_x: Callable[[ExperimentLogItem], float],
            get_y: Callable[[ExperimentLogItem], float],
            ax: matplotlib.axes.Axes,
            get_label: Callable[[ExperimentLog], str] = lambda log: log['experiment_id'],
            category: str = DEFAULT_CATEGORY_KEY,
            item_filter: Callable[[ExperimentLogItem], bool] = lambda item: True,
            **plot_kwargs,
    ):
        self._plot_logs(
            logs=self.logs,
            get_x=get_x,
            get_y=get_y,
            ax=ax,
            get_label=get_label,
            category=category,
            item_filter=item_filter,
            **plot_kwargs
        )

    @staticmethod
    def _plot_logs(
            logs: Iterable[ExperimentLog],
            get_x: Callable[[ExperimentLogItem], float],
            get_y: Callable[[ExperimentLogItem], float],
            ax: matplotlib.axes.Axes,
            get_label: Callable[[ExperimentLog], str] = lambda log: None,
            category: str = DEFAULT_CATEGORY_KEY,
            item_filter: Callable[[ExperimentLogItem], bool] = lambda item: True,
            moving_average_window_size: Optional[int] = None,
            moving_average_plot_kwargs: dict[str, Any] = None,
            color: Any = None,
            **plot_kwargs,
    ):
        lines: list[Line2D] = []

        for log in logs:
            log_items = [item for item in log['logs_by_category'][category] if item_filter(item)]

            x = np.array([get_x(item) for item in log_items])
            y = np.array([get_y(item) for item in log_items])

            col = color
            if color is SAME_COLOR:
                if len(lines) > 0:
                    col = lines[-1].get_color()
                else:
                    col = None

            lines.extend(LogAnalyzer._plot(
                x=x, y=y, ax=ax,
                label=get_label(log),
                moving_average_window_size=moving_average_window_size,
                moving_average_plot_kwargs=moving_average_plot_kwargs,
                color=col,
                **plot_kwargs
            ))

        return lines

    @staticmethod
    def _plot(
            x: np.ndarray | list[float],
            y: np.ndarray | list[float],
            ax: plt.Axes | go.Figure,
            label: Optional[str] = None,
            moving_average_window_size: Optional[int] = None,
            moving_average_plot_kwargs: dict[str, Any] = None,
            **plot_kwargs
    ):
        plot_line = ax.plot(x, y, label=label, **plot_kwargs)[0]
        lines: list[Line2D] = [plot_line]

        if moving_average_window_size is not None:
            cumsum_vec = np.cumsum(np.insert(y, 0, 0))
            ma_vec = (
                             cumsum_vec[moving_average_window_size:] - cumsum_vec[:-moving_average_window_size]
                     ) / moving_average_window_size
            lines.append(ax.plot(
                x[moving_average_window_size - 1:],
                ma_vec,
                color=plot_line.get_color(),
                **(moving_average_plot_kwargs or {})
            )[0])

        return lines

    @staticmethod
    def get_log_items(log: ExperimentLog, category: str = DEFAULT_CATEGORY_KEY):
        # alias
        return get_log_items(log, category)
