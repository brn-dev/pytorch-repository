import copy
import glob
import numbers
import os.path
from itertools import groupby
from typing import Callable, Any, Optional, Iterable

import matplotlib.axes
import numpy as np

from src.experiment_logging.experiment_log import ExperimentLog, ExperimentLogItem, DEFAULT_CATEGORY_KEY, \
    load_experiment_log, get_log_items
from src.hyper_parameters import TYPE_KEY, FQ_TYPE_KEY
from src.utils import dict_diff


class LogAnalyzer:

    def __init__(self):
        self.logs: list[ExperimentLog] = []

    def load_log(self, file_path: str):
        self.logs.append(load_experiment_log(file_path))

    def load_log_folder(self, folder_path: str, log_filter: Callable[[ExperimentLog], bool] = lambda log: True):
        files = glob.glob(os.path.join(folder_path, '*.json'))
        for file in files:
            log = load_experiment_log(file)
            if log_filter(log):
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
            category: str = DEFAULT_CATEGORY_KEY,
            plot_individual: bool = False,
            mean_plot_kwargs: dict[str, Any] = None,
            fill_plot_kwargs: dict[str, Any] = None,
            individual_plot_kwargs: dict[str, Any] = None,
    ):
        mean_plot_kwargs = mean_plot_kwargs or {}
        fill_plot_kwargs = fill_plot_kwargs or {'alpha': 0.2}
        individual_plot_kwargs = individual_plot_kwargs or {'linewidth': 0.5}

        logs_by_group: dict[str, list[ExperimentLog]] = {}
        for log in self.logs:
            group = get_log_group(log)
            if group in logs_by_group:
                logs_by_group[group].append(log)
            else:
                logs_by_group[group] = [log]

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

            mins: list[float] = []
            maxs: list[float] = []
            means: list[float] = []

            for x, ys in points.items():
                mins.append(min(ys))
                maxs.append(max(ys))
                means.append(np.mean(ys))

            xs = list(points.keys())

            line = ax.plot(xs, means, label=f'({len(group_logs)}x) {group}', **mean_plot_kwargs)[0]
            ax.fill_between(xs, mins, maxs, **fill_plot_kwargs)

            if plot_individual:
                self._plot_logs(
                    logs=group_logs,
                    get_x=get_x,
                    get_y=get_y,
                    ax=ax,
                    category=category,
                    color=line.get_color(),
                    **individual_plot_kwargs
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
            **plot_kwargs,
    ):
        for log in logs:
            log_items = [item for item in log['logs_by_category'][category] if item_filter(item)]

            x = np.array([get_x(item) for item in log_items])
            y = np.array([get_y(item) for item in log_items])

            ax.plot(x, y, label=get_label(log), **plot_kwargs)


    @staticmethod
    def get_log_items(log: ExperimentLog, category: str = DEFAULT_CATEGORY_KEY):
        # alias
        return get_log_items(log, category)


# import glob
# import matplotlib.pyplot as plt
# from src.experiment_logging.log_analyzer import LogAnalyzer
# from src.experiment_logging.experiment_log import *
# from src.datetime import get_current_timestamp
# la = LogAnalyzer()
# la.load_log_folder('notebooks/experiment_logs/HalfCheetah-v4/sac/')
# fig: plt.Figure
# ax: plt.Axes
# fig, ax = plt.subplots(figsize=(20, 12))
# fig.text(.85, .95, get_current_timestamp())
# ax.grid()
# # ax.set(xlabel='steps', ylabel='step_time')
# la.plot_logs(
#     lambda item: item['step'],
#     lambda item: item['scores']['mean'] if item['scores'] is not None else 0 ,
#     ax=ax,
#     linewidth=1,
# )
# ax.legend()
# fig.show()
