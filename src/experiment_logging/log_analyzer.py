import glob
import os.path
from typing import Callable

import matplotlib.axes
import numpy as np

from src.experiment_logging.experiment_log import ExperimentLog, ExperimentLogItem, DEFAULT_CATEGORY_KEY, \
    load_experiment_log, get_log_items
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

    def find_hyper_parameter_diffs(self, reference_log_id: str = None):
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

            differences.append((
                log_id,
                dict_diff(reference_log['hyper_parameters'], log['hyper_parameters'])
            ))

        return differences


    def plot_logs(
            self,
            get_x: Callable[[ExperimentLogItem], float],
            get_y: Callable[[ExperimentLogItem], float],
            ax: matplotlib.axes.Axes,
            category: str = DEFAULT_CATEGORY_KEY,
            item_filter: Callable[[ExperimentLogItem], bool] = lambda item: True,
            **plot_kwargs,
    ):
        for log in self.logs:
            log_items = [item for item in log['logs_by_category'][category] if item_filter(item)]

            x = np.array([get_x(item) for item in log_items])
            y = np.array([get_y(item) for item in log_items])

            ax.plot(x, y, label=log['experiment_id'], **plot_kwargs)

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
# la.load_log_folder('notebooks/experiment_logs/sac/')
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
