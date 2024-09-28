import glob
import os.path
from typing import Callable, Iterable, Any, Optional

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_logging.experiment_log import ExperimentLog, ExperimentLogItem, DEFAULT_CATEGORY_KEY, \
    load_experiment_log


class LogAnalyzer:

    def __init__(self):
        self.logs: list[ExperimentLog] = []

    def load_log(self, file_path: str):
        self.logs.append(load_experiment_log(file_path))

    def load_log_folder(self, folder_path: str):
        files = glob.glob(os.path.join(folder_path, '*.json'))
        self.logs.extend((
            load_experiment_log(file)
            for file in files
        ))

    def plot_logs(
            self,
            get_x: Callable[[ExperimentLogItem], float],
            get_y: Callable[[ExperimentLogItem], float],
            ax: matplotlib.axes.Axes,
            category: str = DEFAULT_CATEGORY_KEY,
            **plot_kwargs,
    ):
        for log in self.logs:
            log_items = log['logs_by_category'][category]

            x = np.array([get_x(item) for item in log_items])
            y = np.array([get_y(item) for item in log_items])

            ax.plot(x, y, label=log['experiment_id'], **plot_kwargs)
