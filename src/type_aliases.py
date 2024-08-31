from typing import Any

import numpy as np
import torch

KwArgs = dict[str, Any]

TensorDict = dict[str, torch.Tensor]

NpArrayDict = dict[np.ndarray]

ShapeDict = dict[str, tuple[int, ...]]

