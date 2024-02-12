import random
import time

import numpy as np
import torch

def get_time_based_seed() -> int:
    return time.time_ns() % (2 ** 32)

def set_random_seed(seed: int | None = 42) -> int:
    if seed is None:
        seed = get_time_based_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
