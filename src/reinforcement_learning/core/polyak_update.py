from typing import Iterable

import torch


def polyak_update(
        params: Iterable[torch.Tensor],
        target_params: Iterable[torch.Tensor],
        tau: float
):
    with torch.no_grad():
        for param, target_param in zip(params, target_params, strict=True):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
