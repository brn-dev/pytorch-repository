from typing import Any, Literal

import numpy as np
import torch

InfoDict = dict[str, Any]


def stack_infos(infos: list[InfoDict]) -> InfoDict:
    return combine_infos(infos, combination_method='stack')


def concat_infos(infos: list[InfoDict]) -> InfoDict:
    return combine_infos(infos, combination_method='concat')


def combine_infos(infos: list[InfoDict], combination_method: Literal['stack', 'concat']) -> InfoDict:
    assert len(infos) > 0
    assert all(info.keys() == infos[0].keys() for info in infos[1:])

    keys = infos[0].keys()

    stacked_info: InfoDict = {}
    for key in keys:
        values: list[Any] = [infos[0][key]]

        for info in infos[1:]:
            values.append(info[key])

        if isinstance(values[0], torch.Tensor):
            match combination_method:
                case 'stack':
                    stacked_info[key] = torch.stack(values)
                case 'concat':
                    if len(values[0].shape) == 0:
                        values = [val.unsqueeze(0) for val in values]
                    stacked_info[key] = torch.cat(values, dim=0)
                case _:
                    raise ValueError(combination_method)
        elif isinstance(values[0], np.ndarray):
            match combination_method:
                case 'stack':
                    stacked_info[key] = np.stack(values)
                case 'concat':
                    stacked_info[key] = np.concatenate(values, axis=0)
                case _:
                    raise ValueError(combination_method)
        else:
            raise ValueError

    return stacked_info


