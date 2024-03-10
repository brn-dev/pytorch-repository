from typing import Literal

import numpy as np


class LayerConnections:
    Presets = Literal['full', 'sequential', 'parallel']
    LayerConnectionsLike = list[list[int, int]] | np.ndarray | Presets

    @staticmethod
    def by_name(
            name: Presets | str,
            num_layers: int,
    ) -> np.ndarray:
        if name == 'full':
            connections = np.array([
                [i, j]
                for i in range(0, num_layers + 1)
                for j in range(i, num_layers + 1)
            ])
        elif name == 'sequential':
            connections = np.array([
                [i, i]
                for i in range(0, num_layers + 1)
            ])
        elif name == 'parallel':
            connections = np.array(
                [[0, i] for i in range(0, num_layers)]
                + [[i, num_layers] for i in range(1, num_layers + 1)]
            )
        else:
            raise ValueError('Unknown connections name')

        return connections.astype(int)

    @staticmethod
    def to_np(layer_connections_like: LayerConnectionsLike, num_layers: int) -> np.ndarray:
        if isinstance(layer_connections_like, str):
            connections = LayerConnections.by_name(layer_connections_like, num_layers)
        else:
            connections = np.array(layer_connections_like).astype(int)
            connections = (connections % (num_layers + 1))

        assert LayerConnections.is_valid(connections, num_layers)

        return connections

    @staticmethod
    def is_valid(connections: np.ndarray, num_layers: int) -> bool:
        return ((connections >= 0).all()
                and (connections <= num_layers).all()
                and len(np.unique(connections[:, 0])) == num_layers + 1
                and len(np.unique(connections[:, 1])) == num_layers + 1
                and (connections[:, 0] <= connections[:, 1]).all()
                )
