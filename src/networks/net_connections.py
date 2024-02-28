from typing import Literal

import numpy as np

class NetConnections:

    ConnectionsPresets = Literal['dense', 'sequential']
    ConnectionsLike = tuple[tuple[int, int]] | list[list[int, int]] | np.ndarray | ConnectionsPresets  # TODO Module

    @staticmethod
    def by_name(
            name: ConnectionsPresets | str,
            num_layers: int,
    ) -> np.ndarray:
        if name == 'dense':
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
        else:
            raise ValueError('Unknown connections name')

        return connections.astype(int)

    @staticmethod
    def to_np(connections_like: ConnectionsLike, num_layers: int) -> np.ndarray:
        if isinstance(connections_like, str):
            connections = NetConnections.by_name(connections_like, num_layers)
        else:
            connections = np.array(connections_like)
            connections = (connections % (num_layers + 1))

            assert NetConnections.is_valid(connections, num_layers)

        return connections.astype(int)

    @staticmethod
    def is_valid(connections: np.ndarray, num_layers: int) -> bool:
        return ((connections >= 0).all()
                and (connections <= num_layers).all()
                and len(np.unique(connections[:, 0])) == num_layers + 1
                and len(np.unique(connections[:, 1])) == num_layers + 1
                and (connections[:, 0] <= connections[:, 1]).all()
                )
