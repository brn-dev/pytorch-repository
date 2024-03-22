from typing import Literal, Iterable, Callable

import numpy as np

from src.integer_sequences import generate_sequence_up_to


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

        if not LayerConnections.is_valid(connections, num_layers):
            raise ValueError(f"Layer connections are not valid for {num_layers} layers - {connections}")

        return connections

    @staticmethod
    def is_valid(connections: np.ndarray, num_layers: int) -> bool:
        return ((connections >= 0).all()
                and (connections <= num_layers).all()
                and len(np.unique(connections[:, 0])) == num_layers + 1
                and len(np.unique(connections[:, 1])) == num_layers + 1
                and (connections[:, 0] <= connections[:, 1]).all()
                )


def _as_set_of_tuples(arr: np.ndarray):
    return set(
        (int(connection[0]), int(connection[1]))
        for connection in arr
    )


class LayerConnectionsBuilder:

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.connections: set[tuple[int, int]] = set()

    def _fetch_sequence(self, sequence: Iterable[int] | Callable[[int], int]) -> set[int]:
        if not isinstance(sequence, Iterable):
            return set(generate_sequence_up_to(sequence, self.num_layers))
        return set(sequence)

    def sequential(self):
        self.connections.update(_as_set_of_tuples(LayerConnections.by_name('sequential', self.num_layers)))
        return self

    def parallel(self):
        self.connections.update(_as_set_of_tuples(LayerConnections.by_name('parallel', self.num_layers)))
        return self

    def full(self):
        self.connections.update(_as_set_of_tuples(LayerConnections.by_name('full', self.num_layers)))

    def step_receives(self, tensor_step: int, incoming_tensor_steps: Iterable[int] | Callable[[int], int]):
        incoming_tensor_steps = self._fetch_sequence(incoming_tensor_steps)
        tensor_step = tensor_step % (self.num_layers + 1)
        self.connections.update([
            (from_idx, tensor_step)
            for from_idx
            in incoming_tensor_steps
            if from_idx <= tensor_step
        ])
        return self

    def every_step_receives(self, incoming_tensor_steps: Iterable[int] | Callable[[int], int]):
        incoming_tensor_steps = self._fetch_sequence(incoming_tensor_steps)
        for to_idx in range(self.num_layers + 1):
            self.step_receives(to_idx, incoming_tensor_steps)
        return self

    def every_step_sends_to(self, tensor_step: int):
        tensor_step = tensor_step % (self.num_layers + 1)
        self.connections.update(
            (from_idx, tensor_step)
            for from_idx
            in range(tensor_step + 1)
        )
        return self

    def steps_receive(
            self,
            from_steps: Iterable[int] | Callable[[int], int],
            to_steps: Iterable[int] | Callable[[int], int]
    ):
        from_steps = self._fetch_sequence(from_steps)
        to_steps = self._fetch_sequence(to_steps)
        for to_step in to_steps:
            self.step_receives(to_step, from_steps)
        return self

    def every_step_to_output(self, excluded_steps: Iterable[int] | Callable[[int], int] = ()):
        excluded_steps = self._fetch_sequence(excluded_steps)
        excluded_steps = set(excluded_steps)
        for step in range(self.num_layers + 1):
            if step not in excluded_steps:
                self.connections.add((step, self.num_layers))
        return self

    def to_np(self):
        return LayerConnections.to_np(
            [
                list(connection)
                for connection
                in sorted(self.connections)
            ],
            num_layers=self.num_layers
        )
