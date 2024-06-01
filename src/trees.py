from dataclasses import dataclass
from queue import Queue
from typing import Generic, TypeVar, Callable, Optional, Iterable

T = TypeVar('T')

@dataclass
class Node(Generic[T]):
    id: str

    parent: Optional['Node[T]']
    children: list['Node[T]']

    value: T

class Forest(Generic[T]):

    def __init__(
            self,
            entries: Iterable[T],
            get_id: Callable[[T], str],
            get_parent_id: Callable[[T], str | None]
    ):
        all_nodes: dict[str, Node[T]] = {
            (entry_id := get_id(entry)): Node(
                id=entry_id,
                parent=None,
                children=[],
                value=entry,
            )
            for entry in entries
        }

        root_nodes: list[Node[T]] = []
        for entry in entries:
            entry_id = get_id(entry)
            parent_entry_id = get_parent_id(entry)

            node = all_nodes[entry_id]
            if parent_entry_id is not None:
                parent_node = all_nodes[parent_entry_id]

                node.parent = parent_node
                parent_node.children.append(node)
            else:
                root_nodes.append(node)

        self.root_nodes = root_nodes
        self.all_nodes = all_nodes

    def __getitem__(self, node_id: str) -> Node[T]:
        return self.all_nodes[node_id]


    def compute_num_descendants(self, node_id: str, discount_factor: float = 1.0):
        descendant_queue: Queue[tuple[Node[T], float]] = Queue()
        descendant_queue.put((self[node_id], 1.0))

        discounted_num_descendants = 0
        while not descendant_queue.empty():
            node, node_discount_factor = descendant_queue.get()

            discounted_num_descendants += len(node.children) * node_discount_factor

            child_discount_factor = node_discount_factor * discount_factor
            for child_node in node.children:
                descendant_queue.put((child_node, child_discount_factor))

        return discounted_num_descendants
