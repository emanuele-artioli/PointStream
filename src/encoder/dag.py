from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from src.encoder.execution_pool import BaseExecutionPool, InlineExecutionPool


@dataclass(frozen=True)
class DAGNode:
    name: str
    func: Any
    dependencies: tuple[str, ...] = ()


class DAGOrchestrator:
    def __init__(self, execution_pool: BaseExecutionPool | None = None) -> None:
        self._nodes: dict[str, DAGNode] = {}
        self._execution_pool = execution_pool or InlineExecutionPool()

    def add_node(self, node: DAGNode) -> None:
        if node.name in self._nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self._nodes[node.name] = node

    def _validate_dependencies(self) -> None:
        for node in self._nodes.values():
            for dependency in node.dependencies:
                if dependency not in self._nodes:
                    raise ValueError(
                        f"Node '{node.name}' depends on unknown node '{dependency}'"
                    )

    def _topological_order(self) -> list[str]:
        self._validate_dependencies()

        incoming_count: dict[str, int] = {name: 0 for name in self._nodes}
        outgoing: dict[str, list[str]] = {name: [] for name in self._nodes}

        for node in self._nodes.values():
            for dependency in node.dependencies:
                incoming_count[node.name] += 1
                outgoing[dependency].append(node.name)

        queue: deque[str] = deque(
            name for name, degree in incoming_count.items() if degree == 0
        )
        ordered: list[str] = []

        while queue:
            current = queue.popleft()
            ordered.append(current)
            for neighbor in outgoing[current]:
                incoming_count[neighbor] -= 1
                if incoming_count[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(self._nodes):
            raise ValueError("Cycle detected in DAG")

        return ordered

    def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = dict(initial_context)
        for node_name in self._topological_order():
            node = self._nodes[node_name]
            dependency_outputs = {
                dependency: context[dependency] for dependency in node.dependencies
            }
            execution_tag = getattr(node.func, "_execution_tag", "cpu")
            context[node.name] = self._execution_pool.execute(
                tag=execution_tag,
                func=node.func,
                context=context,
                deps=dependency_outputs,
            )
            context[f"{node.name}__tag"] = execution_tag
        return context

    def shutdown(self) -> None:
        self._execution_pool.shutdown()
