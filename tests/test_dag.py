from __future__ import annotations

import unittest

from src.encoder.dag import DAGNode, DAGOrchestrator
from src.shared.tags import cpu_bound, gpu_bound


class TestDAG(unittest.TestCase):
    def test_dag_executes_dependencies_in_order(self) -> None:
        orchestrator = DAGOrchestrator()

        @cpu_bound
        def a(context, deps):
            return 2

        @gpu_bound
        def b(context, deps):
            return deps["a"] + 3

        orchestrator.add_node(DAGNode(name="a", func=a))
        orchestrator.add_node(DAGNode(name="b", func=b, dependencies=("a",)))

        result = orchestrator.run(initial_context={})
        self.assertEqual(result["a"], 2)
        self.assertEqual(result["b"], 5)
        self.assertEqual(result["a__tag"], "cpu")
        self.assertEqual(result["b__tag"], "gpu")

    def test_unknown_dependency_raises(self) -> None:
        orchestrator = DAGOrchestrator()

        def c(context, deps):
            return 0

        orchestrator.add_node(DAGNode(name="c", func=c, dependencies=("missing",)))
        with self.assertRaises(ValueError):
            orchestrator.run(initial_context={})

    def test_cycle_detection_raises(self) -> None:
        orchestrator = DAGOrchestrator()

        def x(context, deps):
            return 1

        def y(context, deps):
            return 2

        orchestrator.add_node(DAGNode(name="x", func=x, dependencies=("y",)))
        orchestrator.add_node(DAGNode(name="y", func=y, dependencies=("x",)))

        with self.assertRaises(ValueError):
            orchestrator.run(initial_context={})


if __name__ == "__main__":
    unittest.main()
