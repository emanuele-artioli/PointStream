from __future__ import annotations

import unittest

from src.encoder.execution_pool import (
    InlineExecutionPool,
    TaggedMultiprocessPool,
    WorkerConfig,
    make_shared_cpu_tensor,
)


class TestExecutionPool(unittest.TestCase):
    def test_inline_pool_executes_callable(self) -> None:
        pool = InlineExecutionPool()

        def func(context, deps):
            return context["value"] + deps["base"]

        output = pool.execute("cpu", func, context={"value": 3}, deps={"base": 4})
        self.assertEqual(output, 7)

    def test_tagged_pool_stub_executes_inline(self) -> None:
        pool = TaggedMultiprocessPool(config=WorkerConfig(cpu_workers=1, gpu_workers=1))

        def func(context, deps):
            return deps["base"] * 2

        cpu_out = pool.execute("cpu", func, context={}, deps={"base": 5})
        gpu_out = pool.execute("gpu", func, context={}, deps={"base": 6})
        self.assertEqual(cpu_out, 10)
        self.assertEqual(gpu_out, 12)
        pool.shutdown()

    def test_shared_cpu_tensor_helper(self) -> None:
        tensor = make_shared_cpu_tensor((2, 3))
        self.assertEqual(tuple(tensor.shape), (2, 3))
        self.assertTrue(tensor.is_shared())


if __name__ == "__main__":
    unittest.main()
