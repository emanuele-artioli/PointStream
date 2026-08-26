from __future__ import annotations

import unittest

from src.shared.tags import cpu_bound, gpu_bound


class TestTags(unittest.TestCase):
    def test_cpu_bound_sets_execution_tag(self) -> None:
        @cpu_bound
        def fn(context, deps):
            return 1

        self.assertEqual(getattr(fn, "_execution_tag", None), "cpu")

    def test_gpu_bound_sets_execution_tag(self) -> None:
        @gpu_bound
        def fn(context, deps):
            return 1

        self.assertEqual(getattr(fn, "_execution_tag", None), "gpu")


if __name__ == "__main__":
    unittest.main()
