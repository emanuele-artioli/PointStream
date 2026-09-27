from __future__ import annotations

import subprocess
import sys


def test_sam_adapter_import_does_not_load_unrelated_metric_backends() -> None:
    code = """
import sys
import src.components
assert 'src.components.metrics' not in sys.modules
from src.components.segmentation.sam31 import Sam31SequenceSegmenter
assert Sam31SequenceSegmenter.__name__ == 'Sam31SequenceSegmenter'
assert 'src.components.metrics' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_explicit_registry_aggregation_still_loads_all_axes() -> None:
    code = """
from src.components import all_registries
registries = all_registries()
assert set(registries) == {
    'appearance', 'background', 'codec', 'detector', 'domain', 'generation',
    'metric', 'motion', 'pose', 'rigid', 'scene', 'segmenter', 'selection',
    'temporal', 'tracking', 'transport'
}
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
