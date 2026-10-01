"""Focused fail-closed checks without torch, CUDA or authored model files."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

BRIDGE = Path(__file__).resolve().parents[2] / 'experiments/generative/gvcrt_native_bridge.py'
spec = importlib.util.spec_from_file_location('gvcrt_bridge', BRIDGE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class Tensor:
    def __init__(self, shape=(2,)):
        self.shape = shape

    def numel(self):
        return 2


class Model:
    def state_dict(self):
        return {'weight': Tensor()}

    def parameters(self):
        return [Tensor()]

    def load_state_dict(self, state, strict):
        assert strict and set(state) == {'weight'}


def test_complete_i_state_uses_actual_student_dictionary():
    receipt = bridge.checked_state(Model(), {'student': {'module.weight': Tensor()}, 'ema_shadow': {}}, 'I')
    assert receipt['selected_dictionary'] == 'student'
    assert receipt['strict_complete_coverage']


@pytest.mark.parametrize('state', [{}, {'weight': Tensor((3,))}, {'weight': Tensor(), 'extra': Tensor()}])
def test_missing_wrong_shape_or_extra_state_fails(state):
    with pytest.raises(ValueError):
        bridge.checked_state(Model(), {'student': state}, 'I')


def test_normalized_duplicate_fails():
    with pytest.raises(ValueError, match='Duplicate'):
        bridge.checked_state(Model(), {'student': {'weight': Tensor(), 'module.weight': Tensor()}}, 'I')


def test_checkpoint_hash_mismatch_fails(tmp_path):
    payload = tmp_path / 'checkpoint'
    payload.write_bytes(b'not a model')
    with pytest.raises(ValueError, match='SHA256'):
        bridge.checked_file(payload, '0' * 64)


def test_source_denial_blocks_pathlib_and_os_open(tmp_path):
    source = tmp_path / 'source.png'
    source.write_bytes(b'hidden')
    script = '''import importlib.util, pathlib, os, sys
spec=importlib.util.spec_from_file_location('bridge',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
attempts=m.deny_sources([sys.argv[2]])
assert not attempts
for operation in (lambda: pathlib.Path(sys.argv[2]).read_bytes(), lambda: os.open(sys.argv[2],os.O_RDONLY)):
 try:operation()
 except PermissionError:pass
 else:raise AssertionError('source access allowed')
assert len(attempts)==2
'''
    subprocess.run([sys.executable, '-c', script, str(BRIDGE), str(source)], check=True)
