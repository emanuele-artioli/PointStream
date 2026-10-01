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


def test_uvg_prepare_rejects_short_registered_source_before_conversion(tmp_path):
    source = tmp_path / 'Jockey_1920x1080_120fps_420_8bit_YUV.yuv'
    source.write_bytes(b'truncated raw input')
    output = tmp_path / 'converted'
    helper = BRIDGE.parent / 'gvcrt_prepare_uvg.py'
    result = subprocess.run([sys.executable, str(helper), '--source', str(source),
                             '--output-dir', str(output), '--ffmpeg', '/must/not/run'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'physical length differs' in result.stderr
    assert not output.exists()


class CoderStub:
    def __init__(self, intra=False):
        self.intra = intra
        self.events = []

    def set_use_two_entropy_coders(self, value):
        self.events.append(('partition', value))

    def compress(self, x, qp):
        self.events.append(('compress', qp))
        result = {'bit_stream': b'native bytes'}
        if self.intra:
            result['x_hat'] = 'I reconstructed pixels'
        return result

    def clear_dpb(self):
        self.events.append(('clear',))

    def add_ref_frame(self, feature, frame):
        self.events.append(('reference', feature, frame))

    def prepare_feature_adaptor_i(self, qp):
        self.events.append(('adapt', qp))


def test_native_p_compressor_needs_no_pixel_return():
    i, p = CoderStub(True), CoderStub()
    result = bridge.compress_frame(i, p, 'input', 3, False, True, 0)
    assert result == {'bit_stream': b'native bytes'}
    assert p.events == [('adapt', 0), ('compress', 3)]
    assert i.events == []


def test_i_initialization_and_both_native_partition_modes():
    i, p = CoderStub(True), CoderStub()
    assert bridge.configure_entropy(i, p, 1088, 1920) == 1
    bridge.compress_frame(i, p, 'input', 1, True, False, 0)
    assert i.events == [('partition', True), ('compress', 1)]
    assert p.events == [('partition', True), ('clear',), ('reference', None, 'I reconstructed pixels')]


def test_official_namespace_wins_over_later_regular_pointstream_src(tmp_path):
    official, shadow = tmp_path / 'official', tmp_path / 'pointstream'
    (official / 'src').mkdir(parents=True)
    (shadow / 'src').mkdir(parents=True)
    (official / 'src/probe.py').write_text('ORIGIN="verified GVC"\n')
    (shadow / 'src/__init__.py').write_text('ORIGIN="foreign PointStream"\n')
    script = '''import importlib.util,sys
spec=importlib.util.spec_from_file_location('bridge',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path[:0]=[sys.argv[2],sys.argv[3]]
m.bind_source_namespace(sys.argv[2])
import src.probe
assert src.probe.ORIGIN=='verified GVC'
assert list(src.__path__)==[sys.argv[2]+'/src']
'''
    subprocess.run([sys.executable, '-c', script, str(BRIDGE), str(official), str(shadow)], check=True)


def test_preloaded_foreign_src_is_rejected(tmp_path):
    official, shadow = tmp_path / 'official', tmp_path / 'pointstream'
    (official / 'src').mkdir(parents=True)
    (shadow / 'src').mkdir(parents=True)
    (shadow / 'src/__init__.py').write_text('ORIGIN="foreign"\n')
    script = '''import importlib.util,sys
spec=importlib.util.spec_from_file_location('bridge',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0,sys.argv[3]); import src
try:m.bind_source_namespace(sys.argv[2])
except RuntimeError as error:assert 'Foreign src' in str(error)
else:raise AssertionError('foreign package accepted')
'''
    subprocess.run([sys.executable, '-c', script, str(BRIDGE), str(official), str(shadow)], check=True)
