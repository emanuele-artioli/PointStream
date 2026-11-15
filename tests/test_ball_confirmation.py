from fractions import Fraction

import numpy as np
import pytest

from experiments.tennis import ball_confirmation as confirmation
from src.pointstream import ball


def test_roi_missing_hint_is_native_and_touched_blocks_are_protected():
    rows = confirmation.roi_map([None, [64, 64], [0, 0]], 3).splitlines()
    assert set(rows[0].split()[1:]) == {'0'}
    cells = rows[1].split()[1:]
    assert cells.count('0') == 4
    assert set(cells) == {'0', '40'}
    assert rows[2].split()[1:].count('0') == 1


def test_patch_variable_length_and_missing_hint(tmp_path):
    frame = np.arange(confirmation.FRAME_BYTES, dtype=np.uint32).astype(np.uint8)
    path = tmp_path / 'source.yuv'
    path.write_bytes(frame.tobytes() * 2)
    patches, rows = confirmation.patch_frames(path, [[0, 0], None], 2)
    assert patches.shape == (2, 6144)
    assert rows[0]['left'] == rows[0]['top'] == 0
    assert np.all(patches[1, :4096] == 16)
    assert np.all(patches[1, 4096:] == 128)
    assert np.array_equal(patches[0, :4096].reshape(64, 64)[16:48, 16:48],
                          frame[:confirmation.W * confirmation.H].reshape(confirmation.H, confirmation.W)[:32, :32])


def test_sender_validation_binds_each_payload_to_centres():
    payload = ball.encode_frames([None], [None], confirmation.W, confirmation.H)
    import hashlib
    row = {'centres': [None], 'ball_payload_hex': payload.hex(),
           'ball_payload_sha256': hashlib.sha256(payload).hexdigest()}
    sender = {'frames': 1, 'size': [confirmation.W, confirmation.H], 'fps': [25, 1],
              'initial_track': [None],
              'points': [dict(row, codec=k, point=p) for k, p in confirmation.POINTS]}
    confirmation.validate_sender(sender, Fraction(25), 1)
    sender['points'][0]['centres'] = [[1, 1]]
    with pytest.raises(ValueError, match='positions mismatch'):
        confirmation.validate_sender(sender, Fraction(25), 1)


def test_archive_binding_detects_tampering(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'frame').write_bytes(b'physical evidence')
    archive = tmp_path / 'archive.tar'
    binding = confirmation.publish_tree(root, archive)
    confirmation.extract_tree(archive, binding, tmp_path / 'copy')
    assert (tmp_path / 'copy' / 'frame').read_bytes() == b'physical evidence'
    binding['members']['frame']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        confirmation.extract_tree(archive, binding, tmp_path / 'bad')


def test_smoke_prefix_reencodes_same_payload():
    import hashlib

    payload = ball.encode_frames([None, None], [None, None], confirmation.W, confirmation.H)
    row = {'centres': [None, None], 'origins': [None, None], 'ball_payload_hex': payload.hex(),
           'ball_payload_sha256': hashlib.sha256(payload).hexdigest()}
    sender = {'frames': 2, 'size': [confirmation.W, confirmation.H], 'fps': [25, 1],
              'initial_track': [None, None],
              'points': [dict(row, codec=k, point=p) for k, p in confirmation.POINTS]}
    prefix = confirmation.prefix_sender(sender, 1)
    confirmation.validate_sender(prefix, Fraction(25), 1)
    assert prefix['points'][0]['centres'] == [None]


def test_both_prospective_specs_obey_fleet_limits():
    from experiments.jobs.inbox import validate_spec

    gpu = confirmation.spec('bases', 'match235_000', '2099-01-01T00:00:00Z')
    cpu = confirmation.spec('controls', 'match235_000', '2099-01-01T00:00:00Z',
                            bases_archive='/data/base.tar', bases_sha='1' * 64,
                            bases_binding='/data/base.json', binding_sha='2' * 64,
                            bases_acceptance='/data/accepted.json', acceptance_sha='3' * 64)
    for declared in (gpu, cpu):
        normalized = validate_spec(declared, now=0)
        assert normalized['cpu_threads'] == 4
        assert normalized['budget_seconds'] == 2700
        assert normalized['scale']['frames'] == {'smoke': 32, 'full': 154}
    assert gpu['entrypoint'] == ['-m', 'experiments.tennis.ball_confirmation_entry']
    assert gpu['validator'] == ['{python}', '-m', 'experiments.tennis.ball_confirmation_entry', '--validate']
    assert gpu['hosts'] == ['gpu2', 'gpu3']
    assert gpu['gpu_models'] == ['RTX A6000']
    assert cpu['entrypoint'] == gpu['entrypoint']
    assert cpu['validator'] == gpu['validator']
    assert gpu['gpu_sharing']['mode'] == 'shared'
    assert gpu['gpu_sharing']['margin_mib'] == 8192
    assert gpu['gpu_memory_mib'] == 12288
    assert cpu['device'] == 'cpu'
