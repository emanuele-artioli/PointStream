"""Synthetic files test receiver-origin validation; no scientific output is asserted."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.tennis import k5d_encoder as encoder, k5d_same_job_receiver as receiver


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict, Path]:
    root = tmp_path / 'publish'
    work = tmp_path / 'publish-work'
    stage = tmp_path / '20261110T000000Z-deadbeef' / 'full'
    root.mkdir()
    work.mkdir()
    stage.mkdir(parents=True)
    identity = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'environment_sha256': 'c' * 64}
    uuid = 'GPU-abcdef12'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    monkeypatch.setenv('PS_CLAIMED_GPU_UUID', uuid)
    (stage / 'dispatch.json').write_text(json.dumps({'code': identity, 'staged_environment': {'sha256': identity['environment_sha256']},
        'environment': {'gpu': {'name': 'NVIDIA RTX A6000', 'uuid': uuid}}}))
    points = []
    for codec, point in encoder.POINTS:
        name = f'{codec}-{point}'
        (root / name).mkdir()
        native_bytes, wire_bytes, baseline_bytes = b'native-' + name.encode(), b'wire-' + name.encode(), b'baseline-' + name.encode()
        (root / name / 'native.bin').write_bytes(native_bytes)
        (root / name / 'stream.psbl').write_bytes(wire_bytes)
        (root / name / 'baseline.psbl').write_bytes(baseline_bytes)
        src = work / f'receiver-{name}'
        src.mkdir()
        yuv = bytes([point % 255]) * 48
        rgb = bytes([(point + 1) % 255]) * 96
        (src / 'native.yuv').write_bytes(yuv)
        (src / 'rendered.rgb').write_bytes(rgb)
        receipt = {'wire_sha256': sha(wire_bytes), 'native_sha256': sha(native_bytes),
                   'wire_bytes': len(wire_bytes), 'native_bytes': len(native_bytes), 'resource_map_sha256': 'f' * 64,
                   'decoded_yuv_sha256': sha(yuv), 'rendered_rgb_sha256': sha(rgb), 'frames': 2,
                   'size': [4, 4], 'resource_id': codec}
        (src / 'receiver.json').write_text(json.dumps(receipt))
        points.append({'codec': codec, 'point': point, 'wire_sha256': sha(wire_bytes), 'native_sha256': sha(native_bytes),
                       'wire_bytes': len(wire_bytes), 'native_bytes': len(native_bytes),
                       'encoder_decoded_sha256': sha(yuv), 'encoder_rendered_sha256': sha(rgb), 'receiver': receipt,
                       'decode_parity': True, 'render_parity': True,
                       'baseline': {'zero_ball_wire_sha256': sha(baseline_bytes), 'metrics': [], 'judges': {}},
                       'streak': {'metrics': [], 'judges': {}, 'reuse': None}})
    report = {'frames': 2, 'size': [4, 4], 'clip': 'synthetic', 'resource_map_sha256': 'f' * 64,
              'tools': {'ffmpeg': {'sha256': '0' * 64}}, 'points': points,
              'checkpoint_prepare': True, 'full_metrics': False}
    (root / 'pilot.json').write_text(json.dumps(report))
    (root / 'source-qualification.json').write_text(json.dumps({'qualified': True}))
    return root, work, identity, stage


def test_actual_receiver_files_are_retained_and_all_tampering_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, work, identity, stage = fixture(tmp_path, monkeypatch)
    receiver.retain(root, work, identity, stage)
    report = json.loads((root / 'pilot.json').read_text())
    marker = receiver.verify(root, report, identity)
    assert marker['producer_job'] == stage.parent.name
    point = report['points'][0]
    evidence = root / f"{point['codec']}-{point['point']}" / 'receiver-evidence'
    pixels = evidence / 'native.yuv'
    pixels.write_bytes(b'x' * pixels.stat().st_size)
    with pytest.raises(ValueError, match='pixels'):
        receiver.verify(root, report, identity)
    pixels.write_bytes((work / f"receiver-{point['codec']}-{point['point']}" / 'native.yuv').read_bytes())
    value = json.loads((evidence / 'receiver.json').read_text())
    value['wire_sha256'] = '0' * 64
    (evidence / 'receiver.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='receiver identity'):
        receiver.verify(root, report, identity)


def test_rejects_foreign_gpu_and_missing_actual_receiver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, work, identity, stage = fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('PS_CLAIMED_GPU_UUID', 'GPU-foreign')
    with pytest.raises(ValueError, match='dispatch identity'):
        receiver.retain(root, work, identity, stage)
    monkeypatch.setenv('PS_CLAIMED_GPU_UUID', 'GPU-abcdef12')
    (work / 'receiver-svt-48' / 'native.yuv').unlink()
    with pytest.raises(ValueError, match='actual receiver output'):
        receiver.retain(root, work, identity, stage)


def test_sealed_checkpoint_binds_producer_and_rejects_pixel_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_checkpoint as cp

    root, work, identity, stage = fixture(tmp_path, monkeypatch)
    receiver.retain(root, work, identity, stage)
    monkeypatch.setattr(cp.enc, 'validate', lambda *_args, **_kw: {})
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda *_: None)
    cp.seal(root, identity)
    digest = cp.oracle.sha(root / 'checkpoint.json')
    cp.verify(root, digest, identity, require_same_job_receiver=True)
    point = encoder.POINTS[0]
    pixels = root / f'{point[0]}-{point[1]}' / 'receiver-evidence' / 'rendered.rgb'
    pixels.write_bytes(b'x' * pixels.stat().st_size)
    with pytest.raises(ValueError, match='checkpoint content'):
        cp.verify(root, digest, identity, require_same_job_receiver=True)


def test_retained_marker_cannot_be_missing_for_new_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_checkpoint as cp

    root, _work, identity, _stage = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cp.enc, 'validate', lambda *_args, **_kw: {})
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda *_: None)
    cp.seal(root, identity)
    with pytest.raises(ValueError, match='producer evidence required'):
        cp.verify(root, cp.oracle.sha(root / 'checkpoint.json'), identity, require_same_job_receiver=True)


def test_standalone_score_refuses_same_job_checkpoint_before_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import sys
    import types
    from experiments.tennis import k5d_checkpoint as cp

    root = tmp_path / 'checkpoint'
    root.mkdir()
    (root / receiver.MARKER).write_text('{}')
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setitem(sys.modules, 'src.codecs.quality', types.ModuleType('src.codecs.quality'))
    with pytest.raises(ValueError, match='require coalesced'):
        cp.score_point(argparse.Namespace(checkpoint=str(root), output=str(tmp_path / 'output')), 'sha', {})


def test_assembly_rejects_wrong_receiver_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_checkpoint as cp

    root, work, identity, stage = fixture(tmp_path, monkeypatch)
    receiver.retain(root, work, identity, stage)
    monkeypatch.setattr(cp.enc, 'validate', lambda *_args, **_kw: {})
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda *_: None)
    cp.seal(root, identity)
    digest = cp.oracle.sha(root / 'checkpoint.json')
    report = json.loads((root / 'pilot.json').read_text())
    marker = json.loads((root / receiver.MARKER).read_text())
    records = []
    for index, point in enumerate(report['points']):
        origin = {**marker, 'producer_job': '20261110T000000Z-foreign0'} if index == 0 else marker
        payload = {'checkpoint_sha256': digest, 'execution_identity': identity, 'point': point,
                   'receiver_replay': point['receiver'], 'receiver_origin': origin,
                   'receiver_origin_role': 'producer job; no later native decode'}
        path = tmp_path / f'point-{index}.json'
        path.write_text(json.dumps(payload))
        records.append({'path': str(path), 'sha256': cp.oracle.sha(path)})
    with pytest.raises(ValueError, match='producer receipt'):
        cp.assemble(root, digest, records, tmp_path / 'assembled', identity)


def test_new_bridge_rejects_extra_h2_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import difflib
    import io
    import tarfile
    from experiments.tennis import k5d_control_bridge as base, k5d_same_job_source_bridge as bridge
    from experiments.tennis import k5d_parent8_child2_source_bridge as parent

    def archive(path: Path, files: dict[str, bytes]) -> str:
        with tarfile.open(path, 'w') as tar:
            for name, value in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    old, new = tmp_path / 'old.tar', tmp_path / 'new.tar'
    before, after = b'old contents\n', b'new contents\n'
    patch = b''.join(difflib.diff_bytes(difflib.unified_diff, before.splitlines(keepends=True),
        after.splitlines(keepends=True), fromfile=b'old/checkpoint.py', tofile=b'new/checkpoint.py'))
    monkeypatch.setattr(bridge, 'PATCHES', {'checkpoint.py': (sha(before), sha(after), sha(patch))})
    monkeypatch.setattr(bridge, 'ADDITIONS', {parent.FIXTURE, 'new.py'})
    monkeypatch.setattr(parent, 'FIXTURE_SHA', sha(b'fixture'))
    old_sha = archive(old, {'checkpoint.py': before, 'model.py': b'unchanged'})
    new_sha = archive(new, {'checkpoint.py': after, 'model.py': b'unchanged', parent.FIXTURE: b'fixture', 'new.py': b'new'})
    old_id, _ = base.manifest(old, old_sha)
    new_id, _ = base.manifest(new, new_sha)
    monkeypatch.setattr(base, 'CALIBRATION_SOURCE', old_id)
    first = {'source_sha256': old_id, 'snapshot_sha256': old_sha, 'environment_sha256': 'a' * 64}
    second = {'source_sha256': new_id, 'snapshot_sha256': new_sha, 'environment_sha256': 'a' * 64}
    assert bridge.bridge(old, first, new, second)['bridge_type'] == 'same-job-receiver-v1'
    changed_sha = archive(new, {'checkpoint.py': after, 'model.py': b'tampered', parent.FIXTURE: b'fixture', 'new.py': b'new'})
    changed_id, _ = base.manifest(new, changed_sha)
    with pytest.raises(ValueError, match='source changes'):
        bridge.bridge(old, first, new, {**second, 'source_sha256': changed_id, 'snapshot_sha256': changed_sha})


def test_assembly_keeps_producer_receipts_without_copying_bulk_pixels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_checkpoint as cp

    root, work, identity, stage = fixture(tmp_path, monkeypatch)
    receiver.retain(root, work, identity, stage)
    monkeypatch.setattr(cp.enc, 'validate', lambda *_args, **_kw: {'passed': True})
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda *_: None)
    cp.seal(root, identity)
    digest = cp.oracle.sha(root / 'checkpoint.json')
    report = json.loads((root / 'pilot.json').read_text())
    marker = json.loads((root / receiver.MARKER).read_text())
    records = []
    for index, point in enumerate(report['points']):
        payload = {'checkpoint_sha256': digest, 'execution_identity': identity, 'point': point,
                   'receiver_replay': point['receiver'], 'receiver_origin': marker,
                   'receiver_origin_role': 'producer job; no later native decode'}
        path = tmp_path / f'point-{index}.json'
        path.write_text(json.dumps(payload))
        records.append({'path': str(path), 'sha256': cp.oracle.sha(path)})
    assembled = tmp_path / 'assembled'
    assert cp.assemble(root, digest, records, assembled, identity) == {'passed': True}
    assert (root / receiver.MARKER).is_file()
    assert not (assembled / receiver.MARKER).exists()
    assert not list(assembled.rglob('receiver-evidence'))
    assert len(list(root.rglob('receiver-evidence'))) == 6
