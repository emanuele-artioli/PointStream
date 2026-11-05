import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.tennis import w7_byteproof as proof
from experiments.tennis import w7_native_yuv_pilot as pilot
from test_w7_byteproof import fake_ivf


def test_native_yuv_policy_accepts_header_difference_only_after_full_yuv_hash(tmp_path, monkeypatch):
    ffmpeg, ffprobe, dav1d = (tmp_path / n for n in ('ffmpeg', 'ffprobe', 'dav1d'))
    for p in (ffmpeg, ffprobe, dav1d):
        p.write_bytes(p.name.encode())
    monkeypatch.setattr(proof.old, 'FFMPEG_SHA', proof.digest(ffmpeg))
    monkeypatch.setattr(proof.old, 'FFPROBE_SHA', proof.digest(ffprobe))
    monkeypatch.setattr(proof.profile, 'DAV1D_SHA', proof.digest(dav1d))
    original = tmp_path / 'original.ivf'
    fake_ivf(original)
    yuv = b'\x11' * (2 * 2 * 3 // 2 * 240)
    row = {'original_ivf_sha256': proof.digest(original), 'width': 2, 'height': 2,
           'historical_native_yuv_sha256': hashlib.sha256(yuv).hexdigest()}
    probe = json.dumps({'streams': [{'codec_name': 'av1', 'width': 2, 'height': 2,
        'pix_fmt': 'yuv420p', 'r_frame_rate': '50/1', 'time_base': '1/50', 'color_range': 'tv'}],
        'frames': [{'best_effort_timestamp': i} for i in range(240)]}).encode()
    def fake_run(cmd, **kwargs):
        from pathlib import Path
        if cmd[0] == str(ffprobe):
            return SimpleNamespace(stdout=probe)
        if cmd[0] == str(dav1d):
            Path(cmd[cmd.index('-o') + 1]).write_bytes(yuv)
        elif cmd[-1].endswith('delivered.mkv'):
            Path(cmd[-1]).write_bytes(b'whole mkv')
        else:
            data = bytearray(original.read_bytes())
            data[28] ^= 1
            Path(cmd[-1]).write_bytes(data)
        return SimpleNamespace(stdout=b'')
    monkeypatch.setattr(proof.subprocess, 'run', fake_run)
    work = tmp_path / 'publish' / 'one'
    result = proof.prove(row, original, work, ffmpeg, ffprobe, dav1d=dav1d, native_yuv=True)
    assert not result['exact_ivf_bytes'] and result['ordered_payloads_equal']
    assert result['native_yuv_proven'] and result['native_yuv_sha256'] == row['historical_native_yuv_sha256']
    assert proof.validate_proof(row, original, work, native_yuv=True)['native_yuv_proven']
    assert not (tmp_path / 'one-native.yuv').exists()
    bad = {**row, 'historical_native_yuv_sha256': '0' * 64}
    with pytest.raises(ValueError, match='native YUV differs'):
        proof.prove(bad, original, tmp_path / 'publish' / 'two', ffmpeg, ffprobe, dav1d=dav1d, native_yuv=True)
    def bad_payload(cmd, **kwargs):
        result = fake_run(cmd, **kwargs)
        if cmd[-1].endswith('decoder-scratch.ivf'):
            from pathlib import Path
            path = Path(cmd[-1])
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
        return result
    monkeypatch.setattr(proof.subprocess, 'run', bad_payload)
    with pytest.raises(ValueError, match='ordered AV1 payloads'):
        proof.prove(row, original, tmp_path / 'publish' / 'three', ffmpeg, ffprobe, dav1d=dav1d, native_yuv=True)


def test_native_pilot_spec_retains_phase_cap_and_pinned_assets():
    from pathlib import Path
    from experiments.jobs.inbox import validate_spec
    spec = json.loads(Path('docs/experiments/w7-inputs/mkv-native-yuv-pilot-spec-20261105.json').read_text())
    validate_spec(spec)
    assert spec['device'] == 'cpu' and spec['budget_seconds'] == 120
    assert spec['validator_seconds'] == 50 and spec['entrypoint'][-1] == 'experiments.tennis.w7_native_yuv_pilot'
    assert [x['name'] for x in spec['staged_inputs']] == list(pilot.NAMES)


def test_validator_independently_reextracts_and_redecodes(tmp_path, monkeypatch):
    import subprocess
    import tarfile
    from pathlib import Path

    stage = tmp_path / 'smoke'
    stage.mkdir()
    rows, inputs = [], {}
    yuv = b'\x23' * (2 * 2 * 3 // 2 * 240)
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'w') as tar:
        for name in pilot.NAMES:
            original = tmp_path / (name + '.ivf')
            fake_ivf(original)
            inputs[name] = original
            row = {'window': name, 'width': 2, 'height': 2, 'original_ivf_sha256': proof.digest(original),
                   'historical_native_yuv_sha256': hashlib.sha256(yuv).hexdigest()}
            rows.append(row)
            work = tmp_path / (name + '-work')
            work.mkdir()
            (work / 'decoder-scratch.ivf').write_bytes(original.read_bytes())
            (work / 'delivered.mkv').write_bytes(name.encode())
            colors = {'color_range': 'tv', 'color_space': None, 'color_transfer': None, 'color_primaries': None}
            record = {**row, **proof.compare_ivf(original, work / 'decoder-scratch.ivf'),
                'profile': proof.profile.PROFILE, 'delivered_mkv_sha256': proof.digest(work / 'delivered.mkv'),
                'delivered_mkv_bytes': len(name), 'native_yuv_proven': True,
                'native_yuv_sha256': row['historical_native_yuv_sha256'], 'native_decoder_sha256': proof.profile.DAV1D_SHA,
                'original_color_metadata': colors, 'delivered_color_metadata': colors}
            (work / 'proof.json').write_text(json.dumps(record))
            tar.add(work, arcname='publish/' + name)
        timing = tmp_path / 'point-timings.json'
        timing.write_text(json.dumps([{'name': name, 'seconds': .2} for name in pilot.NAMES]))
        tar.add(timing, arcname='publish/point-timings.json')
    inputs.update(ffmpeg=tmp_path / 'ffmpeg', ffprobe=tmp_path / 'ffprobe', dav1d=tmp_path / 'dav1d')
    (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': proof.digest(archive)}}))
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 1, 'timing_contaminated': False}))
    import time
    (stage.parent / 'state.json').write_text(json.dumps({'created': time.time() - 5}))
    dispatch = {'campaign_clock': {'started': 100, 'phase_started': 110, 'budget_end': 220}}
    monkeypatch.setattr(pilot, 'context', lambda stage: (rows, inputs, {}, dispatch))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    now = iter([111, 112, 113, 114, 115, 116, 117])
    monkeypatch.setattr(pilot.time, 'monotonic', lambda: next(now))
    probe = json.dumps({'streams': [{'codec_name': 'av1', 'width': 2, 'height': 2,
        'pix_fmt': 'yuv420p', 'r_frame_rate': '50/1', 'time_base': '1/50', 'color_range': 'tv'}],
        'frames': [{'best_effort_timestamp': i} for i in range(240)]}).encode()
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == str(inputs['ffmpeg']):
            Path(cmd[-1]).write_bytes(inputs[pilot.NAMES[0]].read_bytes())
        elif cmd[0] == str(inputs['dav1d']):
            Path(cmd[cmd.index('-o') + 1]).write_bytes(yuv)
        return SimpleNamespace(stdout=probe)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    target = tmp_path / 'validation.json'
    pilot.validate(stage, target)
    assert json.loads(target.read_text())['passed']
    assert sum(cmd[0] == str(inputs['ffmpeg']) for cmd in calls) == 3
    assert sum(cmd[0] == str(inputs['dav1d']) for cmd in calls) == 3
    now = iter([111, 112, 113, 114, 115, 116, 117])
    monkeypatch.setattr(pilot.time, 'monotonic', lambda: next(now))
    def bad_run(cmd, **kwargs):
        result = fake_run(cmd, **kwargs)
        if cmd[0] == str(inputs['dav1d']):
            Path(cmd[cmd.index('-o') + 1]).write_bytes(b'changed')
        return result
    monkeypatch.setattr(subprocess, 'run', bad_run)
    with pytest.raises(ValueError, match='independent native YUV'):
        pilot.validate(stage, target)
    assert not json.loads(target.read_text())['passed']
