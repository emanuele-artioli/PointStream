import struct

import pytest

from experiments.tennis import w7_byteproof as proof


def fake_ivf(path, *, timestamp_delta=0, frame=100):
    header = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 960, 544, 50, 1, 240, 0)
    body = b''.join(struct.pack('<IQ', 1, i + (timestamp_delta if i == frame else 0)) + bytes([i % 256]) for i in range(240))
    path.write_bytes(header + body)


def test_exact_ivf_proof_rejects_any_header_or_payload_byte_change(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    fake_ivf(a)
    b.write_bytes(a.read_bytes())
    assert proof.compare_ivf(a, b)['exact_ivf_bytes']
    raw = bytearray(b.read_bytes())
    raw[-1] ^= 1
    b.write_bytes(raw)
    got = proof.compare_ivf(a, b)
    assert not got['exact_ivf_bytes'] and not got['ordered_payloads_equal']
    fake_ivf(b, timestamp_delta=1)
    with pytest.raises(ValueError, match='timestamps'):
        proof.compare_ivf(a, b)


def test_ivf_parser_rejects_short_and_extra_streams(tmp_path):
    p = tmp_path / 'input.ivf'
    fake_ivf(p)
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated'):
        proof.ivf(p)
    fake_ivf(p)
    p.write_bytes(p.read_bytes() + b'x')
    with pytest.raises(ValueError, match='extra'):
        proof.ivf(p)


def test_inventory_is_complete_and_retains_unverified_scorer_identity():
    import json
    from pathlib import Path
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    assert len(inventory['points']) == 53
    assert len(inventory['controls']) == 18
    assert len(inventory['new_points']) == 4
    for row in inventory['points'] + inventory['controls'] + inventory['new_points']:
        assert row['frames'] == 240
        assert row['source_and_scorer']['ticks'] == list(range(0, 240, 3))
        assert row['source_and_scorer']['score_source_identity'] is None
        assert len(row['original_ivf_sha256']) == len(row['encode_archive_sha256']) == 64


def test_smoke_cost_requires_complete_fit_and_positive_times():
    assert proof.smoke_forecast(10, 400, 400, 20) == 395
    with pytest.raises(ValueError, match='exceeds'):
        proof.smoke_forecast(30, 400, 400, 20)
    with pytest.raises(ValueError, match='invalid'):
        proof.smoke_forecast(float('nan'), 400, 400, 20)


def test_saved_proof_requires_exact_ivf_and_real_billed_mkv(tmp_path):
    import json
    original = tmp_path / 'original.ivf'
    fake_ivf(original)
    work = tmp_path / 'proof'
    work.mkdir()
    (work / 'decoder-scratch.ivf').write_bytes(original.read_bytes())
    mkv = work / 'delivered.mkv'
    mkv.write_bytes(b'entire container')
    row = {'original_ivf_sha256': proof.digest(original), 'window': 'test'}
    record = {**row, **proof.compare_ivf(original, work / 'decoder-scratch.ivf'),
              'profile': proof.profile.PROFILE, 'delivered_mkv_sha256': proof.digest(mkv),
              'delivered_mkv_bytes': mkv.stat().st_size}
    (work / 'proof.json').write_text(json.dumps(record))
    assert proof.validate_proof(row, original, work)['window'] == 'test'
    mkv.write_bytes(b'changed')
    with pytest.raises(ValueError, match='billed MKV'):
        proof.validate_proof(row, original, work)


def test_representative_three_are_predeclared_and_coverage_is_required():
    import json
    from pathlib import Path
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    rows = proof.representative_three(inventory)
    assert [(r['window'], r['name']) for r in rows] == [
        ('med000', 'control_s8_crf63'), ('med019', 's2_o0_crf68'), ('med000', 's2_o20_crf70')]
    inventory['points'].pop()
    with pytest.raises(ValueError, match='inventory'):
        proof.representative_three(inventory)


def test_smoke_validator_reextracts_real_mkv_and_rejects_mismatch(tmp_path, monkeypatch):
    import json
    import subprocess
    import tarfile
    from types import SimpleNamespace
    from experiments.tennis import w7_byteproof_fleet as fleet

    stage = tmp_path / 'smoke'
    stage.mkdir()
    inputs = {}
    rows = []
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'w') as tar:
        for name in fleet.NAMES:
            source = tmp_path / (name + '.ivf')
            fake_ivf(source)
            inputs[name] = source
            row = {'window': name, 'width': 960, 'height': 544, 'original_ivf_sha256': proof.digest(source)}
            rows.append(row)
            work = tmp_path / (name + '-work')
            work.mkdir()
            (work / 'decoder-scratch.ivf').write_bytes(source.read_bytes())
            (work / 'delivered.mkv').write_bytes(name.encode())
            record = {**row, **proof.compare_ivf(source, work / 'decoder-scratch.ivf'),
                      'profile': proof.profile.PROFILE, 'delivered_mkv_sha256': proof.digest(work / 'delivered.mkv'),
                      'delivered_mkv_bytes': len(name),
                      'original_color_metadata': {'color_range': 'tv', 'color_space': None, 'color_transfer': None, 'color_primaries': None},
                      'delivered_color_metadata': {'color_range': 'tv', 'color_space': None, 'color_transfer': None, 'color_primaries': None}}
            (work / 'proof.json').write_text(json.dumps(record))
            tar.add(work, arcname='publish/' + name)
    inputs.update(ffmpeg=tmp_path / 'ffmpeg', ffprobe=tmp_path / 'ffprobe')
    (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': proof.digest(archive)}}))
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 1, 'timing_contaminated': False}))
    dispatch = {'campaign_clock': {'started': 100, 'phase_started': 110, 'budget_end': 650}}
    monkeypatch.setattr(fleet, 'context', lambda stage: (rows, inputs, {}, dispatch))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(fleet.time, 'monotonic', lambda: 111)
    probe_json = json.dumps({'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544,
        'pix_fmt': 'yuv420p', 'r_frame_rate': '50/1', 'time_base': '1/50', 'color_range': 'tv'}],
        'frames': [{'best_effort_timestamp': i} for i in range(240)]}).encode()
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == str(inputs['ffmpeg']):
            Path = __import__('pathlib').Path
            Path(cmd[-1]).write_bytes(inputs[fleet.NAMES[0]].read_bytes())
            return SimpleNamespace(stdout=b'')
        return SimpleNamespace(stdout=probe_json)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    target = tmp_path / 'validation.json'
    fleet.validate(stage, target)
    assert json.loads(target.read_text())['passed']
    assert sum(cmd[0] == str(inputs['ffmpeg']) for cmd in calls) == 3
    def bad_run(cmd, **kwargs):
        value = fake_run(cmd, **kwargs)
        if cmd[0] == str(inputs['ffmpeg']):
            from pathlib import Path
            p = Path(cmd[-1])
            raw = bytearray(p.read_bytes())
            raw[-1] ^= 1
            p.write_bytes(raw)
        return value
    monkeypatch.setattr(subprocess, 'run', bad_run)
    with pytest.raises(ValueError, match='extraction differs'):
        fleet.validate(stage, target)
    assert not json.loads(target.read_text())['passed']


def test_pilot_context_binds_real_dispatch_staging_and_source_hashes(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from experiments.tennis import w7_byteproof_fleet as fleet

    root = tmp_path / 'local'
    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    root.mkdir()
    rows, staged, declared = [], [], []
    for name in fleet.NAMES:
        cache = root / name
        tree = cache / 'tree'
        source = tree / 'publish' / 'streams' / 'native.ivf'
        source.parent.mkdir(parents=True)
        fake_ivf(source)
        archive = cache / 'archive.tar'
        archive.write_bytes(name.encode())
        sha = proof.digest(archive)
        rows.append({'encode_archive_sha256': sha, 'ivf_member': 'publish/streams/native.ivf',
                     'original_ivf_sha256': proof.digest(source)})
        staged.append({'name': name, 'path': str(tree), 'source': '/remote/archive.tar', 'sha256': sha, 'extract': True})
        declared.append({'name': name, 'sha256': sha, 'extract': True})
    bins = root / 'env' / 'bin'
    bins.mkdir(parents=True)
    for n in ('ffmpeg', 'ffprobe'):
        (bins / n).write_bytes(n.encode())
    monkeypatch.setattr(fleet.old, 'FFMPEG_SHA', proof.digest(bins / 'ffmpeg'))
    monkeypatch.setattr(fleet.old, 'FFPROBE_SHA', proof.digest(bins / 'ffprobe'))
    inventory = tmp_path / 'inventory.json'
    inventory.write_text('{}')
    monkeypatch.setattr(fleet, 'INVENTORY', inventory)
    monkeypatch.setattr(fleet, 'INVENTORY_SHA', proof.digest(inventory))
    monkeypatch.setattr(proof, 'representative_three', lambda _: rows)
    spec = {'device': 'cpu', 'cpu_threads': 1, 'single_cpu_campaign': True, 'budget_seconds': 540,
            'smoke': {'seconds': 150}, 'full': {'seconds': 30}, 'validator_seconds': 60,
            'environment': {'sha256': fleet.old.ENV_SHA}, 'staged_inputs': declared}
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'local_root': str(root), 'inputs': staged,
        'environment': {'bin': str(bins), 'sha256': fleet.old.ENV_SHA}}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [3]}))
    _, paths, _, _ = fleet.context(stage)
    assert all(Path(paths[name]).exists() for name in fleet.NAMES)
    paths[fleet.NAMES[0]].write_bytes(b'changed')
    with pytest.raises(ValueError, match='original IVF'):
        fleet.context(stage)


def test_mismatched_reconstruction_is_preserved_and_requires_decode_gate(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    ffmpeg, ffprobe = tmp_path / 'ffmpeg', tmp_path / 'ffprobe'
    ffmpeg.write_bytes(b'ffmpeg')
    ffprobe.write_bytes(b'ffprobe')
    monkeypatch.setattr(proof.old, 'FFMPEG_SHA', proof.digest(ffmpeg))
    monkeypatch.setattr(proof.old, 'FFPROBE_SHA', proof.digest(ffprobe))
    original = tmp_path / 'original.ivf'
    fake_ivf(original)
    row = {'original_ivf_sha256': proof.digest(original), 'width': 960, 'height': 544}
    probe_json = json.dumps({'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544,
        'pix_fmt': 'yuv420p', 'r_frame_rate': '50/1', 'time_base': '1/50', 'color_range': 'tv'}],
        'frames': [{'best_effort_timestamp': i} for i in range(240)]}).encode()
    def fake_run(cmd, **kwargs):
        from pathlib import Path
        if cmd[0] == str(ffprobe):
            return SimpleNamespace(stdout=probe_json)
        path = Path(cmd[-1])
        if path.name == 'delivered.mkv':
            path.write_bytes(b'mkv')
        else:
            data = bytearray(original.read_bytes())
            data[28] ^= 1
            path.write_bytes(data)
        return SimpleNamespace(stdout=b'')
    monkeypatch.setattr(proof.subprocess, 'run', fake_run)
    work = tmp_path / 'proof'
    with pytest.raises(ValueError, match='budgeted native-YUV'):
        proof.prove(row, original, work, ffmpeg, ffprobe)
    assert not json.loads((work / 'proof.json').read_text())['exact_ivf_bytes']
    assert (work / 'delivered.mkv').exists()
