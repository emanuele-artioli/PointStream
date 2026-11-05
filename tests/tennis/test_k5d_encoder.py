from fractions import Fraction
import json

import numpy as np
import pytest

from experiments.tennis import k5d_encoder as enc
from src.pointstream import ball, ball_stream


def test_crop_is_fixed_and_shifted_at_edges():
    image = np.arange(100 * 100).reshape(100, 100)
    assert np.array_equal(enc.crop64(image, 0, 0), image[:64, :64])
    assert np.array_equal(enc.crop64(image, 99, 99), image[36:, 36:])
    with pytest.raises(ValueError):
        enc.crop64(image, 100, 0)


def fixture_output(tmp_path, monkeypatch):
    monkeypatch.setattr(enc.oracle, "validate_report", lambda _: None)
    qualification = {"qualified": True, "frames": 2, "clip": "test", "rows": [
        {"label_kind": 1, "label_x": 1.0, "label_y": 1.0},
        {"label_kind": 0, "label_x": 0.0, "label_y": 0.0}]}
    (tmp_path / "source-qualification.json").write_text(json.dumps(qualification))
    labels = [(1, 1.0, 1.0), (0, 0.0, 0.0)]
    judges = {name: enc.prediction_rows(labels, [(False, 0., 0.), (False, 0., 0.)]) for name in ("wasb", "balltrack")}
    rows = [{"frame": i, "frame_psnr": 30., "full_lpips": .1, "full_dists": .1,
             "crop_lpips": .1 if i == 0 else None, "crop_dists": .1 if i == 0 else None} for i in range(2)]
    (tmp_path / "resources.json").write_text("{}")
    map_hash = enc.oracle.sha(tmp_path / "resources.json")
    points = []
    for kind, point in enc.POINTS:
        directory = tmp_path / f"{kind}-{point}"
        directory.mkdir()
        native = directory / "native.bin"
        native.write_bytes(b"native")
        stream = ball_stream.Stream(64, 64, Fraction(30), 2, "dcvc-uf" if kind == "dcvc" else "svt-av1-ivf", kind,
                                    b"native", ball.encode_frames([None, None], [None, None], 64, 64))
        wire = directory / "stream.psbl"
        wire.write_bytes(ball_stream.pack(stream))
        zero = directory / "baseline.psbl"
        zero.write_bytes(ball_stream.pack(stream))
        native_sha, wire_sha = enc.oracle.sha(native), enc.oracle.sha(wire)
        points.append({"codec": kind, "point": point, "native_sha256": native_sha, "wire_sha256": wire_sha,
                       "native_bytes": native.stat().st_size, "wire_bytes": wire.stat().st_size,
                       "native_record": {"stream_sha256": native_sha},
                       "receiver": {"wire_sha256": wire_sha, "native_sha256": native_sha, "frames": 2, "resource_map_sha256": map_hash,
                                    "decoded_yuv_sha256": "decode", "rendered_rgb_sha256": "render"},
                       "decode_parity": True, "render_parity": True, "encoder_decoded_sha256": "decode",
                       "encoder_rendered_sha256": "render", "wire_kbps": wire.stat().st_size * 8 / (2 / 30) / 1000,
                       "baseline": {"native_bytes": native.stat().st_size, "native_sha256": native_sha,
                                    "native_kbps": native.stat().st_size * 8 / (2 / 30) / 1000,
                                    "zero_ball_wire_bytes": zero.stat().st_size, "zero_ball_wire_sha256": enc.oracle.sha(zero),
                                    "zero_ball_wire_kbps": zero.stat().st_size * 8 / (2 / 30) / 1000,
                                    "metrics": rows, "judges": judges},
                       "streak": {"guard_off": True, "metrics": rows, "judges": judges}})
    report = {"frames": 2, "clip": "test", "size": [64, 64], "fps": "30", "duration_seconds": 2 / 30, "full_metrics": True, "points": points, "resource_map_sha256": map_hash}
    (tmp_path / "pilot.json").write_text(json.dumps(report))
    return report


def test_validator_rejects_incomplete_points_metric_and_wire(tmp_path, monkeypatch):
    report = fixture_output(tmp_path, monkeypatch)
    assert enc.validate(tmp_path)["points"] == 6
    report["points"].pop()
    (tmp_path / "pilot.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="six point"):
        enc.validate(tmp_path)


def test_validator_checks_frame_metrics_and_parity(tmp_path, monkeypatch):
    report = fixture_output(tmp_path, monkeypatch)
    report["points"][0]["baseline"]["metrics"][0]["crop_lpips"] = None
    (tmp_path / "pilot.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metric"):
        enc.validate(tmp_path)
    report["points"][0]["baseline"]["metrics"][0]["crop_lpips"] = .1
    report["points"][0]["render_parity"] = False
    (tmp_path / "pilot.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="parity"):
        enc.validate(tmp_path)


def test_baseline_wire_and_physical_rate_are_checked(tmp_path, monkeypatch):
    report = fixture_output(tmp_path, monkeypatch)
    report['points'][0]['baseline']['native_kbps'] += 1
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='accounting'):
        enc.validate(tmp_path)
    report['points'][0]['baseline']['native_kbps'] -= 1
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    (tmp_path / 'dcvc-0' / 'baseline.psbl').write_bytes(b'bad')
    with pytest.raises(ValueError):
        enc.validate(tmp_path)


def test_failed_source_is_fidelity_only_with_bound_gate(tmp_path, monkeypatch):
    report = fixture_output(tmp_path, monkeypatch)
    source_path = tmp_path / 'source-qualification.json'
    source = json.loads(source_path.read_text())
    source['qualified'] = False
    source['identities'] = {'archive': {'sha256': 'archive'}}
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='source qualification failed'):
        enc.validate(tmp_path)
    gate_path = tmp_path / 'fidelity-gate.json'
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': 'full-retrospective-benchmark', 'archive_sha256': 'archive',
            'calibration_job_id': 'job', 'evidence_sha256': 'a' * 64,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'full_cost']}
    gate_path.write_text(json.dumps(gate))
    report['fidelity_gate'] = {'sha256': enc.oracle.sha(gate_path)}
    report['source_task_qualified'] = False
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    assert enc.validate(tmp_path)['frames'] == 2
    report['source_task_qualified'] = True
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='relabeled'):
        enc.validate(tmp_path)


def test_integer_sse_psnr_matches_original_float64_on_uint8_frames():
    rng = np.random.default_rng(702)
    inputs = [rng.integers(0, 256, size=(72, 128, 3), dtype=np.uint8) for _ in range(4)]
    inputs += [np.zeros((72, 128, 3), dtype=np.uint8)]
    controls = [rng.integers(0, 256, size=x.shape, dtype=np.uint8) for x in inputs[:4]]
    identical = inputs[-1].copy()
    controls.append(identical)
    one_pixel = identical.copy()
    one_pixel[0, 0, 0] = 1
    inputs.append(identical)
    controls.append(one_pixel)
    labels = [(-1, 0., 0.)] * len(inputs)
    rows = enc.metrics(np.stack(inputs), np.stack(controls), labels, None)
    for row, a, b in zip(rows, inputs, controls):
        assert row['frame_psnr'] == enc.old.psnr(a, b)


def test_guard_reuse_requires_equal_pixels_and_runs_each_judge_once(monkeypatch):
    image = np.zeros((1, 3, 3, 3), dtype=np.uint8)
    calls = {'metrics': 0, 'judges': 0}

    def fake_metrics(*_):
        calls['metrics'] += 1
        return [{'record': 1}]

    def fake_judges(_):
        calls['judges'] += 1
        return {'record': 1}

    monkeypatch.setattr(enc, 'metrics', fake_metrics)
    a, b, c, d, provenance = enc.score_arms(image, image, image.copy(), [(-1, 0., 0.)], None, fake_judges, True)
    assert a is c and b is d and provenance and calls == {'metrics': 1, 'judges': 1}
    changed = image.copy()
    changed[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='differs'):
        enc.score_arms(image, image, changed, [(-1, 0., 0.)], None, fake_judges, True)


def test_generic_decoder_resource_bytes_are_separate_from_wire(tmp_path):
    image, video, decoder = (tmp_path / n for n in ('i', 'v', 'd'))
    image.write_bytes(b'123')
    video.write_bytes(b'12345')
    decoder.write_bytes(b'1')
    ledger = enc.installed_decoder_resources(image, video, decoder)
    assert ledger['dcvc']['checkpoint_total_bytes'] == 8
    assert ledger['dcvc']['image_checkpoint']['sha256'] == enc.oracle.sha(image)
    assert ledger['svt']['decoder']['bytes'] == 1
    assert 'excluded from per-clip wire' in ledger['scope']


def test_confirmation_fidelity_gate_is_narrow_and_separate(tmp_path):
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only',
            'purpose': '188-frame-confirmation', 'archive_sha256': 'archive',
            'calibration_job_id': 'job', 'evidence_sha256': 'a' * 64,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'full_cost'],
            'clip': 'tracknet/game10/Clip3', 'max_frames': 188}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    digest = enc.oracle.sha(path)
    for frames in (16, 188):
        assert enc.fidelity_gate(path, digest, 'archive', purpose='188-frame-confirmation',
                                 clip=gate['clip'], frames=frames) == gate
    for frames in (48, 189):
        with pytest.raises(ValueError, match='confirmation'):
            enc.fidelity_gate(path, digest, 'archive', purpose='188-frame-confirmation',
                              clip=gate['clip'], frames=frames)
    with pytest.raises(ValueError, match='confirmation'):
        enc.fidelity_gate(path, digest, 'archive', purpose='188-frame-confirmation',
                          clip='tracknet/game9/Clip5', frames=188)
    with pytest.raises(ValueError, match='purpose'):
        enc.fidelity_gate(path, digest, 'archive', purpose='full-retrospective-benchmark')
    gate['checks'].remove('full_cost')
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='calibration'):
        enc.fidelity_gate(path, enc.oracle.sha(path), 'archive', purpose='188-frame-confirmation',
                          clip=gate['clip'], frames=188)


def test_checkpoint_assembly_passes_real_report_validator(tmp_path, monkeypatch):
    from experiments.tennis import k5d_checkpoint as cp

    root = tmp_path / 'prepared'
    root.mkdir()
    report = fixture_output(root, monkeypatch)
    scored_points = json.loads(json.dumps(report['points']))
    report['checkpoint_prepare'] = True
    for point in report['points']:
        point['baseline'].update(metrics=[], judges={})
        point['streak'].update(metrics=[], judges={}, reuse=None)
    (root / 'pilot.json').write_text(json.dumps(report))
    identity = {'source_sha256': 's', 'snapshot_sha256': 'p', 'environment_sha256': 'e'}
    cp.seal(root, identity)
    sha = enc.oracle.sha(root / 'checkpoint.json')
    receipts = []
    for i, point in enumerate(scored_points):
        point['streak']['reuse'] = 'disabled'
        path = tmp_path / f'point-{i}.json'
        path.write_text(json.dumps({'checkpoint_sha256': sha, 'execution_identity': identity, 'point': point,
                                    'receiver_replay': {'decoded_yuv_sha256': 'decode', 'rendered_rgb_sha256': 'render'}}))
        receipts.append({'path': str(path), 'sha256': enc.oracle.sha(path)})
    result = cp.assemble(root, sha, receipts, tmp_path / 'assembled', identity)
    assert result['frames'] == 2 and result['points'] == 6 and result['full_metrics']


def test_checkpoint_parity_compares_full_scores_and_ignores_relocated_command(tmp_path, monkeypatch):
    import shutil
    from experiments.tennis import k5d_checkpoint as cp

    prior = tmp_path / 'prior'
    prior.mkdir()
    report = fixture_output(prior, monkeypatch)
    report['conversion'] = {'yuv_sha256': 'same', 'floor_psnr': 30., 'command': ['/old/output']}
    (prior / 'pilot.json').write_text(json.dumps(report))
    source_path = prior / 'source-qualification.json'
    source = json.loads(source_path.read_text())
    source['scores'] = {'wasb': {'hits': 1}}
    source_path.write_text(json.dumps(source))
    current = tmp_path / 'current'
    shutil.copytree(prior, current)
    report['conversion']['command'] = ['/new/output']
    (current / 'pilot.json').write_text(json.dumps(report))
    assert cp.compare_equivalence(prior, current, enc.oracle.sha(prior / 'pilot.json'))['passed']
    report['points'][0]['streak']['metrics'][0]['full_lpips'] = 0.2
    (current / 'pilot.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        cp.compare_equivalence(prior, current, enc.oracle.sha(prior / 'pilot.json'))
