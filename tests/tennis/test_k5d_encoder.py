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
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'archive_sha256': 'archive',
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
