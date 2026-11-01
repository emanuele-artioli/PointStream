from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from experiments.tennis import k5c_independent_ball as pilot
from src.pointstream import ball


def test_one_cpu_is_required_for_all_six_sequential_points() -> None:
    assert len(pilot.POINTS) == 6
    assert len(set(pilot.POINTS)) == 6
    assert [pilot.one_cpu({7}, "1") for _ in pilot.POINTS] == [[7]] * 6
    for affinity, allowance in [({7, 8}, "1"), ({7}, "2"), (set(), "1")]:
        with pytest.raises(ValueError):
            pilot.one_cpu(affinity, allowance)


def test_packet_roundtrip_and_ledger_count_all_physical_headers(tmp_path) -> None:
    native = tmp_path / "stream.ivf"
    native.write_bytes(b"native-physical-header-and-payload")
    packet = tmp_path / "ball.psb"
    track: ball.Track = [(10, 11), None]
    apps = [ball.Appearance((123, 120, 110), 0, 0, 0), None]
    pilot.packet_write(packet, track, apps, 64, 64, Fraction(30))
    assert pilot.packet_read(packet, 64, 64, Fraction(30), 2) == (track, apps)
    assert packet.stat().st_size > len(ball.encode_frames(track, apps, 64, 64))
    record = pilot.ledger(native, packet, 2 / 30)
    assert record["total_bytes"] == len(native.read_bytes()) + len(packet.read_bytes())
    assert record["kbps"] == pytest.approx(record["total_bytes"] * 8 / (2 / 30) / 1000)
    with pytest.raises(ValueError):
        pilot.packet_read(packet, 64, 64, Fraction(30), 3)


def test_denominators_missing_detections_and_false_positives() -> None:
    labels = [(1, 10.0, 10.0), (1, 20.0, 20.0), (0, 0.0, 0.0), (-1, 0.0, 0.0)]
    pred = [(True, 10.0, 10.0), (False, 0.0, 0.0), (True, 8.0, 8.0), (True, 0.0, 0.0)]
    result = pilot.frame_results(pred, labels)
    assert [x["frame"] for x in result["frames"]] == [0, 1, 2, 3]
    assert result["counts"]["positives"] == 2
    assert result["counts"]["negatives"] == 1
    assert result["counts"]["false_on_negative"] == 1
    assert result["rates"]["detection_rate"] == 0.5
    assert result["frames"][1]["detected"] is False
    with pytest.raises(ValueError):
        pilot.frame_results(pred[:-1], labels)


def test_judges_receive_images_not_ball_coordinates() -> None:
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    seen = []

    def judge(pixels):
        seen.append(pixels)
        return [(False, 0.0, 0.0)] * len(pixels)

    result = pilot.judge_images(images, [(1, 1.0, 1.0), (0, 0.0, 0.0)], judge, judge)
    assert len(seen) == 2 and all(x is images for x in seen)
    assert result["wasb"]["counts"]["positives"] == 1


def test_realistic_window_schema_rejects_unknown_and_missing_labels(tmp_path) -> None:
    import json

    path = tmp_path / "windows.json"
    row = {"id": pilot.WINDOW, "dataset": "tracknet", "frames": 2,
           "labels": [[1, 3, 4], [0, 0, 0]], "size": [64, 64], "fps": 30}
    path.write_text(json.dumps({"windows": [row]}))
    assert pilot.load_window(path, 2)["id"] == pilot.WINDOW
    row["labels"] = [[4, 3, 4]]
    path.write_text(json.dumps({"windows": [row]}))
    with pytest.raises(ValueError):
        pilot.load_window(path, 2)
    row["labels"] = [[4, 3, 4], [0, 0, 0]]
    path.write_text(json.dumps({"windows": [row]}))
    with pytest.raises(ValueError):
        pilot.load_window(path, 2)


def test_all_missing_ball_fits_all_points() -> None:
    source = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    fits = pilot.fit_streak([None, None], source, [source] * 6, (16, 16))
    assert len(fits) == 6
    assert all(f["track"] == [None, None] and f["appearances"] == [None, None] for f in fits)


def test_real_streak_fit_packet_and_render_parity(tmp_path) -> None:
    blank = np.full((2, 32, 32, 3), 25, dtype=np.uint8)
    positions: ball.Track = [(14, 15), (16, 15)]
    apps: list[ball.Appearance | None] = [ball.Appearance((220, 180, 130), 1, 3, 3)] * 2
    source = pilot.render_frames(blank, positions, apps)
    fits = pilot.fit_streak(positions, source, [blank], (32, 32))
    assert len(fits) == 1
    fitted = fits[0]
    assert any(p is not None for p in fitted["track"])
    packet = tmp_path / "ball.psb"
    pilot.packet_write(packet, fitted["track"], fitted["appearances"], 32, 32, Fraction(30))
    track, appearances = pilot.packet_read(packet, 32, 32, Fraction(30), 2)
    assert np.array_equal(pilot.render_frames(blank, track, appearances),
                          pilot.render_frames(blank, fitted["track"], fitted["appearances"]))
    assert pilot.psnr(source, pilot.render_frames(blank, track, appearances)) > pilot.psnr(source, blank)


def test_receipt_must_bind_all_evidence(tmp_path, monkeypatch) -> None:
    import json

    config = tmp_path / "config"
    config.write_bytes(b"author-reported config")
    weight = tmp_path / "weights"
    weight.write_bytes(b"checkpoint")
    manifest = tmp_path / "windows"
    manifest.write_bytes(b"selected manifest")
    monkeypatch.setattr(pilot, "CONFIG_SHA", pilot.sha(config))
    monkeypatch.setattr(pilot, "WASB_SHA", pilot.sha(weight))
    receipt = tmp_path / "receipt.json"
    row = {"scope": "author-reported-split-not-training-log", "window": pilot.WINDOW,
           "wasb_sha256": pilot.sha(weight), "config_sha256": pilot.sha(config),
           "window_manifest_sha256": pilot.sha(manifest), "train_games": list(range(1, 8)),
           "test_games": [8, 9, 10]}
    receipt.write_text(json.dumps(row))
    assert pilot.admit(receipt, config, weight, manifest)["window"] == pilot.WINDOW
    row["window_manifest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        pilot.admit(receipt, config, weight, manifest)


def test_validator_checks_saved_bytes_and_judge_denominators(tmp_path) -> None:
    import json

    frames = 2
    predictions = [(True, 3.0, 4.0), (False, 0.0, 0.0)]
    labels = [(1, 3.0, 4.0), (0, 0.0, 0.0)]
    judge = pilot.frame_results(predictions, labels)
    judges = {"wasb": judge, "balltrack": judge}
    points = []
    for kind, number in pilot.POINTS:
        directory = tmp_path / f"{kind}-{number}"
        directory.mkdir()
        native = directory / ("stream.psdc" if kind == "dcvc" else "stream.ivf")
        native.write_bytes(b"native bytes")
        original = directory / "decoded.yuv"
        original.write_bytes(b"pixels")
        (directory / "receiver.yuv").write_bytes(b"pixels")
        packet = directory / "ball.psb"
        pilot.packet_write(packet, [None] * frames, [None] * frames, 32, 32, Fraction(30))
        points.append({"codec": kind, "point": number, "native_record": {"encoder_decoded_sha256": pilot.sha(original), "receiver_decoded_sha256": pilot.sha(original)},
                       "baseline": {"sha256": pilot.sha(native), "bytes": native.stat().st_size, "frame_psnr": 30.0, "judges": judges},
                       "streak": {"ledger": pilot.ledger(native, packet, frames / 30), "ball_on": False,
                                  "roundtrip": True, "guard_off": False, "frame_psnr": 30.0, "judges": judges}})
    manifest = {"windows": [{"id": pilot.WINDOW, "dataset": "tracknet", "frames": frames,
                            "labels": [[1, 3, 4], [0, 0, 0]], "size": [32, 32], "fps": 30}]}
    (tmp_path / "windows.json").write_text(json.dumps(manifest))
    record: dict[str, Any] = {"reference_labels": labels, "identities": {"window_manifest_sha256": pilot.sha(tmp_path / "windows.json")},
              "window": pilot.WINDOW, "frames": frames, "fps": "30", "size": [32, 32],
              "quality_space": "RGB PSNR", "on_cuda": {"wasb": "cuda:0", "balltrack": "cuda:0"},
              "source_judges": judges, "points": points}
    (tmp_path / "pilot.json").write_text(json.dumps(record))
    assert pilot.validate(tmp_path)["passed"]
    record["source_judges"]["wasb"]["frames"][0]["label_x"] = 99
    (tmp_path / "pilot.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="judge labels"):
        pilot.validate(tmp_path)
    record["source_judges"]["wasb"]["frames"][0]["label_x"] = 3
    (tmp_path / "pilot.json").write_text(json.dumps(record))
    (tmp_path / "dcvc-0" / "stream.psdc").write_bytes(b"changed")
    with pytest.raises(ValueError):
        pilot.validate(tmp_path)


def test_cost_projection_requires_timing_and_reserves_both_ceilings(tmp_path) -> None:
    import json

    with pytest.raises((FileNotFoundError, KeyError, ValueError)):
        pilot.cost_projection(tmp_path, 200)
    execution = tmp_path / "execution.json"
    dispatch = tmp_path / "dispatch.json"
    execution.write_text(json.dumps({"exit_code": 0, "seconds": 50}))
    dispatch.write_text(json.dumps({"campaign_clock": {"started": 100, "phase_started": 110, "budget_end": 600}}))
    result = pilot.cost_projection(tmp_path, 200)
    assert result["projected_with_reserve_seconds"] == 247.5
    execution.write_text(json.dumps({"exit_code": 0, "seconds": 100}))
    with pytest.raises(ValueError, match="ceiling"):
        pilot.cost_projection(tmp_path, 200)
    execution.write_text(json.dumps({"exit_code": 0, "seconds": 50}))
    with pytest.raises(ValueError, match="ceiling"):
        pilot.cost_projection(tmp_path, 350)
    dispatch.write_text(json.dumps({"campaign_clock": {"started": 100}}))
    with pytest.raises(KeyError):
        pilot.cost_projection(tmp_path, 200)
