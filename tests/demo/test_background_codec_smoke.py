from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from demo.experiments import background_smoke as smoke
from demo.experiments import background_smoke_core as core
from demo.experiments import dcvc_uf_adapter as adapter


# ----------------------------------------------------------- pure checks

def test_container_round_trip_and_rejections() -> None:
    packed = adapter.pack_container(b"native", structure="htl", frame_count=8, force_intra=False)
    header, native = adapter.unpack_container(packed)
    assert native == b"native" and header == {"structure": "htl", "frame_count": 8, "force_intra": False, "header_bytes": 9}
    with pytest.raises(ValueError, match="not a version-1"):
        adapter.unpack_container(b"XXXX" + packed[4:])
    with pytest.raises(ValueError, match="truncated"):
        adapter.unpack_container(packed[:5])


def test_hierarchical_display_order_and_padding_are_explicit() -> None:
    assert adapter.display_schedule(8, 1, force_intra=False) == [[i] for i in range(8)]
    assert adapter.display_schedule(8, 8, force_intra=False) == [[0], list(range(1, 8))]
    ht32 = adapter.display_schedule(32, 8, force_intra=False)
    assert ht32 == [[0], list(range(1, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 32))]
    assert [i for nal in ht32 for i in nal] == list(range(32))
    assert adapter.display_schedule(3, 8, force_intra=True) == [[0], [1], [2]]
    assert all(adapter.reset_flag(i, 8, adapter.RESET_INTERVAL) == 0 for i in range(64))
    assert adapter.reset_flag(25, 8, 32) == 1


def frame(value: int, shape=(36, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(value)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_rgb_metrics_definitions() -> None:
    ref = [frame(1), frame(2)]
    same = core.sequence_metrics(ref, ref, expected_shape=(36, 64, 3))
    assert same["mean_frame_psnr_db"] == 99.0 and same["temporal_reconstruction_error"] == 0.0
    shifted = [np.clip(f.astype(int) + 4, 0, 255).astype(np.uint8) for f in ref]
    masks = [np.zeros((36, 64), bool) for _ in ref]
    masks[0][:10] = True
    result = core.sequence_metrics(ref, shifted, expected_shape=(36, 64, 3), masks=masks)
    assert result["pooled_mse_psnr_db"] == pytest.approx(core.psnr_from_mse(float(np.mean(result["frame_mse"]))))
    assert result["mask_region"]["inside_mask_pooled_psnr_db"] is not None
    expected_temporal = np.abs((shifted[1].astype(float) - shifted[0]) - (ref[1].astype(float) - ref[0])).mean() / 255
    assert result["temporal_reconstruction_error"] == pytest.approx(expected_temporal)


def test_frame_conventions_are_never_guessed() -> None:
    with pytest.raises(ValueError, match="declared RGB"):
        core.require_rgb_frame(frame(1), channel_order="BGR")
    with pytest.raises(ValueError, match="uint8"):
        core.require_rgb_frame(frame(1).astype(np.float32))
    with pytest.raises(ValueError, match="differs from expected"):
        core.require_rgb_frame(frame(1), expected_shape=(1080, 1920, 3))
    with pytest.raises(ValueError, match="unit-range"):
        core.unit_float_to_uint8(np.full((3, 2, 2), 1.5), layout="CHW", value_range="unit")
    assert core.lpips_input(np.zeros((2, 2, 3), np.uint8)).min() == -1.0


def test_rate_counts_actual_file_bytes(tmp_path: Path) -> None:
    assert core.rate_kbps(1000, frames=8) == pytest.approx(8 * 1000 * 30 / (1000 * 8))
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"12345")
    b.write_bytes(b"123")
    assert core.stream_file_bytes([a, b]) == 8
    with pytest.raises(ValueError, match="more than once"):
        core.stream_file_bytes([a, a])


def test_cut_lengths_order_and_identities(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(bytes([index]))
        rows.append({"index": 120 + index, "path": str(path), "sha256": core.sha256_bytes(bytes([index]))})
    core.validate_cut(rows, start=120, length=8)
    with pytest.raises(ValueError, match="only 8- or 32"):
        core.validate_cut(rows[:4], start=120, length=4)
    with pytest.raises(ValueError, match="order"):
        core.validate_cut([rows[1], rows[0], *rows[2:]], start=120, length=8)
    changed_path = rows[3]["path"]
    assert isinstance(changed_path, str)
    Path(changed_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        core.validate_cut(rows, start=120, length=8)
    with pytest.raises(ValueError, match="display index 1"):
        core.compare_frame_identities(["a", "b"], ["a", "c"])


def test_part_names_stay_inside_the_plan() -> None:
    assert smoke.parse_parts("codec", "f001c3-htl-pre,f001c3-av1,f002-htl-ft") == ["f001c3-htl-pre", "f001c3-av1", "f002-htl-ft"]
    for kind, value in (("codec", "f001c3-htl-mid"), ("codec", "f003-ld-pre"), ("drift", "f001c3-htl-pre"),
                        ("drift", "f002-ld-pre"), ("latent", "f001c3-b4"), ("latent", "f001c3-b8"), ("codec", "")):
        with pytest.raises(ValueError):
            smoke.parse_parts(kind, value)


def test_degradation_flags_use_both_thresholds() -> None:
    parts = {
        "f001c3-ld-pre": {"status": "passed", "mean_frame_psnr_db": 36.0, "container_bytes": 1000},
        "f001c3-ld-ft": {"status": "passed", "mean_frame_psnr_db": 33.5, "container_bytes": 900},
        "f001c3-hts-pre": {"status": "passed", "mean_frame_psnr_db": 34.0, "container_bytes": 1000},
        "f001c3-hts-ft": {"status": "passed", "mean_frame_psnr_db": 34.0, "container_bytes": 2500},
        "f001c3-htl-ft": {"status": "passed", "mean_frame_psnr_db": 34.0, "container_bytes": 1000},
    }
    flags = smoke.degradation_flags(parts)
    assert flags["f001c3-ld-ft"]["degraded"] and flags["f001c3-hts-ft"]["degraded"]
    assert flags["f001c3-htl-ft"].startswith("untested")


def test_drift_summary_requires_exact_frames() -> None:
    rows = [{"index": 120 + i, "psnr_db": 40 - i * 0.1} for i in range(32)]
    summary = core.summarize_drift(rows)
    assert summary["last8_minus_first8_db"] == pytest.approx(-2.4)
    with pytest.raises(ValueError):
        core.summarize_drift(rows[:31])


def test_adapter_environment_requires_claimed_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(smoke.PartBlocked, match="claimed GPU"):
        smoke.adapter_env()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    monkeypatch.setenv("PS_STAGE_DIR", "/tmp/stage")
    env = smoke.adapter_env()
    assert env["PYTHONPATH"] == str(core.DCVC_ROOT) and env["PS_STAGE_DIR"] == "/tmp/stage"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1234"


def test_bounded_commands_time_out() -> None:
    with pytest.raises(TimeoutError):
        smoke.run_command(["sleep", "5"], timeout=0.2)


# ------------------------------------------------- runner on fixtures

SHAPE = (36, 64, 3)


@pytest.fixture
def fixture_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(core, "FRAME_SHAPE", SHAPE)
    monkeypatch.setattr(core, "HEIGHT", SHAPE[0])
    monkeypatch.setattr(core, "WIDTH", SHAPE[1])
    monkeypatch.setattr(smoke, "SCRATCH_ROOT", tmp_path / "scratch")
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for index in range(32):
        path = data / f"{index:05d}.png"
        pixels = frame(index)
        Image.fromarray(pixels).save(path)
        rows.append({"index": 120 + index, "path": str(path), "sha256": core.sha256_file(path, timeout=30)["sha256"],
                     "chunk": f"chunk_{120 + 30 * (index // 30):05d}", "chunk_position": index % 30, "rgb_sha256": core.rgb_identity(pixels)})
    mask = data / "union.npy"
    np.save(mask, np.zeros(SHAPE[:2], bool))
    checkpoints = {}
    for key in ("uf_image", "uf_video:htl:pretrained", "uf_video:ld:pretrained", "uf_video:ld:factory001:s1"):
        path = data / (key.replace(":", "_") + ".pth.tar")
        path.write_bytes(key.encode())
        checkpoints[key] = {"path": str(path), "sha256": core.sha256_bytes(key.encode()), "bytes": len(key)}
    manifest = {
        "schema": "pointstream.background-smoke.inputs.v1", "provenance": {},
        "frames": {"f001c3": rows},
        "masks": {"f001c3": {row["chunk"]: {"path": str(mask), "sha256": core.sha256_file(mask, timeout=30)["sha256"], "shape": list(SHAPE[:2])} for row in rows}},
        "checkpoints": checkpoints,
        "environment": {"stage_smoke": {"lpips": "lpips fixture", "dcvc": {
            "files": {rel: {"sha256": digest} for rel, digest in core.DCVC_REFERENCE_SHA256.items()},
            "adapter_check": {"reference": {"matches_reference": True}}}}},
    }
    manifest_path = data / "selected-inputs.json"
    manifest_path.write_text(json.dumps(manifest))
    job = tmp_path / "inbox" / "20261004T000000Z-abcdef12"
    stage = job / "smoke"
    stage.mkdir(parents=True)
    (job / "ready.json").write_text(json.dumps({"git_head": "f" * 40, "source_sha256": "e" * 64}))
    (job / "environment.json").write_text(json.dumps({"host": "gpu5", "gpu": {"uuid": "GPU-test", "name": "fixture"}}))
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.delenv("PS_JOB_DIR", raising=False)
    # Fixtures exercise our runner/gates, not LPIPS or CUDA correctness.
    monkeypatch.setattr(smoke.Lpips, "score", lambda self, ref, rec: {"status": "fixture", "mean": 0.0})
    return manifest_path


def fake_adapter(*, corrupt_repeat: bool = False):
    """Lossless stand-in for the DCVC process: containers carry raw pixels."""

    def run(stage, action, arguments, *, timeout, report):
        plan = json.loads(Path(arguments[arguments.index("--plan") + 1]).read_text())
        if action == "encode":
            streams = []
            for stream in plan["streams"]:
                frames = np.stack([np.asarray(Image.open(Path(stream["frames_dir"]) / f"im{i + 1:05d}.png")) for i in range(stream["frame_count"])])
                buffer = io.BytesIO()
                np.save(buffer, frames)
                container = adapter.pack_container(buffer.getvalue(), structure="htl", frame_count=len(frames), force_intra=plan["force_intra"])
                Path(stream["container"]).write_bytes(container)
                if stream.get("i_recon_png"):
                    Image.fromarray(frames[0]).save(stream["i_recon_png"])
                nals = [{"type": "I"}] + [{"type": "P"}] * (len(frames) - 1)
                streams.append({"name": stream["name"], "container": stream["container"], "container_bytes": len(container),
                                "container_header_bytes": 9, "native_stream_bytes": len(container) - 9, "nals": nals})
            payload = {"streams": streams, "peak_memory": {"torch_max_reserved_mib": 10.0}, "load_seconds": 0.1}
        else:
            for stream in json.loads((report.parent / "encode-plan.json").read_text())["streams"]:
                assert not Path(stream["frames_dir"]).exists(), "source frames must be sealed during decode"
            names = [s["name"] for s in plan["streams"]] + plan["repeat"]
            streams = []
            for position, name in enumerate(names):
                stream = next(s for s in plan["streams"] if s["name"] == name)
                repeat = position >= len(plan["streams"])
                out = Path(stream["out_dir"] + ("_repeat" if repeat else ""))
                out.mkdir()
                _header, native = adapter.unpack_container(Path(stream["container"]).read_bytes())
                frames = np.load(io.BytesIO(native))
                hashes = []
                for index, pixels in enumerate(frames):
                    if repeat and corrupt_repeat:
                        pixels = 255 - pixels
                    Image.fromarray(pixels).save(out / f"im{index + 1:05d}.png")
                    hashes.append(core.rgb_identity(pixels))
                streams.append({"name": name, "repeat": repeat, "out_dir": str(out), "decoded_png_sha256": hashes,
                                "nal_types": ["I"] + ["P"] * (len(frames) - 1)})
            payload = {"streams": streams, "inputs": "container bytes only", "peak_memory": {"torch_max_reserved_mib": 12.0}, "load_seconds": 0.1}
        payload.update(checkpoints={"strict_load": True}, reference={"matches_reference": True})
        report.write_text(json.dumps(payload))
        return payload

    return run


def test_codec_runner_end_to_end_on_fixtures(fixture_stage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "run_adapter", fake_adapter())
    args = argparse.Namespace(parts="f001c3-htl-pre", stage_seconds=120, manifest=[fixture_stage])
    assert smoke.run("codec", args) == 0
    stage = core.stage_dir()
    result = json.loads((stage / "result.json").read_text())
    case = result["parts"]["f001c3-htl-pre"]
    assert case["status"] == "passed", case
    assert case["mean_frame_psnr_db"] == 99.0 and case["encoder_i_recon_equals_decoder"] is True
    assert case["kbps"] == pytest.approx(core.rate_kbps(case["container_bytes"], frames=8))
    assert case["checkpoints"]["image"]["sha256"] == core.sha256_bytes(b"uf_image")
    assert Path(case["sheet"]).is_file() and case["lpips"]["status"] == "fixture"
    assert (stage / "f001c3-htl-pre" / "src").is_dir(), "sealed sources are restored after decode"
    assert not (stage.parent.parent.parent / "scratch").exists() or not any((stage.parent.parent.parent / "scratch").iterdir())
    assert result["citable"] is False and result["provenance"]["gpu_uuid"] == "GPU-test"
    passed, checks = smoke.validate("codec")
    assert passed, checks


def test_decoder_state_dependence_fails_the_case_and_the_gate(fixture_stage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "run_adapter", fake_adapter(corrupt_repeat=True))
    smoke.run("codec", argparse.Namespace(parts="f001c3-htl-pre", stage_seconds=120, manifest=[fixture_stage]))
    result = json.loads((core.stage_dir() / "result.json").read_text())
    assert result["parts"]["f001c3-htl-pre"]["status"] == "failed"
    target = core.stage_dir().parent / "validation.json"
    monkeypatch.setenv("PS_VALIDATION_PATH", str(target))
    assert smoke.run_validator("codec") == 1
    assert json.loads(target.read_text())["passed"] is False


def test_drift_runner_compares_one_stream_with_four_reset_streams(fixture_stage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "run_adapter", fake_adapter())
    # Create the structurally valid, source/checkpoint-matched B2 pair first.
    smoke.run("codec", argparse.Namespace(parts="f001c3-ld-pre,f001c3-ld-ft", stage_seconds=120, manifest=[fixture_stage]))
    evidence = core.stage_dir() / "result.json"
    stage = core.stage_dir().parent / "full"
    stage.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    assert smoke.run("drift", argparse.Namespace(parts="f001c3-ld-pre", stage_seconds=120, manifest=[fixture_stage], codec_evidence=evidence)) == 0
    case = json.loads((core.stage_dir() / "result.json").read_text())["parts"]["f001c3-ld-pre"]
    assert case["status"] == "passed", case
    assert case["segment_decode_order"][:4] == ["seg3", "seg0", "seg1", "seg2"]
    assert case["one_stream"]["drift"]["frame_count"] == 32
    assert case["four_reset_streams"]["kbps"] == pytest.approx(core.rate_kbps(case["four_reset_streams"]["container_bytes_sum"], frames=32))
    assert len(case["frame_ids"]) == 32
    assert smoke.validate("drift")[0]


def test_missing_checkpoint_identity_blocks_the_case(fixture_stage: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "run_adapter", fake_adapter())
    smoke.run("codec", argparse.Namespace(parts="f001c3-hts-pre", stage_seconds=120, manifest=[fixture_stage]))
    case = json.loads((core.stage_dir() / "result.json").read_text())["parts"]["f001c3-hts-pre"]
    assert case["status"] == "blocked" and "uf_video:hts:pretrained" in case["reason"]
    assert not smoke.validate("codec")[0]


@pytest.mark.parametrize("mutation", ["no_container", "corrupt_container", "no_lpips", "no_strict_load", "missing_metrics", "changed_source", "changed_manifest"])
def test_codec_validator_rejects_incomplete_or_changed_artifacts(fixture_stage, monkeypatch, mutation):
    monkeypatch.setattr(smoke, "run_adapter", fake_adapter())
    smoke.run("codec", argparse.Namespace(parts="f001c3-htl-pre", stage_seconds=120, manifest=[fixture_stage]))
    path = core.stage_dir() / "result.json"
    value = json.loads(path.read_text())
    record = value["parts"]["f001c3-htl-pre"]
    if mutation == "no_container":
        Path(record["container"]).unlink()
    elif mutation == "corrupt_container":
        target = Path(record["container"])
        raw = target.read_bytes()
        target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif mutation == "no_lpips":
        record["lpips_mean"] = None
    elif mutation == "no_strict_load":
        record["strict_load"] = False
    elif mutation == "changed_source":
        value_manifest = json.loads(fixture_stage.read_text())
        Path(value_manifest["frames"]["f001c3"][0]["path"]).write_bytes(b"changed source")
    elif mutation == "changed_manifest":
        fixture_stage.write_text("{}")
    else:
        record.pop("frame_psnr_db")
    path.write_text(json.dumps(value))
    assert smoke.validate("codec")[0] is False


@pytest.mark.parametrize("kind,name", [("codec", "f001c3-htl-pre"), ("drift", "f001c3-ld-pre"), ("latent", "f001c3-b6")])
def test_status_only_records_never_pass_validation(fixture_stage, kind, name):
    core.stage_dir().joinpath("result.json").write_text(json.dumps({
        "kind": kind, "citable": False, "parts_requested": [name], "parts": {name: {"status": "passed"}},
        "provenance": {"code_revision": "f" * 40, "gpu_uuid": "GPU-test"}}))
    assert smoke.validate(kind)[0] is False


def test_drift_requires_a_b2_pair_before_any_adapter(fixture_stage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not start the model")
    monkeypatch.setattr(smoke, "run_adapter", forbidden)
    with pytest.raises(smoke.PartBlocked, match="B2 LD pair"):
        smoke.run("drift", argparse.Namespace(parts="f001c3-ld-pre", stage_seconds=120, manifest=[fixture_stage]))


def test_inventory_preserves_partial_identities_before_later_failure(fixture_stage, monkeypatch):
    stage = smoke.Stage("inventory", 120, [])
    def completed():
        stage.selected["checkpoints"] = {"uf_image": {"sha256": "a" * 64}}
        return {"verified": True}
    stage.run_part("ckpt-image", completed)
    stage.run_part("env", lambda: smoke._blocked("import unavailable"))
    value = json.loads(stage.path("partial-inputs", "ckpt-image.json").read_text())
    assert value["checkpoints"]["uf_image"]["sha256"] == "a" * 64


def test_missing_environment_blocks_inference(fixture_stage, monkeypatch):
    value = json.loads(fixture_stage.read_text())
    value.pop("environment")
    fixture_stage.write_text(json.dumps(value))
    with pytest.raises(smoke.PartBlocked, match="environment"):
        smoke.run("codec", argparse.Namespace(parts="f001c3-htl-pre", stage_seconds=120, manifest=[fixture_stage]))


def test_preview_reads_the_saved_manifest_schema_and_corrects_hnerv_rate(tmp_path, monkeypatch):
    preview = tmp_path / "preview"
    (preview / "canvas").mkdir(parents=True)
    clips = []
    for factory in ("factory001", "factory002"):
        panels = {}
        for i in (0, 2, 4, 6):
            rel = f"canvas/{factory}_frame_{i:02d}.jpg"
            (preview / rel).write_bytes(b"panel")
            panels[str(i)] = rel
        clips.append({"factory": factory, "segment_start_in_holdout": 120, "segment_length": 8,
            "panels": panels, "methods": {"hnerv": {"rate_kbps": 9000, "rgb_psnr_vs_filled_db": 28},
            "uf_ld": {"rate_kbps": 401.94, "rgb_psnr_vs_filled_db": 36.72}}})
    (preview / "canvas/manifest.json").write_text(json.dumps({"clips": clips, "qp": 21}))
    stage = tmp_path / "smoke"
    stage.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setattr(core, "PREVIEW_ROOT", preview)
    value = smoke.inventory_preview(smoke.Stage("inventory", 120, []))
    assert value["status"] == "passed"
    row = value["reuse_table"][0]
    assert row["frames"] == 8 and row["display_rates_kbps"]["uf_ld"] == 401.94
    assert "hnerv" not in row["display_rates_kbps"]
    assert row["hnerv_setup_inclusive_estimate_kbps"] == 9000
    clips[0]["panels"] = {}
    (preview / "canvas/manifest.json").write_text(json.dumps({"clips": clips}))
    assert smoke.inventory_preview(smoke.Stage("inventory", 120, []))["status"] == "inconclusive"


def test_missing_lpips_inventory_blocks_before_model_work(fixture_stage, monkeypatch):
    value = json.loads(fixture_stage.read_text())
    value["environment"]["stage_smoke"]["lpips"] = "blocked: no cached weights"
    fixture_stage.write_text(json.dumps(value))
    with pytest.raises(smoke.PartBlocked, match="LPIPS-Alex"):
        smoke.run("codec", argparse.Namespace(parts="f001c3-htl-pre", stage_seconds=120, manifest=[fixture_stage]))
