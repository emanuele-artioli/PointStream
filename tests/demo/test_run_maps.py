from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demo.experiments import run_maps as run_maps_mod
from demo.experiments.run_maps import (
    INDEX_SCHEMA,
    _entry_from_sidecar,
    build_index,
    discover_clips,
    run_maps,
    run_one_map,
)
from demo.pipeline.maps.contract import OverlayPayloadError, payload_kbps


def _write_clip(path: Path, n: int = 4, fps: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 48, 64
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), f"failed to open VideoWriter for {path}"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[8:40, 10 + i : 50] = (0, 200, 40)
        writer.write(frame)
    writer.release()
    return path


def test_discover_clips_dir_and_file(tmp_path: Path) -> None:
    a = _write_clip(tmp_path / "clip_01.mp4")
    _write_clip(tmp_path / "clip_02.mp4")
    (tmp_path / "notes.txt").write_text("no")
    found = discover_clips(tmp_path)
    assert [p.name for p in found] == ["clip_01.mp4", "clip_02.mp4"]
    assert discover_clips(a) == [a]


def test_run_maps_canny_ok_depth_skipped(tmp_path: Path) -> None:
    clip = _write_clip(tmp_path / "clips" / "factory.mp4")
    out = tmp_path / "maps"
    index = run_maps(clips=[clip], maps=["canny", "depth"], out=out, max_frames=3)

    assert index["schema"] == INDEX_SCHEMA
    sidecar = out / "factory" / "canny" / "sidecar.json"
    assert sidecar.is_file()
    canny = next(e for e in index["entries"] if e["map"] == "canny")
    assert canny["status"] == "ok"
    assert canny["payload_kbps"] == pytest.approx(
        payload_kbps(canny["payload_bytes"], canny["duration_s"])
    )
    assert canny["payload_kbps"] != pytest.approx(
        payload_kbps(canny["preview_bytes"], canny["duration_s"])
    )
    assert canny["preview_bytes"] > canny["payload_bytes"]
    assert "extract_ms_p50" in canny
    assert canny["preview_url"]
    assert not Path(canny["payload_path"]).name.startswith("preview_")

    skipped_maps = {row["map"] for row in index["skipped"]}
    from demo.pipeline.maps.model_paths import MODELS

    if MODELS.get("yolo26s_depth") is None:
        assert "depth" in skipped_maps
        assert any("Do not auto-download" in (row.get("reason") or "") for row in index["skipped"])
        assert (out / "index.json").is_file()


def test_failed_map_does_not_abort_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _write_clip(tmp_path / "clip_01.mp4")

    def _boom(*_a, **_k):
        raise RuntimeError("extractor exploded")

    monkeypatch.setitem(run_maps_mod.RUNNERS, "yoloe", _boom)
    monkeypatch.setitem(run_maps_mod.REQUIRED_KEYS, "yoloe", ())
    index = run_maps(clips=[clip], maps=["canny", "yoloe"], out=tmp_path / "maps", max_frames=2)
    assert any(e["map"] == "canny" and e["status"] == "ok" for e in index["entries"])
    assert any(row["map"] == "yoloe" and "exploded" in row["error"] for row in index["failed"])
    assert (tmp_path / "maps" / "index.json").is_file()


def test_index_rejects_preview_as_payload() -> None:
    with pytest.raises(OverlayPayloadError):
        from demo.pipeline.maps.contract import MapStream

        MapStream(
            map="canny",
            backend="x",
            payload_path="/tmp/preview_canny.mp4",
            payload_bytes=10,
            preview_path="/tmp/preview_canny.mp4",
            preview_bytes=10,
            duration_s=1.0,
            n_frames=1,
            fps=1.0,
            extract_ms_p50=1.0,
            extract_ms_p95=1.0,
            pack_ms_p50=0.0,
            codec_ms_p50=0.0,
            decode_ms_p50=0.0,
            gpu="cpu",
        )


def test_build_index_lists_payload_not_lpips(tmp_path: Path) -> None:
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"")
    sidecar = {
        "map": "canny",
        "backend": "opencv",
        "payload_path": str(tmp_path / "payload.bin"),
        "payload_bytes": 2500,
        "payload_kbps": payload_kbps(2500, 2.0),
        "preview_path": str(tmp_path / "preview"),
        "preview_bytes": 1_000_000,
        "duration_s": 2.0,
        "n_frames": 60,
        "fps": 30.0,
        "extract_ms_p50": 1.5,
        "extract_ms_p95": 2.0,
        "pack_ms_p50": 0.1,
        "codec_ms_p50": 0.0,
        "decode_ms_p50": 0.2,
        "teleop_ok": True,
        "gpu": "cpu",
        "kind": "native",
    }
    (tmp_path / "preview").mkdir()
    (tmp_path / "preview" / "000000.png").write_bytes(b"\x89PNG")
    result = {
        "clip": "a",
        "map": "canny",
        "status": "ok",
        "sidecars": [sidecar],
    }
    index = build_index(clips=[clip], results=[result], maps_root=tmp_path)
    entry = index["entries"][0]
    assert entry["payload_kbps"] == pytest.approx(10.0)
    assert "lpips" not in entry
    assert "LPIPS" not in index["caption"]
    assert "payload" in index["caption"].lower()


def test_run_one_unknown_alias(tmp_path: Path) -> None:
    clip = _write_clip(tmp_path / "x.mp4")
    result = run_one_map("nope", clip, clip_id="x", maps_root=tmp_path, max_frames=1)
    assert result["status"] == "failed"
    assert "unknown" in result["error"]


def test_publish_copies_preview_not_payload(tmp_path: Path) -> None:
    from demo.pitch.publish_site import copy_maps_gallery

    src = tmp_path / "maps_src"
    preview = src / "factory" / "canny" / "preview"
    preview.mkdir(parents=True)
    (preview / "000000.png").write_bytes(b"\x89PNG")
    (preview / "000001.png").write_bytes(b"\x89PNG")
    payload = src / "factory" / "canny" / "payload.bin"
    payload.write_bytes(b"CNNY")
    (src / "factory" / "canny" / "yoloe-26n-seg.pt").write_bytes(b"no")
    index = {
        "schema": INDEX_SCHEMA,
        "caption": "Bitrate is the native payload, not this preview video.",
        "clips": {"factory": {"source": "x", "ok": {}}},
        "entries": [
            {
                "clip": "factory",
                "map": "canny",
                "payload_kbps": 3.2,
                "payload_bytes": 80,
                "preview_bytes": 9000,
                "preview_url": "factory/canny/preview/000000.png",
                "preview_dir": "factory/canny/preview",
                "preview_kind": "image_dir",
                "preview_path": str(preview),
                "extract_ms_p50": 1.1,
                "decode_ms_p50": 0.2,
                "status": "ok",
            }
        ],
        "skipped": [],
        "failed": [],
    }
    (src / "index.json").write_text(__import__("json").dumps(index))
    dest = tmp_path / "site"
    copy_maps_gallery(dest, maps_src=src)
    assert (dest / "maps" / "index.json").is_file()
    assert (dest / "maps" / "factory" / "canny" / "preview" / "000000.png").is_file()
    assert (dest / "maps" / "factory" / "canny" / "preview" / "000001.png").is_file()
    assert not (dest / "maps" / "factory" / "canny" / "payload.bin").exists()
    assert not list(dest.rglob("*.pt"))


def test_canny_entry_uses_mask_png_sequence_not_mp4(tmp_path: Path) -> None:
    preview = tmp_path / "clip_01" / "canny" / "preview"
    preview.mkdir(parents=True)
    (preview / "000000.png").write_bytes(b"\x89PNG")
    (preview / "000001.png").write_bytes(b"\x89PNG")
    mp4 = tmp_path / "clip_01" / "canny" / "preview.mp4"
    mp4.write_bytes(b"ftyp")
    entry = _entry_from_sidecar(
        {
            "map": "canny",
            "payload_kbps": 12.0,
            "payload_bytes": 80,
            "preview_path": str(preview),
            "preview_bytes": 9000,
            "n_frames": 2,
            "fps": 30.0,
        },
        clip_id="clip_01",
        maps_root=tmp_path,
    )
    assert entry["preview_kind"] == "image_dir"
    assert entry["preview_url"] == "clip_01/canny/preview/000000.png"
    assert entry["preview_dir"] == "clip_01/canny/preview"
    assert entry.get("overlay_url") is None


def test_pose_entry_uses_rgba_png_sequence(tmp_path: Path) -> None:
    pose = tmp_path / "clip_01" / "pose"
    preview = pose / "preview"
    preview.mkdir(parents=True)
    (preview / "000000.png").write_bytes(b"\x89PNG")
    entry = _entry_from_sidecar(
        {
            "map": "dwpose",
            "payload_kbps": 17.9,
            "payload_bytes": 2247,
            "preview_path": str(preview),
            "preview_bytes": 100,
            "n_frames": 1,
            "fps": 30.0,
        },
        clip_id="clip_01",
        maps_root=tmp_path,
    )
    assert entry["preview_kind"] == "image_dir"
    assert entry["preview_url"] == "clip_01/pose/preview/000000.png"
    assert entry["preview_dir"] == "clip_01/pose/preview"
    assert entry.get("overlay_url") is None


def test_entry_sets_overlay_url_when_webm_exists(tmp_path: Path) -> None:
    preview = tmp_path / "clip_01" / "yoloe_masks" / "preview"
    preview.mkdir(parents=True)
    (preview / "000000.png").write_bytes(b"\x89PNG")
    webm = tmp_path / "clip_01" / "yoloe_masks" / "preview.webm"
    webm.write_bytes(b"webm")
    entry = _entry_from_sidecar(
        {
            "map": "yoloe_masks",
            "payload_kbps": 12.0,
            "payload_bytes": 80,
            "preview_path": str(preview),
            "preview_bytes": 9000,
            "n_frames": 1,
            "fps": 30.0,
        },
        clip_id="clip_01",
        maps_root=tmp_path,
    )
    assert entry["overlay_url"] == "clip_01/yoloe_masks/preview.webm"
    assert entry["preview_dir"] == "clip_01/yoloe_masks/preview"


def test_publish_prefers_overlay_webm_over_png_sequence(tmp_path: Path) -> None:
    from demo.pitch.publish_site import copy_maps_gallery

    src = tmp_path / "maps_src"
    preview = src / "factory" / "canny" / "preview"
    preview.mkdir(parents=True)
    (preview / "000000.png").write_bytes(b"\x89PNG")
    (preview / "000001.png").write_bytes(b"\x89PNG")
    webm = src / "factory" / "canny" / "preview.webm"
    webm.write_bytes(b"webm")
    (src / "factory" / "canny" / "payload.bin").write_bytes(b"CNNY")
    index = {
        "schema": INDEX_SCHEMA,
        "caption": "Bitrate is the native payload, not this preview video.",
        "clips": {"factory": {"source": "x", "ok": {}}},
        "entries": [
            {
                "clip": "factory",
                "map": "canny",
                "payload_kbps": 3.2,
                "payload_bytes": 80,
                "preview_bytes": 9000,
                "preview_url": "factory/canny/preview/000000.png",
                "preview_dir": "factory/canny/preview",
                "overlay_url": "factory/canny/preview.webm",
                "preview_kind": "image_dir",
                "status": "ok",
            }
        ],
        "skipped": [],
        "failed": [],
    }
    (src / "index.json").write_text(__import__("json").dumps(index))
    dest = tmp_path / "site"
    copy_maps_gallery(dest, maps_src=src)
    assert (dest / "maps" / "factory" / "canny" / "preview.webm").is_file()
    assert not (dest / "maps" / "factory" / "canny" / "preview" / "000000.png").exists()
    assert not (dest / "maps" / "factory" / "canny" / "payload.bin").exists()


def test_inspector_stays_default_tab() -> None:
    html = (Path(__file__).resolve().parents[2] / "demo" / "pitch" / "interactive_demo.html").read_text()
    assert 'data-tab="inspector"' in html
    assert 'id="tab-inspector" class="tab-content active"' in html
    assert 'data-tab="maps"' not in html
    assert "lpips" not in html[html.index("tab-inspector") : html.index("tab-benchmark")].lower()
    assert "payload_kbps" in html
    inspector = html[html.index("tab-inspector") : html.index("tab-benchmark")]
    assert "av1_240" in inspector or "AV1 240p" in inspector
    assert "compare-board" not in html
    assert "STREAM_LATENCY_MS" in html
    assert "18.4" in html
    assert "value=\"empty\"" in inspector
    assert "Error / Distortion Heatmap" in inspector
    assert "Stacked 3-Panel" not in inspector
    assert "47-Byte Skeleton Telemetry" not in inspector
    assert 'id="maps-left"' in html
    assert 'id="maps-right"' in html
    assert "select-canny-rung" not in inspector
    assert "image_dir" in html or "preview_dir" in html
    assert "overlay_url" in html

