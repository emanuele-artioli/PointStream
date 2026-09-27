from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from demo.pipeline.maps.contract import OverlayPayloadError, MapStream, read_sidecar
from demo.pipeline.maps.encode import (
    MASK_STREAM_SCHEMA,
    decode_coco_rle,
    encode_coco_rle,
    frame_record,
    instance_record,
    pack_rle_stream,
    union_mask_from_frame,
    unpack_rle_stream,
)
from demo.pipeline.maps.model_paths import MODELS
from demo.pipeline.maps.sam31_masks import (
    SAM_NOTE,
    main as sam_main,
    write_sam31_map,
)
from demo.pipeline.maps.yoloe_masks import (
    YOLOE_INSTALL_HINT,
    instances_from_result,
    load_boxes_from,
    load_classes,
    main as yoloe_main,
    write_mask_map,
    write_yoloe_map,
)


def _blob_mask(h: int = 16, w: int = 24) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:10, 4:15] = 1
    mask[8:12, 10:20] = 1
    return mask


def test_coco_rle_roundtrip() -> None:
    mask = _blob_mask()
    rle = encode_coco_rle(mask)
    out = decode_coco_rle(rle)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, mask)


def test_coco_rle_empty_is_all_false_not_true() -> None:
    empty = np.zeros((8, 12), dtype=np.uint8)
    rle = encode_coco_rle(empty)
    out = decode_coco_rle(rle)
    assert out.sum() == 0
    assert not bool(out.all())


def test_skip_frame_empty_mask_is_not_whole_true() -> None:
    rec = frame_record(0, [])
    union = union_mask_from_frame(rec, 8, 10)
    assert union.shape == (8, 10)
    assert int(union.sum()) == 0
    assert not bool(union.all())
    assert instance_record(np.zeros((8, 10), dtype=np.uint8), class_id=0, class_name="hand") is None

    blob, _codec = pack_rle_stream(
        [rec], height=8, width=10, fps=30.0, classes=["hand"]
    )
    doc = unpack_rle_stream(blob)
    assert doc["schema"] == MASK_STREAM_SCHEMA
    assert doc["mask_empty_policy"] == "skip_frame"
    assert doc["frames"][0]["instances"] == []
    restored = union_mask_from_frame(doc["frames"][0], 8, 10)
    assert int(restored.sum()) == 0
    assert not np.array_equal(restored, np.ones((8, 10), dtype=np.uint8))


def test_rle_stream_roundtrip_two_instances() -> None:
    h, w = 16, 20
    a = np.zeros((h, w), dtype=np.uint8)
    a[1:4, 2:6] = 1
    b = np.zeros((h, w), dtype=np.uint8)
    b[8:14, 10:18] = 1
    rec = frame_record(
        0,
        [
            instance_record(a, class_id=0, class_name="hand"),
            instance_record(b, class_id=4, class_name="tool"),
        ],
    )
    blob, codec = pack_rle_stream([rec], height=h, width=w, fps=10.0, classes=["hand", "tool"])
    doc = unpack_rle_stream(blob)
    assert len(doc["frames"][0]["instances"]) == 2
    union = union_mask_from_frame(doc["frames"][0], h, w)
    assert np.array_equal(union, a | b)
    assert codec in {"zstd", "zlib"}


def _empty_predict(_frame: np.ndarray) -> list[dict]:
    return []


def _blob_predict(frame: np.ndarray) -> list[dict]:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[2:8, 3:12] = 1
    return [
        {
            "mask": mask,
            "class_id": 0,
            "class_name": "hand",
            "score": 0.91,
            "bbox": [3.0, 2.0, 12.0, 8.0],
        }
    ]


def test_write_yoloe_skip_frame_without_loading_model(tmp_path: Path) -> None:
    frames = [np.zeros((16, 20, 3), dtype=np.uint8), np.zeros((16, 20, 3), dtype=np.uint8)]
    stream = write_yoloe_map(
        frames,
        tmp_path,
        predict_fn=_empty_predict,
        classes=["hand", "tool"],
        n_warmup=0,
        n_runs=1,
        fps=10.0,
    )
    sidecar = read_sidecar(tmp_path / "sidecar.json")
    payload = Path(sidecar["payload_path"])
    assert payload.is_file()
    assert payload.name in {"masks.rle.zst", "payload.bin"}
    assert not payload.name.startswith("preview_")
    assert sidecar["map"] == "yoloe_masks"
    assert sidecar["n_skipped"] == 2
    assert sidecar["n_instances"] == 0
    preview = tmp_path / "preview_masks"
    assert preview.is_dir()
    assert sidecar["preview_path"] == str(preview)
    doc = unpack_rle_stream(payload.read_bytes())
    for rec in doc["frames"]:
        union = union_mask_from_frame(rec, 16, 20)
        assert int(union.sum()) == 0
        assert not bool(union.all())
    assert stream.mask_empty_policy == "skip_frame"


def test_yoloe_preview_is_masks_only_rgba(tmp_path: Path) -> None:
    frame = np.full((16, 20, 3), 90, dtype=np.uint8)
    write_yoloe_map(
        [frame],
        tmp_path,
        predict_fn=_blob_predict,
        classes=["hand"],
        n_warmup=0,
        n_runs=1,
        fps=10.0,
    )
    png = tmp_path / "preview_masks" / "000000.png"
    vis = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert vis is not None
    assert vis.shape[2] == 4
    assert int(vis[:, :, 3].max()) > 0
    # Background must be transparent — not a copy of the RGB video.
    bg = vis[:, :, 3] == 0
    assert bool(bg.any())
    assert int(vis[bg][:, :3].sum()) == 0


def test_profile_pack_accepts_ultralytics_results(tmp_path: Path) -> None:
    mask = _blob_mask(16, 20)

    class FakeMasks:
        data = mask[None, ...]

    class FakeBoxes:
        cls = np.array([0.0])
        conf = np.array([0.9])
        xyxy = np.array([[4.0, 3.0, 15.0, 10.0]])

    class FakeResults:
        masks = FakeMasks()
        boxes = FakeBoxes()
        names = {0: "hand"}

    stream = write_mask_map(
        [np.zeros((16, 20, 3), dtype=np.uint8)],
        tmp_path,
        predict_fn=_empty_predict,
        classes=["hand"],
        map_name="yoloe_masks",
        backend="stub",
        fps=10.0,
        n_warmup=0,
        n_runs=1,
        profile_extract=lambda _frame: FakeResults(),
    )
    assert stream.payload_bytes > 0
    assert stream.pack_ms_p50 >= 0.0


def test_write_sam31_skip_frame_without_loading_model(tmp_path: Path) -> None:
    frames = [np.zeros((12, 18, 3), dtype=np.uint8)]
    stream = write_sam31_map(
        frames,
        tmp_path,
        predict_fn=_empty_predict,
        n_warmup=0,
        n_runs=1,
        fps=5.0,
    )
    sidecar = read_sidecar(tmp_path / "sidecar.json")
    assert sidecar["map"] == "sam31_masks"
    assert sidecar["backend"] == "sam3.pt"
    assert sidecar["note"] == SAM_NOTE
    assert sidecar["temporal_id"] is False
    assert "teleop_ok" in sidecar
    payload = Path(sidecar["payload_path"])
    doc = unpack_rle_stream(payload.read_bytes())
    union = union_mask_from_frame(doc["frames"][0], 12, 18)
    assert int(union.sum()) == 0
    assert stream.kind == "native"


def test_inspector_can_swap_yoloe_and_sam_schema(tmp_path: Path) -> None:
    frames = [np.zeros((16, 20, 3), dtype=np.uint8)]
    yoloe_dir = tmp_path / "yoloe_masks"
    sam_dir = tmp_path / "sam31_masks"
    write_yoloe_map(
        frames, yoloe_dir, predict_fn=_blob_predict, classes=["hand"], n_warmup=0, n_runs=1
    )
    write_sam31_map(frames, sam_dir, predict_fn=_blob_predict, n_warmup=0, n_runs=1)
    yoloe_payload = Path(read_sidecar(yoloe_dir / "sidecar.json")["payload_path"])
    sam_payload = Path(read_sidecar(sam_dir / "sidecar.json")["payload_path"])
    yoloe = unpack_rle_stream(yoloe_payload.read_bytes())
    sam = unpack_rle_stream(sam_payload.read_bytes())
    assert yoloe["schema"] == sam["schema"] == MASK_STREAM_SCHEMA
    assert set(yoloe["frames"][0]["instances"][0]) == set(sam["frames"][0]["instances"][0])
    assert yoloe["frames"][0]["instances"][0]["class_name"] == "hand"


def test_boxes_from_yoloe_dir(tmp_path: Path) -> None:
    frames = [np.zeros((16, 20, 3), dtype=np.uint8)]
    write_yoloe_map(
        frames, tmp_path, predict_fn=_blob_predict, classes=["hand"], n_warmup=0, n_runs=1
    )
    boxes = load_boxes_from(tmp_path)
    assert boxes[0][0]["class_name"] == "hand"
    assert boxes[0][0]["xyxy"] == [3.0, 2.0, 12.0, 8.0]


def test_instances_from_result_duck_type() -> None:
    mask = np.zeros((10, 12), dtype=np.float32)
    mask[2:6, 3:9] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask[None, ...]),
        boxes=SimpleNamespace(
            cls=np.asarray([0.0]),
            conf=np.asarray([0.8]),
            xyxy=np.asarray([[3.0, 2.0, 9.0, 6.0]]),
        ),
        names={0: "hand"},
    )
    instances = instances_from_result(result, 10, 12, {0: "hand"})
    assert len(instances) == 1
    assert instances[0]["class_name"] == "hand"
    assert int(instances[0]["mask"].sum()) > 0


def test_shared_av1_recipe_is_crf63() -> None:
    from demo.pipeline.maps.av1_crf import av1_output_args

    args = av1_output_args()
    assert args[args.index("-vf") + 1] == "scale=426:240"
    assert args[args.index("-c:v") + 1] == "libsvtav1"
    assert args[args.index("-preset") + 1] == "7"
    assert args[args.index("-crf") + 1] == "63"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert "-b:v" not in args


def test_load_classes_from_prompts_yaml() -> None:
    classes = load_classes()
    assert classes == ["hand", "tool", "workbench"]


def test_payload_rejected_if_preview_masks_used() -> None:
    with pytest.raises(OverlayPayloadError):
        MapStream(
            map="yoloe_masks",
            backend="yoloe-26n-seg.pt",
            payload_path="/tmp/preview_masks",
            payload_bytes=1,
            preview_path="/tmp/preview_masks",
            preview_bytes=1,
            duration_s=1.0,
            n_frames=1,
            fps=30.0,
            extract_ms_p50=0.0,
            extract_ms_p95=0.0,
            pack_ms_p50=0.0,
            codec_ms_p50=0.0,
            decode_ms_p50=0.0,
            gpu="cpu",
        )


def test_yoloe_cli_import_error_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise ImportError(YOLOE_INSTALL_HINT)

    monkeypatch.setattr("demo.pipeline.maps.yoloe_masks.run_yoloe_clip", boom)
    code = yoloe_main(["--clip", str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "out")])
    assert code == 2


def test_sam_cli_import_error_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise ImportError("ultralytics.SAM is not importable")

    monkeypatch.setattr("demo.pipeline.maps.sam31_masks.run_sam31_clip", boom)
    code = sam_main(["--clip", str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "out")])
    assert code == 2


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as caught:
        yoloe_main(["--help"])
    assert caught.value.code == 0
    with pytest.raises(SystemExit) as caught:
        sam_main(["--help"])
    assert caught.value.code == 0


@pytest.mark.integration
def test_live_yoloe_weights_are_local_files_not_hub_ids() -> None:
    path = MODELS["yoloe26_seg"]
    if path is None:
        pytest.skip("MODELS['yoloe26_seg'] missing — do not auto-download")
    assert path.is_file()
    assert path.suffix == ".pt"
    assert "yoloe-26" in path.name
    assert path.is_absolute()


@pytest.mark.integration
def test_live_sam3_weights_are_local_files() -> None:
    path = MODELS["sam3"]
    if path is None:
        pytest.skip("MODELS['sam3'] missing — do not auto-download")
    assert path.is_file()
    assert path.name == "sam3.pt"
    assert MODELS["sam31"] is None or MODELS["sam31"].name == "sam3.1_multiplex.pt"


@pytest.mark.integration
def test_live_yoloe_require_does_not_use_a_hub_id() -> None:
    path = MODELS["yoloe26_seg"]
    if path is None:
        pytest.skip("MODELS['yoloe26_seg'] missing — do not auto-download")
    from demo.pipeline.maps.model_paths import require

    resolved = require("yoloe26_seg")
    assert resolved == path
    assert resolved.is_absolute()
    assert "/" not in resolved.name
