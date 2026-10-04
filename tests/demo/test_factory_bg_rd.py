"""Factory background training set and segment cuts."""

from __future__ import annotations

from pathlib import Path

from demo.experiments.factory_bg_rd import (
    CLIP3_DROP_S,
    HT_LONG_SEGMENT,
    av1_segment_cmd,
    chunk_spans,
    hnerv_decoder_params,
    hnerv_embed_dim,
    hnerv_import_stub,
    hnerv_modelsize_for_segment,
    uf_test_config,
    kept_starts,
    segment_starts,
    segments_for,
    source_frame_names,
    write_video_folder,
)
from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_PRESET


def test_clip3_aisle_and_blown_out_seconds_stay_out() -> None:
    batches = [
        {"start_s": 0.0, "train": True},
        {"start_s": 210.0, "train": False},
        {"start_s": 240.0, "train": False},
        {"start_s": 420.0, "train": False},
        {"start_s": 390.0, "train": True},
    ]
    assert kept_starts(batches, CLIP3_DROP_S) == [0.0, 390.0]


def test_source_names_follow_the_original_frame_index() -> None:
    assert source_frame_names(30.0)[0] == "000900.jpg"
    assert source_frame_names(30.0)[-1] == "000929.jpg"


def test_video_folder_uses_one_shared_frame_list(tmp_path: Path) -> None:
    src = tmp_path / "inpainted"
    src.mkdir()
    frames = []
    for index in range(30):
        path = src / f"src_{index:02d}.jpg"
        path.write_bytes(b"x")
        frames.append(path)
    dest = write_video_folder(tmp_path / "set", [("clip_t000000", frames)])
    description = __import__("json").loads((dest / "description.json").read_text())
    assert description["frames"][0] == "000000.jpg"
    assert description["frames"][-1] == "000029.jpg"
    assert (dest / "clip_t000000" / "000000.jpg").resolve() == frames[0].resolve()


def test_segments_are_the_same_cuts_for_av1_and_the_neural_bitstream() -> None:
    assert segment_starts(300, 1) == list(range(300))
    assert segment_starts(300, 300) == [0]
    assert segment_starts(300, 32)[-1] == 256
    assert 300 not in segments_for("hts")
    assert HT_LONG_SEGMENT in segments_for("htl")
    assert segments_for("ld")[-1] == 300


def test_av1_segment_keeps_crf_63_and_sets_the_gop_to_the_segment() -> None:
    cmd = av1_segment_cmd(Path("in.mp4"), Path("out.mp4"), scale="426:240", gop=32)
    assert "libsvtav1" in cmd
    assert AV1_CRF in cmd
    assert AV1_PRESET in cmd
    assert cmd[cmd.index("-g") + 1] == "32"
    assert "keyint=32:keyint-min=32" in cmd


def test_holdout_chunks_cover_ten_seconds_and_keep_a_short_tail() -> None:
    assert chunk_spans(300)[0] == (0, 30)
    assert chunk_spans(300)[-1] == (270, 300)
    assert len(chunk_spans(300)) == 10
    assert chunk_spans(299)[-1] == (270, 299)
    assert chunk_spans(22) == []


def test_uf_score_config_is_png_with_one_intra_for_the_segment() -> None:
    config = uf_test_config(Path("/tmp/src"), {"s0": {"height": 1080, "width": 1920, "intra_period": -1, "frames": 8}})
    holdout = config["test_classes"]["holdout"]
    assert holdout["src_type"] == "png"
    assert holdout["sequences"]["s0"]["intra_period"] == -1
    assert holdout["sequences"]["s0"]["frames"] == 8


def test_hnerv_segment_keeps_the_trained_decoder_budget() -> None:
    # factory002 flat set is 1200 frames. 0.5 * 1.5e6 / 1200 / 144 == 4.34.
    assert hnerv_embed_dim(1200) == 4
    assert hnerv_decoder_params(1200) == 808_800
    assert hnerv_modelsize_for_segment(1200, 1200) == 1.5
    assert hnerv_embed_dim(1440) == 3


def test_hnerv_stub_satisfies_the_unused_encoded_video_import(tmp_path: Path) -> None:
    root = hnerv_import_stub(tmp_path)
    encoded = (root / "pytorchvideo" / "data" / "encoded_video.py").read_text()
    assert "class EncodedVideo" in encoded
    assert "class VideoReader" in (root / "decord.py").read_text()
