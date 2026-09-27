from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo.evaluation.pose_backends import (
    BACKENDS,
    COCO_WB_BODY,
    COCO_WB_FACE,
    COCO_WB_FEET,
    COCO_WB_LEFT_HAND,
    COCO_WB_RIGHT_HAND,
    FrameWholeBody,
    WholeBodyPerson,
    extract_rtm_wholebody_hands,
    people_from_rtm,
    wholebody_to_frame_hands,
)
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.dwpose import (
    MAGIC_BODY,
    MAGIC_COMBINED,
    MAGIC_FACE,
    LandmarkInstance,
    compress_body,
    compress_face,
    decompress_body,
    decompress_face,
    instance_from_part,
    main as dwpose_main,
    pack_combined_payload,
    pack_frame_parts,
    read_packet_stream,
    write_packet_stream,
)
from demo.pipeline.maps.pose_delta import decode_pose_stream, encode_dwb1, transcode_dwb1


def _person() -> WholeBodyPerson:
    kpts = np.zeros((133, 3), dtype=np.float32)
    # body 17
    for i in range(17):
        kpts[i] = (100 + i * 8, 80 + (i % 5) * 12, 0.9)
    # feet
    for i in range(17, 23):
        kpts[i] = (120 + i, 220, 0.8)
    # face 68
    for i in range(23, 91):
        kpts[i] = (160 + (i % 10) * 3, 40 + (i // 10), 0.85)
    # left hand 21
    for i in range(21):
        kpts[91 + i] = (60 + i * 4, 140 + (i % 4) * 5, 0.95)
    # right hand 21
    for i in range(21):
        kpts[112 + i] = (400 + i * 4, 150 + (i % 4) * 5, 0.92)
    return WholeBodyPerson(keypoints=kpts)


def test_coco_wholebody_slices() -> None:
    person = _person()
    assert person.body.shape == (17, 3)
    assert person.feet.shape == (6, 3)
    assert person.face.shape == (68, 3)
    assert person.left_hand.shape == (21, 3)
    assert person.right_hand.shape == (21, 3)
    assert COCO_WB_BODY == slice(0, 17)
    assert COCO_WB_FEET == slice(17, 23)
    assert COCO_WB_FACE == slice(23, 91)
    assert COCO_WB_LEFT_HAND == slice(91, 112)
    assert COCO_WB_RIGHT_HAND == slice(112, 133)


def test_people_from_rtm_splits_without_rtmlib() -> None:
    kpts = np.zeros((2, 133, 2), dtype=np.float32)
    scores = np.ones((2, 133), dtype=np.float32) * 0.7
    kpts[0, 0] = (10, 20)
    people = people_from_rtm(kpts, scores)
    assert len(people) == 2
    assert people[0].body[0, 0] == pytest.approx(10)
    assert people[0].body[0, 2] == pytest.approx(0.7)
    assert people[0].face.shape[0] == 68


def test_backends_keep_existing_and_alias_dwpose() -> None:
    assert BACKENDS["rtm_wholebody"] is extract_rtm_wholebody_hands
    assert "mp_live" in BACKENDS
    assert "mp_offline_gt" in BACKENDS
    assert "rtm_hand" in BACKENDS
    assert "hamer" in BACKENDS
    assert BACKENDS["dwpose"].__name__ == "extract_dwpose_wholebody"


def test_face_pf_roundtrip() -> None:
    bbox = [int((20 / 255.0) * 1920), int((10 / 255.0) * 1080), int((80 / 255.0) * 1920), int((70 / 255.0) * 1080)]
    x1, y1, x2, y2 = bbox
    inst = LandmarkInstance(
        confidence=0.8,
        bbox=bbox,
        landmarks_pixel=[[x1 + (i % 10) * 8, y1 + (i // 10) * 6] for i in range(68)],
    )
    packet = compress_face([inst], 1920, 1080)
    assert packet[:2] == MAGIC_FACE
    back = decompress_face(packet, 1920, 1080)
    assert len(back) == 1
    assert len(back[0].landmarks_pixel) == 68
    orig = np.array(inst.landmarks_pixel)
    rec = np.array(back[0].landmarks_pixel)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    assert np.max(np.abs(orig - rec)) < max(bw, bh) / 255.0 + 1.5


def test_body_pb_roundtrip() -> None:
    bbox = [int((15 / 255.0) * 1920), int((15 / 255.0) * 1080), int((90 / 255.0) * 1920), int((90 / 255.0) * 1080)]
    x1, y1, x2, y2 = bbox
    inst = LandmarkInstance(
        confidence=0.9,
        bbox=bbox,
        landmarks_pixel=[[x1 + 4 + i * 6, y1 + 4 + i * 5] for i in range(17)],
    )
    packet = compress_body([inst], 1920, 1080)
    assert packet[:2] == MAGIC_BODY
    back = decompress_body(packet, 1920, 1080)
    assert len(back) == 1
    assert len(back[0].landmarks_pixel) == 17
    orig = np.array(inst.landmarks_pixel)
    rec = np.array(back[0].landmarks_pixel)
    assert np.mean(np.abs(orig - rec)) < 3.0


def test_hands_keypoint_compressor_with_coco_wb_topology() -> None:
    person = _person()
    frame = FrameWholeBody(frame_idx=0, people=[person], width=1920, height=1080)
    hands = wholebody_to_frame_hands(frame)
    assert len(hands.hands) == 2
    packet = KeypointCompressor.compress_frame(hands, 1920, 1080)
    assert packet[:2] == KeypointCompressor.MAGIC
    rec = KeypointCompressor.decompress_frame(packet, 1920, 1080)
    assert len(rec) == 2
    sidecar = MapStream(
        map="dwpose_hands",
        backend="dw-ll_ucoco_384.onnx",
        payload_path="/tmp/dwpose_hands.pk.bin",
        payload_bytes=len(packet),
        preview_path="/tmp/preview_dwpose_hands.mp4",
        preview_bytes=10,
        duration_s=1.0,
        n_frames=1,
        fps=30.0,
        extract_ms_p50=1.0,
        extract_ms_p95=2.0,
        pack_ms_p50=0.1,
        codec_ms_p50=0.0,
        decode_ms_p50=0.1,
        gpu="cpu",
        extra={"topology": "coco_wb_hand21"},
    )
    assert sidecar.to_sidecar()["topology"] == "coco_wb_hand21"


def test_instance_from_part_rejects_flying_noise() -> None:
    noise = np.zeros((68, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    noise[:, 0] = rng.uniform(0, 1920, 68)
    noise[:, 1] = rng.uniform(0, 1080, 68)
    noise[:, 2] = 0.12
    assert instance_from_part(noise, 1920, 1080, conf_thr=0.4, min_visible=12) is None
    blank = np.zeros((17, 3), dtype=np.float32)
    assert instance_from_part(blank, 1920, 1080, conf_thr=0.5, min_visible=5) is None


def test_pack_frame_parts_three_magics() -> None:
    frame = FrameWholeBody(frame_idx=3, people=[_person()], width=1920, height=1080)
    hand_pkt, face_pkt, body_pkt, hands, face_inst, body_inst = pack_frame_parts(frame)
    assert hand_pkt[:2] == b"PK"
    assert face_pkt[:2] == b"PF"
    assert body_pkt[:2] == b"PB"
    assert isinstance(hands, FrameHandPose)
    assert len(face_inst) == 1 and len(face_inst[0].landmarks_pixel) == 68
    assert len(body_inst) == 1 and len(body_inst[0].landmarks_pixel) == 17
    assert instance_from_part(frame.people[0].face, 1920, 1080) is not None


def test_packet_stream_roundtrip(tmp_path: Path) -> None:
    packets = [compress_body([], 64, 64), compress_body([], 64, 64)]
    path = write_packet_stream(packets, tmp_path / "dwpose_body.pb.bin")
    back = read_packet_stream(path)
    assert back == packets
    assert packets[0][:2] == MAGIC_BODY


def _moving_frames(n: int, *, hands: int = 1, face: bool = False, body: bool = False) -> list[tuple[bytes, bytes, bytes]]:
    frames: list[tuple[bytes, bytes, bytes]] = []
    for i in range(n):
        shift = i % 7
        hand_inst = []
        for h in range(hands):
            x1 = 200 + h * 80 + shift
            y1 = 300 + (i % 3)
            x2 = x1 + 140
            y2 = y1 + 160
            landmarks = [[x1 + 4 + (k % 5) + shift, y1 + 6 + k] for k in range(21)]
            hand_inst.append(
                SingleHand(
                    handedness="Right" if h == 0 else "Left",
                    confidence=0.9,
                    bbox=[x1, y1, x2, y2],
                    landmarks_norm=[],
                    landmarks_pixel=landmarks,
                )
            )
        hand_pkt = KeypointCompressor.compress_frame(FrameHandPose(frame_idx=i, hands=hand_inst), 1920, 1080)
        face_pkt = compress_face([], 1920, 1080)
        body_pkt = compress_body([], 1920, 1080)
        if face and i % 5 == 0:
            bbox = [400, 80, 520, 200]
            face_pkt = compress_face(
                [
                    LandmarkInstance(
                        confidence=0.8,
                        bbox=bbox,
                        landmarks_pixel=[[bbox[0] + (k % 8) * 3, bbox[1] + k // 2] for k in range(68)],
                    )
                ],
                1920,
                1080,
            )
        if body:
            x1, y1 = 100 + shift, 60
            body_pkt = compress_body(
                [
                    LandmarkInstance(
                        confidence=0.85,
                        bbox=[x1, y1, x1 + 200, y1 + 400],
                        landmarks_pixel=[[x1 + 8 + k, y1 + 10 + k * 4] for k in range(17)],
                    )
                ],
                1920,
                1080,
            )
        frames.append((hand_pkt, face_pkt, body_pkt))
    return frames


def test_wholebody_euro_damps_a_one_frame_spike() -> None:
    from demo.evaluation.pose_backends import WholeBodyEuroSmoother, WholeBodyPerson

    smoother = WholeBodyEuroSmoother()
    base = np.zeros((133, 3), dtype=np.float32)
    base[:, 0] = 100 + np.arange(133) * 0.4
    base[:, 1] = 80 + (np.arange(133) % 17)
    base[:, 2] = 0.9
    spike = base.copy()
    spike[:, 0] = base[:, 0] + 12
    held = [base, spike, base.copy(), base.copy()]
    out = [smoother.smooth([WholeBodyPerson(keypoints=frame)], i, fps=30.0)[0].keypoints for i, frame in enumerate(held)]
    assert out[0][0, 0] == pytest.approx(float(base[0, 0]))
    assert float(out[1][0, 0]) < float(spike[0, 0])
    assert float(out[1][0, 0]) > float(base[0, 0])
    assert abs(float(out[3][0, 0]) - float(base[0, 0])) < abs(float(out[1][0, 0]) - float(base[0, 0]))


def test_dwb2_roundtrip_and_omits_absent_parts() -> None:
    blank = [
        (
            KeypointCompressor.compress_frame(FrameHandPose(frame_idx=i, hands=[]), 1920, 1080),
            compress_face([], 1920, 1080),
            compress_body([], 1920, 1080),
        )
        for i in range(30)
    ]
    packed = pack_combined_payload([f[0] for f in blank], [f[1] for f in blank], [f[2] for f in blank])
    assert packed[:4] == MAGIC_COMBINED == b"DWB2"
    assert decode_pose_stream(packed) == blank
    legacy = encode_dwb1(blank)
    assert len(packed) * 10 < len(legacy)

    moving = _moving_frames(40, hands=1, face=True, body=True)
    packed_m = pack_combined_payload([f[0] for f in moving], [f[1] for f in moving], [f[2] for f in moving])
    assert decode_pose_stream(packed_m) == moving
    assert len(packed_m) < len(encode_dwb1(moving))
    assert transcode_dwb1(encode_dwb1(moving)) == packed_m


def test_empty_frame_still_has_magic() -> None:
    assert compress_face([], 100, 100) == MAGIC_FACE + bytes([0])
    assert compress_body([], 100, 100) == MAGIC_BODY + bytes([0])
    empty_pose = FrameHandPose(frame_idx=0, hands=[])
    pkt = KeypointCompressor.compress_frame(empty_pose, 100, 100)
    assert pkt[:2] == b"PK" and pkt[2] == 0


def test_dwpose_cli_exits_2_when_onnx_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(key: str) -> Path:
        raise FileNotFoundError(
            f"MODELS[{key!r}] is missing under /home/itec/emanuele/Models. "
            f"Known missing keys: ['dwpose_pose']. Do not auto-download."
        )

    monkeypatch.setattr("demo.pipeline.maps.dwpose.require", _missing)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    code = dwpose_main(["--clip", str(clip), "--out", str(tmp_path / "o")])
    assert code == 2
    err = capsys.readouterr().err
    assert "Do not auto-download" in err


def test_dwpose_sidecars_are_native(tmp_path: Path) -> None:
    for name in ("dwpose_hands", "dwpose_face", "dwpose_body"):
        payload = tmp_path / f"{name}.bin"
        payload.write_bytes(b"PK\x00")
        stream = MapStream(
            map=name,
            backend="dw-ll_ucoco_384.onnx",
            payload_path=str(payload),
            payload_bytes=3,
            preview_path=str(tmp_path / f"preview_{name}.mp4"),
            preview_bytes=99,
            duration_s=1.0,
            n_frames=1,
            fps=30.0,
            extract_ms_p50=1.0,
            extract_ms_p95=2.0,
            pack_ms_p50=0.1,
            codec_ms_p50=0.0,
            decode_ms_p50=0.1,
            gpu="cpu",
            extra={"topology": "coco_wb_hand21"} if name.endswith("hands") else {},
        )
        text = write_sidecar(stream, tmp_path / f"{name}.json").read_text()
        assert f'"map": "{name}"' in text
        if name.endswith("hands"):
            assert "coco_wb_hand21" in text


@pytest.mark.integration
def test_dwpose_cli_integration_requires_rtmlib_and_onnx() -> None:
    pytest.importorskip("rtmlib")
    from demo.pipeline.maps.model_paths import MODELS

    if MODELS.get("dwpose_pose") is None or MODELS.get("dwpose_det") is None:
        pytest.skip("DWPose ONNX missing")
    pytest.skip("requires a real clip on gpu1")
