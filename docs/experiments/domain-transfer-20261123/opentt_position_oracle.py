"""Charge an official OpenTTGames position trajectory on a measured, matched-rate AV1 base."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
from statistics import mean
import subprocess
import time

import cv2
import numpy as np
import torch

from experiments.tennis import cohort_ball_rate, k5d_opentt_join_witness as native_join
from experiments.tennis import k5d_opentt_source as source
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream, codec, tracked_patch


HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "k5d-inputs"
FIRST, N, FPS = 1066, 240, 120
W, H = 1280, 720
BYTES = 16000
VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
WITNESS_SHA256 = "7ae07b383514ffa0f96286dc22f900446c15d3b3376e4a50027183500f05736d"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cpu_seconds():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def run(command, timeout=120):
    return subprocess.check_output(command, timeout=timeout)


def check_source(video, labels, ffprobe, ffmpeg):
    clip = source.metadata(
        INPUTS / "opentt-source-selection-20261104.json", source.SELECTION_SHA, "test_2", labels
    )
    assert sha(video) == clip["video_sha256"]
    assert (
        clip["start_zero_based_provisional"],
        clip["visible"],
        clip["absent"],
        clip["unknown"],
    ) == (FIRST, 85, 6, 149)
    record = json.loads(
        run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_streams",
                "-show_entries",
                "packet=pts:stream=width,height,time_base,nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    native = record["streams"][0]
    step = 1 / (Fraction(native["time_base"]) * FPS)
    assert (native["width"], native["height"], Fraction(native["avg_frame_rate"])) == (
        1920,
        1080,
        FPS,
    )
    assert step.denominator == 1 and int(native["nb_frames"]) == clip["video_frames"]
    assert sorted(int(p["pts"]) for p in record["packets"]) == list(
        range(0, int(native["nb_frames"]) * int(step), int(step))
    )
    evidence = INPUTS / "opentt-test2-join-evidence-20261105.json"
    witness = source.join_witness(
        INPUTS / "opentt-test2-reviewed-join-20261105.json", WITNESS_SHA256, clip, evidence
    )
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.isOpened() and float(cap.get(cv2.CAP_PROP_FPS)) == FPS
        published = json.loads(evidence.read_text())["frames"]
        assert sha(evidence) == witness["evidence_sha256"]
        base = Fraction(native["time_base"])
        ticks = list(range(1085, 1088))
        timestamp = [tick * int(step) for tick in ticks]
        assert [r["pts"] for r in published] == timestamp
        sequential, _, pts = native_join.extract(Path(ffmpeg), video, 1085, None, base, timestamp)
        fast, _, fast_pts = native_join.extract(Path(ffmpeg), video, 1085, 9, base, timestamp)
        assert sequential == fast and pts == fast_pts == timestamp
        verified = []
        decoded = source.source_rgb(cap, cv2, 1085, 3)
        for original, local, prior in zip(sequential, decoded, published, strict=True):
            rgb = np.frombuffer(original, np.uint8).reshape(local.shape)
            difference = int(np.max(np.abs(rgb.astype(np.int16) - local.astype(np.int16))))
            assert difference <= 4
            verified.append(
                dict(
                    tick=prior["frame_index"],
                    pts=prior["pts"],
                    source_label=prior["label_xy_source"],
                    previous_linux_rgb_sha256=prior["raw_rgb_sha256"],
                    local_ffmpeg_rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                    local_opencv_rgb_sha256=hashlib.sha256(local).hexdigest(),
                    maximum_local_rgb_difference=difference,
                )
            )
        return (
            clip,
            native,
            dict(local_source_join_passed=True, sequential_seek_exact=True, frames=verified),
        )
    finally:
        cap.release()


def source_frames(video, destination):
    cap = cv2.VideoCapture(str(video))
    full = np.memmap(destination, dtype=np.uint8, mode="w+", shape=(N, H, W, 3))
    try:
        hashes = []
        for i, original in enumerate(source.source_rgb(cap, cv2, FIRST, N)):
            full[i] = cv2.resize(original, (W, H), interpolation=cv2.INTER_AREA)
            hashes.append(hashlib.sha256(full[i]).hexdigest())
    finally:
        cap.release()
    full.flush()
    assert len(hashes) == N
    return full, hashes


def payload_for(clip):
    labels = {r["frame"]: r for r in clip["labels"]}
    centers = []
    kinds = []
    for tick in range(FIRST, FIRST + N):
        row = labels.get(tick)
        kind = -1 if row is None else row["kind"]
        center = (round(row["x"] * 2 / 3), round(row["y"] * 2 / 3)) if kind == 1 else None
        assert center is None or 0 <= center[0] < W and 0 <= center[1] < H
        centers.append(center)
        kinds.append(kind)
    assert [kinds.count(k) for k in (1, 0, -1)] == [85, 6, 149]
    appearance = ball.Appearance((235, 235, 235), 2, 0, 3)
    looks = [appearance if center is not None else None for center in centers]
    payload = ball.encode_frames(centers, looks, W, H)
    assert ball.decode_frames(payload, N, W, H) == (centers, looks)
    return payload, centers, kinds


def price(width, height, raw, payload, tools, scratch, deadline):
    ffmpeg, encoder = (tools[k]["path"] for k in ("ffmpeg", "SvtAv1EncApp"))
    tiny = scratch / f"source-{width}.yuv"
    run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(FPS),
            "-i",
            str(raw),
            "-vf",
            f"scale={width}:{height}:flags=area:out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-f",
            "rawvideo",
            str(tiny),
        ]
    )
    assert tiny.stat().st_size == N * width * height * 3 // 2
    bank = {}
    paths = {}
    resource_id = f"svtav1-4.0.1-scaled-{width}x{height}"

    def pack(data):
        return ball_stream.pack(
            ball_stream.Stream(W, H, Fraction(FPS), N, "svt-av1-ivf", resource_id, data, payload)
        )

    def evaluate(q):
        if q not in bank:
            path = scratch / f"{width}-{q}.ivf"
            timing = svtav1.run(
                svtav1.encode_command(
                    encoder,
                    tiny,
                    path,
                    width=width,
                    height=height,
                    fps=Fraction(FPS),
                    frames=N,
                    crf=q / 4,
                    preset=4,
                    full_range=False,
                ),
                90,
            )
            data = path.read_bytes()
            codec.check_native(
                {"width": width, "height": height, "fps": [FPS, 1], "frames": N}, data
            )
            priced = pack(data)
            bank[q] = dict(
                width=width,
                crf=q / 4,
                bytes=len(data),
                ball_bytes=len(priced),
                native_sha256=sha(path),
                ball_sha256=hashlib.sha256(priced).hexdigest(),
                timing=timing,
            )
            paths[q] = path
            assert cpu_seconds() < deadline - 45
        return bank[q]

    for field in ("bytes", "ball_bytes"):
        cohort_ball_rate.search(evaluate, field, BYTES)
    pairs = [
        (a, b)
        for a in bank.values()
        for b in bank.values()
        if a["ball_bytes"] <= BYTES and b["bytes"] <= BYTES
    ]
    if not pairs:
        return dict(points=list(bank.values()), chosen=None)
    candidate, ordinary = min(pairs, key=lambda v: cohort_ball_rate.pair_order(*v, BYTES))
    return dict(
        points=list(bank.values()),
        chosen=(candidate, ordinary),
        candidate_path=paths[int(candidate["crf"] * 4)],
        ordinary_path=paths[int(ordinary["crf"] * 4)],
        pack=pack,
    )


def patch(original, xy, side):
    radius = side // 2
    x, y = xy
    out = original[max(0, y - radius) : min(H, y + radius), max(0, x - radius) : min(W, x + radius)]
    return cv2.copyMakeBorder(
        out,
        max(0, radius - y),
        max(0, y + radius - H),
        max(0, radius - x),
        max(0, x + radius - W),
        cv2.BORDER_REPLICATE,
    )


def tensor(crop):
    return torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None].float() / 255


def distance(net, a, b):
    s, d = tensor(a), tensor(b)
    return quality.masked_dists(net, d, s, torch.ones_like(s[:, :1], dtype=torch.bool)).item()


def mse(a, b):
    return float(np.mean(np.square(a.astype(np.float32) - b.astype(np.float32)), dtype=np.float64))


def psnr(value):
    return 10 * math.log10(255**2 / value)


def score(net, original, frames, centers, kinds, looks, velocity, width, height, deadline):
    rows = []
    for t, reference in enumerate(original):
        plain, base = [
            cv2.resize(
                tracked_patch.yuv_to_rgb(v[t], width, height, False),
                (W, H),
                interpolation=cv2.INTER_CUBIC,
            )
            for v in frames
        ]
        candidate = ball.render(base, t, centers, velocity, looks)
        assert np.array_equal(candidate, base) or kinds[t] == 1
        frame = dict(tick=FIRST + t, kind=kinds[t], center=centers[t], scores={})
        for name, decoded in (("av1", plain), ("same_base", base), ("pointstream", candidate)):
            row = {
                "whole_rgb_mse": mse(reference, decoded),
                "rgb_sha256": hashlib.sha256(decoded).hexdigest(),
            }
            if centers[t] is not None:
                x, y = patch(reference, centers[t], 64), patch(decoded, centers[t], 64)
                row.update(
                    dists64=distance(net, x, y),
                    mse64=mse(x, y),
                    mse12=mse(x[26:38, 26:38], y[26:38, 26:38]),
                )
            frame["scores"][name] = row
        rows.append(frame)
        assert cpu_seconds() < deadline
    return rows


def main(args):
    began = cpu_seconds()
    wall = time.monotonic()
    deadline = began + 600
    args.out.mkdir()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert sha(args.vgg) == VGG_SHA256
    tools = {name: svtav1.tool(name) for name in ("ffmpeg", "ffprobe", "dav1d", "SvtAv1EncApp")}
    assert "v4.0.1" in tools["SvtAv1EncApp"]["version"]
    clip, native, verified = check_source(
        args.video, args.labels, tools["ffprobe"]["path"], tools["ffmpeg"]["path"]
    )
    raw = args.out / "source.rgb"
    original, original_sha256 = source_frames(args.video, raw)
    payload, centers, kinds = payload_for(clip)
    net = quality.load_dists(args.vgg, "cpu")
    sample = patch(original[0], centers[0], 64)
    assert sample.shape == (64, 64, 3) and abs(distance(net, sample, sample)) < 1e-5
    prepared_seconds = cpu_seconds() - began
    assert 1.5 * prepared_seconds + 90 < 600
    search = []
    for width, height in ((320, 180), (160, 90)):
        result = price(width, height, raw, payload, tools, args.out, deadline)
        search.append(dict(width=width, height=height, points=result["points"]))
        if result["chosen"] is not None:
            break
    if result["chosen"] is None:
        (args.out / "search.json").write_text(json.dumps(search, indent=2))
        raise RuntimeError("No physical affordable pair")
    candidate, ordinary = result["chosen"]
    packet = result["pack"](result["candidate_path"].read_bytes())
    (args.out / "trajectory.psbl").write_bytes(packet)
    received = ball_stream.unpack(packet)
    received_centers, looks = ball.decode_frames(
        received.ball_payload, received.frames, received.width, received.height
    )
    assert received_centers == centers and received.fps == FPS and received.frames == N
    assert (received.width, received.height, received.duration) == (W, H, 2)
    assert (
        candidate["ball_bytes"] == len(packet)
        and hashlib.sha256(packet).hexdigest() == candidate["ball_sha256"]
    )
    decoded_paths = []
    native_receipts = {}
    for name, stream in (
        ("ordinary", result["ordinary_path"].read_bytes()),
        ("pointstream_base", received.native),
    ):
        meta = {"width": width, "height": height, "fps": [FPS, 1], "frames": N}
        path = args.out / f"display-{name}.yuv"
        repeated = args.out / f"repeat-{name}.yuv"
        for target in (path, repeated):
            rec = tracked_patch.decode_native(stream, target, meta, tools["dav1d"]["path"])
            if target == path:
                native_receipts[name] = rec
            else:
                assert rec["decoded_yuv_sha256"] == native_receipts[name]["decoded_yuv_sha256"]
        decoded_paths.append(
            np.memmap(path, dtype=np.uint8, mode="r", shape=(N, height * 3 // 2, width))
        )
    velocity = ball.velocities(received_centers)
    rows = score(
        net,
        original,
        decoded_paths,
        received_centers,
        kinds,
        looks,
        velocity,
        width,
        height,
        deadline,
    )
    summary = {}
    for mode in ("av1", "same_base", "pointstream"):
        all_rows = [r["scores"][mode] for r in rows]
        visible = [r["scores"][mode] for r in rows if r["kind"] == 1]
        summary[mode] = dict(
            whole_rgb_psnr=psnr(mean(r["whole_rgb_mse"] for r in all_rows)),
            dists64=mean(r["dists64"] for r in visible),
            rgb_psnr64=psnr(mean(r["mse64"] for r in visible)),
            rgb_psnr12=psnr(mean(r["mse12"] for r in visible)),
        )
    maximum_error = max(
        math.hypot(
            centers[r["frame"] - FIRST][0] - r["x"] * 2 / 3,
            centers[r["frame"] - FIRST][1] - r["y"] * 2 / 3,
        )
        for r in clip["labels"]
        if r["kind"] == 1
    )
    record = dict(
        scope="One reviewed, label-driven OpenTTGames transfer screen",
        revision=sha(Path(__file__)),
        tools=tools,
        model_weights_sha256=VGG_SHA256,
        opencv=cv2.__version__,
        torch=torch.__version__,
        source_sha256=clip["video_sha256"],
        label_zip_sha256=clip["label_archive_sha256"],
        label_markup_sha256=clip["ball_member_sha256"],
        reviewed_witness_sha256=WITNESS_SHA256,
        native=native,
        source_interval=[FIRST, FIRST + N],
        exact_decode_witness=verified,
        source_display_frame_sha256=original_sha256,
        frame_count=N,
        fps=[FPS, 1],
        cap_bytes=BYTES,
        search=search,
        delivered=dict(ordinary=ordinary, pointstream=candidate),
        payload_bytes=len(payload),
        displayed_position_error_max=maximum_error,
        serialized_visible=85,
        explicit_absent=6,
        unknown=149,
        native_replays=native_receipts,
        summary=summary,
        rows=rows,
        cpu_seconds=cpu_seconds() - began,
        wall_seconds=time.monotonic() - wall,
    )
    (args.out / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                pointstream_bytes=len(packet),
                ordinary_bytes=ordinary["bytes"],
                metrics=summary,
                elapsed_seconds=record["wall_seconds"],
                cpu_seconds=record["cpu_seconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "labels", "vgg", "out"):
        p.add_argument(name, type=Path)
    main(p.parse_args())
