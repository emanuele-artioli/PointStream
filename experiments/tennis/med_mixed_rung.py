"""Source-denied rerasterization of a paid base with bit-identical opaque foreground."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zlib

import cv2
import numpy as np

from experiments.tennis import common_al028_finish_isolation as sandbox
from experiments.tennis import common_player_al028_finish as finish
from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis import med_split_native as split
from experiments.tennis import w8_receiver_native_bootstrap as native
from experiments.tennis.player_requalification import array_sha, sha256

SYDNEY_REVIEW_TICKS = (9, 32, 44, 68, 92, 115, 127, 151, 175, 198, 210, 234)
REVIEW_SHAPE = (1080, 1920, 3)
REVIEW_FRAME_BYTES = 1080 * 1920 * 3


def config_for(a):
    channel = mixed.decode(a.joint, a.cache, a.carrier, a.carrier_sha256, allow_wider_base=True)
    root = a.witness / "publish"
    options = dict(
        carrier=str(a.carrier),
        carrier_sha256=a.carrier_sha256,
        manifest=str(a.manifest),
        ffmpeg=str(Path(sys.prefix) / "bin/ffmpeg"),
        joint=str(a.joint),
        cache=str(a.cache),
        original_joint=str(a.original_joint),
        witness_frames=str(root / "frames.npy"),
        witness_receipt=str(root / "finish.json"),
        masks={n: str(root / (n + "-masks.npz")) for n in channel["profile"]["order"]},
        ticks=mixed.smoke_ticks(channel["start"], channel["stop"])
        if a.frames == 4
        else list(range(240)),
    )
    if a.split:
        options["split"] = str(a.split)
    assert a.frames == {"smoke": 4, "full": 240}[os.environ["PS_STAGE"]]
    inputs = [
        Path(v) for k, v in options.items() if k not in ("carrier_sha256", "masks", "ticks")
    ] + [Path(v) for v in options["masks"].values()]
    return options, inputs


def run(c, proof):
    finish._isolation(proof)
    if os.environ["CUDA_VISIBLE_DEVICES"] or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("requires one source-denied CPU")
    cv2.setNumThreads(1)
    started = time.monotonic()
    channel = mixed.decode(
        Path(c["joint"]),
        Path(c["cache"]),
        Path(c["carrier"]),
        c["carrier_sha256"],
        allow_wider_base=True,
    )
    old = Path(c["original_joint"]).read_bytes()
    new = Path(c["joint"]).read_bytes()
    op = packet._embedded_profile(old)
    revised_profile = packet._embedded_profile(new)
    comparison = {
        **revised_profile,
        "player": {**revised_profile["player"], "coded_size": op["player"]["coded_size"]},
    }
    if op != comparison or tuple(op["player"]["coded_size"]) != mixed.BASE_SIZE:
        raise ValueError("paid headers differ beyond the native base")
    orig, _ = packet.wire.unpack(old, op)
    revised, _ = packet.wire.unpack(new, revised_profile)
    packet._ivf(orig["base"], *mixed.BASE_SIZE, tuple(op["player"]["recipe"]["fps"]))
    if orig["poses"] != revised["poses"] or orig["trajectory"] != revised["trajectory"]:
        raise ValueError("foreground channels differ between rungs")
    report = json.loads(Path(c["witness_receipt"]).read_text())
    finish._isolation(report["isolation"])
    if (
        report["ticks"] != list(range(240))
        or report["wire"]["joint_sha256"] != packet.wire.sha(old)
        or report["wire"]["cache_sha256"] != channel["wire"]["cache_sha256"]
        or report["order"] != channel["profile"]["order"]
        or not report["kernels"]
    ):
        raise ValueError("missing complete isolated paid GPU witness")
    original_carrier = packet.wire.pack(
        {**channel["profile"], "coded_size": op["player"]["coded_size"]},
        [
            ("base", "frame", orig["base"]),
            ("keypoints", "frame", orig["poses"]),
            *(
                (n, "reference", v)
                for n, v in sorted(
                    packet.wire.unpack(
                        Path(c["cache"]).read_bytes(),
                        packet._embedded_profile(Path(c["cache"]).read_bytes()),
                    )[0].items()
                )
            ),
        ],
    )
    if (
        packet.wire.sha(original_carrier) != report["carrier_sha256"]
        or sha256(Path(c["witness_frames"])) != report["frames_file_sha256"]
    ):
        raise ValueError("original generated and finished displays not bound to paid foreground")
    witness = np.load(c["witness_frames"], mmap_mode="r", allow_pickle=False)
    assert witness.shape == (240, 1080, 1920, 3) and witness.dtype == np.uint8
    masks = {}
    for name, filename in c["masks"].items():
        path = Path(filename)
        assert sha256(path) == report["mask_files"][name]
        with np.load(path, allow_pickle=False) as f:
            assert np.array_equal(f["t"], np.arange(240))
            masks[name] = f["masks"]
        assert masks[name].shape == (240, 784, 512) and masks[name].dtype == bool
    out = Path(c["output"])
    out.mkdir()
    ff = Path(c["ffmpeg"])
    assert sha256(ff) == manifest.NATIVE
    pieces = None
    raw = None
    if "split" in c:
        assert new == old
        pieces, actual_wire = split.decode(Path(c["split"]), old, channel, out, ff)
        coded_sha = actual_wire["joint_sha256"]
        native_sha = ",".join(p[2] for p in pieces)
        ref_rung = False
    else:
        ivf, yuv = out / "base.ivf", out / "base.yuv"
        ivf.write_bytes(revised["base"])
        subprocess.run(
            native.decode_argv(ff, ivf, yuv),
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        width, height = channel["coded_size"]
        rows_per_frame = height * 3 // 2
        if yuv.stat().st_size != 240 * rows_per_frame * width:
            raise ValueError("decoded native base has the wrong size or frame count")
        raw = np.memmap(yuv, np.uint8, "r", shape=(240, rows_per_frame, width))
        actual_wire = channel["wire"]
        coded_sha, native_sha = sha256(ivf), sha256(yuv)
        ref_rung = new == old
    rows = []
    ticks = c["ticks"]
    assert ticks in (mixed.smoke_ticks(channel["start"], channel["stop"]), list(range(240)))
    stage_start = time.monotonic()
    with (out / "frames.zseq").open("xb") as dst:
        for t in ticks:
            if pieces is not None:
                base = split.canvas(pieces, t, channel)
            else:
                assert raw is not None
                base = mixed.canvas(raw[t], t, channel)
            orig_frame = witness[t]
            ref_row = report["rows"][t]
            assert array_sha(orig_frame) == ref_row["frame_sha256"]
            union = np.zeros((1080, 1920), bool)
            if channel["start"] <= t < channel["stop"]:
                local = t - channel["start"]
                for n in channel["profile"]["order"]:
                    union |= finish.sam.project(masks[n][t], channel["geometry"][n][2][local])
            assert int(union.sum()) == ref_row["coverage_pixels"]
            base[union] = orig_frame[union]
            digest = array_sha(base)
            if ref_rung and digest != ref_row["frame_sha256"]:
                assert raw is not None
                current_canvas = mixed.canvas(raw[t], t, channel)
                outside = ~union
                diffs = np.abs(
                    base[outside].astype(np.int16) - orig_frame[outside].astype(np.int16)
                )
                diagnostic = {
                    "tick": t,
                    "native_yuv_sha256": native_sha,
                    "witness_native_yuv_sha256": report["yuv_sha256"],
                    "base_sha256": array_sha(current_canvas),
                    "witness_base_sha256": ref_row["base_sha256"],
                    "frame_sha256": digest,
                    "witness_frame_sha256": ref_row["frame_sha256"],
                    "outside_pixels": int(outside.sum()),
                    "different_outside_pixels": int(np.any(diffs, axis=1).sum()),
                    "different_outside_channels": int(np.count_nonzero(diffs)),
                    "max_outside_channel_change": int(diffs.max(initial=0)),
                    "opencv_version": cv2.__version__,
                    "opencv_cpu_features": cv2.getCPUFeaturesLine(),
                    "opencv_threads": cv2.getNumThreads(),
                    "opencv_optimized": cv2.useOptimized(),
                    "opencv_ipp": cv2.ipp.useIPP(),
                }
                raise ValueError(
                    f"source-denied original-rung replay differs at {t}: {json.dumps(diagnostic)}"
                )
            coded = zlib.compress(base.tobytes(), 1)
            dst.write(struct.pack(">I", len(coded)))
            dst.write(coded)
            rows.append(
                dict(
                    tick=t,
                    frame_sha256=digest,
                    composed_pixels=int(union.sum()),
                    wire_bytes=4 + len(coded),
                )
            )
    if channel["coded_size"] != mixed.BASE_SIZE and all(
        row["frame_sha256"] == report["rows"][row["tick"]]["frame_sha256"] for row in rows
    ):
        raise ValueError("wider coded base did not change any displayed pixels")
    loop = time.monotonic() - stage_start
    result = dict(
        ticks=ticks,
        rows=rows,
        carrier_sha256=c["carrier_sha256"],
        wire=actual_wire,
        witness_sha256=sha256(Path(c["witness_receipt"])),
        original_rung=ref_rung,
        all_witness_frames_matched=ref_rung,
        coded_size=channel["coded_size"],
        codec_input_sha256=coded_sha,
        native_sha256=sha256(ff),
        yuv_sha256=native_sha,
        frames_sha256=sha256(out / "frames.zseq"),
        loop_seconds=loop,
        elapsed_seconds=time.monotonic() - started,
        isolation=proof,
    )
    (out / "rung.json").write_text(json.dumps(result) + "\n")
    return result


def cli_run(a):
    c, inputs = config_for(a)
    path = Path(os.environ["PS_SCRATCH_DIR"]) / "isolated-replay"
    now = time.monotonic()
    r = sandbox.launch(
        source_root=Path(__file__).resolve().parents[2],
        output=path,
        worker_config=c,
        inputs=inputs,
        denied=a.deny,
        seconds=470 if a.frames == 4 else 1220,
        kind="rung",
    )
    assert r["yuv_sha256"] == a.native_sha and r["original_rung"] == a.reference
    stage = Path(os.environ["PS_STAGE_DIR"])
    shutil.copy(path / "frames.zseq", stage / "frames.zseq")
    shutil.copy(path / "rung.json", stage / "rung.json")
    (stage / "timing.json").write_text(json.dumps(dict(seconds=time.monotonic() - now)))
    if a.review_samples:
        export_review_samples(stage, r)


def review_rows(r):
    if r["ticks"] == list(range(240)):
        return [r["rows"][tick] for tick in SYDNEY_REVIEW_TICKS]
    if r["ticks"] == mixed.smoke_ticks(0, 240):
        return r["rows"][:1]
    raise ValueError("expected a complete Sydney replay or its registered smoke ticks")


def export_review_samples(stage, r):
    rows = review_rows(r)
    offset = 0
    with (
        (stage / "frames.zseq").open("rb") as src,
        (stage / "review-samples.zseq").open("xb") as out,
    ):
        for row in r["rows"]:
            if row in rows:
                src.seek(offset)
                block = src.read(row["wire_bytes"])
                if len(block) != row["wire_bytes"]:
                    raise ValueError("truncated complete replay")
                out.write(block)
            offset += row["wire_bytes"]
    report = {
        "rung_sha256": sha256(stage / "rung.json"),
        "frames_sha256": r["frames_sha256"],
        "sample_sha256": sha256(stage / "review-samples.zseq"),
        "rows": rows,
    }
    (stage / "review-samples.json").write_text(json.dumps(report) + "\n")


def read_review_samples(stage, report):
    """Read authenticated RGB samples with a fixed decompression extent."""
    with (stage / "review-samples.zseq").open("rb") as samples:
        for row in report["rows"]:
            header = samples.read(4)
            if len(header) != 4:
                raise ValueError("truncated review sample header")
            n = struct.unpack(">I", header)[0]
            if n + 4 != row["wire_bytes"] or not 0 < n <= REVIEW_FRAME_BYTES + 65536:
                raise ValueError("invalid compressed review sample size")
            block = samples.read(n)
            unpack = zlib.decompressobj()
            raw = unpack.decompress(block, REVIEW_FRAME_BYTES + 1)
            if (
                len(block) != n
                or len(raw) != REVIEW_FRAME_BYTES
                or not unpack.eof
                or unpack.unused_data
                or unpack.unconsumed_tail
            ):
                raise ValueError("invalid review sample payload")
            rgb = np.frombuffer(raw, np.uint8).reshape(REVIEW_SHAPE)
            if array_sha(rgb) != row["frame_sha256"]:
                raise ValueError("review sample pixels differ from the sealed replay")
            yield row["tick"], rgb
        if samples.read(1):
            raise ValueError("trailing data in the review samples")


def review_receipt(stage, r):
    path = stage / "review-samples.json"
    report = json.loads(path.read_text())
    if (
        report["rung_sha256"] != sha256(stage / "rung.json")
        or report["frames_sha256"] != r["frames_sha256"]
        or report["sample_sha256"] != sha256(stage / "review-samples.zseq")
        or report["rows"] != review_rows(r)
    ):
        raise ValueError("review samples are unbound from the sealed replay")
    return report


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "rung.json").read_text())
    path = stage / "frames.zseq"
    assert sha256(path) == r["frames_sha256"]
    assert (
        r["isolation"]["interfaces"] == ["lo"]
        and len(r["isolation"]["denials"]) >= 4
        and r["isolation"]["mutation_denied"]
    )
    with path.open("rb") as f:
        for row in r["rows"]:
            n = struct.unpack(">I", f.read(4))[0]
            assert n + 4 == row["wire_bytes"]
            rgb = np.frombuffer(zlib.decompress(f.read(n)), np.uint8).reshape(1080, 1920, 3)
            assert array_sha(rgb) == row["frame_sha256"]
        assert not f.read(1)
    timing = json.loads((stage / "timing.json").read_text())["seconds"]
    assert len(r["rows"]) == len(r["ticks"]) and len(r["ticks"]) in (4, 240)
    n = len(r["ticks"])
    forecast = 1.5 * (timing - r["loop_seconds"] + r["loop_seconds"] * 240 / n) + 80
    assert (n == 240 and timing <= 1320) or (n == 4 and forecast <= 1320)
    checks = dict(
        native=True,
        rgba_integrity=True,
        source_denial=True,
        shared_foreground=True,
        forecast=True,
    )
    if (stage / "review-samples.json").exists():
        report = review_receipt(stage, r)
        for _ in read_review_samples(stage, report):
            pass
        checks.update(
            review_manifest_sha256=sha256(stage / "review-samples.json"),
            review_rung_sha256=report["rung_sha256"],
            review_sample_sha256=report["sample_sha256"],
        )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            dict(
                passed=True,
                checks=checks,
                seconds=timing,
                forecast=forecast,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    for k in ("carrier", "manifest", "joint", "cache", "original-joint", "witness"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--carrier-sha256")
    p.add_argument("--native-sha")
    p.add_argument("--frames", type=int)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--review-samples", action="store_true")
    p.add_argument("--deny", type=Path, action="append", default=[])
    a = p.parse_args()
    validate() if a.validate else cli_run(a)
