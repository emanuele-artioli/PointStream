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


def config_for(a):
    channel = mixed.decode(a.joint, a.cache, a.carrier, a.carrier_sha256)
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
        Path(c["joint"]), Path(c["cache"]), Path(c["carrier"]), c["carrier_sha256"]
    )
    old = Path(c["original_joint"]).read_bytes()
    new = Path(c["joint"]).read_bytes()
    op = packet._embedded_profile(old)
    if op != packet._embedded_profile(new):
        raise ValueError("paid headers differ beyond the native base")
    orig, _ = packet.wire.unpack(old, op)
    revised, _ = packet.wire.unpack(new, op)
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
        channel["profile"],
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
        assert yuv.stat().st_size == 240 * 144 * 160
        raw = np.memmap(yuv, np.uint8, "r", shape=(240, 144, 160))
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
                raise ValueError(f"source-denied original-rung replay differs at {t}")
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
    loop = time.monotonic() - stage_start
    result = dict(
        ticks=ticks,
        rows=rows,
        carrier_sha256=c["carrier_sha256"],
        wire=actual_wire,
        witness_sha256=sha256(Path(c["witness_receipt"])),
        original_rung=ref_rung,
        all_witness_frames_matched=ref_rung,
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
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    native=True,
                    rgba_integrity=True,
                    source_denial=True,
                    shared_foreground=True,
                    forecast=True,
                ),
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
    p.add_argument("--deny", type=Path, action="append", default=[])
    a = p.parse_args()
    validate() if a.validate else cli_run(a)
