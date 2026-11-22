"""Replay real DCVC-UF containers against the 50-fps rally's native reviewed supports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

import cv2
import numpy as np

from src.codecs import dcvc_uf_worker, quality
from src.pointstream.tracked_patch import yuv_to_rgb

WEIGHTS = {
    "image": "b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302",
    "video": "934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3",
    "vgg": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
}
PACKETS = {
    18: "63134ca2b7995b78404815625dfea84df50cc462c459c697303fc131d7393ff4",
    24: "a021ab92d217700a65046af5932c80f7cbb4f67819c2d2133b3193a1162efa39",
}
ARCHIVE = "2dd9fd2e821b9cbfce6de838166ac5f77d0ff0dafc156b75fa9d2a4f7a0d888f"
REGIONS = "1bff8cee33528e776f135ef62e6246a2af22bb2024d818c91bfeb6be27fe09c7"
FRAMES = "4c891777b33b10ba92fd94fa1a59ba360aacf175e7bfd67c382206058d59550a"


SOURCES = {
    10: "0360ccc4ec8eea097f872285551080d36de520b8ec84ab4f0ccf30b5e9284a7a",
    30: "c0e89a4fee71fde3f53b012da149a1a06d721a7c3ed1bc8d9dc90e9052bcc4c6",
    50: "fd936ebf15975927e52681aba02139dc3077e88e0212a20cd5ad8feb69dce915",
    70: "b09733f794643094908d90fd6ab3eeca57b99d230b9d7d01285e8a6943eaabaf",
    90: "bdfe0b6899eee9c7f14c0fe7a6de7e0758ee7b6e4f04dcefe5bd90d01e4c3128",
    110: "fd89a7916e40c9d3f6c25437012c38a4f428973b7a9c125d8df57b11361e499f",
    130: "30b7d1e52a908fd2ebf31eb24d7963a11ff9c7c6dae427de042f25cae992c1b8",
    150: "a9b53ac29b46e0f03b094a49deec15ba32d1f7a1e20051a3daf74d2d1f43b59e",
    170: "708ee6708f0841b2a691b541a39ac08a0139512a2128bd274105ae9260dd3b94",
    190: "5dc6e7b97953f5c937a5267f8d17f3a5575882eeea479eeb82d27ac26a37b68a",
    210: "f76942341785b4a286375f46dd87b6ac0cdbda58fe10d4dfbc6d2021b325f06d",
    230: "67573fcfac629aa7a1a85ca3bb502fb9140f39bae15dd76c3324a529d6a1d7e6",
}


def sha(p):
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(a, archive, qp, work, *, window="nad000", fps=50, packet_sha=None):
    expected_packet = PACKETS[qp] if packet_sha is None else packet_sha
    tag = f"dcvc-s2-q{qp}"
    data = archive.extractfile(f"publish/{window}/{tag}.psdc").read()
    prior = json.load(archive.extractfile(f"publish/{window}/{tag}-reports.json"))
    assert hashlib.sha256(data).hexdigest() == expected_packet
    packet = work / f"{tag}.psdc"
    packet.write_bytes(data)
    raw = work / f"{tag}.yuv"
    plan = dict(
        structure="htl",
        width=960,
        height=544,
        frame_count=240,
        src_type="yuv420",
        image_ckpt=str(a.image.resolve()),
        video_ckpt=str(a.video.resolve()),
        image_sha256=WEIGHTS["image"],
        video_sha256=WEIGHTS["video"],
        container=str(packet),
        out_file=str(raw),
        profile=False,
        deterministic=True,
        stream_policy="legacy",
    )
    plan_path = work / f"{tag}-plan.json"
    plan_path.write_text(json.dumps(plan))
    report = work / f"{tag}-decode.json"
    command, env, cwd = dcvc_uf_worker.dcvc_command(Path(sys.prefix), "decode", plan_path, report)
    started = time.monotonic()
    proc = subprocess.run(
        command, env={**os.environ, **env}, cwd=cwd, capture_output=True, timeout=180
    )
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors="replace")[-2000:])
    r = json.loads(report.read_text())
    expected = prior["decode"]["decode-a"]["passes"][0]["frame_sha256"]
    comparisons = [sum(a != b for a, b in zip(p["frame_sha256"], expected)) for p in r["passes"]]
    identity = dict(
        qp=qp,
        original_count=len(expected),
        deterministic=r["deterministic"],
        pass_lengths=[len(p["frame_sha256"]) for p in r["passes"]],
        different_frames_by_pass=comparisons,
        packet_matches=r["container_sha256"] == expected_packet,
    )
    if not (
        len(expected) == 240
        and r["deterministic"]
        and identity["packet_matches"]
        and identity["pass_lengths"] == [240, 240]
        and comparisons == [0, 0]
    ):
        raise ArithmeticError(json.dumps(identity))
    assert raw.stat().st_size == 240 * 960 * 544 * 3 // 2
    frames = np.memmap(raw, "uint8", "r", shape=(240, 816, 960))
    assert all(hashlib.sha256(f.tobytes()).hexdigest() == h for f, h in zip(frames, expected))
    return frames, dict(
        qp=qp,
        kbps=len(data) * 8 * fps / (240 * 1000),
        bytes=len(data),
        packet_sha256=expected_packet,
        passes=2,
        historical_frame_hashes_matched=240,
        decoder_environment=r["environment"],
        raw_sha256=sha(raw),
        decode_seconds=time.monotonic() - started,
    )


def image(blob, gray=False):
    a = cv2.imdecode(
        np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    )
    return a if gray else a[:, :, ::-1].copy()


def measure(net, native, meta, qp, ticks, records, regions, frames):
    import torch

    rows = []

    def score(rgb, source, mask):
        def tensor(im):
            return (
                torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].to("cuda").float()
                / 255
            )

        return float(
            quality.masked_dists(
                net, tensor(rgb), tensor(source), torch.from_numpy(mask > 0)[None, None].to("cuda")
            ).item()
        )

    squared = []
    for tick in ticks:
        canvas = cv2.resize(
            yuv_to_rgb(native[tick], 960, 544, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )
        source = image(frames.read(SOURCES[tick] + ".png"))
        squared.append(float(np.mean((source.astype(np.float32) - canvas) ** 2, dtype=np.float64)))
        for r in (r for r in records if r["t"] == tick):
            prefix = f"{tick:03d}/{r['region']}/"
            ref = image(regions.read(prefix + "source.png"))
            mask = image(regions.read(prefix + "mask.png"), True)
            x, y, right, bot = r["rect"]
            assert np.array_equal(ref, source[y:bot, x:right])
            av1 = score(image(regions.read(prefix + "plain_match.png")), ref, mask)
            assert abs(av1 - r["av1_dists"]) <= 1e-4
            value = score(canvas[y:bot, x:right], ref, mask)
            assert np.isfinite(value)
            rows.append(
                dict(
                    t=tick,
                    region=r["region"],
                    qp=qp,
                    dists=value,
                    anchor_delta=av1 - r["av1_dists"],
                )
            )
    meta["whole_rgb_psnr"] = float(10 * np.log10(255**2 / np.mean(squared)))
    meta["mean"] = {
        region: float(
            np.mean([r["dists"] for r in rows if r["qp"] == qp and r["region"] == region])
        )
        for region in ("p1", "p2", "ball")
    }
    return rows


def run(a):
    import torch

    start = time.monotonic()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and "6000 Ada" in torch.cuda.get_device_name(0)
    assert (sha(a.old), sha(a.regions), sha(a.frames)) == (ARCHIVE, REGIONS, FRAMES)
    assert all(sha(getattr(a, k)) == v for k, v in WEIGHTS.items())
    net = quality.load_dists(a.vgg, "cuda")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    rows = []
    summary = []
    attempts = []
    with (
        tarfile.open(a.old) as archive,
        zipfile.ZipFile(a.regions) as regions,
        zipfile.ZipFile(a.frames) as frames,
    ):
        records = json.loads(regions.read("regions.json"))
        ticks = sorted({r["t"] for r in records})
        assert ticks == list(range(10, 240, 20))
        ticks = ticks[:1] if a.count == 1 else ticks
        for qp in list(PACKETS)[: a.count]:
            for attempt in range(2):
                directory = scratch / f"replay-{qp}-{attempt}"
                directory.mkdir()
                try:
                    native, meta = replay(a, archive, qp, directory)
                except ArithmeticError as exc:
                    attempts.append(dict(attempt=attempt, qualified=False, **json.loads(str(exc))))
                    continue
                attempts.append(dict(attempt=attempt, qp=qp, qualified=True))
                break
            else:
                continue
            rows.extend(measure(net, native, meta, qp, ticks, records, regions, frames))
            summary.append(meta)
    result = dict(
        rows=rows,
        points=summary,
        ticks=ticks,
        attempts=attempts,
        elapsed_seconds=time.monotonic() - start,
        raw_input_sha256=ARCHIVE,
        metric=quality.dists_record(a.vgg),
        gpu=torch.cuda.get_device_name(0),
    )
    (publish / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def validate():
    with tarfile.open(Path(os.environ["PS_STAGE_DIR"]) / "published.tar") as archive:
        r = json.load(archive.extractfile("publish/result.json"))
    smoke = os.environ["PS_STAGE"] == "smoke"
    forecast = 3 * r["elapsed_seconds"] + 100
    checks = dict(
        historical_pixels=all(p["historical_frame_hashes_matched"] == 240 for p in r["points"]),
        objective=len(r["rows"]) == (3 if smoke else 72)
        and all(abs(row["anchor_delta"]) <= 1e-4 for row in r["rows"]),
        archive=r["raw_input_sha256"] == ARCHIVE,
        time_gate=not smoke or forecast < 550,
    )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=all(checks.values()), checks=checks, forecast_seconds=forecast))
    )
    if not all(checks.values()):
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("old", "regions", "frames", "image", "video", "vgg"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--count", type=int, choices=(1, 2))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
