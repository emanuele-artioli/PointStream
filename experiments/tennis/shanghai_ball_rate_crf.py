"""Create prospective Shanghai ball-only PSBL and native AV1 controls without labels."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from src.codecs import svtav1
from src.pointstream import ball, ball_stream

SOURCE_SHA = "0afb0fb191881def1235285b58fe54814556c83fed7815cd99a00b92d26bc86c"
TOOLS = {
    "SvtAv1EncApp": "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4",
    "dav1d": "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6",
    "ffmpeg": "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2",
}


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(a):
    start = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    assert sha(a.source) == SOURCE_SHA
    if a.arm == "ball":
        assert sha(a.payload) == a.payload_sha256
    tools = {k: svtav1.tool(k) for k in TOOLS}
    assert all(tools[k]["sha256"] == v for k, v in TOOLS.items())
    payload = b""
    if a.arm == "ball":
        track, apps = ball.decode_frames(a.payload.read_bytes(), 240, 960, 544)
        payload = ball.encode_frames(track[: a.frames], apps[: a.frames], 960, 544)
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "shanghai-rate"
    work.mkdir()
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    stage = Path(os.environ["PS_STAGE_DIR"])
    target = round(20400 * a.frames / 25 / 8)
    groups = []
    for width, height in ((320, 192), (640, 384)):
        src = work / f"{width}.yuv"
        command = [
            tools["ffmpeg"]["path"],
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(a.source),
            "-frames:v",
            str(a.frames),
            "-vf",
            f"scale={width}:{height}:flags=area,format=yuv420p",
            "-f",
            "rawvideo",
            str(src),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=150)
        assert src.stat().st_size == a.frames * width * height * 3 // 2
        for kind in (a.arm,):
            lower, upper = 160, 280
            seen = set()
            records = []
            for iteration in range(10 if a.frames == 240 else 2):
                tbr = lower if iteration == 0 else upper if iteration == 1 else (lower + upper) // 2
                if tbr in seen:
                    break
                seen.add(tbr)
                ivf = work / f"{width}-{kind}-{iteration}.ivf"
                command = svtav1.encode_command(
                    tools["SvtAv1EncApp"]["path"],
                    src,
                    ivf,
                    width=width,
                    height=height,
                    fps=Fraction(25),
                    frames=a.frames,
                    crf=tbr / 4,
                    preset=4,
                    full_range=False,
                )
                then = time.monotonic()
                proc = subprocess.run(command, check=True, capture_output=True, timeout=240)
                if not ivf.exists() or not ivf.stat().st_size:
                    raise ValueError(proc.stderr.decode(errors="replace")[-2000:])
                native = ivf.read_bytes()
                assert len(svtav1.ivf_frames(native)) == a.frames
                packed = (
                    native
                    if kind == "ordinary"
                    else ball_stream.pack(
                        ball_stream.Stream(
                            960,
                            544,
                            Fraction(25),
                            a.frames,
                            "svt-av1-ivf",
                            f"svtav1-4.2-scaled-{width}x{height}",
                            native,
                            payload,
                        )
                    )
                )
                if kind == "ball":
                    check = ball_stream.unpack(packed)
                    assert check.native == native and check.ball_payload == payload
                path = work / f"{width}-{kind}-{iteration}.packet"
                path.write_bytes(packed)
                records.append(
                    {
                        "crf": tbr / 4,
                        "native_bytes": len(native),
                        "bytes": len(packed),
                        "sha256": sha(path),
                        "native_sha256": hashlib.sha256(native).hexdigest(),
                        "seconds": time.monotonic() - then,
                        "file": path.name,
                    }
                )
                if a.frames == 240:
                    if len(packed) < target:
                        upper = tbr
                    else:
                        lower = tbr
                    if abs(len(packed) - target) / target <= 0.002:
                        break
            best = min(records, key=lambda r: (abs(r["bytes"] - target), r["bytes"], r["crf"]))
            selected = (work / best["file"]).read_bytes()
            native = selected if kind == "ordinary" else ball_stream.unpack(selected).native
            ivf = work / f"{width}-{kind}-selected.ivf"
            ivf.write_bytes(native)
            yuv = work / f"{width}-{kind}-decoded.yuv"
            subprocess.run(
                svtav1.decode_command(tools["dav1d"]["path"], ivf, yuv, threads=1),
                check=True,
                capture_output=True,
                timeout=150,
            )
            assert yuv.stat().st_size == a.frames * width * height * 3 // 2
            suffix = "ivf" if kind == "ordinary" else "psbl"
            dest = pub / f"{width}-{kind}.{suffix}"
            dest.write_bytes(selected)
            best["decoded_yuv_sha256"] = sha(yuv)
            best["selected_file"] = dest.name
            groups.append(
                {
                    "width": width,
                    "height": height,
                    "kind": kind,
                    "trials": records,
                    "selected": best,
                }
            )
    data = {
        "stage": os.environ["PS_STAGE"],
        "frames": a.frames,
        "fps": 25,
        "source_sha256": sha(a.source),
        "arm": a.arm,
        "rate_mode": "crf-quarter-bisection",
        "payload_sha256": sha(a.payload) if a.arm == "ball" else None,
        "payload_prefix_sha256": hashlib.sha256(payload).hexdigest(),
        "logical_canvas": [960, 544],
        "target_bytes": target,
        "groups": groups,
        "tools": tools,
        "elapsed_seconds": time.monotonic() - start,
    }
    (pub / "report.json").write_text(json.dumps(data, indent=2) + "\n")
    (stage / "report.json").write_text(json.dumps(data, indent=2) + "\n")


def validate():
    data = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "report.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    checks = {
        "frames": data["frames"] == n,
        "groups": len(data["groups"]) == 2
        and all(g["kind"] == data["arm"] for g in data["groups"]),
        "input": data["source_sha256"] == SOURCE_SHA,
        "native_decode": all(
            len(g["selected"]["decoded_yuv_sha256"]) == 64 for g in data["groups"]
        ),
    }
    if n == 24:
        infer = sum(x["seconds"] for g in data["groups"] for x in g["trials"])
        checks["forecast"] = 1.5 * infer * (240 / 24) * (10 / 2) + 300 < 1800
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n"
    )
    assert all(checks.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--payload", type=Path)
    p.add_argument("--payload-sha256")
    p.add_argument("--frames", type=int)
    p.add_argument("--arm", choices=("ordinary", "ball"))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
