"""Price a native AV1 stream spanning two selected broadcast windows at one cadence."""

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import time

from experiments.tennis.player_requalification import sha256
from src.codecs import svtav1

W, H, RATE = 640, 384, 25
FRAME = W * H * 3 // 2
WEIGHT = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def write_input(target, raw_inputs, expected, half):
    with target.open("xb") as dst:
        for i, (file, signed) in enumerate(zip(raw_inputs, expected, strict=True)):
            dig = hashlib.sha256()
            with gzip.open(file, "rb") as f:
                for t in range(240):
                    block = f.read(FRAME)
                    assert len(block) == FRAME
                    dig.update(block)
                    if (i == 0 and t >= 240 - half) or (i == 1 and t < half):
                        dst.write(block)
                assert not f.read(1)
            assert dig.hexdigest() == signed
    assert target.stat().st_size == half * 2 * FRAME


def tensor(img):
    import cv2
    import numpy as np
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255


def selected_scores(path, support, rows, model, half):
    import cv2
    import numpy as np
    import torch

    from src.codecs import quality
    from src.pointstream.tracked_patch import yuv_to_rgb

    outputs = []
    video = np.memmap(path, np.uint8, "r", shape=(half * 2, H * 3 // 2, W))
    for row in rows:
        shot, tick = row["shot"], row["tick"]
        i = tick - (240 - half) if shot == 0 else half + tick
        if (shot == 0 and i < 0) or (shot == 1 and tick >= half):
            continue
        pix = {}
        for tag, (name, signed) in row["assets"].items():
            assert Path(name).name == name and sha256(support / name) == signed
            pix[tag] = cv2.imread(
                str(support / name), cv2.IMREAD_GRAYSCALE if tag == "mask" else cv2.IMREAD_COLOR
            )
        m = torch.from_numpy((pix["mask"] > 0).copy())[None, None]
        x, y, xx, yy = row["xy"]
        rgb = cv2.resize(
            yuv_to_rgb(video[i], W, H, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )
        display = tensor(cv2.cvtColor(rgb[y:yy, x:xx], cv2.COLOR_RGB2BGR))
        ref = tensor(pix["source"])
        with torch.inference_mode():
            calibration = float(quality.masked_dists(model, tensor(pix["positive"]), ref, m))
            value = float(quality.masked_dists(model, display, ref, m))
            if not outputs:
                assert abs(float(quality.masked_dists(model, ref, ref, m))) < 1e-5
        error = abs(calibration - row["positive_dists"])
        assert math.isfinite(value) and 0 <= value <= 2 and error <= 2e-4
        outputs.append(
            dict(
                shot=shot,
                tick=tick,
                role=row["role"],
                stream_tick=i,
                dists=value,
                calibration_error=error,
            )
        )
    assert len(outputs) == (8 if half == 48 else 48)
    return outputs


def run(a):
    import cv2
    import numpy as np
    import torch

    from src.codecs import quality

    half = a.frames // 2
    stage = os.environ["PS_STAGE"]
    assert a.frames == {"smoke": 96, "full": 480}[stage] and sha256(a.weights) == WEIGHT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    start = time.monotonic()
    support = a.samples
    contract = json.loads((support / "grid.json").read_text())
    scratch, dst = Path(os.environ["PS_SCRATCH_DIR"]), Path(os.environ["PS_STAGE_DIR"])
    data = scratch / "paired.yuv"
    write_input(data, [a.first, a.second], contract["input_sha256"], half)
    tools = {n: svtav1.tool(n) for n in ("SvtAv1EncApp", "dav1d", "ffprobe")}
    net = quality.load_dists(a.weights, "cpu")
    by_q = {}

    def measure(q4):
        if q4 not in by_q:
            work = scratch / f"q{q4}"
            work.mkdir()
            r = svtav1.code(
                data,
                work,
                width=W,
                height=H,
                fps=Fraction(RATE),
                frames=a.frames,
                crf=q4 / 4,
                preset=4,
                threads=1,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=240,
            )
            assert r["temporal_units"] == r["decoded_frames"] == a.frames
            by_q[q4] = r
        return by_q[q4]["file_bytes"]

    if stage == "smoke":
        measure(154)
        picks = {"qualification": 154}
    else:
        measure(112)
        measure(208)
        picks = {}
        for name, cap in sorted(contract["physical_caps"].items(), key=lambda v: v[1]):
            for _ in range(4):
                above = [q for q, v in by_q.items() if v["file_bytes"] > cap]
                below = [q for q, v in by_q.items() if v["file_bytes"] <= cap]
                assert above and below
                hi = min(below, key=lambda q: cap - by_q[q]["file_bytes"])
                lo = min(above, key=lambda q: by_q[q]["file_bytes"] - cap)
                if cap - measure(hi) <= 0.005 * cap or hi - lo <= 1:
                    break
                fraction = math.log(cap / measure(lo)) / math.log(measure(hi) / measure(lo))
                q = max(lo + 1, min(hi - 1, round(lo + fraction * (hi - lo))))
                if q in by_q:
                    choices = [x for x in range(lo + 1, hi) if x not in by_q]
                    if not choices:
                        break
                    q = min(choices, key=lambda x: abs(x - q))
                measure(q)
            picks[name] = min(
                (q for q, v in by_q.items() if v["file_bytes"] <= cap),
                key=lambda q: cap - by_q[q]["file_bytes"],
            )
        assert len(by_q) <= 10
    report = {}
    for name, q in picks.items():
        r = by_q[q]
        path = Path(r["stream"])
        again = scratch / (name + "-independent.yuv")
        subprocess.run(
            svtav1.decode_command(tools["dav1d"]["path"], path, again, threads=1),
            check=True,
            capture_output=True,
            timeout=180,
        )
        rawsha = sha256(again)
        assert again.stat().st_size == FRAME * a.frames and rawsha == sha256(Path(r["decoded"]))
        images = json.loads(
            subprocess.check_output(
                [
                    tools["ffprobe"]["path"],
                    "-v",
                    "error",
                    "-show_frames",
                    "-show_entries",
                    "frame=key_frame,pict_type",
                    "-of",
                    "json",
                    str(path),
                ],
                timeout=180,
            )
        )["frames"]
        assert len(images) == a.frames and images[0]["key_frame"] == 1
        scored = selected_scores(again, support, contract["rows"], net, half)
        physical = dst / (name + ".ivf")
        copyfile(path, physical)
        assert sha256(physical) == r["stream_sha256"]
        report[name] = dict(
            bytes=r["file_bytes"],
            kbps=r["file_bytes"] * 8 * RATE / (a.frames * 1000),
            stream_sha256=r["stream_sha256"],
            raw_sha256=rawsha,
            crf=q / 4,
            keyframes=[i for i, v in enumerate(images) if v["key_frame"]],
            at_return=images[half],
            dists=float(np.mean([s["dists"] for s in scored])),
            per_shot=[
                float(np.mean([s["dists"] for s in scored if s["shot"] == i])) for i in (0, 1)
            ],
            rows=scored,
        )
    forecast = 1.5 * by_q[next(iter(by_q))]["encode_seconds"] * 5 * 10 + 120
    result = dict(
        frames=a.frames,
        cadence=[25, 1],
        width=W,
        height=H,
        source_files=contract["source_files"],
        raw_input_sha256=contract["input_sha256"],
        toolchain=tools,
        weights_sha256=WEIGHT,
        independent_decode=True,
        physical_caps=contract["physical_caps"],
        points=report,
        trials={
            q: {
                k: r[k]
                for k in (
                    "file_bytes",
                    "stream_sha256",
                    "encode_command",
                    "decode_command",
                    "encode_seconds",
                    "encode_cpu_seconds",
                    "decode_cpu_seconds",
                )
            }
            for q, r in by_q.items()
        },
        elapsed=time.monotonic() - start,
        full_forecast_seconds=forecast,
        stage=stage,
    )
    (dst / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    validate_report(dst, result)


def validate_report(dst, r):
    smoke = r["stage"] == "smoke"
    n = 96 if smoke else 480
    checks = {
        "duration": r["frames"] == n and r["cadence"] == [25, 1] and r["independent_decode"],
        "point_count": len(r["points"]) == (1 if smoke else 2),
        "prices": all(
            (smoke or v["bytes"] <= r["physical_caps"][name])
            and v["bytes"] == (dst / (name + ".ivf")).stat().st_size
            for name, v in r["points"].items()
        ),
        "hashes": all(
            sha256(dst / (name + ".ivf")) == v["stream_sha256"] for name, v in r["points"].items()
        ),
        "masks": all(
            len(v["rows"]) == (8 if smoke else 48)
            and max(x["calibration_error"] for x in v["rows"]) <= 2e-4
            for v in r["points"].values()
        ),
        "forecast": not smoke or r["full_forecast_seconds"] < 2050,
    }
    ok = all(checks.values())
    (dst / "two_window_validation.json").write_text(
        json.dumps(dict(passed=ok, checks=checks)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    c = argparse.ArgumentParser(description=__doc__)
    for k in ("first", "second", "samples", "weights"):
        c.add_argument("--" + k, type=Path)
    c.add_argument("--frames", type=int)
    c.add_argument("--validate", action="store_true")
    args = c.parse_args()
    if args.validate:
        d = Path(os.environ["PS_STAGE_DIR"])
        r = json.loads((d / "result.json").read_text())
        validate_report(d, r)
        copyfile(d / "two_window_validation.json", os.environ["PS_VALIDATION_PATH"])
    else:
        run(args)
