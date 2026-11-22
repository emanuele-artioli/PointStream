"""Measure source-defined player support and full-display RGB errors on authentic decodes."""

import argparse
from hashlib import file_digest
import json
import math
import struct
import zlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.tennis.player_requalification import array_sha
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

WEIGHT = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
SOURCE = {
    "000": "3bf7ddaae0db5c48b1072fccb15f9f162e41fa5999886b9a5bde58dc0495c630",
    "019": "1c13479eb3b6f8d175eb6f5020c629896e6b01f719014ec995a06b4c6c5a3e9c",
    "024": "4d602291ebea0fcb9698d86d3f93ce4c3076a13005e33079a82975b8b64605f2",
}


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255


def main(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert sha(a.weights) == WEIGHT and sha(a.source) == SOURCE[a.clip]
    start = {"000": 0, "019": 92, "024": 69}[a.clip]
    schema = json.loads(a.supports.read_text())
    assert schema["ticks"] == list(range(10, 240, 20)) and schema["source_start"] == 0
    supports = {}
    for r in schema["rows"]:
        if not r["area"]:
            continue
        key = (r["tick"], r["object_index"])
        path = a.supports.parent / r["mask"]
        assert sha(path) == r["sha256"]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        assert int(mask.sum()) == r["area"] and key not in supports
        supports[key] = (r["xyxy"], torch.from_numpy(mask.copy())[None, None])
    assert len(supports) == schema["nonempty"]
    arms = json.loads(a.arms.read_text())
    assert arms and len({r["name"] for r in arms}) == len(arms)
    model = quality.load_dists(a.weights, "cpu")
    a.out.mkdir(exist_ok=False)
    summary = []
    for arm in arms:
        dest = a.out / (arm["name"] + ".json")
        path = Path(arm["path"])
        assert sha(path) == arm["sha256"]
        h, w = arm["height"], arm["width"]
        seq = None
        rgb = arm["kind"] == "rgb_npy"
        if rgb:
            vid = np.load(path, mmap_mode="r", allow_pickle=False)
            assert vid.dtype == np.uint8 and vid.shape == (240, 1080, 1920, 3)
        elif arm["kind"] == "rgb_zseq":
            seq = path.open("rb")
            receipt_path = Path(arm["receipt"])
            assert sha(receipt_path) == arm["receipt_sha256"]
            receipt = json.loads(receipt_path.read_text())
            assert receipt["frames_sha256"] == arm["sha256"] and receipt["ticks"] == list(
                range(240)
            )
        else:
            assert arm["kind"] == "native_yuv420p" and path.stat().st_size == h * w * 240 * 3 // 2
            vid = np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w))
        cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 50
        accumulated = [0.0, 0.0]
        scores = []
        cpu0 = time.process_time()
        max_source_self = 0.0
        for t in range(240):
            ok, bgr = cap.read()
            assert ok
            src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert src.shape == (1080, 1920, 3)
            if seq is not None:
                size = struct.unpack(">I", seq.read(4))[0]
                assert size + 4 == receipt["rows"][t]["wire_bytes"]
                image = np.frombuffer(zlib.decompress(seq.read(size)), np.uint8).reshape(
                    1080, 1920, 3
                )
                assert array_sha(image) == receipt["rows"][t]["frame_sha256"]
            elif rgb:
                image = vid[t]
            else:
                image = cv2.resize(
                    yuv_to_rgb(vid[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
            bucket = int(not start <= t < start + 120)
            accumulated[bucket] += cv2.norm(src, image, cv2.NORM_L2SQR)
            for who in (0, 1):
                if (t, who) not in supports:
                    continue
                (x0, y0, x1, y1), m = supports[t, who]
                ref = tensor(src[y0:y1, x0:x1])
                display = tensor(image[y0:y1, x0:x1])
                with torch.inference_mode():
                    if not scores:
                        reference = float(quality.masked_dists(model, ref, ref, m))
                        assert abs(reference) < 1e-5
                        max_source_self = abs(reference)
                    value = float(quality.masked_dists(model, display, ref, m))
                if not math.isfinite(value):
                    raise ValueError("nonfinite DISTS")
                scores.append(dict(tick=t, player_index=who, active=not bucket, dists=value))
            if t == 10:
                forecast = (time.process_time() - cpu0) * 24 * 1.5 + 30
                if forecast > 1500:
                    raise ValueError(
                        f"per-arm CPU pilot projects {forecast:.1f} seconds above 1,500"
                    )
        cap.release()
        if seq is not None:
            assert not seq.read(1)
            seq.close()
        assert len(scores) == schema["nonempty"]

        def mean(rows):
            return sum(r["dists"] for r in rows) / len(rows)

        def psnr(sse, n):
            return 10 * math.log10(255**2 * (n * 1920 * 1080 * 3) / sse)

        result = dict(
            arm=arm,
            source_sha256=SOURCE[a.clip],
            supports_sha256=sha(a.supports),
            method="Original W7 source-SAM masks on 12 fixed ticks, native RGB, tight support plus 32-pixel margin, masked DISTS; all-frame native RGB PSNR",
            native_fps=50,
            frames=240,
            active_start=start,
            active_frames=120,
            fallback_frames=120,
            body_rows=scores,
            mean_dists=mean(scores),
            active_dists=mean([r for r in scores if r["active"]]),
            fallback_dists=mean([r for r in scores if not r["active"]]),
            rgb_psnr=psnr(sum(accumulated), 240),
            active_rgb_psnr=psnr(accumulated[0], 120),
            fallback_rgb_psnr=psnr(accumulated[1], 120),
            source_self_abs=max_source_self,
            weights_sha256=WEIGHT,
            torch=torch.__version__,
            cv2=cv2.__version__,
            device="cpu",
            cpu_seconds=time.process_time() - cpu0,
            forecast_cpu_seconds=forecast,
        )
        dest.write_text(json.dumps(result, indent=2) + "\n")
        summary.append(
            {
                k: result[k]
                for k in (
                    "arm",
                    "mean_dists",
                    "active_dists",
                    "fallback_dists",
                    "rgb_psnr",
                    "active_rgb_psnr",
                    "fallback_rgb_psnr",
                    "cpu_seconds",
                )
            }
        )
        print(arm["name"], round(result["mean_dists"], 6), round(result["rgb_psnr"], 4), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("source", "weights", "supports", "arms", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--clip", choices=SOURCE, required=True)
    main(p.parse_args())
