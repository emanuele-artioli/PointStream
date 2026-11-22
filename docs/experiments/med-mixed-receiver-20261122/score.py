"""Measure source-defined player support and full-display RGB errors on authentic decodes."""

import argparse
from hashlib import file_digest, sha256
import json
import math
import struct
import sys
import zlib
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.tennis.player_requalification import array_sha
from src.codecs import quality
from src.pointstream import ball
from src.pointstream.tracked_patch import yuv_to_rgb

WEIGHT = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
SOURCE = {
    "al031": "054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76",
    "000": "3bf7ddaae0db5c48b1072fccb15f9f162e41fa5999886b9a5bde58dc0495c630",
    "019": "1c13479eb3b6f8d175eb6f5020c629896e6b01f719014ec995a06b4c6c5a3e9c",
    "024": "4d602291ebea0fcb9698d86d3f93ce4c3076a13005e33079a82975b8b64605f2",
}


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255


def capture_sample(folder, tick, source, display, supports, ball_rows):
    """Save preselected native crops and small context images, without resizing metric inputs."""
    folder.mkdir(exist_ok=True, parents=True)
    boxes = {}
    for index, name in enumerate(("player-0", "player-1")):
        support = supports.get((tick, index))
        if support:
            boxes[name] = list(support[0])
    if ball_rows and ball_rows[-1]["tick"] == tick:
        x, y = ball_rows[-1]["crop_xy"]
        boxes["ball"] = [x, y, x + 64, y + 64]
    recorded = {}
    for domain, rgb in [("source", source), ("display", display)]:
        parts = {name: rgb[y0:y1, x0:x1] for name, (x0, y0, x1, y1) in boxes.items()}
        parts["context"] = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
        for name, part in parts.items():
            file = folder / f"tick{tick:03d}-{domain}-{name}.png"
            assert cv2.imwrite(str(file), cv2.cvtColor(part, cv2.COLOR_RGB2BGR))
            recorded[file.name] = sha(file)
    return dict(tick=tick, source_crop_xyxy=boxes, files_sha256=recorded)


def main(a):
    torch.set_num_threads(getattr(a, "threads", 2))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert sha(a.weights) == WEIGHT and sha(a.source) == SOURCE[a.clip]
    frame_count = getattr(a, "frame_count", 240)
    assert frame_count in (24, 240)
    start = {"000": 0, "019": 92, "024": 69, "al031": 0}[a.clip]
    native_fps, active_frames = (25, 240) if a.clip == "al031" else (50, 120)
    schema = json.loads(a.supports.read_text())
    expected_ticks = list(range(240)) if active_frames == 240 else list(range(10, 240, 20))
    assert schema["ticks"] == expected_ticks and schema["source_start"] == 0
    ball_positions = None
    if a.ball_payload or a.ball_receipt:
        assert a.ball_payload and a.ball_receipt and a.clip == "al031"
        br = json.loads(a.ball_receipt.read_text())
        assert br["source_sha256"] == SOURCE[a.clip] and br["payload_sha256"] == sha(a.ball_payload)
        ball_positions, appearance = ball.decode_frames(a.ball_payload.read_bytes(), 240, 960, 544)
        assert (
            ball.encode_frames(ball_positions, appearance, 960, 544) == a.ball_payload.read_bytes()
        )
        assert sum(p is not None for p in ball_positions) == br["present"]
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
        streamed = arm["kind"] == "rgb_zseq_stdin"
        if not streamed:
            assert sha(path) == arm["sha256"]
        sequence_hash = sha256()
        h, w = arm["height"], arm["width"]
        seq = None
        rgb = arm["kind"] == "rgb_npy"
        if rgb:
            vid = np.load(path, mmap_mode="r", allow_pickle=False)
            assert vid.dtype == np.uint8 and vid.shape == (240, 1080, 1920, 3)
        elif arm["kind"] in ("rgb_zseq", "rgb_zseq_stdin"):
            seq = sys.stdin.buffer if streamed else path.open("rb")
            receipt_path = Path(arm["receipt"])
            assert sha(receipt_path) == arm["receipt_sha256"]
            receipt = json.loads(receipt_path.read_text())
            assert receipt["frames_sha256"] == arm["sha256"] and receipt["ticks"] == list(
                range(240)
            )
        else:
            assert arm["kind"] == "native_yuv420p" and path.stat().st_size == h * w * 240 * 3 // 2
            vid = np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w))
        video = getattr(a, "canonical", None)
        canonical = None
        if video:
            canonical = json.loads(a.canonical_receipt.read_text())
            assert (
                sha(video) == canonical["canonical_sha256"]
                and canonical["source_sha256"] == SOURCE[a.clip]
            )
            assert (
                canonical["supports_sha256"] == sha(a.supports) and canonical["fps"] == native_fps
            )
        cap = cv2.VideoCapture(str(video or a.source), cv2.CAP_FFMPEG)
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == native_fps
        accumulated = [0.0, 0.0]
        scores = []
        ball_rows = []
        samples = []
        cpu0 = time.process_time()
        max_source_self = 0.0
        for t in range(frame_count):
            ok, bgr = cap.read()
            assert ok
            src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert src.shape == (1080, 1920, 3)
            if canonical:
                assert array_sha(src) == canonical["source_rgb_hashes"][t]
            if "source_rgb_hashes" in schema:
                assert array_sha(src) == schema["source_rgb_hashes"][t]
            if seq is not None:
                header = seq.read(4)
                size = struct.unpack(">I", header)[0]
                assert size + 4 == receipt["rows"][t]["wire_bytes"]
                payload = seq.read(size)
                assert len(payload) == size
                sequence_hash.update(header)
                sequence_hash.update(payload)
                image = np.frombuffer(zlib.decompress(payload), np.uint8).reshape(1080, 1920, 3)
                assert array_sha(image) == receipt["rows"][t]["frame_sha256"]
            elif rgb:
                image = vid[t]
            else:
                image = cv2.resize(
                    yuv_to_rgb(vid[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
            bucket = int(not start <= t < start + active_frames)
            accumulated[bucket] += cv2.norm(src, image, cv2.NORM_L2SQR)
            if ball_positions is not None and ball_positions[t] is not None:
                cx, cy = ball_positions[t]
                x0 = int(np.clip(round((cx + 0.5) * 1920 / 960 - 0.5) - 32, 0, 1920 - 64))
                y0 = int(np.clip(round((cy + 0.5) * 1080 / 544 - 0.5) - 32, 0, 1080 - 64))
                ref_ball = tensor(src[y0 : y0 + 64, x0 : x0 + 64])
                dec_ball = tensor(image[y0 : y0 + 64, x0 : x0 + 64])
                with torch.inference_mode():
                    ones = torch.ones((1, 1, 64, 64), dtype=torch.bool)
                    if not ball_rows:
                        assert (
                            abs(float(quality.masked_dists(model, ref_ball, ref_ball, ones))) < 1e-5
                        )
                    value = float(quality.masked_dists(model, dec_ball, ref_ball, ones))
                assert math.isfinite(value)
                ball_rows.append(
                    dict(tick=t, sender_xy=[int(cx), int(cy)], crop_xy=[x0, y0], dists=value)
                )
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
            if getattr(a, "sample_dir", None) and t in (10, 90, 170, 230):
                samples.append(capture_sample(a.sample_dir, t, src, image, supports, ball_rows))
            if t == 10:
                forecast = (time.process_time() - cpu0) * 24 * 1.5 + 30
                if forecast > 1500:
                    raise ValueError(
                        f"per-arm CPU pilot projects {forecast:.1f} seconds above 1,500"
                    )
        cap.release()
        if seq is not None:
            if frame_count == 240:
                assert not seq.read(1) and sequence_hash.hexdigest() == arm["sha256"]
            if not streamed:
                seq.close()
        assert len(scores) == sum(t < frame_count for t, _ in supports)

        def mean(rows):
            return sum(r["dists"] for r in rows) / len(rows) if rows else None

        def psnr(sse, n):
            return 10 * math.log10(255**2 * (n * 1920 * 1080 * 3) / sse) if n else None

        active_observed = sum(start <= t < start + active_frames for t in range(frame_count))
        result = dict(
            arm=arm,
            source_sha256=SOURCE[a.clip],
            supports_sha256=sha(a.supports),
            method=(
                schema["method"]
                if active_frames == 240
                else "Original W7 source-SAM masks on 12 fixed ticks, native RGB, tight support plus 32-pixel margin, masked DISTS; all-frame native RGB PSNR"
            ),
            native_fps=native_fps,
            frames=frame_count,
            active_start=start,
            active_frames=active_observed,
            fallback_frames=frame_count - active_observed,
            body_rows=scores,
            mean_dists=mean(scores),
            active_dists=mean([r for r in scores if r["active"]]),
            fallback_dists=mean([r for r in scores if not r["active"]])
            if active_frames < 240
            else None,
            rgb_psnr=psnr(sum(accumulated), frame_count),
            active_rgb_psnr=psnr(accumulated[0], active_observed),
            fallback_rgb_psnr=psnr(accumulated[1], frame_count - active_observed)
            if active_frames < 240
            else None,
            source_self_abs=max_source_self,
            weights_sha256=WEIGHT,
            torch=torch.__version__,
            cv2=cv2.__version__,
            device="cpu",
            cpu_seconds=time.process_time() - cpu0,
            forecast_cpu_seconds=forecast,
        )
        if samples:
            result["samples"] = samples
        if ball_positions is not None:
            assert len(ball_rows) == sum(p is not None for p in ball_positions[:frame_count])
            result["ball"] = dict(
                method="64x64 native RGB crop at every transmitted source-fitted center",
                payload_sha256=br["payload_sha256"],
                total_ticks=frame_count,
                present=len(ball_rows),
                rows=ball_rows,
                dists=mean(ball_rows) if ball_rows else None,
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
    for key in ("ball-payload", "ball-receipt"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--canonical", type=Path)
    p.add_argument("--canonical-receipt", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--frame-count", type=int, choices=(24, 240), default=240)
    p.add_argument("--clip", choices=SOURCE, required=True)
    main(p.parse_args())
