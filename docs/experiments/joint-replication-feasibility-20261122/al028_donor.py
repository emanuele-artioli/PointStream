"""Source-only donor-background preparation from paid 25 fps player/ball inputs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from src.pointstream import ball

HERE = Path(__file__).resolve().parent
SOURCE = Path("/private/tmp/ps-al028-review-source-20261122.mp4")
CARRIER = Path("/private/tmp/ps-al028-common25-packet-20261122/players.pscp")
BALL = Path("/private/tmp/ps-al028-ball-sender-full.payload")
OUT = Path("/private/tmp/ps-al028-donor-base-20261122")
PILOT = HERE.parent / "clean-court-rally-20261118/pilot.py"
H, W = 544, 960


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu():
    return sum(
        resource.getrusage(x).ru_utime + resource.getrusage(x).ru_stime
        for x in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def main(*, portable=False):
    start = cpu()
    cv2.setNumThreads(1)
    if sha(SOURCE) != "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934" or sha(
        CARRIER
    ) != (
        "731cc34e1fcb7c0874cd5722ab63241a123d31fbcd699e199cee4163cdcbc0dc"
        if portable
        else "85d7e48e33718522d39b9f4719f6376a6e470a6b4a5b8bb9bb31304bb84c39e3"
    ):
        raise ValueError("source or carrier identity")
    payload = BALL.read_bytes()
    positions, looks = ball.decode_frames(payload, 240, 960, 544)
    if ball.encode_frames(positions, looks, 960, 544) != payload:
        raise ValueError("ball payload does not round-trip")
    _, _, tracks = packet.unpack(CARRIER.read_bytes())
    spec = importlib.util.spec_from_file_location("donor_pilot", PILOT)
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    OUT.mkdir(exist_ok=False)
    source = OUT / "source.yuv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(SOURCE),
            "-vf",
            "select=between(n\\,0\\,239),scale=960:544:flags=area:out_range=tv,format=yuv420p",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            str(source),
        ],
        check=True,
        timeout=120,
    )
    if source.stat().st_size != 240 * 960 * 544 * 3 // 2:
        raise ValueError("native source count")
    frames = np.memmap(source, np.uint8, "r", shape=(240, 816, 960))
    coords = np.array(
        [
            (-1, -1)
            if pos is None
            else (max(0, 2 * pos[0] - 16), max(0, round(pos[1] * 1080 / 544) - 16))
            for pos in positions
        ]
    )
    masks = np.stack([pilot.mask_at(t, frames[t], tracks, coords) for t in range(240)])
    if not all(np.count_nonzero(mask) > 0 for mask in masks):
        raise ValueError("unexpected empty foreground mask")
    orb = cv2.ORB_create(nfeatures=8000)
    reference = cv2.cvtColor(frames[0], cv2.COLOR_YUV2BGR_I420)
    _, _ = reference.shape[:2]
    k0, d0 = orb.detectAndCompute(reference, cv2.bitwise_not(masks[0]))
    if d0 is None:
        raise ValueError("no source registration features")

    def align(indices):
        homs = []
        rows = []
        for t in indices:
            image = cv2.cvtColor(frames[t], cv2.COLOR_YUV2BGR_I420)
            keys, desc = orb.detectAndCompute(image, cv2.bitwise_not(masks[t]))
            if desc is None:
                raise ValueError("no donor features")
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, d0, k=2)
            good = [
                a
                for pair in pairs
                if len(pair) == 2
                for a, b in [pair]
                if a.distance < 0.7 * b.distance
            ]
            a = np.float32([keys[x.queryIdx].pt for x in good])
            b = np.float32([k0[x.trainIdx].pt for x in good])
            hom, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
            if hom is None or inliers is None or int(inliers.sum()) < 50:
                raise ValueError("donor alignment failed")
            homs.append(hom / hom[2, 2])
            rows.append({"tick": int(t), "matches": len(good), "inliers": int(inliers.sum())})
        return homs, rows

    check_ticks = np.linspace(0, 23, 3).round().astype(int)
    before = cpu()
    smoke_h, smoke_rows = align(check_ticks)
    smoke_plate, smoke_holes = pilot.plate_for(
        [pilot.planes(frames[t])[0] for t in check_ticks], masks[check_ticks], smoke_h, 1
    )
    if smoke_plate.shape != (H, W):
        raise ValueError("donor smoke shape")
    smoke_cpu = cpu() - before
    projected = (cpu() - start) + 1.5 * smoke_cpu * (17 / 3) * 3 + 500
    if projected > 1200:
        raise ValueError(f"donor forecast exceeds bound: {projected}")
    ticks = np.linspace(0, 239, 17).round().astype(int)
    homs, rows = align(ticks)
    plates = []
    holes = []
    for plane, scale in enumerate((1, 0.5, 0.5)):
        plate, missing = pilot.plate_for(
            [pilot.planes(frames[t])[plane] for t in ticks], masks[ticks], homs, scale
        )
        plates.append(plate)
        holes.append(missing)
    cleaned = np.memmap(OUT / "clean.yuv", np.uint8, "w+", shape=frames.shape)
    for tick in range(240):
        current = np.empty((3, 3))
        for y in range(3):
            for x in range(3):
                current[y, x] = np.interp(tick, ticks, [h[y, x] for h in homs])
        inverse = np.linalg.inv(current)
        cleaned[tick] = frames[tick]
        for plane, (dst, plate, scale) in enumerate(
            zip(pilot.planes(cleaned[tick]), plates, (1, 0.5, 0.5), strict=True)
        ):
            matrix = np.diag([scale, scale, 1])
            transform = matrix @ inverse @ np.linalg.inv(matrix)
            support = (
                masks[tick] > 0
                if scale == 1
                else cv2.resize(masks[tick], (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0
            )
            warp = cv2.warpPerspective(
                plate,
                transform,
                (int(W * scale), int(H * scale)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            dst[support] = warp[support]
            if not np.array_equal(dst[~support], pilot.planes(frames[tick])[plane][~support]):
                raise ValueError("outside-support pixels changed")
    cleaned.flush()
    report = {
        "source_sha256": sha(SOURCE),
        "source_yuv_sha256": sha(source),
        "carrier_sha256": sha(CARRIER),
        "ball_payload_sha256": sha(BALL),
        "clean_sha256": sha(OUT / "clean.yuv"),
        "frames": 240,
        "fps": 25,
        "donors": rows,
        "holes_yuv": holes,
        "mask_coverage": int(np.count_nonzero(masks)),
        "outside_support_changed": 0,
        "smoke_donors": smoke_rows,
        "smoke_holes": smoke_holes,
        "smoke_cpu": smoke_cpu,
        "projected_cpu": projected,
        "cpu_seconds": cpu() - start,
        "ball_positions": sum(p is not None for p in positions),
        "scope": "Source-derived donor fill with source-model boxes and fitted ball hints; no human labels, no per-pixel coverage, quality or receiver proof.",
    }
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUT / "result.json"),
                "holes_yuv": holes,
                "cpu_seconds": report["cpu_seconds"],
                "projected_cpu": projected,
            }
        )
    )


if __name__ == "__main__":
    main()
