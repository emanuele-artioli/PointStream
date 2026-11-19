"""Source-derived court plate, masked replacement and matched AV1 CPU pilot."""

import argparse
import base64
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import warnings

import cv2
import numpy as np
from experiments.tennis import common_player_packet as packet
from src.codecs import svtav1

ROOT = Path("/Users/manu/Datasets/pointstream-submission-20261118")
SOURCE = ROOT / "ps-native-ctl-20261118-full/source.yuv"
CARRIER = Path(
    "/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp"
)
BALL = Path("/private/tmp/ps-low-ball-stage-20261118/ball-input.npz")
LABELS = (
    ROOT
    / "foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json"
)
W, H, N = 960, 544, 240


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    return sum(
        r.ru_utime + r.ru_stime
        for r in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def mask_at(t, frame, tracks, coords):
    mask = np.zeros((H, W), np.uint8)
    for run in tracks.values():
        x, y, w, h = run[0]["boxes"][t]
        x, y, w, h = x / 2, y * H / 1080, w / 2, h * H / 1080
        x0 = max(0, math.floor(x - w * 0.25))
        x1 = min(W, math.ceil(x + w * 1.25))
        y0 = max(0, math.floor(y - h * 0.15))
        y1 = min(H, math.ceil(y + h * 1.15 + 12))
        mask[y0:y1, x0:x1] = 255
    x, y = coords[t]
    if x >= 0:
        x0 = max(0, int(x / 2) - 5)
        y0 = max(0, int(y * H / 1080) - 5)
        mask[
            y0 : min(H, math.ceil((y + 32) * H / 1080) + 5),
            x0 : min(W, math.ceil((x + 32) / 2) + 5),
        ] = 255
    # Sender-side color cue over the court, including ticks with missing tracks.
    bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 85, 130), (44, 255, 255))
    yellow[:145] = 0
    yellow[:, :100] = 0
    yellow[:, 850:] = 0
    count, components, stats, _ = cv2.connectedComponentsWithStats(yellow)
    small = np.zeros_like(mask)
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if 1 <= area <= 120 and w <= 40 and h <= 25:
            small[components == k] = 255
    return cv2.bitwise_or(mask, cv2.dilate(small, np.ones((9, 9), np.uint8)))


def planes(frame):
    return [
        frame[:H],
        frame[H:].reshape(2, H // 2, W // 2)[0],
        frame[H:].reshape(2, H // 2, W // 2)[1],
    ]


def plate_for(values, mask_values, homographies, scale):
    size = (int(W * scale), int(H * scale))
    warped = []
    for frame, mask, hom in zip(values, mask_values, homographies):
        s = np.diag([scale, scale, 1])
        trans = s @ hom @ np.linalg.inv(s)
        result = cv2.warpPerspective(frame, trans, size, flags=cv2.INTER_LINEAR).astype(np.float32)
        valid = (
            cv2.warpPerspective(
                cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
                trans,
                size,
                flags=cv2.INTER_NEAREST,
                borderValue=255,
            )
            == 0
        )
        result[~valid] = np.nan
        warped.append(result)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plate = np.nanmedian(np.stack(warped), axis=0)
    missing = ~np.isfinite(plate)
    plate[missing] = 0
    return cv2.inpaint(
        np.rint(plate).astype(np.uint8), missing.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA
    ), int(missing.sum())


def build(out):
    if (
        sha(SOURCE) != "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c"
        or sha(CARRIER) != "3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd"
    ):
        raise ValueError("source/carrier identity")
    tracks = packet.unpack(CARRIER.read_bytes())[2]
    coords = np.load(BALL, allow_pickle=False)["coords"]
    src = np.memmap(SOURCE, dtype=np.uint8, mode="r", shape=(N, H * 3 // 2, W))
    masks = np.stack([mask_at(t, src[t], tracks, coords) for t in range(N)])
    ticks = np.linspace(0, N - 1, 17).round().astype(int)
    orb = cv2.ORB_create(nfeatures=8000)
    base = cv2.cvtColor(src[0], cv2.COLOR_YUV2BGR_I420)
    keys0, d0 = orb.detectAndCompute(base, cv2.bitwise_not(masks[0]))
    homs = []
    matches = []
    for t in ticks:
        frame = cv2.cvtColor(src[t], cv2.COLOR_YUV2BGR_I420)
        keys, desc = orb.detectAndCompute(frame, cv2.bitwise_not(masks[t]))
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, d0, k=2)
        good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
        a = np.float32([keys[m.queryIdx].pt for m in good])
        b = np.float32([keys0[m.trainIdx].pt for m in good])
        hom, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if hom is None or int(inliers.sum()) < 50:
            raise ValueError("insufficient camera alignment")
        homs.append(hom / hom[2, 2])
        matches.append(dict(tick=int(t), inliers=int(inliers.sum()), matches=len(good)))
    plates = []
    holes = []
    for p, scale in zip(range(3), (1, 0.5, 0.5)):
        plate, hole = plate_for([planes(src[t])[p] for t in ticks], masks[ticks], homs, scale)
        plates.append(plate)
        holes.append(hole)
    cv2.imwrite(str(out / "plate-y.png"), plates[0])
    frames = np.memmap(out / "clean.yuv", dtype=np.uint8, mode="w+", shape=src.shape)

    for t in range(N):
        current = np.empty((3, 3))
        for r in range(3):
            for c in range(3):
                current[r, c] = np.interp(t, ticks, [h[r, c] for h in homs])
        inverse = np.linalg.inv(current)
        frames[t] = src[t]
        for p, (destination, plate, scale) in enumerate(
            zip(planes(frames[t]), plates, (1, 0.5, 0.5))
        ):
            s = np.diag([scale, scale, 1])
            trans = s @ inverse @ np.linalg.inv(s)
            support = (
                masks[t] > 0
                if p == 0
                else cv2.resize(masks[t], (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0
            )
            warped = cv2.warpPerspective(
                plate,
                trans,
                (int(W * scale), int(H * scale)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            destination[support] = warped[support]
            if np.any(destination[~support] != planes(src[t])[p][~support]):
                raise ValueError("outside modified")
        for tick in (10, 90, 170):
            if t == tick:
                cv2.imwrite(
                    str(out / f"tick-{t}-source.png"), cv2.cvtColor(src[t], cv2.COLOR_YUV2BGR_I420)
                )
                cv2.imwrite(
                    str(out / f"tick-{t}-clean.png"),
                    cv2.cvtColor(frames[t], cv2.COLOR_YUV2BGR_I420),
                )
                cv2.imwrite(str(out / f"tick-{t}-mask.png"), masks[t])
    frames.flush()
    reviewed = json.loads(LABELS.read_text())["frames"]
    coverage = []
    for item in reviewed:
        t = item["tick"]
        body = np.zeros((1080, 1920), np.uint8)
        for player in item["players"]:
            raw = base64.b64decode(player["mask"]["png_base64"])
            m = cv2.imdecode(np.frombuffer(raw, np.uint8), 0)
            body |= m
        body = cv2.resize(body, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ball = item["ball"]
        center = ball.get("center_xy")
        ball_covered = (
            None
            if center is None
            else bool(
                masks[t, min(H - 1, int(center[1] * H / 1080)), min(W - 1, int(center[0] / 2))]
            )
        )
        coverage.append(
            dict(
                tick=t,
                body_pixels=int(body.sum()),
                body_covered=int((body & (masks[t] > 0)).sum()),
                ball=ball,
                ball_center_covered=ball_covered,
            )
        )
    np.save(out / "removal-masks.npy", masks > 0)
    return dict(
        source_sha256=sha(SOURCE),
        clean_sha256=sha(out / "clean.yuv"),
        carrier_sha256=sha(CARRIER),
        ball_input_sha256=sha(BALL),
        labels_sha256=sha(LABELS),
        donors=matches,
        plate_spatial_inpaint_pixels=holes,
        masks_pixels=int((masks > 0).sum()),
        reviewed_coverage=coverage,
        outside_support_changed=0,
    )


def encode(source, out, frames):
    return svtav1.code(
        source,
        out,
        width=W,
        height=H,
        fps=Fraction(50),
        frames=frames,
        crf=48,
        preset=4,
        threads=2,
        full_range=False,
        encoder="/opt/homebrew/bin/SvtAv1EncApp",
        decoder="/opt/homebrew/bin/dav1d",
        timeout=140,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    begin = cpu()
    cv2.setNumThreads(1)
    args.output.mkdir(exist_ok=False, parents=True)
    prep = build(args.output)
    smoke = []
    for mode, source in [("plain", SOURCE), ("clean", args.output / "clean.yuv")]:
        target = args.output / "smoke" / mode
        target.mkdir(parents=True)
        smoke.append(encode(source, target, 24))
    elapsed = cpu() - begin
    projected = (
        elapsed
        + 2 * sum(row["encode_cpu_seconds"] + row["decode_cpu_seconds"] for row in smoke) * 10
    )
    if projected > 600 or time.monotonic() - start > 260:
        raise ValueError(f"projected CPU exceeds cap: {projected}")
    rows = []
    for mode, source in [("plain", SOURCE), ("clean", args.output / "clean.yuv")]:
        target = args.output / "full" / mode
        target.mkdir(parents=True)
        row = encode(source, target, 240)
        if row["temporal_units"] != 240 or row["decoded_frames"] != 240:
            raise ValueError("incomplete decode")
        row.update(
            mode=mode,
            decoded_sha256=sha(Path(row["decoded"])),
            complete_kbps=row["file_bytes"] / 600,
        )
        rows.append(row)
    result = dict(
        preparation=prep,
        smoke=smoke,
        rows=rows,
        cpu_seconds=cpu() - begin,
        wall_seconds=time.monotonic() - start,
        script_sha256=sha(Path(__file__)),
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        opencv=cv2.__version__,
        numpy=np.__version__,
    )
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("cpu_seconds", "wall_seconds")}))
    print([(r["mode"], r["file_bytes"]) for r in rows])


if __name__ == "__main__":
    main()
