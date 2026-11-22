"""Source-aligned /002 GrabCut and donor-court receiver base."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis.k6b_rate_streams import decode_keypoints
from src.codecs import svtav1

SOURCE_HASH = "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
BOX_HASHES = {
    "nadal_kyrgios__002_at120__person_000000000000": "562fc155839b75f7746392c3611fa6f53179fd801a564bbf025f50b4ba2da75a",
    "nadal_kyrgios__002_at120__person_000000000002": "396a29f77bb9b20bd74ec2740ee91cffcc8d393fe38031a5e9e8d9f68d810015",
}
W, H, TOTAL = 960, 544, 240


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def module():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/clean-court-grabcut-20261119/refine.py"
    )
    spec = importlib.util.spec_from_file_location("court_refine", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def extract(path, name):
    with tarfile.open(path) as archive:
        f = archive.extractfile(name)
        if f is None:
            raise ValueError(f"missing {name}")
        return f.read()


def run(args):
    start = time.monotonic()
    cv2.setNumThreads(1)
    if len(os.sched_getaffinity(0)) != 1 or sha(args.source) != SOURCE_HASH:
        raise ValueError("source identity or CPU affinity")
    n = args.frames
    if n != (24 if os.environ["PS_STAGE"] == "smoke" else TOTAL):
        raise ValueError("stage frames")
    refine = module()
    pilot = refine.pilot
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    work = scratch / "court002"
    work.mkdir()
    publish = scratch / "publish"
    publish.mkdir()
    ff, encoder, decoder = (svtav1.tool(x) for x in ("ffmpeg", "SvtAv1EncApp", "dav1d"))
    source = work / "source.yuv"
    command = [
        ff["path"],
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(args.source),
        "-vf",
        "scale=960:544:flags=area,format=yuv420p",
        "-frames:v",
        str(n),
        "-f",
        "rawvideo",
        str(source),
    ]
    subprocess.run(command, check=True, timeout=300)
    if source.stat().st_size != n * W * H * 3 // 2:
        raise ValueError("source frame count")
    src = np.memmap(source, np.uint8, "r", shape=(n, H * 3 // 2, W))
    kps = extract(args.transport, "publish/packets/nadal_kyrgios__002@120/keypoints.kps")
    meta, units = decode_keypoints(kps)
    if (
        meta["native_frames"] != TOTAL
        or meta["fps"] != [50, 1]
        or meta["clip_local_source_start"] != 0
    ):
        raise ValueError("keypoint timeline")
    tracks = {}
    for unit in units:
        name = unit["name"]
        if name not in BOX_HASHES or not np.array_equal(unit["t"], np.arange(TOTAL)):
            raise ValueError("keypoint unit")
        blob = extract(args.boxes, "publish/" + name + ".npz")
        if hashlib.sha256(blob).hexdigest() != BOX_HASHES[name]:
            raise ValueError("box NPZ identity")
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            if not np.array_equal(data["t"], unit["t"]) or data["box"].shape != (TOTAL, 4):
                raise ValueError("box timeline")
            boxes = data["box"].copy()
        tracks[name] = [{"boxes": boxes, "kp": unit["kp"]}]
    if set(tracks) != set(BOX_HASHES):
        raise ValueError("incomplete player set")
    coords = np.full((n, 2), -1, dtype=np.int32)
    track_hash = None
    if args.ball:
        track_hash = sha(args.ball)
        if track_hash != "43a816146a72daebfab62db5a87dcb651d431aa408feb859f7df9b7dc60f0859":
            raise ValueError("ball track identity")
        track = json.loads(args.ball.read_text())
        if (
            track["source_sha256"] != SOURCE_HASH
            or track["canvas_size"] != [1920, 1080]
            or track["fps"] != [50, 1]
            or len(track["centers"]) != TOTAL
        ):
            raise ValueError("ball track alignment")
        for t, point in enumerate(track["centers"][:n]):
            if point is not None:
                coords[t] = [
                    max(0, min(1888, int(point[0]) - 16)),
                    max(0, min(1048, int(point[1]) - 16)),
                ]
    masks = np.stack([refine.mask_at(t, src[t], tracks, coords) for t in range(n)])
    ticks = np.linspace(0, n - 1, 3 if n == 24 else 17).round().astype(int)
    orb = cv2.ORB_create(nfeatures=8000)
    image0 = cv2.cvtColor(src[0], cv2.COLOR_YUV2BGR_I420)
    key0, desc0 = orb.detectAndCompute(image0, cv2.bitwise_not(masks[0]))
    homs, donors = [], []
    for tick in ticks:
        image = cv2.cvtColor(src[tick], cv2.COLOR_YUV2BGR_I420)
        keys, desc = orb.detectAndCompute(image, cv2.bitwise_not(masks[tick]))
        if desc is None or desc0 is None:
            raise ValueError("missing ORB features")
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, desc0, k=2)
        good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
        if len(good) < 50:
            raise ValueError("insufficient feature correspondences")
        a = np.float32([keys[m.queryIdx].pt for m in good])
        b = np.float32([key0[m.trainIdx].pt for m in good])
        hom, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if hom is None or int(inliers.sum()) < 50:
            raise ValueError("insufficient ORB alignment")
        homs.append(hom / hom[2, 2])
        donors.append({"tick": int(tick), "matches": len(good), "inliers": int(inliers.sum())})
    plates = []
    holes = []
    for plane, scale in enumerate((1, 0.5, 0.5)):
        plate, hole = pilot.plate_for(
            [pilot.planes(src[t])[plane] for t in ticks], masks[ticks], homs, scale
        )
        plates.append(plate)
        holes.append(hole)
    clean_path = publish / "clean.yuv"
    frames = np.memmap(clean_path, np.uint8, "w+", shape=src.shape)
    changed_outside = 0
    for t in range(n):
        hom = np.array(
            [[np.interp(t, ticks, [h[r, c] for h in homs]) for c in range(3)] for r in range(3)]
        )
        inverse = np.linalg.inv(hom)
        frames[t] = src[t]
        for plane, (destination, plate, scale) in enumerate(
            zip(pilot.planes(frames[t]), plates, (1, 0.5, 0.5))
        ):
            s = np.diag([scale, scale, 1])
            trans = s @ inverse @ np.linalg.inv(s)
            support = (
                masks[t] > 0
                if plane == 0
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
            changed_outside += int(
                np.count_nonzero(destination[~support] != pilot.planes(src[t])[plane][~support])
            )
    frames.flush()
    if changed_outside:
        raise ValueError("pixels outside mask changed")
    rows = []
    for mode, path in (("original", source), ("clean", clean_path)):
        low = work / (mode + "-low.yuv")
        subprocess.run(
            [
                ff["path"],
                "-v",
                "error",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv420p",
                "-video_size",
                "960x544",
                "-framerate",
                "50",
                "-i",
                str(path),
                "-vf",
                "scale=240:136:flags=area,format=yuv420p",
                "-frames:v",
                str(n),
                "-f",
                "rawvideo",
                str(low),
            ],
            check=True,
            timeout=300,
        )
        target = work / mode
        target.mkdir()
        row = svtav1.code(
            low,
            target,
            width=240,
            height=136,
            fps=Fraction(50),
            frames=n,
            crf=60,
            preset=4,
            threads=1,
            full_range=False,
            encoder=encoder["path"],
            decoder=decoder["path"],
            timeout=600,
        )
        if row["decoded_frames"] != n or row["temporal_units"] != n:
            raise ValueError("IVF frame count")
        stream = publish / (mode + "-crf60.ivf")
        shutil.copyfile(row["stream"], stream)
        rows.append(
            {
                "mode": mode,
                "bytes": stream.stat().st_size,
                "kbps": stream.stat().st_size * 8 * 50 / n / 1000,
                "stream_sha256": sha(stream),
                "decoded_sha256": sha(Path(row["decoded"])),
                "low_source_sha256": sha(low),
            }
        )
    report = {
        "stage": os.environ["PS_STAGE"],
        "frames": n,
        "fps": 50,
        "source_mp4_sha256": sha(args.source),
        "source_yuv_sha256": sha(source),
        "keypoints_sha256": hashlib.sha256(kps).hexdigest(),
        "box_hashes": BOX_HASHES,
        "mask_policy": (
            "paid player poses and boxes, source yellow cue and sender ball-track windows"
            if args.ball
            else "paid player poses and boxes plus source yellow cue; no authenticated ball track"
        ),
        "ball_track_sha256": track_hash,
        "ball_track_present": int(np.count_nonzero(coords[:, 0] >= 0)),
        "mask_pixels": int(np.count_nonzero(masks)),
        "donors": donors,
        "holes_yuv": holes,
        "outside_support_changed": changed_outside,
        "clean_yuv_sha256": sha(clean_path),
        "clean_yuv_bytes": clean_path.stat().st_size,
        "outputs": rows,
        "encoder": encoder,
        "decoder": decoder,
        "ffmpeg": ff,
        "opencv": cv2.__version__,
        "wall_seconds": time.monotonic() - start,
    }
    output = json.dumps(report, indent=2) + "\n"
    (publish / "receipt.json").write_text(output)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(output)


def validate(args):
    root = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((root / "result.json").read_text())
    n = 24 if r["stage"] == "smoke" else 240
    checks = {
        "frames": r["frames"] == n and r["fps"] == 50,
        "source": r["source_mp4_sha256"] == SOURCE_HASH,
        "mask": r["mask_pixels"] > 0 and r["outside_support_changed"] == 0,
        "donors": len(r["donors"]) == (3 if n == 24 else 17)
        and all(x["inliers"] >= 50 for x in r["donors"]),
        "outputs": len(r["outputs"]) == 2,
        "raw_size": r["clean_yuv_bytes"] == n * W * H * 3 // 2,
    }
    if n == 24:
        checks["forecast"] = r["wall_seconds"] * 10 * 1.5 + 120 < args.full_limit
    if args.require_ball:
        checks["ball_track"] = (
            r["ball_track_sha256"]
            == "43a816146a72daebfab62db5a87dcb651d431aa408feb859f7df9b7dc60f0859"
            and r["ball_track_present"] > 0
        )
        if n == 240:
            checks["ball_count"] = r["ball_track_present"] == 231
    with tarfile.open(root / "published.tar") as archive:
        for row in r["outputs"]:
            blob = archive.extractfile("publish/" + row["mode"] + "-crf60.ivf").read()
            checks[row["mode"]] = (
                hashlib.sha256(blob).hexdigest() == row["stream_sha256"]
                and len(svtav1.ivf_frames(blob)) == n
            )
        entry = archive.getmember("publish/clean.yuv")
        checks["raw_archive"] = entry.size == r["clean_yuv_bytes"]
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "projected_full_seconds": r["wall_seconds"] * 10 * 1.5 + 120 if n == 24 else None,
    }
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(str(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "boxes", "transport", "ball"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--require-ball", action="store_true")
    p.add_argument("--full-limit", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate(args) if args.validate else run(args)


if __name__ == "__main__":
    main()
