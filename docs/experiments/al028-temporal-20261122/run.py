"""Frozen Al028 12-pair source-flow display metric; source-model body supports."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.pointstream.tracked_patch import yuv_to_rgb

ROOT = Path("/private/tmp")
SOURCE = ROOT / "ps-al028-review-source-20261122.mp4"
AV1 = ROOT / "ps-nov22-al028-exact-budget/ordinary-52.75/decoded.yuv"
DCVC = ROOT / "ps-nov22-al028-q24-full/full.yuv"
REVIEW = ROOT / "ps-al028-source-annotations-20261122"
EXPECTED = {
    SOURCE: "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934",
    AV1: "c60215a513435b726bf9c3dd398b651b1c9add176245a354eb1076ee21445bf1",
    DCVC: "bac9b77792126c748a7640eee2ce80649f54c2509d5973623d06e1ed35b931bb",
}
TICKS = list(range(10, 240, 20))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def main(a):
    began = time.process_time()
    cv2.setNumThreads(1)
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ValueError(f"input identity: {path}")
    review = json.loads((REVIEW / "review.json").read_text())
    if (
        review["source_video_sha256"] != EXPECTED[SOURCE]
        or review["source_video_fps"] != [25, 1]
        or [r["frame_index"] for r in review["frames"]] != TICKS
    ):
        raise ValueError("review source identity or cadence")
    labels = {r["frame_index"]: r for r in review["frames"]}
    ordinary = np.memmap(AV1, np.uint8, "r", shape=(240, 288, 320))
    neural = np.memmap(DCVC, np.uint8, "r", shape=(240, 816, 960))
    arms = {"ordinary": ordinary, "neural": neural}
    sizes = {"ordinary": (320, 192), "neural": (960, 544)}
    rates = {"ordinary": 20.360833333333332, "neural": 20.411666666666665}
    joint = None
    if a.joint:
        info_path = a.joint / "result.json"
        if sha(info_path) != a.result_sha256:
            raise ValueError("joint result identity")
        info = json.loads(info_path.read_text())
        if info["frames"] != 240 or abs(info["joint_warm_kbps"]["20"] - 20.4075) > 1e-8:
            raise ValueError("joint frame count or rate")
        joint = np.load(a.joint / "frames-20.npy", mmap_mode="r")
        if joint.shape != (240, 1080, 1920, 3) or joint.dtype != np.uint8:
            raise ValueError("joint dimensions")
        h = hashlib.sha256()
        for f in joint:
            h.update(f)
        if h.hexdigest() != info["rgb_stream_sha256"]["20"]:
            raise ValueError("full joint RGB digest")
        rates["joint_cached"] = 20.4075
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    if not cap.isOpened() or [
        cap.get(x)
        for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_COUNT,
        )
    ] != [1920, 1080, 25, 492]:
        raise ValueError("source decode geometry/cadence")
    selected = set(TICKS[: a.count])
    needed = selected | {t - 1 for t in selected}
    source = {}
    for t in range(max(needed) + 1):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("short native source")
        if t in selected:
            r = labels[t]
            p = REVIEW / r["source"]
            if sha(p) != r["source_sha256"] or not np.array_equal(cv2.imread(str(p)), bgr):
                raise ValueError("source review pixel identity")
        if t in needed:
            source[t] = cv2.resize(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (960, 544), interpolation=cv2.INTER_AREA
            )
    cap.release()
    yy, xx = np.mgrid[:544, :960]
    rows = []

    def display(arr, t, w, h):
        image = cv2.resize(
            yuv_to_rgb(arr[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )
        return cv2.resize(image, (960, 544), interpolation=cv2.INTER_AREA)

    for index, t in enumerate(sorted(selected)):
        flow = dis_flow(
            cv2.cvtColor(source[t], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(source[t - 1], cv2.COLOR_RGB2GRAY),
        )
        x, y = xx + flow[:, :, 0], yy + flow[:, :, 1]
        valid = (x >= 0) & (y >= 0) & (x < 959) & (y < 543)
        masks = {"whole": valid}
        for person in labels[t]["players"]:
            p = REVIEW / person["proposal"]
            if sha(p) != person["sha256"]:
                raise ValueError("proposal identity")
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (1080, 1920) or int(np.count_nonzero(m)) != person["pixels"]:
                raise ValueError("proposal geometry")
            ys, xs = np.nonzero(m)
            x0 = max(0, int(xs.min()) - 32)
            x1 = min(1920, int(xs.max()) + 33)
            y0 = max(0, int(ys.min()) - 32)
            y1 = min(1080, int(ys.max()) + 33)
            bounded = np.zeros_like(m)
            bounded[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            masks[person["id"]] = valid & (
                cv2.resize(bounded, (960, 544), interpolation=cv2.INTER_NEAREST) > 0
            )
        current = {"source": source[t]}
        previous = {"source": source[t - 1]}
        for name, arr in arms.items():
            w, h = sizes[name]
            current[name] = display(arr, t, w, h)
            previous[name] = display(arr, t - 1, w, h)
        if joint is not None:
            current["joint_cached"] = cv2.resize(joint[t], (960, 544), interpolation=cv2.INTER_AREA)
            previous["joint_cached"] = cv2.resize(
                joint[t - 1], (960, 544), interpolation=cv2.INTER_AREA
            )
        delta = {k: luma(v) - luma(warp(previous[k], flow)) for k, v in current.items()}
        for region, mask in masks.items():
            count = int(mask.sum())
            if count == 0:
                raise ValueError("empty valid mask")
            for name, residual in delta.items():
                error = float(np.mean(np.abs(residual[mask] - delta["source"][mask])))
                if name == "source" and error != 0:
                    raise ValueError("source self error")
                rows.append(
                    {
                        "tick": t,
                        "prior_tick": t - 1,
                        "region": region,
                        "arm": name,
                        "valid_pixels": count,
                        "error": error,
                        "energy": float(np.mean(np.abs(residual[mask]))),
                    }
                )
        elapsed = time.process_time() - began
        if index == 1 and a.count == 2 and elapsed * 6 * 1.5 + 30 >= 600:
            raise ValueError("full projection exceeds CPU budget")
        if elapsed >= 600:
            raise ValueError("CPU budget exceeded")
    means = {
        name: {
            region: {
                key: float(
                    np.mean([r[key] for r in rows if r["arm"] == name and r["region"] == region])
                )
                for key in ("energy", "error")
            }
            for region in masks
        }
        for name in current
    }
    output = {
        "frames": 240,
        "fps": 25,
        "ticks": sorted(selected),
        "means": means,
        "rows": rows,
        "rates_kbps": rates,
        "source_sha256": EXPECTED[SOURCE],
        "ordinary_yuv_sha256": EXPECTED[AV1],
        "neural_yuv_sha256": EXPECTED[DCVC],
        "review_sha256": sha(REVIEW / "review.json"),
        "opencv": cv2.__version__,
        "cpu_seconds": time.process_time() - began,
        "joint_result_sha256": a.result_sha256,
    }
    if a.controls:
        old = json.loads(a.controls.read_text())
        for name in ("source", "ordinary", "neural"):
            if means[name] != old["means"][name] or [r for r in rows if r["arm"] == name] != [
                r for r in old["rows"] if r["arm"] == name
            ]:
                raise ValueError("frozen control reproduction mismatch")
        output["controls_exact"] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps(means))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, choices=(2, 12), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--joint", type=Path)
    p.add_argument("--result-sha256")
    p.add_argument("--controls", type=Path)
    main(p.parse_args())
