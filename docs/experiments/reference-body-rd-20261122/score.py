"""Score frozen appearance-rung renders on source-aligned body masks."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import k4, k6b
from experiments.tennis.k4b_pose import player_masks
from experiments.visor.h3b import dists_crops
from src.codecs import quality
from src.segmentation.masks import ClipMasks, decompress


HERE = Path(__file__).resolve().parent
RGB = HERE.parent / "common-reference-q90-20261122/rendered-crop-metrics.json"
LABEL_SHA = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
ANCHOR_SHA = "965b8879a821b90dd1e6afba415b8eed269a35a768e1f9891e0f4df96a37a51a"
WEIGHT_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
MODES = {"q65": "avif_q65", "q78": "avif_q78", "q90": "avif_q90", "lossless": "lossless"}
W, H = 1920, 1080


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_rgb(source, raw=None):
    # Match the original player_requalification_preview.source_crops conversion.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(source),
        "-frames:v",
        "24",
        "-vf",
        r"select=between(n\,0\,23),scale=1920:1080:flags=area:out_range=tv,format=yuv420p",
        "-filter_threads",
        "1",
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "pipe:1",
    ]
    data = raw.read_bytes() if raw else subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(data, dtype=np.uint8).reshape(24, H * 3 // 2, W)
    result = []
    for frame in frames:
        y = (frame[:H].astype(np.float32) - 16.0) / 219.0
        u, v = frame[H:].reshape(2, H // 2, W // 2).astype(np.float32)
        cb = (cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        cr = (cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        red = y + 2.0 * (1.0 - 0.2126) * cr
        blue = y + 2.0 * (1.0 - 0.0722) * cb
        green = (y - 0.2126 * red - 0.0722 * blue) / (1.0 - 0.2126 - 0.0722)
        result.append(
            np.clip(np.rint(np.stack((red, green, blue), axis=2) * 255), 0, 255).astype(np.uint8)
        )
    return result, hashlib.sha256(data).hexdigest(), cmd


def read_inputs(a):
    prior = json.loads(RGB.read_text())
    assert digest(a.scratch / "source.mp4") == prior["source_sha256"]
    assert digest(a.labels) == LABEL_SHA and digest(a.scratch / "anchors.tar") == ANCHOR_SHA
    assert digest(a.weights) == WEIGHT_SHA
    frames, raw_sha, cmd = source_rgb(a.scratch / "source.mp4", a.yuv)
    human = {}
    for row in json.loads(a.labels.read_text())["frames"]:
        if row["tick"] >= 24:
            continue
        for person in row["players"]:
            assert person["status"] == "visible"
            mask = (
                cv2.imdecode(
                    np.frombuffer(base64.b64decode(person["mask"]["png_base64"]), np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                > 0
            )
            assert mask.shape == (H, W) and mask.sum() == person["mask"]["foreground_pixels"]
            human[(row["tick"], person["id"])] = mask
    assert len(human) == 2
    prefix = "publish/nadal_kyrgios__000@120/"
    with tarfile.open(a.scratch / "anchors.tar") as f:
        aliases = json.load(f.extractfile(prefix + "features.json"))["aliases"]
        sam = ClipMasks.from_doc(
            json.loads(decompress(f.extractfile(prefix + "sam-masks.rle").read()))
        )
    assert len(sam.frames) == 240 and (sam.width, sam.height, sam.fps) == (960, 540, 50)
    people = ["person:000000000000", "person:000000000001"]
    masks = {}
    for t in range(24):
        for p, m in player_masks(sam.frames[t], aliases, people).items():
            masks[(t, p)] = k4.upsample(m, (W, H))
    assert len(masks) == 48
    renders, boxes, reproduced = {}, {}, {}
    for mode, key in MODES.items():
        archive = a.scratch / (mode + ".tar")
        assert digest(archive) == prior["archives"][key]
        with tarfile.open(archive) as f:
            for p in people:
                unit = "nadal_kyrgios__000_at120__" + p.replace(":", "_")
                member = f.extractfile("publish/" + unit + "@tennis_only.npz").read()
                reference = prior["metrics"][unit][key]
                assert hashlib.sha256(member).hexdigest() == reference["render_sha256"]
                with np.load(io.BytesIO(member), allow_pickle=False) as data:
                    assert np.array_equal(data["t"], np.arange(24))
                    box, crops = data["box"], data["crops"]
                    assert crops.shape == (24, 784, 512, 3)
                    if p in boxes:
                        assert np.array_equal(box, boxes[p])
                    boxes[p] = box
                    renders[(mode, p)] = crops
                    sse = 0
                    for t, render in enumerate(crops):
                        diff = render.astype(np.int32) - k6b.crop_image(frames[t], box[t]).astype(
                            np.int32
                        )
                        sse += int(np.square(diff).sum(dtype=np.int64))
                    db = 10 * math.log10(255**2 * crops.size / sse)
                    reproduced[f"{mode}/{p}"] = dict(observed=db, previous=reference["rgb_psnr_db"])
    return people, frames, human, masks, boxes, renders, reproduced, raw_sha, cmd


def compare(net, truth, outputs, fullmask, box):
    x, y, w, h = k6b.int_box(box)
    assert min(x, y) >= 0 and x + w <= W and y + h <= H
    mask = fullmask[y : y + h, x : x + w]
    coverage = float(mask.sum() / fullmask.sum())
    assert mask.any()
    yy, xx = np.nonzero(mask)
    # The same surrounding context for each candidate, clipped to the delivered player crop.
    ys = slice(max(0, int(yy.min()) - 32), min(h, int(yy.max()) + 33))
    xs = slice(max(0, int(xx.min()) - 32), min(w, int(xx.max()) + 33))
    ref = truth[y : y + h, x : x + w][ys, xs].copy()
    submask = mask[ys, xs].copy()
    names = list(outputs)
    crops = [outputs[name][ys, xs].copy() for name in names]
    dists = dists_crops(net, crops, ref, submask, "cpu")
    vals = {}
    for mode, crop, score in zip(names, crops, dists, strict=True):
        diff = (crop.astype(np.int32) - ref.astype(np.int32))[submask]
        sse = int(np.square(diff).sum(dtype=np.int64))
        vals[mode] = dict(dists=score, sum_squared_rgb=sse, samples=diff.size)
        assert math.isfinite(score)
    return dict(pixels=int(submask.sum()), coverage=coverage, scores=vals)


def run(a):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    core = time.process_time()
    people, frames, human, masks, boxes, renders, repro, raw_sha, cmd = read_inputs(a)
    # The existing crop scores are empirical authentication of source conversion and registration.
    worst = max(abs(v["observed"] - v["previous"]) for v in repro.values())
    print(f"Maximum full-crop reproduction difference {worst:.8f} dB", flush=True)
    assert worst < 1e-5, repro
    net = quality.load_dists(a.weights, "cpu")
    rows = []
    with torch.inference_mode():
        for t, frame in enumerate(frames):
            for p in people:
                b = boxes[p][t]
                _, _, w, h = k6b.int_box(b)
                out = {
                    m: cv2.resize(renders[(m, p)][t], (w, h), interpolation=cv2.INTER_LANCZOS4)
                    for m in MODES
                }
                if t == 0 and p == people[0]:
                    x, y, _, _ = k6b.int_box(b)
                    start = time.process_time()
                    smoke = compare(
                        net,
                        frame,
                        {"source": frame[y : y + h, x : x + w], "q65": out["q65"]},
                        masks[t, p],
                        b,
                    )
                    elapsed = time.process_time() - start
                    assert abs(smoke["scores"]["source"]["dists"]) < 1e-5
                    assert smoke["scores"]["source"]["sum_squared_rgb"] == 0
                    assert elapsed * 2 * 50 * 1.5 + time.process_time() - core < 2400
                for name, maskdict in [("source_model", masks), ("human", human)]:
                    if (t, p) not in maskdict:
                        continue
                    row = compare(net, frame, out, maskdict[t, p], b)
                    row.update(tick=t, person=p, support=name)
                    rows.append(row)
            if time.process_time() - core > 2400:
                raise RuntimeError("CPU time cap exhausted")
    summary = {}
    for name in ("source_model", "human"):
        summary[name] = {}
        for mode in MODES:
            per = {}
            for p in people:
                items = [
                    r["scores"][mode] for r in rows if r["support"] == name and r["person"] == p
                ]
                per[p] = dict(
                    dists=float(np.mean([r["dists"] for r in items])),
                    rgb_psnr=10
                    * math.log10(
                        255**2
                        * sum(r["samples"] for r in items)
                        / sum(r["sum_squared_rgb"] for r in items)
                    ),
                    ticks=len(items),
                )
            summary[name][mode] = {"per_player": per}
            for metric in ("rgb_psnr", "dists"):
                summary[name][mode][metric] = float(np.mean([v[metric] for v in per.values()]))
    result = dict(
        summary=summary,
        rows=rows,
        full_crop_controls=repro,
        source_yuv_sha256=raw_sha,
        source_sha256=digest(a.scratch / "source.mp4"),
        source_command=cmd,
        labels_sha256=LABEL_SHA,
        source_masks_tar_sha256=ANCHOR_SHA,
        weights_sha256=WEIGHT_SHA,
        script_sha256=digest(Path(__file__)),
        cpu_seconds=time.process_time() - core,
        cv2=cv2.__version__,
        numpy=np.__version__,
        torch=str(torch.__version__),
        ffmpeg_sha256=digest(Path(shutil.which("ffmpeg"))),
        cost_forecast_basis_cpu_seconds=elapsed,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps({k: {m: round(v["dists"], 5) for m, v in s.items()} for k, s in summary.items()})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("scratch", "weights", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--yuv", type=Path)
    run(parser.parse_args())
