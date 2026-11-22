"""Apply K6's original full-height/crop decision to both frozen TEST source tracks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import k4, k4b_features, k4b_pose, k6
from experiments.visor.h3 import rgb_memmap
from experiments.tennis.player_requalification import sha256
from src.segmentation.masks import ClipMasks, decompress


def median_heights(clip, aliases, ids):
    all_heights = {p: [] for p in ids}
    for frame in clip.frames:
        for p, small in k4b_pose.player_masks(frame, aliases, ids).items():
            body = k4.upsample(small, k6.FRAME)
            ys = np.flatnonzero(body.any(axis=1))
            if ys.size:
                all_heights[p].append(int(ys[-1] - ys[0] + 1))
    return {p: float(np.median(v)) if v else 0.0 for p, v in all_heights.items()}


def main(a):
    import torch

    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        r = json.loads((stage / "result.json").read_text())
        count = r["frames"]
        assert count in (24, 240)
        assert len(r["tracks"]) == len(r["selected_ids"]) <= 2 and r["gpu"]["cuda"]
        for p, v in r["tracks"].items():
            assert (
                0 <= v["fitted"] <= v["mask"] <= count and v["fitted"] + len(v["missing"]) == count
            )
            assert v["setting"] == ("whole" if r["median_height_px"][p] >= 162 else "crop1")
            assert v["shape"] == [count, 17, 3] and len(v["mask_box_sha256"]) == 64
        assert (
            len(r["output_archive_sha256"]) == 64
            and sha256(stage / "published.tar") == r["output_archive_sha256"]
        )
        if count == 24 and r["estimate_full_s"] >= 655:
            raise ValueError(f"Forecast {r['estimate_full_s']:.1f} exceeds 655 s allowance")
        verdict = dict(
            passed=True,
            checks=dict(
                source=r["source_sha256"],
                selected=r["selected_ids"],
                coverage={p: v["fitted"] for p, v in r["tracks"].items()},
                crop_inferences=r["crop_count"],
                forecast=r["estimate_full_s"],
            ),
        )
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(verdict) + "\n")
        return
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and a.frames in (24, 240)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    began = time.monotonic()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    pub = scratch / "publish"
    pub.mkdir(exist_ok=False)
    root = a.features / "publish"
    original = json.loads((root / "admission.json").read_text())
    prefix = root / ("test__" + a.event + "@120")
    win = json.loads((prefix / "window.json").read_text())
    assert (
        original["source_sha256"] == win["sha256"] == sha256(a.source) == a.source_sha
        and win["start"] == 0
        and win["frames"] == 240
    )
    for name, digest in win["files"].items():
        assert sha256(prefix / name) == digest
    assert sha256(a.pose) == "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"
    clip = ClipMasks.load(prefix / "sam-masks.rle")
    features = json.loads((prefix / "features.json").read_text())
    aliases = features["aliases"]
    whole = json.loads(decompress((prefix / "pose.json.zst").read_bytes()))
    assert len(clip.frames) == len(whole) == 240
    ids = original["ids"]
    assert len(set(ids)) == len(ids) and 1 <= len(ids) <= 2
    heights = median_heights(clip, aliases, ids)
    yuv = scratch / "source.yuv"
    conv = k4b_features.convert("ffmpeg", dict(win, frames=a.frames), a.source, yuv)
    if a.frames == 240:
        assert conv["sha256"] == win["conversion"]["sha256"]
    rgb = rgb_memmap(yuv, scratch / "rgb.npy", a.frames, 1920, 1080, False, "cuda")
    maskboxes = {p: [] for p in ids}
    masks = []
    for t in range(a.frames):
        frame = {
            p: k4.upsample(mask, k6.FRAME)
            for p, mask in k4b_pose.player_masks(clip.frames[t], aliases, ids).items()
            if mask.any()
        }
        for p in ids:
            bounds = cv2.boundingRect(frame[p].astype(np.uint8)) if p in frame else (0, 0, 0, 0)
            maskboxes[p].append(bounds)
        masks.append(frame)
    runtime = time.monotonic()
    models = k4.Poses([("yolo11x-pose", str(a.pose))])
    crops = k4b_pose.Crops(models)
    results = k6.pose_window(crops, rgb, a.frames, masks, whole[: a.frames], ids, heights)
    meta = models.record()
    if crops.count["crop1"] and not all(v.startswith("cuda") for v in meta["on_cuda"].values()):
        raise ValueError("Native pose estimator did not execute on CUDA")
    rows = {}
    for p in ids:
        rr = results[p]
        fit = np.array([bool(v and v.get("fitted")) for v in rr])
        kps = np.zeros((a.frames, 17, 3), np.float32)
        for t in np.flatnonzero(fit):
            kps[t] = np.asarray(rr[t]["kps"], np.float32)
        mb = np.asarray(maskboxes[p], np.int32)
        assert np.isfinite(kps).all() and ((kps[:, :, 2] >= 0) & (kps[:, :, 2] <= 1)).all()
        pfile = pub / (p.replace(":", "_") + ".npz")
        np.savez_compressed(
            pfile, t=np.arange(a.frames, dtype=np.int32), kp=kps, fitted=fit, mask_bounds_xywh=mb
        )
        confidences = (kps[:, :, 2] >= k4.KP_CONF).sum(axis=1)
        rows[p] = dict(
            fitted=int(fit.sum()),
            mask=int(np.count_nonzero(mb[:, 2] * mb[:, 3])),
            missing=np.flatnonzero(~fit).tolist(),
            crop_confident_joints_min=int(confidences[fit].min()) if fit.any() else None,
            confident_joints_median=float(np.median(confidences[fit])) if fit.any() else None,
            whole_fitted=sum(bool(v and v["whole_fitted"]) for v in rr),
            setting="whole" if heights[p] >= 162 else "crop1",
            shape=list(kps.shape),
            kp_sha256=hashlib.sha256(kps.tobytes()).hexdigest(),
            mask_box_sha256=hashlib.sha256(mb.tobytes()).hexdigest(),
            npz_sha256=sha256(pfile),
        )
    wall = time.monotonic() - began
    inference = time.monotonic() - runtime
    projected = 1.5 * (wall - inference + inference * 240 / a.frames) + 65
    summary = dict(
        event=a.event,
        source_sha256=a.source_sha,
        prior_feature_archive_sha256=a.feature_sha,
        frames=a.frames,
        selected_ids=ids,
        median_height_px=heights,
        tracks=rows,
        whole_frame_original=original["tracks"],
        native_rgb_source_yuv_sha256=conv["sha256"],
        supported_rules={
            "crop_below_fraction": k6.WHOLE_FRAME_ABOVE,
            "pose": k4.POSE_CONF,
            "joint": k4.KP_CONF,
        },
        fitted_both=len(ids) == 2 and all(rows[p]["fitted"] == a.frames for p in ids),
        trained_weights_sha256=sha256(a.pose),
        model=meta,
        gpu=dict(
            cuda=torch.cuda.is_available(),
            name=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
        ),
        crop_count=crops.count["crop1"],
        yolo_inference_s=crops.seconds["crop1"],
        registration_wall_s=runtime - began,
        total_s=wall,
        estimate_full_s=projected,
    )
    (pub / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    dest = stage / "published.tar"
    with tarfile.open(dest, "x") as z:
        z.add(pub, arcname="publish")
    summary["output_archive_sha256"] = sha256(dest)
    (stage / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("source", "features", "pose"):
        p.add_argument("--" + n, type=Path)
    for n in ("event", "feature-sha", "source-sha"):
        p.add_argument("--" + n)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    main(p.parse_args())
