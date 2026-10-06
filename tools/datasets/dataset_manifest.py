"""Inspect labelled datasets, smoke-read one labelled sample, and write read-only manifests.

Runs on a GPU host. Archives are copied to host-local scratch (``/dev/shm``) and
extracted there; nothing is extracted onto the shared NFS home. Needs numpy,
Pillow, opencv-python-headless, python-docx and ``ffprobe``/``ffmpeg`` on PATH.

    python dataset_manifest.py inspect openttgames --root /home/itec/emanuele/Datasets/OpenTTGames \
        --scratch /dev/shm/ps-dl/inspect --out facts.json
    python dataset_manifest.py manifest openttgames --root ... --facts facts.json \
        --known-hashes known.tsv --out /home/itec/emanuele/Datasets/manifests/OpenTTGames.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

# Static provenance: where each dataset came from and under what terms.
SOURCES: dict[str, dict[str, Any]] = {
    "openttgames": {
        "name": "OpenTTGames",
        "homepage": "https://lab.osai.ai/",
        "urls": ["https://lab.osai.ai/datasets/openttgames/data/{game_1..game_5,test_1..test_7}.{mp4,zip}"],
        "licence": "CC BY-NC-SA 4.0",
        "access": "public direct download, no registration",
        "citation": "Voeikov et al., TTNet: Real-time temporal and spatial video analysis of table tennis, CVPRW 2020",
    },
    "racketvision": {
        "name": "RacketVision",
        "homepage": "https://github.com/OrcustD/RacketVision",
        "urls": ["https://huggingface.co/datasets/linfeng302/RacketVision"],
        "licence": "MIT (Hugging Face dataset card)",
        "access": "public Hugging Face dataset, not gated",
        "citation": "RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis, AAAI 2026 (arXiv:2511.17045)",
    },
    "tracknet": {
        "name": "TrackNet (tennis)",
        "homepage": "https://nol.cs.nctu.edu.tw/ndo3je6av9/ (unreachable 2026-10-05)",
        "urls": [
            "https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut (Dataset.zip, file id 1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw; mirror linked from github.com/yastrebksv/TrackNet)",
        ],
        "failed_urls": [
            "https://drive.google.com/open?id=1GzJZeEPEi8lJjEAAtVnHhRdX8TVR14yK (original link: not publicly retrievable)",
        ],
        "licence": "none stated in the release (Readme.docx has no licence); academic dataset of broadcast footage",
        "access": "public Google Drive mirror; the official NYCU host is unreachable",
        "citation": "Huang et al., TrackNet, AVSS 2019 (arXiv:1907.03698)",
    },
    "egohos": {
        "name": "EgoHOS",
        "homepage": "https://github.com/owenzlz/EgoHOS",
        "urls": ["https://drive.google.com/file/d/1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK (from download_datasets.sh)"],
        "licence": "none stated in the repository; images derive from Ego4D, EPIC-KITCHENS, THU-READ, Escape Room and YouTube",
        "access": "public Google Drive file, no registration",
        "citation": "Zhang et al., Fine-Grained Egocentric Hand-Object Segmentation, ECCV 2022",
    },
    "visor": {
        "name": "EPIC-KITCHENS VISOR",
        "homepage": "https://epic-kitchens.github.io/VISOR/",
        "urls": [
            "https://data.bris.ac.uk/datasets/tar/2v6cgv1x04ol22qp9rm9x2j6a7.zip (doi:10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7)",
            "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/... (EPIC-KITCHENS-55 videos)",
            "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/<P>/videos/... (EPIC-KITCHENS-100 extension videos)",
        ],
        "licence": "CC BY-NC 4.0 (VISOR README and EPIC-KITCHENS); data.bris page lists Non-Commercial Government Licence v2",
        "access": "public direct download, no registration",
        "citation": "Darkhalil et al., EPIC-KITCHENS VISOR, NeurIPS 2022 D&B; Damen et al., EPIC-KITCHENS-100, IJCV 2022",
    },
    "hot3d": {
        "name": "HOT3D-Clips",
        "homepage": "https://facebookresearch.github.io/hot3d/",
        "urls": ["https://huggingface.co/datasets/bop-benchmark/hot3d"],
        "licence": "HOT3D Dataset License Agreement (2024-05-29): sequences and non-hand annotations CC BY-SA 4.0; hand annotations CC BY-NC-SA 4.0; 3D models CC BY-SA 4.0 with a no-sale clause. Accepted by the user on 2026-10-05.",
        "access": "public Hugging Face dataset, not gated; licence binds on access",
        "citation": "Banerjee et al., HOT3D, CVPR 2025",
    },
}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def ffprobe(path: str) -> dict[str, Any]:
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration", "-of", "json", path,
    ])
    return json.loads(out)["streams"][0]


def decode_frame(video: str, index: int, fps: float) -> np.ndarray:
    """Decode frame ``index`` with an accurate seek; returns RGB."""
    import cv2

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    got = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index} of {video}")
    if got != index:
        raise RuntimeError(f"seek landed on {got}, wanted {index} in {video}")
    return frame[..., ::-1].copy()


def stage(src: str, scratch: Path) -> Path:
    """Copy one archive to host-local scratch with one sequential read."""
    scratch.mkdir(parents=True, exist_ok=True)
    dst = scratch / os.path.basename(src)
    if not dst.exists() or dst.stat().st_size != os.path.getsize(src):
        shutil.copyfile(src, dst)
    return dst


def extract(archive: Path, dest: Path) -> Path:
    marker = dest / f".extracted-{archive.name}"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    marker.touch()
    return dest


def runs_of(frames: list[int]) -> list[int]:
    if not frames:
        return []
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f != prev + 1:
            out.append(prev - start + 1)
            start = f
        prev = f
    out.append(prev - start + 1)
    return out


def summary(xs: list[int]) -> dict[str, float]:
    return {"n": len(xs), "min": min(xs), "median": float(np.median(xs)), "max": max(xs)} if xs else {"n": 0}


def save_overlay(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=88)
    return str(path)


def blend(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, np.float32)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- inspectors


OTT_CHANNELS = ["table", "human", "scoreboard"]


def inspect_openttgames(root: Path, scratch: Path) -> dict[str, Any]:
    raw = root / "raw"
    work = scratch / "openttgames"
    per_video = {}
    totals = collections.Counter()
    for z in sorted(raw.glob("*.zip")):
        name = z.stem
        d = extract(stage(str(z), work / "zips"), work / name)
        ball = json.load(open(d / "ball_markup.json"))
        events = json.load(open(d / "events_markup.json"))
        masks = sorted(int(p.stem) for p in (d / "segmentation_masks").glob("*.png"))
        probe = ffprobe(str(raw / f"{name}.mp4"))
        n_frames = int(probe["nb_frames"])
        mask_sizes = collections.Counter(Image.open(d / "segmentation_masks" / f"{m}.png").size for m in masks[::100])
        per_video[name] = {
            "split": "train" if name.startswith("game") else "test",
            "video": probe,
            "frames": n_frames,
            "ball_labelled_frames": len(ball),
            "ball_visible_frames": sum(1 for p in ball.values() if p["x"] >= 0),
            "mask_frames": len(masks),
            "mask_frames_equal_ball_frames": set(masks) == {int(k) for k in ball},
            "labelled_fraction": round(len(masks) / n_frames, 4),
            "labelled_run_length": summary(runs_of(masks)),
            "events": dict(collections.Counter(events.values())),
            "mask_sizes_sampled": {f"{w}x{h}": c for (w, h), c in mask_sizes.items()},
        }
        totals.update(frames=n_frames, mask_frames=len(masks), ball_frames=len(ball), events=len(events))

    # Smoke: test_2, first labelled frame.
    name, frame = "test_2", min(int(k) for k in json.load(open(work / "test_2" / "ball_markup.json")))
    fps = 120.0
    rgb = decode_frame(str(raw / f"{name}.mp4"), frame, fps)
    mask = np.array(Image.open(work / name / "segmentation_masks" / f"{frame}.png").convert("RGB"))
    ball = json.load(open(work / name / "ball_markup.json"))[str(frame)]
    up = np.array(Image.fromarray(mask).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    # Channel order verified visually on this overlay: R=table, G=human, B=scoreboard.
    ov = blend(blend(blend(rgb, up[..., 0], (255, 0, 255)), up[..., 1], (0, 255, 0)), up[..., 2], (0, 0, 255))
    draw = Image.fromarray(ov)
    ImageDraw.Draw(draw).ellipse([ball["x"] - 12, ball["y"] - 12, ball["x"] + 12, ball["y"] + 12], outline=(255, 255, 0), width=4)
    smoke = {
        "sample": f"{name} frame {frame}",
        "frame_shape": list(rgb.shape),
        "mask_shape": list(mask.shape),
        "mask_fraction_per_channel": {c: round(float((mask[..., i] > 127).mean()), 4) for i, c in enumerate(OTT_CHANNELS)},
        "ball_xy": [ball["x"], ball["y"]],
        "overlay": save_overlay(np.array(draw), scratch / "smoke" / "OpenTTGames.jpg"),
    }
    return {
        "label_format": {
            "ball_markup.json": "{frame: {x, y}} in 1920x1080 pixels; (-1, -1) = ball absent",
            "events_markup.json": "{frame: bounce | net | empty_event}",
            "segmentation_masks/<frame>.png": "RGB PNG, one binary channel each: R=table, G=human, B=scoreboard (verified visually); stored at 320x128, i.e. downscaled 6x horizontally and 8.4x vertically from 1920x1080, so edges are blocky when upsampled",
            "mask_provenance": "model_aided (per the OpenTTGames page)",
        },
        "classes": {"masks": ["human (players and umpire; rackets not included)", "table", "scoreboard"], "points": ["ball"], "events": ["bounce", "net", "empty_event"]},
        "clips": {"videos": len(per_video), "train": 5, "test": 7, **dict(totals)},
        "resolution": "1920x1080", "fps": 120,
        "per_video": per_video,
        "smoke_read": smoke,
    }


def inspect_racketvision(root: Path, scratch: Path) -> dict[str, Any]:
    work = scratch / "racketvision"
    for t in sorted(root.glob("*.tar")):
        extract(stage(str(t), work / "tars"), work / "data")
    d = work / "data"
    sports = {}
    for sp in ["badminton", "tabletennis", "tennis"]:
        vids = sorted((d / sp / "videos").glob("*.mp4"))
        res, fps, frames = collections.Counter(), collections.Counter(), 0
        for v in vids:
            p = ffprobe(str(v))
            res[f"{p['width']}x{p['height']}"] += 1
            fps[p["avg_frame_rate"]] += 1
            frames += int(p.get("nb_frames", 0))
        ball_rows = ball_vis = 0
        for b in glob.glob(str(d / sp / "all" / "*" / "csv" / "*_ball.csv")):
            rows = list(csv.DictReader(open(b)))
            ball_rows += len(rows)
            ball_vis += sum(r["Visibility"] == "1" for r in rows)
        rk = glob.glob(str(d / sp / "all" / "*" / "racket" / "*" / "*.json"))
        inst = sum(len(json.load(open(r))) for r in rk)
        sports[sp] = {
            "clips": len(vids), "frames": frames, "resolution": dict(res), "fps": dict(fps),
            "ball_labelled_frames": ball_rows, "ball_visible": ball_vis,
            "ball_labelled_fraction": round(ball_rows / frames, 4),
            "racket_labelled_frames": len(rk), "racket_instances": inst,
            "racket_labelled_fraction": round(len(rk) / frames, 4),
            "splits": {k: len(json.load(open(d / sp / "info" / f"{k}.json"))) for k in ["train", "val", "test"]},
        }
    # Smoke: first tennis racket-labelled frame.
    r = sorted(glob.glob(str(d / "tennis" / "all" / "*" / "racket" / "*" / "*.json")))[0]
    m = re.search(r"all/(match\d+)/racket/(\d+)/(\d+)\.json$", r)
    match, rally, frame = m.group(1), m.group(2), int(m.group(3))
    video = str(d / "tennis" / "videos" / f"{match}_{rally}.mp4")
    rgb = decode_frame(video, frame, 0)
    inst = json.load(open(r))
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    for k in inst:
        x, y, w, h = k["bbox_xywh"]
        dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
        for kx, ky, kv in k["keypoints"]:
            dr.ellipse([kx - 5, ky - 5, kx + 5, ky + 5], fill=(255, 0, 0) if kv else (90, 90, 90))
    balls = {int(row["Frame"]): row for row in csv.DictReader(open(d / "tennis" / "all" / match / "csv" / f"{rally}_ball.csv"))}
    near = min(balls, key=lambda f: abs(f - frame))
    smoke = {
        "sample": f"tennis {match}_{rally} frame {frame}",
        "frame_shape": list(rgb.shape),
        "rackets": len(inst),
        "nearest_ball_label": {"frame": near, **{k: balls[near][k] for k in ("X", "Y", "Visibility")}},
        "overlay": save_overlay(np.array(img), scratch / "smoke" / "RacketVision.jpg"),
    }
    return {
        "label_format": {
            "<sport>/all/<match>/csv/<rally>_ball.csv": "Frame,Visibility,X,Y (0-indexed frame, 1920x1080 px); sparse rows",
            "<sport>/all/<match>/racket/<rally>/<frame>.json": "list of {bbox_xywh, keypoints[5][x,y,vis]}; keypoints top,bottom,handle,left,right",
            "info/*_det_coco.json, *_pose_coco.json": "COCO detection (3 racket categories) and pose (1 category, 5 keypoints)",
            "<sport>/interp_ball, merged_racket": "interpolated ball trajectories and merged racket predictions (not ground truth)",
            "data_traj/*.pkl": "pre-built trajectory-forecast sets",
        },
        "classes": {"points": ["ball"], "boxes_keypoints": ["badminton_racket", "tabletennis_racket", "tennis_racket"], "masks": []},
        "clips": {sp: {k: v[k] for k in ("clips", "frames")} for sp, v in sports.items()},
        "resolution": "1920x1080", "fps": "per clip: 24-60 (see per_sport)",
        "per_sport": sports,
        "smoke_read": smoke,
    }


def inspect_tracknet(root: Path, scratch: Path) -> dict[str, Any]:
    import docx

    work = scratch / "tracknet"
    d = extract(stage(str(root / "Dataset.zip"), work / "zips"), work / "x") / "Dataset"
    readme = "\n".join(p.text for p in docx.Document(str(d / "Readme.docx")).paragraphs if p.text.strip())
    clips = sorted(d.glob("game*/Clip*"))
    vis, rows_n, jpg, sizes = collections.Counter(), 0, 0, collections.Counter()
    for c in clips:
        rows = list(csv.DictReader(open(c / "Label.csv")))
        rows_n += len(rows)
        vis.update(r["visibility"] for r in rows)
        js = sorted(c.glob("*.jpg"))
        jpg += len(js)
        sizes[Image.open(js[0]).size] += 1
    c = clips[0]
    row = next(r for r in csv.DictReader(open(c / "Label.csv")) if r["visibility"] == "1")
    img = Image.open(c / row["file name"]).convert("RGB")
    x, y = int(row["x-coordinate"]), int(row["y-coordinate"])
    ImageDraw.Draw(img).ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 0), width=3)
    return {
        "label_format": {"game<N>/Clip<M>/Label.csv": "file name,visibility,x-coordinate,y-coordinate,status; one row per frame; visibility 0=out of frame, 1=easy, 2=hard, 3=occluded; status 0=flying, 1=hit, 2=bouncing"},
        "classes": {"points": ["ball"], "masks": []},
        "clips": {"games": len(list(d.glob("game*"))), "clips": len(clips), "frames": jpg, "label_rows": rows_n, "visibility": dict(vis)},
        "resolution": {f"{w}x{h}": n for (w, h), n in sizes.items()}, "fps": 30,
        "storage": "JPEG frames only (no video files)",
        "readme_excerpt": readme[:600],
        "smoke_read": {"sample": f"{c.relative_to(d)}/{row['file name']}", "ball_xy": [x, y], "frame_size": list(img.size),
                       "overlay": save_overlay(np.array(img), scratch / "smoke" / "TrackNet.jpg")},
    }


EGOHOS_CLASSES = ["background", "left_hand", "right_hand", "object1_left", "object1_right", "object1_both",
                  "object2_left", "object2_right", "object2_both"]


def inspect_egohos(root: Path, scratch: Path) -> dict[str, Any]:
    work = scratch / "egohos"
    d = extract(stage(str(root / "data.zip"), work / "zips"), work / "x") / "data"
    splits = {}
    for sp in ["train", "val", "test_indomain", "test_outdomain"]:
        ims = sorted((d / sp / "image").glob("*"))
        labs = sorted((d / sp / "label").glob("*"))
        src = collections.Counter(re.split(r"[_\-]", p.name)[0] for p in ims)
        sizes = collections.Counter(f"{Image.open(p).size[0]}x{Image.open(p).size[1]}" for p in ims[::20])
        splits[sp] = {"images": len(ims), "labels": len(labs), "sources": dict(src), "sizes_sampled": dict(sizes.most_common(6))}
    img_p = sorted((d / "train" / "image").glob("*.jpg"))[0]
    lab = np.array(Image.open(str(img_p).replace("/image/", "/label/").replace(".jpg", ".png")))
    im = np.array(Image.open(img_p).convert("RGB").resize((lab.shape[1], lab.shape[0])))
    ov = blend(blend(im, lab == 1, (255, 0, 0)), lab == 2, (0, 0, 255))
    for k in range(3, 9):
        ov = blend(ov, lab == k, (255, 255, 0))
    counts = {EGOHOS_CLASSES[k]: int((lab == k).sum()) for k in np.unique(lab)}
    return {
        "label_format": {"<split>/label/<name>.png": "single-channel PNG, values 0-8: " + ", ".join(f"{i}={c}" for i, c in enumerate(EGOHOS_CLASSES))},
        "classes": {"masks": EGOHOS_CLASSES[1:]},
        "clips": {"images": sum(s["images"] for s in splits.values()), "note": "independent frames sampled sparsely from source videos; no temporal sequences"},
        "resolution": "mixed (see per_split sizes)", "fps": None,
        "per_split": splits,
        "hand_mask_extent": "includes the visible forearm and arm (sleeves included) up to the image border; checked visually on 12 random train images",
        "contact_boundaries": "not present in this release archive (only image/ and label/ per split)",
        "smoke_read": {"sample": f"train/{img_p.name}", "label_shape": list(lab.shape), "pixel_counts": counts,
                       "overlay": save_overlay(ov, scratch / "smoke" / "EgoHOS.jpg")},
    }


def decode_hot3d_rle(m: dict[str, Any]) -> np.ndarray:
    """HOT3D-Clips masks: flat row-major [start, length, start, length, ...]."""
    flat = np.zeros(m["height"] * m["width"], bool)
    r = m["rle"]
    for s, n in zip(r[0::2], r[1::2]):
        flat[s:s + n] = True
    return flat.reshape(m["height"], m["width"])


def inspect_hot3d(root: Path, scratch: Path) -> dict[str, Any]:
    splits = json.load(open(root / "clip_splits.json"))
    n_tar = {d: len(list((root / d).glob("clip-*.tar"))) for d in ["train_aria", "test_aria"]}
    work = scratch / "hot3d"
    tar = sorted((root / "train_aria").glob("clip-*.tar"))[0]
    d = extract(stage(str(tar), work / "tars"), work / tar.stem)
    members = collections.Counter(p.name.split(".", 1)[1] for p in d.iterdir() if p.name[0].isdigit())
    test_tar = sorted((root / "test_aria").glob("clip-*.tar"))[0]
    with tarfile.open(test_tar) as t:
        test_members = sorted({n.split(".", 1)[1] for n in t.getnames() if n[0].isdigit()})
    frame = "000000"
    cams = json.load(open(d / f"{frame}.cameras.json"))
    objs = json.load(open(d / f"{frame}.objects.json"))
    hands = json.load(open(d / f"{frame}.hands.json"))
    rgb = np.array(Image.open(d / f"{frame}.image_214-1.jpg").convert("RGB"))
    ov, names = rgb, []
    for inst_list in objs.values():
        for inst in (inst_list if isinstance(inst_list, list) else [inst_list]):
            m = inst.get("masks_modal", {}).get("214-1")
            if m:
                ov = blend(ov, decode_hot3d_rle(m), (255, 255, 0))
                names.append(inst["object_name"])
    img = Image.fromarray(ov)
    dr = ImageDraw.Draw(img)
    for side, colour in (("left", (255, 0, 0)), ("right", (0, 0, 255))):
        box = hands.get(side, {}).get("boxes_amodal", {}).get("214-1")
        if box:
            dr.rectangle(box, outline=colour, width=4)
    return {
        "label_format": {
            "train_aria/clip-<id>.tar": "150 frames per clip; per frame: image_214-1.jpg (RGB fisheye 1408x1408), image_1201-{1,2}.jpg (mono SLAM 640x480), cameras.json, hands.json, hand_crops.json, objects.json, info.json; __hand_shapes.json__ per clip",
            "objects.json": "per object: 6DoF pose, boxes_amodal, masks_amodal (rendered from GT pose), masks_modal (SAM2-based), visibilities; masks are {height,width,rle} with rle = flat row-major [start,length,...]",
            "hands.json": "left/right: MANO and UmeTrack pose, boxes_amodal, visibilities. No hand masks: they must be rendered from MANO (plus __hand_shapes.json__) through the fisheye camera model",
            "test_aria": f"no labels: members are {test_members}",
        },
        "classes": {"masks": "33 rigid objects (object_models/models_info.json); hands only as MANO/UmeTrack pose + boxes", "boxes": ["left_hand", "right_hand"]},
        "clips": {"train_aria_tars": n_tar["train_aria"], "test_aria_tars": n_tar["test_aria"], "frames_per_clip": 150,
                  "train_aria_frames": 150 * n_tar["train_aria"],
                  "splits_in_clip_splits_json": {k: ({a: len(b) for a, b in v.items()} if isinstance(v, dict) else len(v)) for k, v in splits.items()}},
        "resolution": {"214-1": "1408x1408 RGB fisheye (FISHEYE624)", "1201-1/2": "640x480 mono"}, "fps": 30,
        "storage": "one tar per clip; Quest3 (mono only) and the visualisation/onboarding folders were not downloaded",
        "sample_clip_members": dict(members),
        "smoke_read": {"sample": f"{tar.relative_to(root)} frame {frame}", "image_shape": list(rgb.shape),
                       "objects_with_modal_mask": names, "hands_present": sorted(hands),
                       "camera_214_1": {k: cams["214-1"]["calibration"][k] for k in ("image_width", "image_height", "projection_model_type")},
                       "overlay": save_overlay(np.array(img), scratch / "smoke" / "HOT3D.jpg")},
    }


VISOR_ZIP = "2v6cgv1x04ol22qp9rm9x2j6a7.zip"


def visor_mask(annotations: list[dict[str, Any]], names: set[str], shape: tuple[int, int]) -> np.ndarray:
    img = Image.new("1", (shape[1], shape[0]))
    dr = ImageDraw.Draw(img)
    for a in annotations:
        if a["name"] in names:
            for poly in a["segments"]:
                if len(poly) >= 3:
                    dr.polygon([tuple(p) for p in poly], fill=1)
    return np.array(img)


def inspect_visor(root: Path, scratch: Path) -> dict[str, Any]:
    work = scratch / "visor"
    zpath = stage(str(root / VISOR_ZIP), work / "zips")
    base = work / "x" / VISOR_ZIP[:-4]
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        wanted = [n for n in names if "/GroundTruth-SparseAnnotations/annotations/" in n and n.endswith(".json")]
        wanted += [n for n in names if n.endswith(("frame_mapping.json", "EPIC_100_noun_classes_v2.csv"))]
        z.extractall(work / "x", members=wanted)
        dense = [n for n in names if "/Interpolations-DenseAnnotations/" in n and n.endswith(".zip")]
        rgb = [n for n in names if "/GroundTruth-SparseAnnotations/rgb_frames/" in n and n.endswith(".zip")]
        dense_bytes = sum(z.getinfo(n).file_size for n in dense)
    mapping = json.load(open(base / "frame_mapping.json"))

    splits, classes, hands_per_image = {}, collections.Counter(), collections.Counter()
    gaps: list[int] = []
    for sp in ["train", "val"]:
        files = sorted((base / "GroundTruth-SparseAnnotations" / "annotations" / sp).glob("*.json"))
        n_img = n_mask = 0
        for f in files:
            va = json.load(open(f))["video_annotations"]
            n_img += len(va)
            frames = sorted(int(re.search(r"frame_(\d+)", e["image"]["name"]).group(1)) for e in va)
            gaps += [b - a for a, b in zip(frames, frames[1:])]
            for e in va:
                n_mask += len(e["annotations"])
                names_here = [a["name"] for a in e["annotations"]]
                classes.update(names_here)
                hands_per_image[sum(n in ("left hand", "right hand") for n in names_here)] += 1
        splits[sp] = {"videos": len(files), "annotated_images": n_img, "masks": n_mask}
    test_videos = sorted({n.rsplit("/", 1)[1][:-4] for n in rgb if "/rgb_frames/test/" in n})
    splits["test"] = {"videos": len(test_videos), "annotations": "not released (images only)"}

    # Smoke: P01_01, first sparse frame with both hands; rasterise on the released frame and
    # on the same frame decoded from the downloaded EPIC-KITCHENS video via frame_mapping.json.
    video_id = "P01_01"
    va = json.load(open(base / "GroundTruth-SparseAnnotations" / "annotations" / "train" / f"{video_id}.json"))["video_annotations"]
    entry = next(e for e in va if {"left hand", "right hand"} <= {a["name"] for a in e["annotations"]})
    with zipfile.ZipFile(zpath) as z:
        inner = next(n for n in rgb if n.endswith(f"/{video_id}.zip"))
        with z.open(inner) as fz, zipfile.ZipFile(fz) as frames_zip:
            sparse_rgb = np.array(Image.open(frames_zip.open(entry["image"]["name"])).convert("RGB"))
    ek_name = mapping[video_id][entry["image"]["name"]]
    ek_index = int(re.search(r"frame_(\d+)", ek_name).group(1)) - 1  # EPIC rgb_frames are 1-indexed
    video = root / "epic_kitchens_videos" / f"{video_id}.MP4"
    probe = ffprobe(str(video))
    vid_rgb = decode_frame(str(video), ek_index, 0)

    def overlay(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = np.array([w / 1920, h / 1080])
        anns = [{**a, "segments": [[[x * scale[0], y * scale[1]] for x, y in p] for p in a["segments"]]} for a in entry["annotations"]]
        out = blend(img, visor_mask(anns, {"left hand"}, (h, w)), (255, 0, 0))
        out = blend(out, visor_mask(anns, {"right hand"}, (h, w)), (0, 0, 255))
        others = {a["name"] for a in anns} - {"left hand", "right hand"}
        return blend(out, visor_mask(anns, others, (h, w)), (255, 255, 0))

    side = [np.array(Image.fromarray(overlay(x)).resize((960, 540))) for x in (sparse_rgb, vid_rgb)]
    smoke_path = save_overlay(np.concatenate(side, 1), scratch / "smoke" / "VISOR.jpg")
    vid_small = np.array(Image.fromarray(vid_rgb).resize((sparse_rgb.shape[1], sparse_rgb.shape[0]))).astype(float)
    frame_mae = float(np.abs(vid_small - sparse_rgb.astype(float)).mean())

    # Dense interpolations of the same video.
    with zipfile.ZipFile(zpath) as z:
        inner = next(n for n in dense if n.endswith(f"/{video_id}_interpolations.zip"))
        with z.open(inner) as fz, zipfile.ZipFile(fz) as dz:
            dj = json.load(dz.open(dz.namelist()[0]))["video_annotations"]
    dense_frames = sorted({int(re.search(r"frame_(\d+)", e["image"]["name"]).group(1)) for e in dj})
    dense_types = collections.Counter(a.get("type") for e in dj[:2000] for a in e["annotations"])

    # Hand extent: grid of 12 sparse frames with hands from different videos.
    tiles = []
    with zipfile.ZipFile(zpath) as z:
        for f in sorted((base / "GroundTruth-SparseAnnotations" / "annotations" / "train").glob("*.json"))[::14][:12]:
            vid = f.stem
            va2 = json.load(open(f))["video_annotations"]
            e2 = next((e for e in va2 if {"left hand", "right hand"} & {a["name"] for a in e["annotations"]}), None)
            if e2 is None:
                continue
            inner = next(n for n in rgb if n.endswith(f"/{vid}.zip"))
            with z.open(inner) as fz, zipfile.ZipFile(fz) as frames_zip:
                im = np.array(Image.open(frames_zip.open(e2["image"]["name"])).convert("RGB"))
            h, w = im.shape[:2]
            sc = (w / 1920, h / 1080)
            anns = [{**a, "segments": [[[x * sc[0], y * sc[1]] for x, y in p] for p in a["segments"]]} for a in e2["annotations"]]
            t = blend(blend(im, visor_mask(anns, {"left hand"}, (h, w)), (255, 0, 0)), visor_mask(anns, {"right hand"}, (h, w)), (0, 0, 255))
            tiles.append(np.array(Image.fromarray(t).resize((480, 270))))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    grid_path = save_overlay(grid, scratch / "smoke" / "VISOR_hands_grid.jpg")

    hand_like = {k: v for k, v in classes.items() if "hand" in k or "glove" in k or "arm" in k}
    return {
        "label_format": {
            "GroundTruth-SparseAnnotations/annotations/<split>/<video>.json": "video_annotations[]: image{image_path,name,subsequence,video}, annotations[]{id,name (open vocabulary),class_id (EPIC_100_noun_classes_v2.csv),exhaustive,in_contact_object (hands),on_which_hand (gloves),segments: polygons in 1920x1080 coordinates}",
            "GroundTruth-SparseAnnotations/rgb_frames/<split>/<P>/<video>.zip": "the annotated sparse frames as JPEG",
            "Interpolations-DenseAnnotations/<split>/<video>_interpolations.zip": "one JSON per video, same schema plus type (1=start/end GT, 0=interpolated) and interpolation id",
            "frame_mapping.json": "VISOR frame name -> EPIC-KITCHENS rgb_frames name (1-indexed); used to index the downloaded videos",
            "epic_kitchens_videos/<video>.MP4": "original EPIC-KITCHENS videos for all 179 VISOR videos (md5-verified against epic-kitchens-download-scripts data/md5.csv)",
        },
        "classes": {"hands": ["left hand", "right hand"], "hand_related": hand_like, "objects": f"{len(classes)} open-vocabulary entity names mapped to EPIC-100 noun classes; top 25: {dict(classes.most_common(25))}"},
        "clips": {"videos": len(mapping), "per_split": splits, "hands_per_annotated_image": dict(hands_per_image),
                  "sparse_gap_frames": summary(gaps),
                  "dense_interpolation_zips": len(dense), "dense_bytes_uncompressed_zips": dense_bytes,
                  "dense_sample": {"video": video_id, "frames_with_masks": len(dense_frames), "video_frames": int(probe.get("nb_frames", 0)),
                                   "coverage": round(len(dense_frames) / max(1, int(probe.get("nb_frames", 1))), 4),
                                   "dense_runs": summary(runs_of(dense_frames)), "type_counts_first_2000": dict(dense_types)}},
        "resolution": {"sparse_frames": f"{sparse_rgb.shape[1]}x{sparse_rgb.shape[0]}", "videos_P01_01": f"{probe['width']}x{probe['height']}"},
        "fps": {"P01_01": probe["avg_frame_rate"], "note": "EPIC-KITCHENS: mostly 59.94 (EK-55) and 50 (EK-100 extension); see per-video probe at use time"},
        "smoke_read": {"sample": f"{video_id} {entry['image']['name']} -> video frame index {ek_index} ({ek_name})",
                       "objects": [a["name"] for a in entry["annotations"]],
                       "sparse_frame_shape": list(sparse_rgb.shape), "video_frame_shape": list(vid_rgb.shape),
                       "mean_abs_diff_video_vs_sparse_frame": round(frame_mae, 2),
                       "overlay": smoke_path, "hands_grid": grid_path},
    }


INSPECTORS: dict[str, Callable[[Path, Path], dict[str, Any]]] = {
    "visor": inspect_visor,
    "hot3d": inspect_hot3d,
    "openttgames": inspect_openttgames,
    "racketvision": inspect_racketvision,
    "tracknet": inspect_tracknet,
    "egohos": inspect_egohos,
}


# --------------------------------------------------------------------------- manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known(paths: list[str]) -> dict[str, tuple[str, str]]:
    """Read ``relpath<TAB>sha256<TAB>how`` lines recorded while downloading, or a previous manifest."""
    known: dict[str, tuple[str, str]] = {}
    for p in paths:
        if p.endswith(".json"):
            for e in json.load(open(p))["files"]:
                known[e["path"]] = (e["sha256"], e["sha256_source"])
            continue
        for line in open(p):
            rel, digest, how = line.rstrip("\n").split("\t")
            known[rel] = (digest, how)
    return known


def build_manifest(name: str, root: Path, facts: dict[str, Any], known: dict[str, tuple[str, str]],
                   exclude: list[str], workers: int, smoke_dir: Path) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not any(p.match(e) for e in exclude))

    def entry(p: Path) -> dict[str, Any]:
        rel = str(p.relative_to(root))
        size = p.stat().st_size
        if rel in known:
            digest, how = known[rel]
        else:
            digest, how = sha256(p), "computed_from_stored_file"
        return {"path": rel, "bytes": size, "sha256": digest, "sha256_source": how}

    with ThreadPoolExecutor(workers) as ex:
        entries = list(ex.map(entry, files))
    # Keep the smoke overlay next to the manifests; scratch is ephemeral.
    smoke = facts.get("smoke_read", {})
    if smoke.get("overlay") and Path(smoke["overlay"]).exists():
        dst = smoke_dir / Path(smoke["overlay"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(smoke["overlay"], dst)
            os.chmod(dst, 0o444)
        smoke["overlay"] = str(dst)
    revision = os.environ.get("PS_TOOL_REVISION", "unknown")
    return {
        "dataset": SOURCES[name]["name"],
        "root": str(root),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": {"script": "tools/datasets/dataset_manifest.py", "git_revision": revision, "host": os.uname().nodename},
        "source": SOURCES[name],
        **facts,
        "storage_note": "Archives are stored as downloaded; extract to host-local disk or /dev/shm, never onto the NFS home.",
        "files": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }


def write_readonly(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; manifests are immutable (write a new version instead)")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    os.chmod(tmp, 0o444)
    tmp.rename(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect")
    a.add_argument("dataset", choices=sorted(INSPECTORS))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--scratch", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("manifest")
    b.add_argument("dataset", choices=sorted(SOURCES))
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--facts", type=Path, required=True)
    b.add_argument("--known-hashes", nargs="*", default=[])
    b.add_argument("--exclude", nargs="*", default=["*.part", "*.tmp"])
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        facts = INSPECTORS[args.dataset](args.root, args.scratch)
        args.out.write_text(json.dumps(facts, indent=1, default=str) + "\n")
        print(json.dumps(facts.get("smoke_read"), indent=1, default=str))
    else:
        facts = json.loads(args.facts.read_text())
        manifest = build_manifest(args.dataset, args.root, facts, load_known(args.known_hashes), args.exclude,
                                  args.workers, args.out.parent / "smoke")
        write_readonly(args.out, manifest)
        print(f"wrote {args.out}: {len(manifest['files'])} files, {manifest['total_bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
