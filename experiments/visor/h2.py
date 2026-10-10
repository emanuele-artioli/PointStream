"""PLAN step H2: HaMeR against WiLoR on egocentric hands, and what their pose stream costs to send.

    python -m experiments.visor.h2 run --part hint --hint-zip ZIP --limit N --speed ... (model and MANO flags)
    python -m experiments.visor.h2 run --part hot3d --hot3d-clip TAR ... --clips N --frames N
    python -m experiments.visor.h2 run --part visor --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --h1-boxes JSON --video V.MP4 ... --items all|ID,ID --frames N
    python -m experiments.visor.h2 validate
    python -m experiments.visor.h2 code --hot3d TAR ... --visor TAR ... --eval-set JSON --mano-right PKL --out DIR
    python -m experiments.visor.h2 report --hint TAR ... --hot3d TAR ... --visor TAR ... --code JSON --out DIR

``run`` fits both models to the same hands with the same box (enlarged ``RESCALE`` times, the same
``ViTDetDataset`` preprocessing) and handedness (docs/experiments.md, 2026-10-09 H2):

* ``hint``: HInt's TEST_epick and TEST_newdays hands, boxes from the labels; 2D distances normalised as
  HaMeR's evaluator does (``hamer/utils/pose_utils.py``). With ``--speed``, timings of both
  regressors and of WiLoR's detector.
* ``hot3d``: HOT3D-Clips hands warped into the dataset's pinhole crop cameras (the hand-tracking
  challenge protocol) at half their focal length, so the enlarged box sees real context rather than
  padding; the box is the projected ground-truth mesh's. 3D errors against the
  motion-capture MANO, and every fit expressed in the fisheye camera ``214-1`` for the coding stage.
* ``visor``: evaluation set v2 windows with H1's recorded hand boxes; rendered silhouettes against
  the mask's hand side (H1's wrist split), and the per-frame parameters for the coding stage.

Each finished unit (a block of HInt hands, a clip, a window) is published as ``<part>/<unit>.npz``
with a ``.json`` beside it, and checkpointed, so a stopped stage resumes with the rest.

``code`` runs the pose-coding grid (``pose_coding``) on the saved parameters; ``report`` applies
``DECISION``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import multiprocessing
import os
import shutil
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import mano_np
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

MODELS = ("hamer", "wilor")
RESCALE = 2.0
ASPECT = (192, 256)  # BBOX_SHAPE both models are run with
PATCH_MARGIN = 16
PCK_THRESHOLDS = (0.05, 0.10, 0.15)
HINT_SPLITS = ("TEST_epick_img", "TEST_newdays_img")
HINT_UNIT = 200
HOT3D_STREAM = "214-1"
HOT3D_CROP = 512
HOT3D_ZOOM = 0.5  # the dataset's crop camera at half its focal length: twice the field of view
NOMINAL_FOCAL = 5000.0 / 256.0  # × the image's longer side, as H1 and the models' demos
SEED = "pointstream-h2"
BATCH = 32
FILL = "visor_dense_sam_fill"

DECISION: dict[str, Any] = {
    "contests": ["hint_visor_pck05_all", "hint_newdays_pck05_all", "hot3d_pa_mpjpe_mm", "hot3d_accel_error_mm",
                 "visor_iou_hand_median"],
    "lower_is_better": ["hot3d_pa_mpjpe_mm", "hot3d_accel_error_mm"],
    "bootstrap": {"resamples": 1000, "interval": 0.95, "seed": 20261009,
                  "unit": {"hint": "image", "hot3d": "clip", "visor": "item"}},
    "tie_break": "faster regressor at batch 1",
    # Pre-registered: HaMeR's VISOR PCK@0.05 (all) within 40-47 with HInt's `bbox` as the normaliser. It
    # failed (51.3): that box is padded 1.24x the keypoints. User, 2026-10-10: normalise by the labelled
    # keypoints' extent (grown to 3:4) and anchor both sets on the paper within MAX_GAP.
    "anchor": {"metric": "hamer pck05_all, keypoint-extent normalisation", "published": {"TEST_epick_img": 43.0,
               "TEST_newdays_img": 48.0}, "max_gap": 4.0, "preregistered": {"range": [40.0, 47.0], "normaliser": "bbox"}},
    "coding": {"max_error_increase": 0.05, "budgets_ms": [0.0, 33.3, 100.0, 266.7],
               "h1_screen": {"max_share": 0.10, "svt_crf62_hand_kbps": 23.4},
               "choice": "per budget, the lowest VISOR kbps among combinations whose HOT3D MPJPE and 2D error against the truth are at most 5% above the uncoded estimate's",
               # User, 2026-10-10 (after the coding dev check): the uncoded reference carries the shape the
               # scheme sends, the track's median betas, so the limit measures pose coding alone; the cost
               # of sending the shape once is reported apart.
               "reference": "uncoded pose with the track's median shape"},
}

HYBRID_DECISION: dict[str, Any] = {
    "hybrid": "WiLoR's joint rotations and shape, HaMeR's global orientation and translation (wrist kept)",
    "hot3d": "wrist-aligned MPJPE below both models (95% paired interval below zero against each; by clip)",
    "hint_margin_pck": 1.0, "visor_margin_iou": 0.01, "visor_split": "wilor",
    "bootstrap": {"resamples": 1000, "seed": 20261010},
    "model_focal": 5000.0, "model_image_size": 256,
}


# ----------------------------------------------------------------- shared helpers

def expanded_size(box: np.ndarray) -> float:
    """The longer side of a box grown to the models' 192:256 aspect, as ``expand_to_aspect_ratio``."""
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    wt, ht = ASPECT
    if h / max(w, 1e-9) < ht / wt:
        h = max(w * ht / wt, h)
    else:
        w = max(h * wt / ht, w)
    return max(w, h)


def content_order(names: list[str], tag: str) -> list[str]:
    return sorted(names, key=lambda n: hashlib.sha256(f"{SEED}:{tag}:{n}".encode()).hexdigest())


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def save_unit(publish: Path, part: str, unit: str, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """Publish a finished unit and checkpoint it."""
    target = publish / part
    target.mkdir(parents=True, exist_ok=True)
    data = npz_bytes(arrays)
    meta = {**meta, "npz_sha256": hashlib.sha256(data).hexdigest()}
    (target / f"{unit}.npz").write_bytes(data)
    write_json(target / f"{unit}.json", meta)
    if os.environ.get("PS_CHECKPOINT_DIR"):
        from experiments.jobs.monitor import save_checkpoint

        save_checkpoint(f"{part}--{unit}.npz", data)
        save_checkpoint(f"{part}--{unit}.json", json.dumps(meta, sort_keys=True, default=str).encode())


def restore_units(publish: Path, part: str) -> dict[str, dict[str, Any]]:
    """Units a previous attempt finished (both files present), copied into ``publish``."""
    directory = Path(os.environ.get("PS_CHECKPOINT_DIR") or "")
    if not os.environ.get("PS_CHECKPOINT_DIR") or not directory.is_dir():
        return {}
    out = {}
    for meta_path in sorted(directory.glob(f"{part}--*.json")):
        unit = meta_path.name[len(part) + 2:-5]
        data_path = directory / f"{part}--{unit}.npz"
        if not data_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != meta.get("npz_sha256"):
            continue
        (publish / part).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_path, publish / part / f"{unit}.npz")
        write_json(publish / part / f"{unit}.json", {**meta, "restored_from_checkpoint": True})
        out[unit] = meta
    return out


# ----------------------------------------------------------------- regressors

class Regressor:
    """HaMeR or WiLoR, loaded as the environment audit loads them."""

    def __init__(self, kind: str, args: argparse.Namespace) -> None:
        import torch

        from experiments.audit.env_smoke import _hand_model

        began = time.time()
        self.kind = kind
        self.model, self.cfg, self.dataset_cls, self.load = _hand_model(kind, args)
        self.load["seconds"] = round(time.time() - began, 2)
        self.faces = np.asarray(self.model.mano.faces, dtype=np.int32)
        self.torch = torch
        self.kernels: dict[str, Any] = {}

    def item(self, image: np.ndarray, box: np.ndarray, right: bool) -> dict[str, Any]:
        """One hand's input, cut from a padded window around the box so the dataset's anti-alias blur
        does not run over the whole frame; the patch is the one the full image would give."""
        size = RESCALE * max(box[2] - box[0], box[3] - box[1]) * ASPECT[1] / ASPECT[0]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = size / 2 + PATCH_MARGIN
        x0, y0 = int(math.floor(cx - half)), int(math.floor(cy - half))
        x1, y1 = int(math.ceil(cx + half)), int(math.ceil(cy + half))
        window = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x1, image.shape[1]), min(y1, image.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
        shifted = np.asarray(box, np.float32) - np.array([x0, y0, x0, y0], np.float32)
        with contextlib.redirect_stdout(io.StringIO()):  # the dataset prints per item
            dataset = self.dataset_cls(self.cfg, np.ascontiguousarray(window[..., ::-1]), shifted[None],
                                       np.array([1.0 if right else 0.0], np.float32), rescale_factor=RESCALE)
            item = dataset[0]
        item["box_center"] = item["box_center"] + np.array([x0, y0], np.float32)
        item["img_size"] = np.array([image.shape[1], image.shape[0]], np.float32)
        return item

    def forward(self, items: list[dict[str, Any]], profile: bool = False) -> dict[str, np.ndarray]:
        """Batched fits; arrays per hand, left hands mirrored back (``joints``, ``verts`` in the
        regressor's camera axes, origin at MANO's), keypoints in image pixels."""
        torch = self.torch
        from torch.utils.data import default_collate

        out_all: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(items), BATCH):
            batch = default_collate(items[start:start + BATCH])
            batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
            with torch.no_grad():
                if profile and not self.kernels:
                    from experiments.audit.env_smoke import profile_cuda

                    out, self.kernels = profile_cuda(lambda: self.model(batch))
                else:
                    out = self.model(batch)
            m = (2 * batch["right"] - 1).float().cpu().numpy()
            joints = out["pred_keypoints_3d"][:, :21].float().cpu().numpy().copy()
            verts = out["pred_vertices"].float().cpu().numpy().copy()
            joints[..., 0] *= m[:, None]
            verts[..., 0] *= m[:, None]
            kp = out["pred_keypoints_2d"][:, :21].float().cpu().numpy().copy()
            kp[..., 0] *= m[:, None]
            size = batch["box_size"].float().cpu().numpy()
            centre = batch["box_center"].float().cpu().numpy()
            cam = out["pred_cam"].float().cpu().numpy().copy()
            cam[:, 1] *= m
            params = out["pred_mano_params"]
            rot = torch.cat([params["global_orient"].reshape(-1, 1, 3, 3), params["hand_pose"].reshape(-1, 15, 3, 3)], 1)
            chunk = {"joints": joints, "verts": verts, "kp2d": kp * size[:, None, None] + centre[:, None, :],
                     "cam_crop": cam, "box_size": size, "box_center": centre,
                     "rotmats": rot.float().cpu().numpy(), "betas": params["betas"].float().cpu().numpy(),
                     "right": batch["right"].float().cpu().numpy()}
            for k, v in chunk.items():
                out_all.setdefault(k, []).append(v)
        return {k: np.concatenate(v, 0) for k, v in out_all.items()}


def full_translation(fit: dict[str, np.ndarray], img_wh: np.ndarray, focal: float | np.ndarray) -> np.ndarray:
    """The crop camera (s, tx, ty) as a translation for a pinhole of ``focal`` centred in the image
    (WiLoR's and HaMeR's ``cam_crop_to_full``)."""
    s, tx, ty = fit["cam_crop"][:, 0], fit["cam_crop"][:, 1], fit["cam_crop"][:, 2]
    bs = fit["box_size"] * s + 1e-9
    centre = fit["box_center"]
    return np.stack([2 * (centre[:, 0] - img_wh[0] / 2) / bs + tx, 2 * (centre[:, 1] - img_wh[1] / 2) / bs + ty,
                     2 * focal / bs], -1)


def pinhole(points: np.ndarray, focal: float, centre: tuple[float, float]) -> np.ndarray:
    return focal * points[..., :2] / points[..., 2:3] + np.asarray(centre)


def procrustes_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean joint error after the best similarity transform (PA-MPJPE), in the inputs' units."""
    mp, mg = pred.mean(0), gt.mean(0)
    p, g = pred - mp, gt - mg
    u, s, vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    u[:, -1] *= d
    r = u @ vt
    scale = s.sum() / max((p ** 2).sum(), 1e-12)
    return float(np.linalg.norm(scale * p @ r - g, axis=1).mean())


def load_models(args: argparse.Namespace) -> dict[str, Regressor]:
    return {kind: Regressor(kind, args) for kind in MODELS}


def mano_dir(args: argparse.Namespace, scratch: Path) -> str:
    """MANO loaders want one directory; staged files keep their cache names."""
    target = scratch / "mano"
    target.mkdir(parents=True, exist_ok=True)
    for name, source in (("MANO_LEFT.pkl", args.mano_left), ("MANO_RIGHT.pkl", args.mano_right)):
        if not (target / name).exists():
            (target / name).symlink_to(Path(source).resolve())
    return str(target)


# ----------------------------------------------------------------- HInt

def hint_entries(zip_path: str, limit: int) -> list[dict[str, Any]]:
    """HInt test hands, per split in content-blind order (``limit`` per split; 0 for all)."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        out = []
        for split in HINT_SPLITS:
            members = [n for n in names if n.split("/")[1:2] == [split] and n.endswith(".json")]
            members = content_order(members, "hint")
            if limit:
                members = members[:limit]
            for member in members:
                hands = json.loads(z.read(member))
                for index, hand in enumerate(hands):
                    out.append({"split": split, "member": member, "image": member[:-5] + ".jpg", "index": index,
                                "right": member.endswith("_r.json"), "box": hand["bbox"][0],
                                "keypoints": hand["keypoints"], "existence": hand["existence"],
                                "occlusion": hand["occlusion"]})
    return out


def run_hint(args: argparse.Namespace, models: dict[str, Regressor], publish: Path) -> dict[str, Any]:
    from PIL import Image

    entries = hint_entries(args.hint_zip, int(args.limit))
    units = [entries[i:i + HINT_UNIT] for i in range(0, len(entries), HINT_UNIT)]
    restored = restore_units(publish, "hint")
    done = len(restored)
    with zipfile.ZipFile(args.hint_zip) as z:
        for number, unit_entries in enumerate(units):
            unit = f"u{number:03d}"
            if unit in restored:
                continue
            began = time.time()
            images: dict[str, np.ndarray] = {}
            for e in unit_entries:
                if e["image"] not in images:
                    images[e["image"]] = np.asarray(Image.open(io.BytesIO(z.read(e["image"]))).convert("RGB"))
            arrays: dict[str, Any] = {
                "member": np.array([e["member"] for e in unit_entries]), "index": np.array([e["index"] for e in unit_entries]),
                "split": np.array([e["split"] for e in unit_entries]), "right": np.array([e["right"] for e in unit_entries]),
                "box": np.array([e["box"] for e in unit_entries], float),
                "gt": np.array([e["keypoints"] for e in unit_entries], float),
                "existence": np.array([e["existence"] for e in unit_entries], float) > 0.5,
                "occlusion": np.array([e["occlusion"] for e in unit_entries], float) > 0.5,
            }
            arrays["norm"] = np.array([expanded_size(np.asarray(e["box"], float)) for e in unit_entries])
            for kind, model in models.items():
                items = [model.item(images[e["image"]], np.asarray(e["box"], np.float32), e["right"]) for e in unit_entries]
                fit = model.forward(items, profile=number == 0)
                arrays[f"{kind}_kp2d"] = fit["kp2d"]
                # Parameters, so combinations of the two models can be scored offline (H2b).
                arrays[f"{kind}_rotvec"] = mano_np.rotvec_of(fit["rotmats"].astype(float))
                arrays[f"{kind}_betas"] = fit["betas"]
                arrays[f"{kind}_cam_crop"] = fit["cam_crop"]
                arrays[f"{kind}_box_size"] = fit["box_size"]
                arrays[f"{kind}_box_center"] = fit["box_center"]
                arrays[f"{kind}_dist"] = np.linalg.norm(fit["kp2d"] - arrays["gt"], axis=-1) / arrays["norm"][:, None]
            save_unit(publish, "hint", unit, arrays, {"unit": unit, "hands": len(unit_entries),
                                                       "images": len(images), "seconds": round(time.time() - began, 2)})
            done += 1
            progress(done)
    return {"hands": len(entries), "units": len(units), "restored_units": sorted(restored),
            "per_split": {s: sum(1 for e in entries if e["split"] == s) for s in HINT_SPLITS}}


def run_speed(args: argparse.Namespace, models: dict[str, Regressor]) -> dict[str, Any]:
    """Regressor time per hand at batch 1 and batch 32, and WiLoR's detector per frame (fp32)."""
    import torch
    from PIL import Image
    from torch.utils.data import default_collate
    from ultralytics import YOLO

    entries = [e for e in hint_entries(args.hint_zip, 64) if e["split"] == "TEST_epick_img"][:BATCH]
    with zipfile.ZipFile(args.hint_zip) as z:
        images = {e["image"]: np.asarray(Image.open(io.BytesIO(z.read(e["image"]))).convert("RGB")) for e in entries}
    out: dict[str, Any] = {"hands": len(entries)}

    def timed(fn: Any) -> float:
        torch.cuda.synchronize()
        began = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - began) * 1000.0

    for kind, model in models.items():
        prep = []
        items = []
        for e in entries:
            began = time.perf_counter()
            items.append(model.item(images[e["image"]], np.asarray(e["box"], np.float32), e["right"]))
            prep.append((time.perf_counter() - began) * 1000.0)
        singles = [{k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in default_collate([it]).items()} for it in items]
        full = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in default_collate(items).items()}
        with torch.no_grad():
            for _ in range(3):
                model.model(full)
            for s in singles[:5]:
                model.model(s)
            one = [timed(lambda s=s: model.model(s)) for s in singles]
            many = [timed(lambda: model.model(full)) / len(items) for _ in range(5)]
        out[kind] = {"batch1_ms_per_hand": round(float(np.median(one)), 2), "batch1_ms_all": [round(v, 2) for v in one],
                     f"batch{len(items)}_ms_per_hand": round(float(np.median(many)), 2),
                     "preprocess_ms_per_hand_cpu": round(float(np.median(prep)), 2),
                     "parameters": int(sum(p.numel() for p in model.model.parameters()))}
    detector = YOLO(args.wilor_detector)
    frames = list(images.values())[:16]
    for f in frames[:2]:
        detector.predict(np.ascontiguousarray(f[..., ::-1]), conf=0.3, device=0, verbose=False)
    times = [timed(lambda f=f: detector.predict(np.ascontiguousarray(f[..., ::-1]), conf=0.3, device=0, verbose=False))
             for f in frames]
    out["wilor_detector"] = {"ms_per_frame": round(float(np.median(times)), 2), "frames": len(frames),
                             "weights_sha256": file_sha256(Path(args.wilor_detector))}
    return out


# ----------------------------------------------------------------- HOT3D

def hot3d_clip_frames(path: str, frames: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A clip's hand shape and its first ``frames`` frames (cameras, hands, crops, image ``214-1``)."""
    from PIL import Image

    with tarfile.open(path) as tar:
        members = {m.name: m for m in tar.getmembers()}

        def read(name: str) -> bytes:
            handle = tar.extractfile(members[name])
            assert handle is not None
            return handle.read()

        shapes = json.loads(read("__hand_shapes.json__"))
        keys = sorted({n.split(".")[0] for n in members if n[:6].isdigit()})[:frames]
        rows = []
        for key in keys:
            rows.append({"key": key, "cameras": json.loads(read(f"{key}.cameras.json")),
                         "hands": json.loads(read(f"{key}.hands.json")),
                         "crops": json.loads(read(f"{key}.hand_crops.json")),
                         "info": json.loads(read(f"{key}.info.json")),
                         "image": np.asarray(Image.open(io.BytesIO(read(f"{key}.image_{HOT3D_STREAM}.jpg"))).convert("RGB"))})
    return shapes, rows


def run_hot3d(args: argparse.Namespace, models: dict[str, Regressor], publish: Path, right_model: mano_np.Mano) -> dict[str, Any]:
    import torch
    from hand_tracking_toolkit import camera
    from hand_tracking_toolkit.camera import PinholePlaneCameraModel
    from hand_tracking_toolkit.dataset import HandSide, decode_hand_crop_params, decode_hand_pose, warp_image
    from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

    gt_model = MANOHandModel(args.mano_dir)
    clips = list(args.hot3d_clip)[:int(args.clips)] if int(args.clips) else list(args.hot3d_clip)
    restored = restore_units(publish, "hot3d")
    done = len(restored)
    for number, clip in enumerate(clips):
        unit = Path(clip).stem
        if unit in restored:
            continue
        began = time.time()
        shapes, rows = hot3d_clip_frames(clip, int(args.frames))
        beta = torch.tensor(shapes["mano"], dtype=torch.float32)
        hands: list[dict[str, Any]] = []
        for t, row in enumerate(rows):
            cam214 = camera.from_json(row["cameras"][HOT3D_STREAM])
            poses = decode_hand_pose(row["hands"])
            crops = decode_hand_crop_params(row["crops"], HOT3D_CROP)
            for side, pose in poses.items():
                if pose.mano is None or HOT3D_STREAM not in crops.get(side, {}):
                    continue
                # HOT3D's crops frame the hand so tightly that its mesh leaves the crop; the regressors'
                # box × RESCALE would then be mostly padding. Same camera, wider view.
                base = crops[side][HOT3D_STREAM]
                crop_cam = PinholePlaneCameraModel(width=HOT3D_CROP, height=HOT3D_CROP,
                                                   f=(base.f[0] * HOT3D_ZOOM, base.f[1] * HOT3D_ZOOM), c=base.c,
                                                   distort_coeffs=[], T_world_from_eye=base.T_world_from_eye)
                hands.append({"t": t, "right": side == HandSide.RIGHT, "pose": pose.mano, "cam214": cam214,
                              "crop_cam": crop_cam, "crop": warp_image(cam214, crop_cam, row["image"])})
        if not hands:
            save_unit(publish, "hot3d", unit, {"t": np.zeros(0, int)}, {"unit": unit, "hands": 0,
                                                                         "clip_sha256": file_sha256(Path(clip))})
            continue
        with torch.no_grad():
            verts_w, lm_w = gt_model(beta[None].repeat(len(hands), 1), torch.stack([h["pose"].mano_theta for h in hands]),
                                     torch.stack([h["pose"].wrist_xform for h in hands]),
                                     torch.tensor([h["right"] for h in hands]))
        verts_w = verts_w.double().numpy()
        gt_w = lm_w.double().numpy()[:, list(mano_np.MANO_TO_OPENPOSE)]
        tip_check = float(np.abs(gt_w[:, [4, 8, 12, 16, 20]] - verts_w[:, list(mano_np.TIPS)]).max())
        n = len(hands)
        f_crop = float(hands[0]["crop_cam"].f[0])
        c_crop = (float(hands[0]["crop_cam"].c[0]), float(hands[0]["crop_cam"].c[1]))
        gt_eye = np.stack([h["crop_cam"].world_to_eye(gt_w[i]) for i, h in enumerate(hands)])
        gt_verts_eye = np.stack([h["crop_cam"].world_to_eye(verts_w[i]) for i, h in enumerate(hands)])
        boxes = np.zeros((n, 4))
        for i in range(n):
            uv = pinhole(gt_verts_eye[i], f_crop, c_crop)
            boxes[i] = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
        arrays: dict[str, np.ndarray] = {
            "t": np.array([h["t"] for h in hands]), "right": np.array([h["right"] for h in hands]), "box_crop": boxes,
            "gt_world": gt_w, "gt_214": np.stack([h["cam214"].world_to_eye(gt_w[i]) for i, h in enumerate(hands)]),
            "crop_focal": np.array([float(h["crop_cam"].f[0]) for h in hands]),
            "R_world_214": np.stack([h["cam214"].T_world_from_eye[:3, :3] for h in hands]),
        }
        checks: dict[str, float] = {"gt_tips_vs_vertices_m": tip_check}
        for kind, model in models.items():
            items = [model.item(h["crop"], boxes[i].astype(np.float32), h["right"]) for i, h in enumerate(hands)]
            fit = model.forward(items, profile=number == 0)
            focal = arrays["crop_focal"]
            t_crop = full_translation(fit, np.array([HOT3D_CROP, HOT3D_CROP]), focal)
            p_eye = fit["joints"] + t_crop[:, None]
            v_eye = fit["verts"] + t_crop[:, None]
            rel_p, rel_g = p_eye - p_eye[:, :1], gt_eye - gt_eye[:, :1]
            arrays[f"{kind}_mpjpe_ra_mm"] = np.linalg.norm(rel_p - rel_g, axis=-1).mean(1) * 1000
            arrays[f"{kind}_pa_mpjpe_mm"] = np.array([procrustes_error(p_eye[i], gt_eye[i]) for i in range(n)]) * 1000
            arrays[f"{kind}_mpvpe_ra_mm"] = np.linalg.norm((v_eye - p_eye[:, :1]) - (gt_verts_eye - gt_eye[:, :1]),
                                                           axis=-1).mean(1) * 1000
            sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            arrays[f"{kind}_err2d_box"] = np.array([np.linalg.norm(pinhole(p_eye[i], focal[i], c_crop) - pinhole(gt_eye[i], focal[i], c_crop),
                                                                   axis=-1).mean() for i in range(n)]) / sizes
            # Fixed world axes for acceleration: root-relative joints rotated out of the moving crop cameras.
            arrays[f"{kind}_rel_world"] = np.stack([hands[i]["crop_cam"].T_world_from_eye[:3, :3] @ rel_p[i].T for i in range(n)]).transpose(0, 2, 1)
            # The fit in the fisheye camera's frame, for the coding stage.
            go, tr = np.zeros((n, 3, 3)), np.zeros((n, 3))
            for i, h in enumerate(hands):
                t_wc = h["crop_cam"].T_world_from_eye
                t_w214 = h["cam214"].T_world_from_eye
                rotation = t_w214[:3, :3].T @ t_wc[:3, :3]
                offset = t_w214[:3, :3].T @ (t_wc[:3, 3] - t_w214[:3, 3])
                rest_root = right_model.shaped(fit["betas"][i])[1][0]
                go[i], tr[i] = mano_np.change_frame(rotation, offset, fit["rotmats"][i, 0], t_crop[i], rest_root, not h["right"])
            rot214 = fit["rotmats"].astype(float).copy()
            rot214[:, 0] = go
            arrays[f"{kind}_rotvec"] = mano_np.rotvec_of(rot214)  # (n, 16, 3): global then 15 joints
            arrays[f"{kind}_betas"] = fit["betas"]
            arrays[f"{kind}_transl"] = tr
            arrays[f"{kind}_box_size"] = fit["box_size"]
            check = []
            for i, h in enumerate(hands):
                mine, _ = right_model.forward(rot214[i:i + 1], fit["betas"][i], tr[i:i + 1], left=not h["right"])
                want = h["cam214"].world_to_eye(h["crop_cam"].eye_to_world(p_eye[i]))
                check.append(float(np.abs(mine[0] - want).max()))
            checks[f"{kind}_numpy_mano_vs_model_m"] = max(check)
        meta = {"unit": unit, "clip": clip, "clip_sha256": file_sha256(Path(clip)), "hands": n, "frames": len(rows),
                "participant": rows[0]["info"].get("participant_id"), "sequence": rows[0]["info"].get("sequence_id"),
                "camera_214": rows[0]["cameras"][HOT3D_STREAM], "crop_principal": c_crop, "fps": 30.0, "checks": checks,
                "seconds": round(time.time() - began, 2)}
        save_unit(publish, "hot3d", unit, arrays, meta)
        done += 1
        progress(done)
    return {"clips": len(clips), "restored_units": sorted(restored)}


# ----------------------------------------------------------------- VISOR

def pack_window(mask: np.ndarray, box: np.ndarray, grow: float = 1.5) -> tuple[np.ndarray, bytes]:
    """The mask inside the box grown ``grow`` times (clipped), as (x0, y0, x1, y1) and packed bits."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = grow * max(box[2] - box[0], box[3] - box[1]) / 2
    x0, y0 = max(int(cx - half), 0), max(int(cy - half), 0)
    x1, y1 = min(int(math.ceil(cx + half)), mask.shape[1]), min(int(math.ceil(cy + half)), mask.shape[0])
    return np.array([x0, y0, x1, y1]), np.packbits(mask[y0:y1, x0:x1]).tobytes()


def unpack_window(window: np.ndarray, data: bytes) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in window)
    bits = np.unpackbits(np.frombuffer(data, np.uint8))[:(y1 - y0) * (x1 - x0)]
    return bits.reshape(y1 - y0, x1 - x0).astype(bool)


def run_visor(args: argparse.Namespace, models: dict[str, Regressor], publish: Path, right_model: mano_np.Mano) -> dict[str, Any]:
    from experiments.visor.b2 import load_mask_sets, select_items
    from experiments.visor.h1 import decode_window, iou, rasterize, wrist_split
    from src.segmentation import visor

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    boxes_all = json.loads(Path(args.h1_boxes).read_text())["items"]
    videos = {Path(v).stem: v for v in args.video}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    restored = restore_units(publish, "visor")
    todo = [it for it in items if it["id"] not in restored]
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
    count = int(args.frames)
    shape = visor.FRAME_SIZE
    focal = NOMINAL_FOCAL * max(shape)
    centre = (shape[1] / 2.0, shape[0] / 2.0)

    def submit(item: dict[str, Any]) -> Any:
        target = scratch / "frames" / f"{item['id']}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        return pool.submit(decode_window, videos[item["video"]], item, count, str(args.archive), str(target),
                           max(1, allowance // 2))

    pending = {i: submit(todo[i]) for i in range(min(2, len(todo)))}
    done = len(restored)
    for number, item in enumerate(todo):
        decoded = pending.pop(number).result()
        if number + 2 < len(todo):
            pending[number + 2] = submit(todo[number + 2])
        began = time.time()
        frames_path = scratch / "frames" / f"{item['id']}.npy"
        frames = np.load(frames_path, mmap_mode="r")
        clips, mask_info = load_mask_sets(item, count, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
        rows = [r for r in boxes_all[item["id"]] if r["t"] < count]
        n = len(rows)
        arrays: dict[str, Any] = {"t": np.array([r["t"] for r in rows], int),
                                  "right": np.array([r["side"] == "right hand" for r in rows]),
                                  "box": np.array([r["box"] for r in rows], float).reshape(-1, 4)}
        masks = [clips[FILL].class_mask(r["t"], r["side"]) for r in rows]
        checks: dict[str, float] = {}
        for kind, model in models.items():
            inputs = [model.item(np.asarray(frames[r["t"]]), np.asarray(r["box"], np.float32), r["side"] == "right hand")
                      for r in rows]
            fit = model.forward(inputs, profile=number == 0) if inputs else None
            ious, ious_mask, forearm, windows, packed = [], [], [], [], []
            if fit is not None:
                t_full = full_translation(fit, np.array([shape[1], shape[0]]), focal)
                for i in range(n):
                    verts2d = pinhole(fit["verts"][i] + t_full[i], focal, centre)
                    joints2d = pinhole(fit["joints"][i] + t_full[i], focal, centre)
                    silhouette = rasterize(verts2d, model.faces, shape)
                    hand, arm = wrist_split(masks[i], joints2d[0], joints2d[9])
                    ious.append(iou(silhouette, hand))
                    ious_mask.append(iou(silhouette, masks[i]))
                    forearm.append(float(arm.sum()) / max(int(masks[i].sum()), 1))
                    window, data = pack_window(hand, arrays["box"][i])
                    windows.append(window)
                    packed.append(data)
                arrays[f"{kind}_rotvec"] = mano_np.rotvec_of(fit["rotmats"].astype(float))
                arrays[f"{kind}_betas"] = fit["betas"]
                arrays[f"{kind}_transl"] = t_full
                arrays[f"{kind}_kp2d"] = np.stack([pinhole(fit["joints"][i] + t_full[i], focal, centre) for i in range(n)])
                arrays[f"{kind}_box_size"] = fit["box_size"]
                sample = list(range(0, n, max(1, n // 8)))
                checks[f"{kind}_numpy_mano_vs_model_m"] = max(
                    float(np.abs(right_model.forward(fit["rotmats"][i:i + 1], fit["betas"][i], t_full[i:i + 1],
                                                     left=not bool(arrays["right"][i]))[0][0] - (fit["joints"][i] + t_full[i])).max())
                    for i in sample)
            arrays[f"{kind}_iou_hand"] = np.array([np.nan if v is None else v for v in ious], float)
            arrays[f"{kind}_iou_mask"] = np.array([np.nan if v is None else v for v in ious_mask], float)
            arrays[f"{kind}_forearm_share"] = np.array(forearm, float)
            arrays[f"{kind}_hand_window"] = np.array(windows, int).reshape(-1, 4)
            arrays[f"{kind}_hand_bits"] = np.frombuffer(b"".join(packed), np.uint8)
            arrays[f"{kind}_hand_bits_len"] = np.array([len(d) for d in packed], int)
        meta = {"unit": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": count,
                "hands": n, "decode": decoded, "mask_set": mask_info[FILL], "focal": focal, "principal": centre,
                "image_shape": list(shape), "checks": checks, "seconds": round(time.time() - began, 2)}
        save_unit(publish, "visor", item["id"], arrays, meta)
        frames_path.unlink()
        done += 1
        progress(done)
    pool.shutdown()
    return {"items": [it["id"] for it in items], "restored_units": sorted(restored),
            "h1_boxes_sha256": file_sha256(Path(args.h1_boxes)), "eval_set_sha256": file_sha256(Path(args.eval_set))}


# ----------------------------------------------------------------- run

def command_run(args: argparse.Namespace) -> int:
    import cv2
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda

    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    args.mano_dir = mano_dir(args, scratch)
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    cv2.setNumThreads(allowance)
    torch.set_num_threads(allowance)
    device = device_record(torch.cuda.get_device_name(0))
    models = load_models(args)
    right_model = mano_np.Mano.load(args.mano_right)
    began = time.time()
    if args.part == "hint":
        result: dict[str, Any] = {"hint": run_hint(args, models, publish)}
        if args.speed:
            result["speed"] = run_speed(args, models)
            write_json(publish / "speed.json", result["speed"])
    elif args.part == "hot3d":
        result = {"hot3d": run_hot3d(args, models, publish, right_model)}
    else:
        result = {"visor": run_visor(args, models, publish, right_model)}
    inputs = {"mano_left_sha256": file_sha256(Path(args.mano_left)), "mano_right_sha256": file_sha256(Path(args.mano_right))}
    if args.part == "hint":
        inputs["hint_zip_sha256"] = file_sha256(Path(args.hint_zip))
    write_json(stage_dir() / "h2.json", {
        "part": args.part, "device": device, "seconds": round(time.time() - began, 2),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "models": {k: {**m.load, "kernels": m.kernels, "on_cuda": model_on_cuda(m.model)} for k, m in models.items()},
        "inputs": inputs,
        "settings": {"rescale": RESCALE, "aspect": ASPECT, "pck_thresholds": PCK_THRESHOLDS, "hint_splits": HINT_SPLITS,
                     "hot3d_stream": HOT3D_STREAM, "hot3d_crop": HOT3D_CROP, "hot3d_zoom": HOT3D_ZOOM, "nominal_focal": NOMINAL_FOCAL,
                     "limit": args.limit, "clips": args.clips, "frames": args.frames, "items": args.items,
                     "decision": DECISION},
        **result,
    })
    return 0


# ----------------------------------------------------------------- reading published units

def load_units(tars: list[str], part: str) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Every ``publish/<part>/<unit>`` in the given stage archives (``published.tar`` or ``partial.tar``);
    a unit in several archives is read once."""
    out: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for path in tars:
        with tarfile.open(path) as tar:
            members = {m.name: m for m in tar.getmembers()}
            for name in sorted(members):
                if not (name.startswith(f"publish/{part}/") and name.endswith(".json")):
                    continue
                unit = name[len(f"publish/{part}/"):-5]
                if unit in out:
                    continue
                meta_handle = tar.extractfile(members[name])
                data_handle = tar.extractfile(members[name[:-5] + ".npz"])
                assert meta_handle is not None and data_handle is not None
                meta = json.loads(meta_handle.read())
                data = data_handle.read()
                meta["npz_hash_ok"] = hashlib.sha256(data).hexdigest() == meta["npz_sha256"]
                with np.load(io.BytesIO(data), allow_pickle=False) as npz:
                    out[unit] = (meta, {k: npz[k] for k in npz.files})
    return [out[k] for k in sorted(out)]


# ----------------------------------------------------------------- validate

def validate_code(stage: Path) -> dict[str, bool]:
    """The coding stage: every combination for both datasets and models, references, choices, IoUs."""
    from experiments.visor import pose_coding as pc

    code = json.loads((stage / "code.json").read_text())
    rows = code["rows"]
    keys = {pc.combo_key(c) for c in pc.combinations()}
    numbers = [v for name in rows for m in MODELS for r in rows[name][m] for k, v in r.items()
               if k.endswith(("_mm", "_px", "kbps")) and isinstance(v, float)]
    h1 = {m: {r["key"]: r for r in rows["visor"][m]}[H1_BASELINE]["kbps"] for m in MODELS}
    return {
        "every_combination_evaluated": all({r["key"] for r in rows[name][m]} == keys for name in ("hot3d", "visor") for m in MODELS),
        "values_finite_and_non_negative": bool(numbers) and all(math.isfinite(v) and v >= 0 for v in numbers),
        "truth_references_present": all("mpjpe_vs_truth_mm" in code["baselines"]["hot3d"][m] for m in MODELS),
        # H1's coding of WiLoR on all 34 items cost 15.2 kbps; the same coding here must land near it.
        "h1_coding_reproduced_within_25pct": code["units"]["visor"] < 34 or abs(h1["wilor"] - 15.2) / 15.2 <= 0.25,
        "finer_steps_cost_more": all({r["key"]: r for r in rows["visor"][m]}["none|all|full|q0.5|previous"]["kbps"] > h1[m] for m in MODELS),
        "choices_for_every_budget": all(set(code["choices"][m]) == {f"{b:g}" for b in DECISION["coding"]["budgets_ms"]} for m in MODELS),
        "uncoded_iou_rendered": all("uncoded" in code["visor_iou"][m] for m in MODELS),
        "shape_once_keeps_the_wrist": all(code["baselines"][name][m]["shape_once_vs_per_frame"]["wrist_offset_m"] < 1e-6
                                          for name in ("hot3d", "visor") for m in MODELS),
        "two_d_errors_bounded": all(code["baselines"]["hot3d"][m]["err2d_vs_truth_px"] < 1408 for m in MODELS),
    }


def validate_hybrid(stage: Path) -> dict[str, bool]:
    out = json.loads((stage / "hybrid.json").read_text())
    c = out["checks"]
    return {
        "hint_models_reproduced_0.5px": c["hint_reproduction_px"] < 0.5,
        "hot3d_models_reproduced_1um": c["hot3d_reproduction_m"] < 1e-6,
        "visor_wilor_iou_reproduced": c["visor_wilor_iou_median_vs_job"] < 0.005,
        "hybrid_keeps_hamer_wrist": c["hybrid_wrist_vs_hamer_m"] < 1e-6 and c["hint_hybrid_wrist_vs_hamer_m"] < 1e-6,
        "every_part_scored": all(k in out["summary"] for k in ("hint", "hot3d", "visor")),
        "decision_made": out["decision"]["adopt_hybrid"] in (True, False),
    }


def validate_result(stage: Path) -> dict[str, bool]:
    if (stage / "hybrid.json").exists():
        return validate_hybrid(stage)
    if (stage / "code.json").exists():
        return validate_code(stage)
    result = json.loads((stage / "h2.json").read_text())
    tar = stage / "published.tar"
    part = result["part"]
    units = load_units([str(tar)], part)
    models = result["models"]
    checks = {
        "units_published": bool(units) and all(m["npz_hash_ok"] for m, _ in units),
        "models_on_cuda": all(m["on_cuda"] and m["kernels"].get("kernel_launches", 0) > 0 for m in models.values()),
        "weights_load_completely": all(m["missing_count"] == 0 and m["unexpected_count"] == 0 for m in models.values()),
        "pinned_wilor_weights": models["wilor"]["checkpoint_sha256"] == "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
    }
    if part == "hint":
        info = result["hint"]
        checks["every_unit_present"] = len(units) == info["units"]
        dist = {k: np.concatenate([a[f"{k}_dist"][a["existence"]] for _, a in units]) for k in MODELS}
        checks["distances_finite"] = all(np.isfinite(v).all() for v in dist.values())
        # Both models land near the labelled joints: a wrong box, flip or scale puts the median far off.
        checks["median_distance_under_0.15_box"] = all(float(np.median(v)) < 0.15 for v in dist.values())
        checks["both_splits"] = all(info["per_split"][s] > 0 for s in HINT_SPLITS)
        if "speed" in result:
            speed = result["speed"]
            checks["speed_measured"] = all(speed[k]["batch1_ms_per_hand"] > 0 for k in MODELS) and speed["wilor_detector"]["ms_per_frame"] > 0
            checks["speed_on_a6000"] = "A6000" in result["device"]["name"]
    elif part == "hot3d":
        checks["every_clip_present"] = len(units) == result["hot3d"]["clips"]
        rows = [(m, a) for m, a in units if m["hands"]]
        checks["hands_found"] = bool(rows)
        checks["gt_fingertips_are_the_tip_vertices"] = all(m["checks"]["gt_tips_vs_vertices_m"] < 1e-5 for m, _ in rows)
        # The numpy MANO the coding stage uses reproduces each model's joints after the change of frame.
        checks["numpy_mano_matches_models_1mm"] = all(m["checks"][f"{k}_numpy_mano_vs_model_m"] < 1e-3 for m, _ in rows for k in MODELS)
        boxes = np.concatenate([a["box_crop"] for _, a in rows])
        checks["gt_boxes_inside_crops_90pct"] = float(np.mean((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= HOT3D_CROP)
                                                              & (boxes[:, 3] <= HOT3D_CROP))) >= 0.9
        checks["median_mpjpe_under_50mm"] = all(float(np.median(np.concatenate([a[f"{k}_mpjpe_ra_mm"] for _, a in rows]))) < 50
                                                for k in MODELS)
    else:
        info = result["visor"]
        checks["every_item_present"] = len(units) == len(info["items"])
        checks["sparse_jpegs_match_decoded_frames"] = all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"])
        checks["mask_set_matches_record"] = all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"]
                                                for m, _ in units)
        checks["numpy_mano_matches_models_1mm"] = all(m["checks"][f"{k}_numpy_mano_vs_model_m"] < 1e-3 for m, _ in units
                                                      for k in MODELS if m["hands"])
        ious = {k: np.concatenate([a[f"{k}_iou_hand"] for _, a in units]) for k in MODELS}
        checks["ious_in_unit_interval"] = all(np.nanmin(v) >= 0 and np.nanmax(v) <= 1 for v in ious.values())
        checks["median_hand_iou_over_0.4"] = all(float(np.nanmedian(v)) > 0.4 for v in ious.values())
        checks["windows_unpack"] = all(_windows_ok(a) for _, a in units)
    return checks


def _windows_ok(a: dict[str, np.ndarray]) -> bool:
    for k in MODELS:
        offsets = np.concatenate([[0], np.cumsum(a[f"{k}_hand_bits_len"])])
        if offsets[-1] != len(a[f"{k}_hand_bits"]):
            return False
        for i in range(min(3, len(offsets) - 1)):
            unpack_window(a[f"{k}_hand_window"][i], a[f"{k}_hand_bits"][offsets[i]:offsets[i + 1]].tobytes())
    return True


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- code (CPU)

H1_BASELINE = "none|all|full|q1|previous"


class CodingSet:
    """One dataset's fits by one model, as hand tracks, with the references distortion is measured against."""

    def __init__(self, name: str, model: str, units: list[tuple[dict[str, Any], dict[str, np.ndarray]]],
                 mano: mano_np.Mano) -> None:
        from experiments.visor import pose_coding as pc

        self.name, self.model, self.mano = name, model, mano
        rot, betas, transl, left, unit_of, truth = [], [], [], [], [], []
        widths = []
        self.units: list[dict[str, Any]] = []
        self.tracks: list[dict[str, Any]] = []
        self.cameras: list[Any] = []
        offset = 0
        for u, (meta, a) in enumerate(units):
            n = int(meta["hands"])
            fps = float(meta["fps"])
            frames = int(meta["frames"])
            if name == "hot3d":
                from hand_tracking_toolkit import camera

                cam = camera.from_json(meta["camera_214"])
                focal, centre = float(cam.f[0]), (float(cam.c[0]), float(cam.c[1]))
                self.cameras.append(cam)
                widths.append(int(cam.width))
            else:
                focal, centre = float(meta["focal"]), tuple(meta["principal"])
                self.cameras.append(None)
                widths.append(int(meta["image_shape"][1]))
            self.units.append({"unit": meta["unit"], "fps": fps, "seconds": frames / fps, "hands": n, "focal": focal,
                               "centre": centre, "rows": slice(offset, offset + n)})
            if not n:
                continue
            rv = a[f"{model}_rotvec"]
            rot.append(mano_np.rodrigues(rv))
            betas.append(a[f"{model}_betas"])
            transl.append(a[f"{model}_transl"])
            left.append(~a["right"].astype(bool))
            unit_of.append(np.full(n, u))
            if name == "hot3d":
                truth.append(a["gt_214"])
            t = a["t"]
            for side in (True, False):
                idx = np.nonzero(a["right"].astype(bool) == side)[0]
                idx = idx[np.argsort(t[idx])]
                if not len(idx):
                    continue
                cuts = np.nonzero(np.diff(t[idx]) != 1)[0] + 1
                for run in np.split(idx, cuts):
                    self.tracks.append({"unit": u, "rows": run + offset, "left": not side, "fps": fps,
                                        "betas": np.median(a[f"{model}_betas"][run], 0)})
            offset += n
        self.rot = np.concatenate(rot)
        self.betas = np.concatenate(betas)
        self.transl = np.concatenate(transl)
        self.left = np.concatenate(left)
        self.unit_of = np.concatenate(unit_of)
        self.truth = np.concatenate(truth) if truth else None
        self.order = np.concatenate([tr["rows"] for tr in self.tracks])
        self.width = max(widths)  # 2D errors are capped here: a joint at the camera's plane projects to infinity
        self.uncoded = mano.joints(self.rot, self.betas, self.transl, self.left)
        self.uncoded_2d = self.project(self.uncoded)
        # The scheme sends each track's median shape once, and the root so that the wrist stays where the
        # frame's own shape put it (MANO turns about a shape-dependent rest wrist, not its origin).
        track_betas = np.zeros_like(self.betas)
        for tr in self.tracks:
            track_betas[tr["rows"]] = tr["betas"]
        shift = mano.rest_root(self.betas) - mano.rest_root(track_betas)
        shift[self.left] = shift[self.left] @ mano_np.MIRROR
        self.transl_sent = self.transl + shift
        self.reference = mano.joints(self.rot, track_betas, self.transl_sent, self.left)
        rv_all = mano_np.rotvec_of(self.rot)
        for tr in self.tracks:
            info = self.units[tr["unit"]]
            rows = tr["rows"]
            tr["x"] = np.concatenate([mano_np.unwrap_rotvecs(rv_all[rows, 0]), mano_np.unwrap_rotvecs(rv_all[rows, 1:]).reshape(len(rows), 45),
                                      pc.root_params(self.transl_sent[rows], info["focal"], info["centre"])], 1)
        self.reference_2d = self.project(self.reference)
        self.truth_2d = self.project(self.truth) if self.truth is not None else None

    def project(self, joints: np.ndarray) -> np.ndarray:
        out = np.zeros(joints.shape[:-1] + (2,))
        for u, info in enumerate(self.units):
            sel = self.unit_of == u
            if not sel.any():
                continue
            if self.cameras[u] is not None:
                out[sel] = self.cameras[u].eye_to_window(joints[sel].reshape(-1, 3)).reshape(-1, 21, 2)
            else:
                out[sel] = pinhole(joints[sel], info["focal"], info["centre"])
        return out

    def decode_joints(self, decoded: list[np.ndarray]) -> np.ndarray:
        """Joints for every frame (dataset row order) from each track's decoded values."""
        from experiments.visor import pose_coding as pc

        x = np.concatenate(decoded)
        rows = self.order
        rot = mano_np.rodrigues(x[:, :48].reshape(-1, 16, 3))
        betas = np.concatenate([np.repeat(tr["betas"][None], len(tr["rows"]), 0) for tr in self.tracks])
        transl = np.zeros((len(x), 3))
        start = 0
        for tr in self.tracks:
            info = self.units[tr["unit"]]
            end = start + len(tr["rows"])
            transl[start:end] = pc.root_transl(x[start:end, 48:51], info["focal"], info["centre"])
            start = end
        joints = self.mano.joints(rot, betas, transl, np.concatenate([np.full(len(tr["rows"]), tr["left"]) for tr in self.tracks]))
        out = np.zeros_like(joints)
        out[rows] = joints
        return out


def joint_errors(joints: np.ndarray, joints_2d: np.ndarray, ref: np.ndarray, ref_2d: np.ndarray,
                 cap: float = math.inf) -> tuple[float, float]:
    """Wrist-aligned 3D error (mm) and 2D error (px, each joint's capped at ``cap``), means over joints and frames."""
    mpjpe = np.linalg.norm((joints - joints[:, :1]) - (ref - ref[:, :1]), axis=-1).mean() * 1000
    px = np.nan_to_num(np.linalg.norm(joints_2d - ref_2d, axis=-1), nan=cap, posinf=cap)
    return float(mpjpe), float(np.minimum(px, cap).mean())


_SETS: dict[tuple[str, str], CodingSet] = {}


def evaluate_smoother(key: tuple[str, str], smoother: tuple[Any, ...], basis_arrays: tuple[np.ndarray, np.ndarray]) -> list[dict[str, Any]]:
    """Every combination that uses ``smoother`` on one dataset and model."""
    from experiments.visor import pose_coding as pc

    data = _SETS[key]
    basis = pc.Basis(*basis_arrays)
    smoothed = [pc.smooth(tr["x"], smoother, tr["fps"]) for tr in data.tracks]
    rows = []
    for hz in pc.SEND_HZ:
        for k in pc.SUBSPACES:
            for scale in pc.STEP_SCALES:
                base = {"smoother": smoother, "send_hz": hz, "fill": "hold", "subspace": k, "step_scale": scale,
                        "prediction": "previous"}
                encoded = [pc.encode(tr["x"], base, tr["fps"], basis, smoothed=s) for tr, s in zip(data.tracks, smoothed)]
                sent_per_unit = np.zeros(len(data.units))
                for tr, (_, sent) in zip(data.tracks, encoded):
                    sent_per_unit[tr["unit"]] += len(sent)
                frames = sum(len(tr["rows"]) for tr in data.tracks)
                rates = {}
                for prediction in pc.PREDICTIONS:
                    bits = pc.pooled_bits([pc.residuals(sym, prediction) for sym, _ in encoded])
                    kbps = [bits * sent_per_unit[u] / info["seconds"] / 1000 for u, info in enumerate(data.units)]
                    rates[prediction] = {"bits_per_sent_frame": bits, "kbps": float(np.mean(kbps))}
                for fill_mode in (("hold",) if hz is None else pc.FILLS):
                    combo = dict(base, fill=fill_mode)
                    decoded = [pc.decode(sym, sent, len(tr["x"]), combo, basis) for tr, (sym, sent) in zip(data.tracks, encoded)]
                    joints = data.decode_joints(decoded)
                    joints_2d = data.project(joints)
                    mp_unc, px_unc = joint_errors(joints, joints_2d, data.reference, data.reference_2d, data.width)
                    row_common = {"mpjpe_vs_uncoded_mm": mp_unc, "err2d_vs_uncoded_px": px_unc,  # uncoded = shape-once reference
                                  "sent_share": float(sent_per_unit.sum() / max(frames, 1)),
                                  "latency_ms": max(pc.latency_ms(combo, tr["fps"]) for tr in data.tracks) if data.tracks else 0.0}
                    if data.truth is not None:
                        assert data.truth_2d is not None
                        mp_true, px_true = joint_errors(joints, joints_2d, data.truth, data.truth_2d, data.width)
                        row_common |= {"mpjpe_vs_truth_mm": mp_true, "err2d_vs_truth_px": px_true}
                    for prediction in pc.PREDICTIONS:
                        combo_p = dict(combo, prediction=prediction)
                        rows.append({"key": pc.combo_key(combo_p), **combo_p, **rates[prediction], **row_common})
    return rows


def coding_choice(rows: dict[str, dict[str, list[dict[str, Any]]]], baselines: dict[str, Any], model: str) -> dict[str, Any]:
    """Per latency budget, the lowest VISOR rate whose HOT3D errors against the truth stay within
    ``DECISION['coding']['max_error_increase']`` of the uncoded estimate's."""
    rule = DECISION["coding"]
    hot = {r["key"]: r for r in rows["hot3d"][model]}
    vis = {r["key"]: r for r in rows["visor"][model]}
    base = baselines["hot3d"][model]
    limit = 1.0 + rule["max_error_increase"]
    out = {}
    for budget in rule["budgets_ms"]:
        eligible = [k for k in vis if k in hot and max(hot[k]["latency_ms"], vis[k]["latency_ms"]) <= budget + 1e-6
                    and hot[k]["mpjpe_vs_truth_mm"] <= limit * base["mpjpe_vs_truth_mm"]
                    and hot[k]["err2d_vs_truth_px"] <= limit * base["err2d_vs_truth_px"]]
        best = min(eligible, key=lambda k: vis[k]["kbps"]) if eligible else None
        out[f"{budget:g}"] = None if best is None else {
            "key": best, "visor_kbps": vis[best]["kbps"], "hot3d_kbps": hot[best]["kbps"],
            "visor_bits_per_sent_frame": vis[best]["bits_per_sent_frame"],
            "hot3d_mpjpe_vs_truth_mm": hot[best]["mpjpe_vs_truth_mm"], "hot3d_err2d_vs_truth_px": hot[best]["err2d_vs_truth_px"],
            "visor_err2d_vs_uncoded_px": vis[best]["err2d_vs_uncoded_px"],
            "latency_ms": max(hot[best]["latency_ms"], vis[best]["latency_ms"]), "eligible": len(eligible)}
    return out


def visor_iou(data: CodingSet, units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], combo: dict[str, Any] | None,
              basis: Any, faces: np.ndarray, tracks: list[int] | None = None) -> np.ndarray:
    """Rendered silhouettes of decoded hands against each mask's hand side (the model's own split), per
    hand-frame (NaN outside ``tracks``)."""
    from experiments.visor import pose_coding as pc
    from experiments.visor.h1 import iou, rasterize

    windows, bits = [], []
    for meta, a in units:
        if not meta["hands"]:
            continue
        offsets = np.concatenate([[0], np.cumsum(a[f"{data.model}_hand_bits_len"])])
        for i in range(int(meta["hands"])):
            windows.append(a[f"{data.model}_hand_window"][i])
            bits.append(a[f"{data.model}_hand_bits"][offsets[i]:offsets[i + 1]].tobytes())
    ious = np.full(len(data.rot), np.nan)
    for t_index, tr in enumerate(data.tracks):
        if tracks is not None and t_index not in tracks:
            continue
        info = data.units[tr["unit"]]
        rows = tr["rows"]
        if combo is None:
            # Each frame with its own shape, as the job rendered it.
            verts = np.stack([data.mano.mesh(data.rot[r:r + 1], data.betas[r], data.transl[r:r + 1], left=tr["left"])[0]
                              for r in rows])
        else:
            symbols, sent = pc.encode(tr["x"], combo, tr["fps"], basis)
            x = pc.decode(symbols, sent, len(tr["x"]), combo, basis)
            rot = mano_np.rodrigues(x[:, :48].reshape(-1, 16, 3))
            transl = pc.root_transl(x[:, 48:51], info["focal"], info["centre"])
            verts = data.mano.mesh(rot, tr["betas"], transl, left=tr["left"])
        shape = tuple(units[0][0]["image_shape"])
        for j, r in enumerate(rows):
            window = windows[r]
            hand = unpack_window(window, bits[r])
            full = np.zeros(shape, bool)
            full[window[1]:window[3], window[0]:window[2]] = hand
            silhouette = rasterize(pinhole(verts[j], info["focal"], info["centre"]), faces, shape)
            value = iou(silhouette, full)
            ious[r] = np.nan if value is None else value
    return ious


_VISOR_UNITS: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []


def iou_task(model: str, combo: dict[str, Any] | None, basis_arrays: tuple[np.ndarray, np.ndarray], tracks: list[int]) -> np.ndarray:
    from experiments.visor import pose_coding as pc

    data = _SETS[("visor", model)]
    return visor_iou(data, _VISOR_UNITS, combo, pc.Basis(*basis_arrays), data.mano.faces, tracks)


def command_code(args: argparse.Namespace) -> int:
    from experiments.visor import pose_coding as pc

    began = time.time()
    mano = mano_np.Mano.load(args.mano_right)
    units = {"hot3d": load_units(args.hot3d, "hot3d"), "visor": load_units(args.visor, "visor")}
    for name, rows in units.items():
        if not rows or not all(m["npz_hash_ok"] for m, _ in rows):
            raise SystemExit(f"{name}: missing or corrupt units")
        if int(args.units):
            units[name] = rows[:int(args.units)]  # a smoke's subset
    _VISOR_UNITS[:] = units["visor"]
    for name in units:
        for model in MODELS:
            _SETS[(name, model)] = CodingSet(name, model, units[name], mano)
    basis_arrays = (mano.hands_mean, mano.hands_components)
    baselines: dict[str, dict[str, Any]] = {"hot3d": {}, "visor": {}}
    for (name, model), data in _SETS.items():
        entry: dict[str, Any] = {"hand_frames": len(data.rot), "tracks": len(data.tracks), "units": len(data.units)}
        shape_mm, shape_px = joint_errors(data.reference, data.reference_2d, data.uncoded, data.uncoded_2d, data.width)
        entry["shape_once_vs_per_frame"] = {"mpjpe_mm": shape_mm, "err2d_px": shape_px,
                                            "wrist_offset_m": float(np.abs(data.reference[:, 0] - data.uncoded[:, 0]).max())}
        if data.truth is not None:
            assert data.truth_2d is not None
            mp, px = joint_errors(data.reference, data.reference_2d, data.truth, data.truth_2d, data.width)
            entry |= {"mpjpe_vs_truth_mm": mp, "err2d_vs_truth_px": px}
            mp, px = joint_errors(data.uncoded, data.uncoded_2d, data.truth, data.truth_2d, data.width)
            entry["per_frame_shape"] = {"mpjpe_vs_truth_mm": mp, "err2d_vs_truth_px": px}
        baselines[name][model] = entry
    workers = max(1, int(args.workers or os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    tasks = [(key, s) for key in _SETS for s in pc.SMOOTHERS]
    results: dict[str, dict[str, list[dict[str, Any]]]] = {"hot3d": {m: [] for m in MODELS}, "visor": {m: [] for m in MODELS}}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(evaluate_smoother, key, s, basis_arrays): key for key, s in tasks}
        for number, future in enumerate(futures):
            name, model = futures[future]
            results[name][model].extend(future.result())
            progress(number + 1)
    choices = {m: coding_choice(results, baselines, m) for m in MODELS}
    by_key = {pc.combo_key(c): c for c in pc.combinations()}
    ious: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        pending = []
        for model in MODELS:
            keys: dict[str, dict[str, Any] | None] = {"uncoded": None, H1_BASELINE: by_key[H1_BASELINE]}
            for choice in choices[model].values():
                if choice is not None:
                    keys[choice["key"]] = by_key[choice["key"]]
            count = len(_SETS[("visor", model)].tracks)
            for label, combo in keys.items():
                shares = [list(range(i, count, workers)) for i in range(min(workers, count))]
                pending.append((model, label, [pool.submit(iou_task, model, combo, basis_arrays, sh) for sh in shares]))
        for number, (model, label, futures_iou) in enumerate(pending):
            values = np.fmax.reduce([f.result() for f in futures_iou])  # each share fills its own tracks
            ious.setdefault(model, {})[label] = {"median": float(np.nanmedian(values)), "mean": float(np.nanmean(values))}
            progress(len(tasks) + number + 1)
    out_dir = Path(args.out) if args.out else stage_dir()
    write_json(out_dir / "code.json", {
        "inputs": {"hot3d": [file_sha256(Path(p)) for p in args.hot3d], "visor": [file_sha256(Path(p)) for p in args.visor],
                   "mano_right_sha256": file_sha256(Path(args.mano_right))},
        "settings": {"smoothers": pc.SMOOTHERS, "send_hz": pc.SEND_HZ, "fills": pc.FILLS, "subspaces": pc.SUBSPACES,
                     "step_scales": pc.STEP_SCALES, "predictions": pc.PREDICTIONS, "angle_step_rad": pc.ANGLE_STEP,
                     "px_step": pc.PX_STEP, "logz_step": pc.LOGZ_STEP, "decision": DECISION["coding"],
                     "shape": "the track's median betas, sent once and not counted"},
        "units": {name: len(rows) for name, rows in units.items()},
        "baselines": baselines, "rows": results, "choices": choices, "visor_iou": ious,
        "seconds": round(time.time() - began, 1),
    })
    return 0


# ----------------------------------------------------------------- report

def pck(dist: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """HaMeR's PCK (%): per joint, correct over labelled, then the mean over joints."""
    labelled = mask.sum(0)
    correct = ((dist <= threshold) & mask).sum(0)
    per_joint = correct[labelled > 0] / labelled[labelled > 0]
    return float(per_joint.mean() * 100) if per_joint.size else float("nan")


def paired_bootstrap(groups: list[Any], stat: Any, rng: np.random.Generator, resamples: int,
                     names: tuple[str, str] = ("hamer", "wilor")) -> dict[str, float]:
    """``stat(groups) -> (a, b)``; the observed difference a - b and its percentile interval over
    resamples of the groups."""
    a, b = stat(groups)
    diffs = []
    for _ in range(resamples):
        pick = rng.integers(0, len(groups), len(groups))
        x, y = stat([groups[i] for i in pick])
        diffs.append(x - y)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {names[0]: a, names[1]: b, "difference": a - b, "low": float(lo), "high": float(hi)}


def keypoint_norm(gt: np.ndarray, existence: np.ndarray) -> np.ndarray:
    """Per hand, the longer side of the labelled (in-frame) keypoints' extent grown to 3:4; NaN under two."""
    out = np.full(len(gt), np.nan)
    for i in range(len(gt)):
        p = gt[i][existence[i]]
        if len(p) >= 2:
            out[i] = expanded_size(np.concatenate([p.min(0), p.max(0)]))
    return out


def hint_summary(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], rng: np.random.Generator) -> dict[str, Any]:
    a = {k: np.concatenate([u[k] for _, u in units]) for k in ("member", "split", "existence", "occlusion", "gt",
                                                               "hamer_kp2d", "wilor_kp2d", "hamer_dist", "wilor_dist")}
    norm = keypoint_norm(a["gt"], a["existence"])
    usable = np.isfinite(norm)
    a["existence"] = a["existence"] & usable[:, None]
    for k in MODELS:
        a[f"{k}_dist_bbox"] = a[f"{k}_dist"]
        a[f"{k}_dist"] = np.linalg.norm(a[f"{k}_kp2d"] - a["gt"], axis=-1) / np.where(usable, norm, 1.0)[:, None]
    out: dict[str, Any] = {"normaliser": "keypoint extent (user, 2026-10-10)", "hands_without_two_keypoints": int((~usable).sum())}
    for split in HINT_SPLITS:
        sel = a["split"] == split
        subsets = {"all": a["existence"][sel], "visible": a["existence"][sel] & ~a["occlusion"][sel],
                   "occluded": a["existence"][sel] & a["occlusion"][sel]}
        table = {k: {name: {f"{t:g}": pck(a[f"{k}_dist"][sel], mask, t) for t in PCK_THRESHOLDS}
                     for name, mask in subsets.items()} for k in MODELS}
        members = a["member"][sel]
        groups = [np.nonzero(members == m)[0] for m in np.unique(members)]
        ex = a["existence"][sel]
        hd, wd = a["hamer_dist"][sel], a["wilor_dist"][sel]

        def stat(gs: list[np.ndarray]) -> tuple[float, float]:
            idx = np.concatenate(gs)
            return pck(hd[idx], ex[idx], 0.05), pck(wd[idx], ex[idx], 0.05)

        table_bbox = {k: {f"{t:g}": pck(a[f"{k}_dist_bbox"][sel], subsets["all"], t) for t in PCK_THRESHOLDS} for k in MODELS}
        out[split] = {"hands": int(sel.sum()), "pck": table, "pck_all_bbox_normaliser": table_bbox,
                      "pck05_all_bootstrap": paired_bootstrap(groups, stat, rng, DECISION["bootstrap"]["resamples"])}
    return out


def accel_errors(meta: dict[str, Any], a: dict[str, np.ndarray], model: str) -> np.ndarray:
    """Per frame triple, |predicted − true| acceleration of root-relative joints in world axes (mm/frame²)."""
    out = []
    gt_rel = a["gt_world"] - a["gt_world"][:, :1]
    pred = a[f"{model}_rel_world"]
    for side in (True, False):
        idx = np.nonzero(a["right"].astype(bool) == side)[0]
        by_t = {int(a["t"][i]): i for i in idx}
        for t, i in by_t.items():
            if t - 1 in by_t and t + 1 in by_t:
                p, n = by_t[t - 1], by_t[t + 1]
                acc_p = pred[n] - 2 * pred[i] + pred[p]
                acc_g = gt_rel[n] - 2 * gt_rel[i] + gt_rel[p]
                out.append(float(np.linalg.norm(acc_p - acc_g, axis=-1).mean() * 1000))
    return np.asarray(out)


def hot3d_summary(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], rng: np.random.Generator) -> dict[str, Any]:
    rows = [(m, a) for m, a in units if m["hands"]]
    out: dict[str, Any] = {"clips": len(rows), "hands": int(sum(m["hands"] for m, _ in rows)),
                           "participants": sorted({m["participant"] for m, _ in rows})}
    for metric in ("mpjpe_ra_mm", "pa_mpjpe_mm", "mpvpe_ra_mm", "err2d_box"):
        out[metric] = {k: float(np.mean(np.concatenate([a[f"{k}_{metric}"] for _, a in rows]))) for k in MODELS}
    # Global orientation (reported, not a contest): the wrist→middle-knuckle direction in world axes.
    gt_rel = np.concatenate([a["gt_world"] - a["gt_world"][:, :1] for _, a in rows])
    out["orientation_error_deg_median"] = {}
    for k in MODELS:
        pred = np.concatenate([a[f"{k}_rel_world"] for _, a in rows])
        cos = np.sum(pred[:, 9] * gt_rel[:, 9], -1) / (np.linalg.norm(pred[:, 9], axis=-1) * np.linalg.norm(gt_rel[:, 9], axis=-1))
        out["orientation_error_deg_median"][k] = float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))
    accel = [{k: accel_errors(m, a, k) for k in MODELS} for m, a in rows]
    out["accel_error_mm"] = {k: float(np.mean(np.concatenate([c[k] for c in accel]))) for k in MODELS}
    resamples = DECISION["bootstrap"]["resamples"]

    def pooled(metric: str) -> Any:
        return lambda gs: tuple(float(np.mean(np.concatenate([g[f"{k}_{metric}"] for g in gs]))) for k in MODELS)

    out["pa_mpjpe_bootstrap"] = paired_bootstrap([a for _, a in rows], pooled("pa_mpjpe_mm"), rng, resamples)
    out["accel_bootstrap"] = paired_bootstrap(accel, lambda gs: tuple(float(np.mean(np.concatenate([g[k] for g in gs])))
                                                                       for k in MODELS), rng, resamples)
    return out


def visor_summary(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], rng: np.random.Generator) -> dict[str, Any]:
    rows = [(m, a) for m, a in units if m["hands"]]
    out: dict[str, Any] = {"items": len(rows), "hand_frames": int(sum(m["hands"] for m, _ in rows))}
    for k in MODELS:
        ious = np.concatenate([a[f"{k}_iou_hand"] for _, a in rows])
        jitter, flips, pairs = [], 0, 0
        for _, a in rows:
            for side in (True, False):
                idx = np.nonzero(a["right"].astype(bool) == side)[0]
                by_t = {int(a["t"][i]): i for i in idx}
                for t, i in by_t.items():
                    if t - 1 in by_t:
                        pairs += 1
                        r0 = mano_np.rodrigues(a[f"{k}_rotvec"][by_t[t - 1], 0])
                        r1 = mano_np.rodrigues(a[f"{k}_rotvec"][i, 0])
                        angle = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))
                        flips += angle > 45
                    if t - 1 in by_t and t + 1 in by_t:
                        kp = a[f"{k}_kp2d"]
                        acc = kp[by_t[t + 1]] - 2 * kp[i] + kp[by_t[t - 1]]
                        jitter.append(float(np.linalg.norm(acc, axis=-1).mean()) / float(a[f"{k}_box_size"][i] / RESCALE))
        out[k] = {"iou_hand_median": float(np.nanmedian(ious)), "iou_hand_mean": float(np.nanmean(ious)),
                  "iou_hand_share_0.6": float(np.nanmean(ious >= 0.6)),
                  "iou_mask_median": float(np.nanmedian(np.concatenate([a[f"{k}_iou_mask"] for _, a in rows]))),
                  "forearm_share_mean": float(np.mean(np.concatenate([a[f"{k}_forearm_share"] for _, a in rows]))),
                  "kp2d_accel_box_median": float(np.median(jitter)) if jitter else None,
                  "orientation_flips_per_1000_pairs": 1000.0 * flips / max(pairs, 1)}
    out["iou_bootstrap"] = paired_bootstrap(
        [a for _, a in rows], lambda gs: tuple(float(np.nanmedian(np.concatenate([g[f"{k}_iou_hand"] for g in gs]))) for k in MODELS),
        rng, DECISION["bootstrap"]["resamples"])
    return out


def decide(summary: dict[str, Any], speed: dict[str, Any], code: dict[str, Any] | None) -> dict[str, Any]:
    rule_anchor = DECISION["anchor"]
    gaps = {split: summary["hint"][split]["pck"]["hamer"]["all"]["0.05"] - published
            for split, published in rule_anchor["published"].items()}
    holds = all(abs(g) <= rule_anchor["max_gap"] for g in gaps.values())
    contests = {
        "hint_visor_pck05_all": summary["hint"]["TEST_epick_img"]["pck05_all_bootstrap"],
        "hint_newdays_pck05_all": summary["hint"]["TEST_newdays_img"]["pck05_all_bootstrap"],
        "hot3d_pa_mpjpe_mm": summary["hot3d"]["pa_mpjpe_bootstrap"],
        "hot3d_accel_error_mm": summary["hot3d"]["accel_bootstrap"],
        "visor_iou_hand_median": summary["visor"]["iou_bootstrap"],
    }
    wins = {k: 0 for k in MODELS}
    verdicts = {}
    for name, b in contests.items():
        lower = name in DECISION["lower_is_better"]
        if b["low"] > 0:
            winner = "wilor" if lower else "hamer"
        elif b["high"] < 0:
            winner = "hamer" if lower else "wilor"
        else:
            winner = None
        if winner:
            wins[winner] += 1
        verdicts[name] = {**b, "winner": winner}
    faster = min(MODELS, key=lambda k: speed[k]["batch1_ms_per_hand"])
    chosen = max(MODELS, key=lambda k: wins[k]) if wins["hamer"] != wins["wilor"] else faster
    out: dict[str, Any] = {"anchor": {"gap_to_paper": gaps, "max_gap": rule_anchor["max_gap"], "holds": holds},
                           "contests": verdicts, "wins": wins, "faster_at_batch1": faster,
                           "estimator": chosen if holds else None,
                           "by": "wins" if wins["hamer"] != wins["wilor"] else "tie, speed"}
    if code is not None and out["estimator"]:
        rule = DECISION["coding"]
        screen = rule["h1_screen"]["max_share"] * rule["h1_screen"]["svt_crf62_hand_kbps"]
        choices = code["choices"][out["estimator"]]
        named = None
        for budget in rule["budgets_ms"]:
            c = choices.get(f"{budget:g}")
            if c is not None and c["visor_kbps"] <= screen:
                named = {"budget_ms": budget, **c, "passes_h1_screen": True}
                break
        if named is None and choices.get("100") is not None:
            named = {"budget_ms": 100.0, **choices["100"], "passes_h1_screen": False}
        out["coding"] = {"screen_kbps": screen, "per_budget": choices, "for_h3": named,
                         "visor_iou": code["visor_iou"][out["estimator"]], "baselines": code["baselines"]}
    return out


def command_report(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(DECISION["bootstrap"]["seed"])
    summary = {"hint": hint_summary(load_units(args.hint, "hint"), rng),
               "hot3d": hot3d_summary(load_units(args.hot3d, "hot3d"), rng),
               "visor": visor_summary(load_units(args.visor, "visor"), rng)}
    speed = None
    for path in args.hint:
        with tarfile.open(path) as tar:
            if "publish/speed.json" in tar.getnames():
                handle = tar.extractfile("publish/speed.json")
                assert handle is not None
                speed = json.loads(handle.read())
    if speed is None:
        raise SystemExit("no speed.json in the HInt archives")
    code = json.loads(Path(args.code).read_text()) if args.code else None
    report = {"inputs": {"hint": {p: file_sha256(Path(p)) for p in args.hint}, "hot3d": {p: file_sha256(Path(p)) for p in args.hot3d},
                         "visor": {p: file_sha256(Path(p)) for p in args.visor},
                         "code": {args.code: file_sha256(Path(args.code))} if args.code else None},
              "summary": summary, "speed": speed, "decision": decide(summary, speed, code), "rule": DECISION}
    write_json(Path(args.out) / "h2-report.json", report)
    print(json.dumps(report["decision"], indent=1, default=str)[:4000])
    return 0


# ----------------------------------------------------------------- hybrid (H2b)

def hybrid_params(mano: mano_np.Mano, a: dict[str, np.ndarray], left: np.ndarray,
                  transl_key: str = "transl") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WiLoR's joint rotations and shape with HaMeR's global orientation, translated so that the wrist
    stays where HaMeR put it."""
    rv = a["wilor_rotvec"].astype(float).copy()
    rv[:, 0] = a["hamer_rotvec"][:, 0]
    shift = mano.rest_root(a["hamer_betas"]) - mano.rest_root(a["wilor_betas"])
    shift[left] = shift[left] @ mano_np.MIRROR
    return rv, a["wilor_betas"].astype(float), a[f"hamer_{transl_key}"].astype(float) + shift


def model_keypoints(mano: mano_np.Mano, rv: np.ndarray, betas: np.ndarray, cam_crop: np.ndarray, box_size: np.ndarray,
                    box_center: np.ndarray, right: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """Image keypoints as the regressors project them: joints in the (unmirrored) model space, the crop
    camera (s, tx, ty) at the models' focal length, then the box (``hamer.models.hamer.forward_step``)."""
    f = HYBRID_DECISION["model_focal"] / HYBRID_DECISION["model_image_size"]
    m = 2 * right.astype(float) - 1
    joints = mano.joints(mano_np.rodrigues(rv), betas, np.zeros((len(rv), 3)), np.zeros(len(rv), bool))
    t = np.stack([cam_crop[:, 1] * m, cam_crop[:, 2], 2 * f / (cam_crop[:, 0] + 1e-9)], -1)
    p = joints + t[:, None] + (0.0 if offset is None else offset[:, None])
    kp = f * p[..., :2] / p[..., 2:3]
    kp[..., 0] *= m[:, None]
    return kp * box_size[:, None, None] + box_center[:, None, :]


def hybrid_hint(mano: mano_np.Mano, units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], rng: np.random.Generator,
                checks: dict[str, float]) -> dict[str, Any]:
    keys = ("member", "split", "existence", "gt", "right", "hamer_kp2d", "wilor_kp2d") + tuple(
        f"{k}_{f}" for k in MODELS for f in ("rotvec", "betas", "cam_crop", "box_size", "box_center"))
    a = {k: np.concatenate([u[k] for _, u in units]) for k in keys}
    right = a["right"].astype(bool)
    kp = {}
    for k in MODELS:
        kp[k] = model_keypoints(mano, a[f"{k}_rotvec"], a[f"{k}_betas"], a[f"{k}_cam_crop"], a[f"{k}_box_size"],
                                a[f"{k}_box_center"], right)
    checks["hint_reproduction_px"] = float(max(np.abs(kp[k] - a[f"{k}_kp2d"]).max() for k in MODELS))
    # In model space every hand is a right hand (no mirror for the wrist shift); the camera is HaMeR's, and
    # the shift keeps the wrist where HaMeR's own shape put it.
    rv, betas, shift = hybrid_params(mano, {**a, "hamer_transl": np.zeros((len(right), 3))}, np.zeros(len(right), bool))
    kp["hybrid"] = model_keypoints(mano, rv, betas, a["hamer_cam_crop"], a["hamer_box_size"], a["hamer_box_center"], right,
                                   offset=shift)
    wrist_h = mano.joints(mano_np.rodrigues(a["hamer_rotvec"]), a["hamer_betas"], np.zeros((len(right), 3)), np.zeros(len(right), bool))
    wrist_x = mano.joints(mano_np.rodrigues(rv), betas, shift, np.zeros(len(right), bool))
    checks["hint_hybrid_wrist_vs_hamer_m"] = float(np.abs(wrist_x[:, 0] - wrist_h[:, 0]).max())
    norm = keypoint_norm(a["gt"], a["existence"])
    usable = np.isfinite(norm)
    ex = a["existence"] & usable[:, None]
    dist = {k: np.linalg.norm(v - a["gt"], axis=-1) / np.where(usable, norm, 1.0)[:, None] for k, v in kp.items()}
    out: dict[str, Any] = {}
    for split in HINT_SPLITS:
        sel = a["split"] == split
        if not sel.any():
            continue  # a smoke's subset may hold one split only
        members = a["member"][sel]
        groups = [np.nonzero(members == m)[0] for m in np.unique(members)]
        d = {k: v[sel] for k, v in dist.items()}
        e = ex[sel]
        out[split] = {"pck05_all": {k: pck(d[k], e, 0.05) for k in d},
                      "hybrid_minus_wilor": paired_bootstrap(
                          groups, lambda gs: (pck(d["hybrid"][np.concatenate(gs)], e[np.concatenate(gs)], 0.05),
                                              pck(d["wilor"][np.concatenate(gs)], e[np.concatenate(gs)], 0.05)),
                          rng, HYBRID_DECISION["bootstrap"]["resamples"], ("hybrid", "wilor"))}
    return out


def hybrid_hot3d(mano: mano_np.Mano, units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], rng: np.random.Generator,
                 checks: dict[str, float]) -> dict[str, Any]:
    from hand_tracking_toolkit import camera

    rows = [(m, a) for m, a in units if m["hands"]]
    per_clip = []
    reproduction = 0.0
    wrist = 0.0
    for meta, a in rows:
        left = ~a["right"].astype(bool)
        cam = camera.from_json(meta["camera_214"])
        joints = {}
        for k in MODELS:
            joints[k] = mano.joints(mano_np.rodrigues(a[f"{k}_rotvec"]), a[f"{k}_betas"], a[f"{k}_transl"], left)
            rel = np.einsum("nab,njb->nja", a["R_world_214"], joints[k] - joints[k][:, :1])
            reproduction = max(reproduction, float(np.abs(rel - a[f"{k}_rel_world"]).max()))
        rv, betas, transl = hybrid_params(mano, a, left)
        joints["hybrid"] = mano.joints(mano_np.rodrigues(rv), betas, transl, left)
        wrist = max(wrist, float(np.abs(joints["hybrid"][:, 0] - joints["hamer"][:, 0]).max()))
        gt = a["gt_214"]
        gt_2d = cam.eye_to_window(gt.reshape(-1, 3)).reshape(-1, 21, 2)
        clip: dict[str, Any] = {}
        for k, p in joints.items():
            d = p[:, 9] - p[:, 0]
            g = gt[:, 9] - gt[:, 0]
            cos = (d * g).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(g, axis=1))
            p2d = cam.eye_to_window(p.reshape(-1, 3)).reshape(-1, 21, 2)
            rel_world = np.einsum("nab,njb->nja", a["R_world_214"], p - p[:, :1])
            clip[k] = {"mpjpe_ra_mm": np.linalg.norm((p - p[:, :1]) - (gt - gt[:, :1]), axis=-1).mean(1) * 1000,
                       "pa_mpjpe_mm": np.array([procrustes_error(p[i], gt[i]) for i in range(len(p))]) * 1000,
                       "orientation_deg": np.degrees(np.arccos(np.clip(cos, -1, 1))),
                       "wrist_mm": np.linalg.norm(p[:, 0] - gt[:, 0], axis=-1) * 1000,
                       "err2d_px": np.minimum(np.nan_to_num(np.linalg.norm(p2d - gt_2d, axis=-1), nan=cam.width,
                                                            posinf=cam.width), cam.width).mean(1),
                       "accel_mm": accel_errors(meta, {"t": a["t"], "right": a["right"], "gt_world": a["gt_world"],
                                                       f"{k}_rel_world": rel_world}, k)}
        per_clip.append(clip)
    checks["hot3d_reproduction_m"] = reproduction
    checks["hybrid_wrist_vs_hamer_m"] = wrist
    variants = ("hamer", "wilor", "hybrid")
    out: dict[str, Any] = {"clips": len(rows), "hands": int(sum(m["hands"] for m, _ in rows))}
    for metric in ("mpjpe_ra_mm", "pa_mpjpe_mm", "wrist_mm", "err2d_px", "accel_mm"):
        out[metric] = {k: float(np.mean(np.concatenate([c[k][metric] for c in per_clip]))) for k in variants}
    out["orientation_deg_median"] = {k: float(np.median(np.concatenate([c[k]["orientation_deg"] for c in per_clip])))
                                     for k in variants}
    for other in MODELS:
        out[f"hybrid_minus_{other}_mpjpe_ra"] = paired_bootstrap(
            per_clip, lambda gs, o=other: tuple(float(np.mean(np.concatenate([g[v]["mpjpe_ra_mm"] for g in gs])))
                                               for v in ("hybrid", o)), rng, HYBRID_DECISION["bootstrap"]["resamples"],
            ("hybrid", other))
    return out


def _visor_item_ious(task: tuple[dict[str, Any], dict[str, np.ndarray], str]) -> dict[str, np.ndarray]:
    from experiments.visor.h1 import iou, rasterize

    meta, a, mano_path = task
    mano = mano_np.Mano.load(mano_path)
    n = int(meta["hands"])
    left = ~a["right"].astype(bool)
    shape = tuple(meta["image_shape"])
    params = {k: (a[f"{k}_rotvec"], a[f"{k}_betas"], a[f"{k}_transl"]) for k in MODELS}
    params["hybrid"] = hybrid_params(mano, a, left)
    splits = {}
    for k in MODELS:
        offsets = np.concatenate([[0], np.cumsum(a[f"{k}_hand_bits_len"])])
        splits[k] = [unpack_window(a[f"{k}_hand_window"][i], a[f"{k}_hand_bits"][offsets[i]:offsets[i + 1]].tobytes())
                     for i in range(n)]
    out = {f"{v}_vs_{s}": np.full(n, np.nan) for v in params for s in MODELS}
    for i in range(n):
        for v, (rv, betas, transl) in params.items():
            verts = mano.mesh(mano_np.rodrigues(rv[i:i + 1]), betas[i], transl[i:i + 1], left=bool(left[i]))[0]
            silhouette = rasterize(pinhole(verts, meta["focal"], tuple(meta["principal"])), mano.faces, shape)
            for s in MODELS:
                window = a[f"{s}_hand_window"][i]
                full = np.zeros(shape, bool)
                full[window[1]:window[3], window[0]:window[2]] = splits[s][i]
                value = iou(silhouette, full)
                out[f"{v}_vs_{s}"][i] = np.nan if value is None else value
    return out


def hybrid_visor(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], mano_path: str, workers: int,
                 rng: np.random.Generator, checks: dict[str, float]) -> dict[str, Any]:
    rows = [(m, a) for m, a in units if m["hands"]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        items = list(pool.map(_visor_item_ious, [(m, a, mano_path) for m, a in rows]))
    split = HYBRID_DECISION["visor_split"]
    saved = np.concatenate([a["wilor_iou_hand"] for _, a in rows])
    mine = np.concatenate([it["wilor_vs_wilor"] for it in items])
    checks["visor_wilor_iou_median_vs_job"] = abs(float(np.nanmedian(mine)) - float(np.nanmedian(saved)))
    out: dict[str, Any] = {"items": len(rows)}
    for s in MODELS:
        out[f"iou_median_vs_{s}_split"] = {v: float(np.nanmedian(np.concatenate([it[f"{v}_vs_{s}"] for it in items])))
                                           for v in ("hamer", "wilor", "hybrid")}
    out["hybrid_minus_wilor"] = paired_bootstrap(
        items, lambda gs: tuple(float(np.nanmedian(np.concatenate([g[f"{v}_vs_{split}"] for g in gs])))
                                for v in ("hybrid", "wilor")), rng, HYBRID_DECISION["bootstrap"]["resamples"], ("hybrid", "wilor"))
    return out


def command_hybrid(args: argparse.Namespace) -> int:
    began = time.time()
    rng = np.random.default_rng(HYBRID_DECISION["bootstrap"]["seed"])
    mano = mano_np.Mano.load(args.mano_right)
    limit = int(args.units)
    units = {part: load_units(getattr(args, part), part) for part in ("hint", "hot3d", "visor")}
    for part, rows in units.items():
        if not rows or not all(m["npz_hash_ok"] for m, _ in rows):
            raise SystemExit(f"{part}: missing or corrupt units")
        if limit:
            units[part] = rows[:limit]
    checks: dict[str, float] = {}
    workers = max(1, int(args.workers or os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    summary = {"hint": hybrid_hint(mano, units["hint"], rng, checks)}
    progress(1)
    summary["hot3d"] = hybrid_hot3d(mano, units["hot3d"], rng, checks)
    progress(2)
    summary["visor"] = hybrid_visor(units["visor"], args.mano_right, workers, rng, checks)
    progress(3)
    rule = HYBRID_DECISION
    a = all(summary["hot3d"][f"hybrid_minus_{o}_mpjpe_ra"]["high"] < 0 for o in MODELS)
    b = all(s in summary["hint"] and summary["hint"][s]["hybrid_minus_wilor"]["low"] > -rule["hint_margin_pck"]
            for s in HINT_SPLITS)
    c = summary["visor"]["hybrid_minus_wilor"]["low"] > -rule["visor_margin_iou"]
    out_dir = Path(args.out) if args.out else stage_dir()
    write_json(out_dir / "hybrid.json", {
        "inputs": {part: [file_sha256(Path(p)) for p in getattr(args, part)] for part in ("hint", "hot3d", "visor")}
        | {"mano_right_sha256": file_sha256(Path(args.mano_right))},
        "units": {part: len(rows) for part, rows in units.items()}, "rule": rule, "checks": checks, "summary": summary,
        "decision": {"a_hot3d_better_than_both": a, "b_hint_within_margin": b, "c_visor_within_margin": c,
                     "adopt_hybrid": bool(a and b and c)},
        "seconds": round(time.time() - began, 1),
    })
    return 0


# ----------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.visor.h2", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--part", choices=("hint", "hot3d", "visor"), required=True)
    run.add_argument("--hamer-checkpoint", required=True)
    run.add_argument("--hamer-config", required=True)
    run.add_argument("--hamer-mean-params", required=True)
    run.add_argument("--wilor-checkpoint", required=True)
    run.add_argument("--wilor-detector", required=True)
    run.add_argument("--mano-left", required=True)
    run.add_argument("--mano-right", required=True)
    run.add_argument("--hint-zip")
    run.add_argument("--limit", default="0", help="HInt hands per split (0: all)")
    run.add_argument("--speed", action="store_true")
    run.add_argument("--hot3d-clip", action="append", default=[])
    run.add_argument("--clips", default="0", help="HOT3D clips to run (0: all given)")
    run.add_argument("--frames", default="150")
    run.add_argument("--eval-set")
    run.add_argument("--archive")
    run.add_argument("--masks")
    run.add_argument("--mask-record")
    run.add_argument("--h1-boxes")
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--items", default="all")
    commands.add_parser("validate")
    code = commands.add_parser("code")
    code.add_argument("--hot3d", action="append", required=True)
    code.add_argument("--visor", action="append", required=True)
    code.add_argument("--mano-right", required=True)
    code.add_argument("--workers", default="")
    code.add_argument("--units", default="0", help="first N units per dataset (0: all); for a smoke")
    code.add_argument("--out", default="", help="default: the stage directory")
    hybrid = commands.add_parser("hybrid")
    for part in ("hint", "hot3d", "visor"):
        hybrid.add_argument(f"--{part}", action="append", required=True)
    hybrid.add_argument("--mano-right", required=True)
    hybrid.add_argument("--workers", default="")
    hybrid.add_argument("--units", default="0", help="first N units per part (0: all); for a smoke")
    hybrid.add_argument("--out", default="", help="default: the stage directory")
    report = commands.add_parser("report")
    report.add_argument("--hint", action="append", required=True)
    report.add_argument("--hot3d", action="append", required=True)
    report.add_argument("--visor", action="append", required=True)
    report.add_argument("--code")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "validate":
        return command_validate(args)
    if args.command == "code":
        return command_code(args)
    if args.command == "hybrid":
        return command_hybrid(args)
    return command_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
