"""PLAN step H3e: a pretrained pose-conditioned generator renders the hand from the coded keypoints
(docs/experiments.md, 2026-10-12).

    python -m experiments.visor.h3e prepare --eval-set JSON --archive DIR --video V.MP4 ... --items ID,ID --frames N \\
        --poses TAR ... --pose-code JSON --mano-left PKL --mano-right PKL
    python -m experiments.visor.h3e judge --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video V.MP4 ... \\
        --items ID,ID --frames N --generated DIR --order NAME --image-ckpt PTH --video-ckpt PTH --lpips-backbone PTH \\
        --inception PTH --poses TAR ... --pose-code JSON --mano-left PKL --mano-right PKL --wilor-checkpoint ...
    python -m experiments.visor.h3e validate
    python -m experiments.visor.h3e report --judge TAR ... --h3c2 TAR ... --h1-report JSON --out DIR

``prepare`` (CPU) writes each window's generator input: frame 0 on JointControlVideo's 832×480 canvas, H2's coded
WiLoR joints (100 ms choice, each track with its median shape, decoded as H3b) at window frames 0, 3, …, 240 in the
camera frame, and the canvas intrinsics. `experiments.visor.h3e_generate` (its own environment) renders the window.
``judge`` (GPU) composites the generated frames inside R over DCVC-UF off the shelf at H3c's nine points (V1: the
hand inside its dilated coded silhouette; V2: all of R), bills each by DCVC-UF's own split (H3c), and judges pose
(WiLoR), realism (Inception features) and flicker. ``report`` applies `DECISION` against H3c2's pixel envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import RESAMPLES, bootstrap_mean, dis_flow, flow_box, hand_kbps, rgb_memmap, seed_of, warp
from experiments.visor.h3b import (CROP_GROW, INCEPTION_SIZE, KID_RESAMPLES, flicker, inception_features, kid_from, kid_kernels,
                                   load_inception, pareto_front, rate_on_front, scaled_size, square_crop)
from experiments.visor.h3c import (PILOT, POSE_GATE_BUDGET, QPS, SCALES, code_dcvc, combo_at, decode_windows, downscale_mask,
                                   downscale_planes, load_window, pose_errors)

#: The decision rule, fixed before any fleet run (docs/experiments.md, H3e entry).
DECISION: dict[str, Any] = {
    "pose_error": "H3c2's: mean over judged hands of the mean 2D joint distance to WiLoR on the source, over the box",
    "judge_frames": "window frames both generated (multiples of 3) and judged by H3c2 (multiples of 5), after frame 0",
    "pixel_envelope": "SVT-AV1 and DCVC-UF off the shelf (H3c2's points) on the same frames: their achievable hull "
                      "(upper concave hull in quality, i.e. lower convex hull in pose error, over rate)",
    "rate": "1.93 kbps keypoints + DCVC-UF's bits on R by its own split: V1 frame 0 on R and later frames on R outside "
            "the pasted masks; V2 frame 0 on R",
    "saving": "V1: -BD-rate of its nine points against the hull (PCHIP, log rate over the overlapping pose range); "
              "without overlap, better/worse at every rate and its best point's 1 - rate / the hull's rate at equal "
              "pose error. V2 (one point): 1 - rate / the hull's rate at equal pose error",
    "decides": "v1",
    "passes_ci_low_above": 0.5,  # V1's mean saving's 95% bootstrap CI by item above 50%: build the route
    "fails_mean_at_most": 0.5,  # V1's mean saving at or below 50%: no generator now, H3d next
}

STRIDE = 3  # window frames per generated frame
GENERATED = 81  # JointControlVideo's clip length (4k+1)
JUDGE_EVERY = 15  # generated (3) and judged by H3c2 (5)
CANVAS = (832, 480)  # JointControlVideo's width, height
DILATE = 0.10  # V1's paste mask: the coded silhouette dilated by this share of the hand's box
FEATHER = 0.02  # and feathered inside by a Gaussian of this share of the box
HFOV_DEG = 90.0  # the generator's camera: H2's WiLoR focal (5000/256 of the frame) is near-orthographic
SEED = "pointstream-h3e"


# ----------------------------------------------------------------- geometry (shared by prepare and judge)


def canvas_geometry(width: int, height: int) -> dict[str, Any]:
    """JointControlVideo's loader: scale so the frame covers the canvas, then centre-crop (integer offsets here)."""
    scale = max(CANVAS[0] / width, CANVAS[1] / height)
    size = (round(width * scale), round(height * scale))
    offset = ((size[0] - CANVAS[0]) // 2, (size[1] - CANVAS[1]) // 2)
    return {"scale": scale, "size": list(size), "offset": list(offset), "source": [width, height]}


def generator_focal(width: int) -> float:
    """A pinhole of `HFOV_DEG` across the frame's width (960 px at 1920)."""
    return 0.5 * width / math.tan(math.radians(HFOV_DEG) / 2)


def generator_joints(joints: np.ndarray, focal_w: float, centre: tuple[float, float], focal: float) -> np.ndarray:
    """H2's camera-frame joints (WiLoR's focal ``focal_w``, hands about 20 m away) for a camera of ``focal``: the
    same 2D projection, each hand's root at the depth that camera implies (``focal / focal_w`` of WiLoR's), and
    MANO's metric depth offsets between the joints. Zero rows (absent hands) stay zero."""
    from experiments.visor.h2 import pinhole

    out = np.zeros_like(joints, dtype=np.float32)
    for k in range(joints.shape[0]):
        for side in (0, 21):
            hand = joints[k, side:side + 21].astype(np.float64)
            if not hand.any():
                continue
            uv = pinhole(hand, focal_w, centre)
            z = hand[0, 2] * focal / focal_w + (hand[:, 2] - hand[0, 2])
            out[k, side:side + 21, 0] = (uv[:, 0] - centre[0]) * z / focal
            out[k, side:side + 21, 1] = (uv[:, 1] - centre[1]) * z / focal
            out[k, side:side + 21, 2] = z
    return out


def canvas_intrinsics(focal: float, centre: tuple[float, float], geometry: dict[str, Any]) -> np.ndarray:
    s, (ox, oy) = geometry["scale"], geometry["offset"]
    return np.array([[focal * s, 0, centre[0] * s - ox], [0, focal * s, centre[1] * s - oy], [0, 0, 1]], np.float32)


def to_canvas(image: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    import cv2

    w, h = geometry["size"]
    ox, oy = geometry["offset"]
    return np.ascontiguousarray(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)[oy:oy + CANVAS[1], ox:ox + CANVAS[0]])


def from_canvas(image: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    """A generated canvas frame back to the window's resolution: edges replicated over the cropped margins, Lanczos."""
    import cv2

    w, h = geometry["size"]
    ox, oy = geometry["offset"]
    padded = cv2.copyMakeBorder(image, oy, h - CANVAS[1] - oy, ox, w - CANVAS[0] - ox, cv2.BORDER_REPLICATE)
    return np.asarray(cv2.resize(padded, tuple(geometry["source"]), interpolation=cv2.INTER_LANCZOS4))


def generated_times(frames: int) -> list[int]:
    """The window frame each generated frame stands for (the last repeats the window's last frame)."""
    return [min(STRIDE * k, frames - 1) for k in range(GENERATED)]


def joints_at(times: list[int], coded_joints: np.ndarray, t: np.ndarray, right: np.ndarray) -> np.ndarray:
    """(len(times), 42, 3): per time, the coded joints of the left hand (0–20) and right hand (21–41), zero if absent."""
    out = np.zeros((len(times), 42, 3), np.float32)
    for k, frame in enumerate(times):
        for r in np.nonzero(t == frame)[0]:
            side = 21 if right[r] else 0
            if not out[k, side:side + 21].any():
                out[k, side:side + 21] = coded_joints[r]
    return out


def pose_setup(args: argparse.Namespace, scratch: Path) -> None:
    """H2's WiLoR units, H2's 100 ms coding and MANO, as `experiments.visor.h3c` sets them up for H3c2."""
    from experiments.visor import mano_np
    from experiments.visor import pose_coding as pc

    code = json.loads(Path(args.pose_code).read_text())
    args.pose_kbps = float(code["choices"]["wilor"][POSE_GATE_BUDGET]["visor_kbps"])
    # Only WiLoR's coded joints are read; H3b's helper also codes HaMeR's rows, with the same combination.
    args.combos = {m: combo_at(code, "wilor", POSE_GATE_BUDGET) for m in ("wilor", "hamer")}
    args.mano = mano_np.Mano.load(args.mano_right)
    args.basis = pc.Basis(args.mano.hands_mean, args.mano.hands_components)
    args.pose_units = {meta["unit"]: (meta, a) for meta, a in load_units(args.poses, "visor")}


def coded_poses(args: argparse.Namespace, item_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from experiments.visor.h3b import item_poses

    meta_p, a = args.pose_units[item_id]
    return meta_p, a, item_poses(meta_p, a, args.mano, args.combos, args.basis)["wilor"]


# ----------------------------------------------------------------- prepare (CPU)


def command_prepare(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import select_items

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    frames = int(args.frames)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    pose_setup(args, scratch)
    if missing := [it["id"] for it in items if it["id"] not in args.pose_units]:
        raise SystemExit(f"no H2 poses for {missing}")
    times = generated_times(frames)
    records = []
    for item, decoded in decode_windows(args, items, frames, scratch):
        width, height = decoded["width"], decoded["height"]
        rgb = rgb_memmap(Path(decoded["path"]), scratch / f"{item['id']}-rgb.npy", 1, width, height, decoded["full_range"], "cpu")
        meta_p, a, wilor = coded_poses(args, item["id"])
        geometry = canvas_geometry(width, height)
        centre = tuple(meta_p["principal"])
        focal = generator_focal(width)
        K = canvas_intrinsics(focal, centre, geometry)
        coded = joints_at(times, wilor["coded_joints"], np.asarray(a["t"]), np.asarray(a["right"], bool))
        joints = generator_joints(coded, float(meta_p["focal"]), centre, focal)
        np.savez(publish / f"{item['id']}.npz", frame0=to_canvas(np.asarray(rgb[0]), geometry), joints=joints, K=K)
        hands = (np.abs(joints).reshape(len(times), 2, -1).sum(-1) > 0)
        record = {"item": item["id"], "frames": frames, "times": times, "geometry": geometry, "K": K.tolist(),
                  "wilor_focal": float(meta_p["focal"]), "generator_focal": focal, "hfov_deg": HFOV_DEG,
                  "principal": list(meta_p["principal"]), "pose_kbps": args.pose_kbps,
                  "coding": args.combos["wilor"], "hands_per_generated_frame": hands.sum(1).tolist(),
                  "generated_frames_with_a_hand": int(hands.any(1).sum()), "pose_unit_hash_ok": bool(meta_p["npz_hash_ok"]),
                  "depth_m": [float(np.median(joints[..., 2][joints[..., 2] > 0]))] if (joints[..., 2] > 0).any() else [],
                  "decode": {k: v for k, v in decoded.items() if k != "path"},
                  "npz_sha256": file_sha256(publish / f"{item['id']}.npz")}
        write_json(publish / f"{item['id']}.json", record)
        records.append(record)
        os.remove(decoded["path"])
        os.remove(scratch / f"{item['id']}-rgb.npy")
        progress(len(records))
    write_json(stage_dir() / "h3e.json", {"command": "prepare", "items": [it["id"] for it in items], "records": records,
                                          "pose_code_sha256": file_sha256(Path(args.pose_code)),
                                          "poses_sha256": [file_sha256(Path(x)) for x in args.poses]})
    return 0


# ----------------------------------------------------------------- judge (GPU)


def silhouette(points: np.ndarray, faces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The projected mesh's filled triangles."""
    import cv2

    mask = np.zeros(shape, np.uint8)
    tris = np.round(np.clip(points, -1e5, 1e5)[faces] * 16).astype(np.int32)
    cv2.fillPoly(mask, list(tris), (1,), lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def paste_masks(args: argparse.Namespace, a: dict[str, np.ndarray], coded: np.ndarray, focal: float, centre: tuple[float, float],
                frames: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per window frame, V1's paste mask (each coded hand's silhouette dilated by `DILATE` of its box) and its
    feathered alpha. The decoder has the coded pose, so neither costs bits."""
    import cv2

    from experiments.visor.h2 import pinhole

    masks = np.zeros((frames, *shape), bool)
    alpha = np.zeros((frames, *shape), np.float32)
    faces = args.mano.faces
    for r, t in enumerate(np.asarray(a["t"])):
        if t >= frames:
            continue
        box = np.asarray(a["box"][r], float)
        side = max(box[2] - box[0], box[3] - box[1])
        sil = silhouette(pinhole(coded[r], focal, centre), faces, shape)
        radius = max(1, int(round(DILATE * side)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        grown = cv2.dilate(sil.astype(np.uint8), kernel).astype(bool)
        sigma = max(0.5, FEATHER * side)
        soft = cv2.GaussianBlur(cv2.erode(grown.astype(np.float32), np.ones((3, 3), np.uint8),
                                          iterations=max(1, int(round(2 * sigma)))), (0, 0), sigma)
        masks[t] |= grown
        alpha[t] = np.maximum(alpha[t], np.where(grown, np.maximum(soft, 0), 0))
    return masks, np.clip(alpha, 0, 1)


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, first: bool) -> None:
    import cv2

    began = time.time()
    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    seconds = frames / float(Fraction(item["fps"]).limit_denominator(1001))
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info = load_window(args, item, frames, decoded)
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    meta_p, a, wilor = coded_poses(args, item["id"])
    focal, centre = float(meta_p["focal"]), tuple(meta_p["principal"])
    tree = Path(args.generated)
    tree = tree / "publish" if (tree / "publish").is_dir() else tree  # an extracted published.tar keeps publish/
    gen_record = json.loads((tree / f"{item['id']}@{args.order}.json").read_text())
    gen_canvas = np.load(tree / f"{item['id']}@{args.order}.npz")["frames"]
    geometry = canvas_geometry(width, height)
    times = generated_times(frames)
    if gen_canvas.shape[0] != len(times):
        raise RuntimeError(f"{item['id']}: {gen_canvas.shape[0]} generated frames, expected {len(times)}")
    generated = {t: from_canvas(gen_canvas[k], geometry) for k, t in enumerate(times) if k == 0 or t != times[k - 1]}
    shown = sorted(t for t in generated if t > 0)
    masks, alpha = paste_masks(args, a, wilor["coded"], focal, centre, frames, (height, width))
    v1_bill = np.stack([regions[t] if t == 0 else regions[t] & ~masks[t] for t in range(frames)])
    v2_bill = np.stack([regions[t] if t == 0 else np.zeros_like(regions[t]) for t in range(frames)])

    rows_at: dict[int, list[int]] = {}
    for r, t in enumerate(np.asarray(a["t"])):
        if t < frames:
            rows_at.setdefault(int(t), []).append(r)
    judge_t = [t for t in shown if t % JUDGE_EVERY == 0 and rows_at.get(t)]
    judge_rows = [r for t in judge_t for r in rows_at[t]]
    feat_t = [t for t in shown if rows_at.get(t)]
    feat_rows = [r for t in feat_t for r in rows_at[t]]

    def composite(t: int, base: np.ndarray, paste: np.ndarray | None, weights: np.ndarray | None) -> np.ndarray:
        """The decode (or generated frame) inside R, the source outside; ``paste`` blended in by ``weights``."""
        out = np.asarray(truth[t]).copy()
        r = regions[t]
        inside = base[r].astype(np.float32)
        if paste is not None and weights is not None:
            w = weights[r][:, None]
            inside = w * paste[r].astype(np.float32) + (1 - w) * inside
        out[r] = np.clip(np.round(inside), 0, 255).astype(np.uint8)
        return out

    variants: list[str] = []
    points: list[dict[str, Any]] = []
    pose_kp: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    flick: list[np.ndarray] = []

    def judge_variant(name: str, frame_of: Any) -> None:
        """WiLoR on the judge frames, Inception on every generated frame with a hand, flicker between generated frames."""
        cache: dict[int, np.ndarray] = {}

        def get(t: int) -> np.ndarray:
            if t not in cache:
                cache[t] = frame_of(t)
            return cache[t]

        items = [nets["wilor"].item(get(int(a["t"][r])), np.asarray(a["box"][r], np.float32), bool(a["right"][r])) for r in judge_rows]
        pose_kp.append(nets["wilor"].forward(items)["kp2d"].astype(np.float32) if items else np.zeros((0, 21, 2), np.float32))
        crops = [square_crop(get(int(a["t"][r])), a["box"][r], CROP_GROW, INCEPTION_SIZE) for r in feat_rows]
        feats.append(inception_features(nets["inception"], crops, device) if crops else np.zeros((0, 2048), np.float16))
        values = np.full(len(shown), np.nan)
        for i, t in enumerate(shown):
            p = t - STRIDE
            if p < 0 or not regions[t].any() or not regions[p].any():
                continue
            fb = flow_box(regions[t], regions[p])
            crop = (slice(fb[1], fb[3]), slice(fb[0], fb[2]))
            truth_t, truth_p = np.asarray(truth[t])[crop], np.asarray(truth[p])[crop]
            flow = dis_flow(cv2.cvtColor(truth_t, cv2.COLOR_RGB2GRAY), cv2.cvtColor(truth_p, cv2.COLOR_RGB2GRAY))
            values[i] = flicker(get(t)[crop], warp(np.ascontiguousarray(get(p)[crop]), flow), truth_t,
                                warp(np.ascontiguousarray(truth_p), flow), regions[t][crop])
        flick.append(values)
        variants.append(name)
        cache.clear()

    # The source (WiLoR's re-run, the reference for every pose error), the generator's own frames (does it follow
    # its input), and V2 (all of R generated; independent of DCVC-UF's point).
    judge_variant("source", lambda t: np.asarray(truth[t]))
    judge_variant("generated", lambda t: generated[t])
    judge_variant("v2", lambda t: composite(t, generated[t], None, None))

    look = judge_t[len(judge_t) // 2] if judge_t else shown[len(shown) // 2]
    ys, xs = np.nonzero(regions[look]) if regions[look].any() else (np.array([0, height - 1]), np.array([0, width - 1]))
    look_box = (slice(max(0, ys.min() - 64), min(height, ys.max() + 65)), slice(max(0, xs.min() - 64), min(width, xs.max() + 65)))
    frames_dir = publish / "frames" / item["id"]
    frames_dir.mkdir(parents=True, exist_ok=True)

    def keep_look(name: str, frame: np.ndarray) -> None:
        cv2.imwrite(str(frames_dir / f"{name.replace('/', '_').replace('@', '-')}-f{look:03d}.png"), cv2.cvtColor(frame[look_box], cv2.COLOR_RGB2BGR))

    keep_look("source", np.asarray(truth[look]))
    keep_look("generated", generated[look])
    keep_look("v2", composite(look, generated[look], None, None))

    for scale in args.scales:
        sw, sh = scaled_size(width, height, scale)
        source = Path(decoded["path"])
        if scale > 1:
            source = work / f"source-{scale}.yuv"
            with source.open("wb") as handle:
                for t in range(frames):
                    handle.write(downscale_planes(np.asarray(planes[t]), width, height, sw, sh).tobytes())
        files = {}
        for name, stack in (("r", regions), ("v1", v1_bill), ("v2", v2_bill)):
            files[name] = work / f"region-{name}-{scale}.npy"
            np.save(files[name], np.stack([downscale_mask(np.asarray(m), sw, sh) for m in stack]))
        for qp in args.qps:
            rec = code_dcvc(args, source, files["r"], (sw, sh), frames, qp, args.video_ckpt, args.video_sha256,
                            work / f"dcvc-{scale}-{qp}", extra_regions={"v1": files["v1"], "v2": files["v2"]})
            if not rec["deterministic"] or not rec["decoder_matches_encoder_intra"]:
                raise RuntimeError(f"{item['id']} scale {scale} QP {qp}: decode is not the encoder's "
                                   f"(first differing frames {rec['decode_passes_differ_at']} of {rec['decode_passes_differ_frames']})")
            rgb = rgb_memmap(Path(rec["decoded"]), work / "dcvc-rgb.npy", frames, sw, sh, full_range, device)

            def decoded_at(t: int, rgb: np.ndarray = rgb, scale: int = scale) -> np.ndarray:
                f = np.asarray(rgb[t])
                return cv2.resize(f, (width, height), interpolation=cv2.INTER_CUBIC) if scale > 1 else f

            name = f"@{qp}/{scale}"
            judge_variant(f"dcvc{name}", lambda t: composite(t, decoded_at(t), None, None))
            judge_variant(f"v1{name}", lambda t: composite(t, decoded_at(t), generated[t], alpha[t]))
            keep_look(f"dcvc{name}", composite(look, decoded_at(look), None, None))
            keep_look(f"v1{name}", composite(look, decoded_at(look), generated[look], alpha[look]))
            bits = rec["extra_region_bits"]
            points.append({"name": name, "qp": qp, "scale": scale, "kbps_r": rec["region_bits"] / seconds / 1000,
                           "kbps_v1_pixels": bits["v1"] / seconds / 1000, "kbps_v2_pixels": bits["v2"] / seconds / 1000,
                           "kbps": rec["bits"] / seconds / 1000, **{k: v for k, v in rec.items() if k not in ("decoded", "extra_region_bits")}})
            shutil.rmtree(work / f"dcvc-{scale}-{qp}")
            os.remove(work / "dcvc-rgb.npy")
        if scale > 1:
            source.unlink()
        for f in files.values():
            f.unlink()

    box_size = np.array([max(a["box"][r][2] - a["box"][r][0], a["box"][r][3] - a["box"][r][1]) for r in judge_rows], float)
    from experiments.visor.h2 import pinhole

    coded_kp = (np.stack([pinhole(wilor["coded_joints"][r], focal, centre) for r in judge_rows]).astype(np.float32)
                if judge_rows else np.zeros((0, 21, 2), np.float32))
    checks: dict[str, Any] = {}
    if judge_rows:
        archived = np.asarray(a["wilor_kp2d"])[judge_rows]
        checks["source_rerun_median_px"] = float(np.median(np.linalg.norm(pose_kp[0] - archived, axis=-1).mean(-1)))
    pasted = np.array([int(masks[t].sum()) for t in range(frames)], np.int64)
    os.remove(decoded["path"])
    shutil.rmtree(work)
    arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64), "pasted_px": pasted,
              "pasted_in_r_px": np.array([int((masks[t] & regions[t]).sum()) for t in range(frames)], np.int64),
              "judge_row": np.array(judge_rows, int), "judge_t": np.array([int(a["t"][r]) for r in judge_rows], int),
              "judge_kp2d": np.stack(pose_kp), "judge_wilor_box_size": box_size, "coded_kp2d": coded_kp,
              "archived_wilor_kp2d": np.asarray(a["wilor_kp2d"])[judge_rows].astype(np.float32) if judge_rows else np.zeros((0, 21, 2), np.float32),
              "feature_row": np.array(feat_rows, int), "features": np.stack(feats), "flicker": np.stack(flick),
              "shown_t": np.array(shown, int)}
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": frames,
            "seconds_coded": seconds, "order": args.order, "variants": variants, "points": points, "look_frame": look,
            "generated": gen_record, "geometry": geometry, "pose_kbps": args.pose_kbps,
            "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "checks": checks,
            "pose_unit_hash_ok": bool(meta_p["npz_hash_ok"]), "seconds": round(time.time() - began, 2)}
    save_unit(publish, "judge", item["id"], arrays, meta)


def command_judge(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import Regressor, mano_dir

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    args.qps = [int(q) for q in args.qps.split(",")]
    args.scales = [int(s) for s in args.scales.split(",")]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "judge")
    todo = [it for it in items if it["id"] not in restored]
    pose_setup(args, scratch)
    windows = decode_windows(args, todo, int(args.frames), scratch)
    if not torch.cuda.is_available():
        raise SystemExit("h3e judge needs a GPU (DCVC-UF)")
    device = "cuda"
    args.mano_dir = mano_dir(args, scratch)
    nets = {"wilor": Regressor("wilor", args), "inception": load_inception(Path(args.inception), device)}
    on_cuda = {"wilor": model_on_cuda(nets["wilor"].model), "inception": next(nets["inception"].parameters()).is_cuda}
    done = len(restored)
    began = time.time()
    for number, (item, decoded) in enumerate(windows):
        process_item(args, item, decoded, nets, device, publish, scratch, first=number == 0)
        done += 1
        progress(done)
    write_json(stage_dir() / "h3e.json", {
        "command": "judge", "device": device_record(torch.cuda.get_device_name(0)), "on_cuda": on_cuda,
        "seconds": round(time.time() - began, 2), "wilor": nets["wilor"].load,
        "inception": {"path": args.inception, "sha256": file_sha256(Path(args.inception))},
        "dcvc": {"image_sha256": args.image_sha256, "video_sha256": args.video_sha256},
        "pose_code_sha256": file_sha256(Path(args.pose_code)), "poses_sha256": [file_sha256(Path(x)) for x in args.poses],
        "settings": {"frames": int(args.frames), "items": args.items, "qps": args.qps, "scales": args.scales, "order": args.order,
                     "stride": STRIDE, "judge_every": JUDGE_EVERY, "dilate": DILATE, "feather": FEATHER, "decision": DECISION},
        "items": [it["id"] for it in items], "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation


def validate_prepare(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    records = result["records"]
    return {
        "every_item_present": bool(records) and len(records) == len(result["items"]),
        "pose_units_verified": all(r["pose_unit_hash_ok"] for r in records),
        "generated_frames": all(len(r["times"]) == GENERATED for r in records),
        "hands_present": all(r["generated_frames_with_a_hand"] > 0 for r in records),
        "plausible_depth": all(r["depth_m"] and 0.1 < r["depth_m"][0] < 3.0 for r in records),
        "jpeg_gate": all(g["holds"] for r in records for g in r["decode"]["jpeg_gate"]),
    }


def validate_judge(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    units = load_units([str(stage / "published.tar")], "judge")
    pts = [p for m, _ in units for p in m["points"]]
    return {
        "every_item_present": bool(units) and len(units) == len(result["items"]) and all(m["npz_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "every_point_coded": all(len(m["points"]) == len(result["settings"]["qps"]) * len(result["settings"]["scales"]) for m, _ in units),
        "every_variant_judged": all(a["judge_kp2d"].shape[0] == len(m["variants"]) and a["features"].shape[0] == len(m["variants"])
                                    for m, a in units),
        "judges_finite": all(np.isfinite(a["judge_kp2d"]).all() and np.isfinite(a["features"].astype(np.float32)).all() for _, a in units),
        "estimate_tracks_actual": all(0.7 < p["estimate_over_actual"] < 1.4 for p in pts),
        "billing_nested": all(p["kbps_v2_pixels"] <= p["kbps_v1_pixels"] + 1e-9 <= p["kbps_r"] + 2e-9 for p in pts),
        # H3c2's tolerance: WiLoR re-run on the source reproduces H2's fit within 1% of the box (median hand).
        "source_rerun_reproduces_h2": all(m["checks"].get("source_rerun_median_px", 0.0)
                                          <= 0.01 * float(np.median(a["judge_wilor_box_size"])) for m, a in units if len(a["judge_row"])),
        "paste_masks_nonempty": all(int(a["pasted_in_r_px"].sum()) > 0 for _, a in units),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "h3e.json").read_text())
    checks = validate_prepare(stage, result) if result["command"] == "prepare" else validate_judge(stage, result)
    checks = {k: bool(v) for k, v in checks.items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


MIN_OVERLAP = 0.001  # pose error (share of the box) both curves must span for a BD-rate


def hull(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The anchor's achievable envelope on (kbps, pose error): the lower convex hull in pose (upper concave hull in
    quality), which switching between its encodes reaches, by rising rate and falling error. The same construction
    as the background's ``experiments.background.g5.envelope`` (G5d, 2026-10-12): a Pareto staircase over several
    anchor configurations (here SVT-AV1 and DCVC-UF at three scales) would understate the anchor."""
    out: list[dict[str, Any]] = []
    for q in sorted(points, key=lambda r: (r["kbps"], r["pose"])):
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            # b on or above the chord a→q in error (on or below it in quality): drop it
            if (a["pose"] - b["pose"]) * (q["kbps"] - a["kbps"]) <= (a["pose"] - q["pose"]) * (b["kbps"] - a["kbps"]):
                out.pop()
            else:
                break
        out.append(q)
    return pareto_front(out, "pose")


def bd_rate_pose(test: list[dict[str, Any]], anchor: list[dict[str, Any]]) -> float | None:
    """BD-rate of ``test`` (its Pareto front) against ``anchor``'s hull on pose error: PCHIP of log rate over the
    error range both span; None when that range is under `MIN_OVERLAP` or either has fewer than two points."""
    from scipy.interpolate import PchipInterpolator

    t, a = pareto_front(test, "pose"), hull(anchor)
    if len(t) < 2 or len(a) < 2:
        return None
    lo, hi = max(t[-1]["pose"], a[-1]["pose"]), min(t[0]["pose"], a[0]["pose"])
    if hi - lo < MIN_OVERLAP:
        return None

    def area(front: list[dict[str, Any]]) -> float:
        pts = sorted(front, key=lambda q: q["pose"])
        f = PchipInterpolator([q["pose"] for q in pts], [math.log(q["kbps"]) for q in pts])
        return float(f.integrate(lo, hi))

    return math.exp((area(t) - area(a)) / (hi - lo)) - 1.0


def saving_against(front: list[dict[str, float]], kbps: float, pose: float) -> tuple[float, str]:
    """H3c2's saving at equal pose error (rate on the anchor's hull, linear in log rate), with its bounds beyond
    the hull's ends."""
    need = rate_on_front(front, "pose", pose)
    if need is not None:
        return 1 - kbps / need, "on the front"
    if pose < min(q["pose"] for q in front):
        return 1 - kbps / max(q["kbps"] for q in front), "better than every pixel point (lower bound)"
    return 1 - kbps / min(q["kbps"] for q in front), "pixels' cheapest point already better (upper bound)"


def pixel_envelope(meta: dict[str, Any], a: dict[str, np.ndarray], h1: dict[str, Any], rows: np.ndarray) -> list[dict[str, Any]]:
    """H3c2's pixel points (SVT-AV1 and DCVC-UF off the shelf) scored on the judged rows ``rows`` only."""
    keep = np.isin(a["judge_row"], rows)
    errors = pose_errors(a["judge_kp2d"][:, keep], a["judge_source_kp2d"][keep], a["judge_box_size"][keep])
    rates = hand_kbps(h1[meta["item"]])
    out = []
    for j, p in enumerate(meta["points"]):
        if p["arm"] in ("svtav1", "reference"):
            out.append({"name": p["name"], "kbps": float(p["kbps_r"] if "kbps_r" in p else rates[f"{p['crf']:g}"]), "pose": float(errors[j])})
    return out


def item_report(meta: dict[str, Any], a: dict[str, np.ndarray], pixel_units: list[tuple[dict[str, Any], dict[str, np.ndarray]]],
                h1: dict[str, Any]) -> dict[str, Any]:
    rows = a["judge_row"]
    # Every pose error, pixels' and generator's, against H3c2's WiLoR re-run on the source and its box size for the
    # same hands (this job's own re-run is checked against H2's fit by the validator).
    p_meta, p_a = next((m, x) for m, x in pixel_units if m["item"] == meta["item"] and "judge_kp2d" in x)
    order = {int(r): i for i, r in enumerate(p_a["judge_row"])}
    if missing := [int(r) for r in rows if int(r) not in order]:
        raise RuntimeError(f"{meta['item']}: rows {missing[:5]} not judged by H3c2")
    index = [order[int(r)] for r in rows]
    source, box = p_a["judge_source_kp2d"][index], p_a["judge_box_size"][index]
    errors = {v: float(pose_errors(a["judge_kp2d"][i][None], source, box)[0]) for i, v in enumerate(meta["variants"])}
    pixels = pixel_envelope(p_meta, p_a, h1, rows)
    front = hull(pixels)
    kp = meta["pose_kbps"]
    v1 = []
    for p in meta["points"]:
        rate = kp + p["kbps_v1_pixels"]
        saving, where = saving_against(front, rate, errors[f"v1{p['name']}"])
        v1.append({"name": p["name"], "kbps": rate, "pose": errors[f"v1{p['name']}"], "saving": saving, "where": where,
                   "pixel_pose_same_point": errors[f"dcvc{p['name']}"], "kbps_r_pixels": p["kbps_r"]})
    cheapest = min(meta["points"], key=lambda p: p["kbps_v2_pixels"])
    v2_rate = kp + cheapest["kbps_v2_pixels"]
    v2_saving, v2_where = saving_against(front, v2_rate, errors["v2"])
    best = max(v1, key=lambda q: q["saving"])
    # The item's statistic: V1's BD-rate against the pixel hull where their pose errors overlap; otherwise
    # "better/worse at every rate" with the single best point's rate at equal quality (a bound beyond the hull).
    bd = bd_rate_pose(v1, pixels)
    if bd is not None:
        statistic = {"saving": -bd, "how": "BD-rate over the overlapping pose range", "bd_rate": bd}
    else:
        own = pareto_front(v1, "pose")
        relation = ("better at every rate" if own[0]["pose"] < front[-1]["pose"] and best["kbps"] <= front[0]["kbps"] else
                    "worse at every rate" if own[-1]["pose"] > front[0]["pose"] and own[0]["kbps"] >= front[0]["kbps"] else
                    "no overlap")
        statistic = {"saving": best["saving"], "how": f"{relation}; best point's rate at equal quality ({best['where']})",
                     "bd_rate": None}
    following = float(pose_errors(a["judge_kp2d"][meta["variants"].index("generated")][None], a["coded_kp2d"], box)[0])
    keypoints = float(pose_errors(a["coded_kp2d"][None], source, box)[0])
    rerun = float(pose_errors(a["judge_kp2d"][meta["variants"].index("source")][None], source, box)[0])
    return {"judged_hands": int(len(rows)), "errors": errors, "source_rerun_vs_h3c2": rerun, "pixel_points": pixels, "hull": front, "v1": v1, "v1_best": best,
            "v1_statistic": statistic,
            "v2": {"kbps": v2_rate, "pose": errors["v2"], "saving": v2_saving, "where": v2_where, "frame0_point": cheapest["name"]},
            "keypoints_pose": keypoints, "generator_vs_its_input": following,
            "flicker": {v: float(np.nanmean(a["flicker"][i])) for i, v in enumerate(meta["variants"]) if v != "source"},
            "generation": {k: meta["generated"][k] for k in ("steps", "seconds", "seconds_per_step", "peak_memory_mib")}}


def realism(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, Any]:
    """KID of hand crops on generated frames against the source's, pooled over items; the interval by item."""
    variants = units[0][0]["variants"]
    owner = np.concatenate([np.full(a["features"].shape[1], i) for i, (_, a) in enumerate(units)])
    source = np.concatenate([a["features"][variants.index("source")] for _, a in units]).astype(np.float64)
    out = {}
    for j, v in enumerate(variants):
        if v == "source":
            continue
        x = np.concatenate([a["features"][j] for _, a in units]).astype(np.float64)
        kxx, kyy, kxy = kid_kernels(x, source)
        rng = np.random.default_rng(seed_of(f"{SEED}:kid:{v}"))
        boots = []
        for _ in range(KID_RESAMPLES):
            idx = np.concatenate([np.nonzero(owner == i)[0] for i in rng.integers(0, len(units), len(units))])
            boots.append(kid_from(kxx[np.ix_(idx, idx)], kyy[np.ix_(idx, idx)], kxy[np.ix_(idx, idx)], rng, subsets=10))
        out[v] = {"kid": kid_from(kxx, kyy, kxy, rng), "kid_low": float(np.percentile(boots, 2.5)),
                  "kid_high": float(np.percentile(boots, 97.5)), "crops": int(len(x))}
    return out


def decide(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": DECISION}
    complete = set(items) == set(PILOT)
    for variant, key in (("v1", "v1_statistic"), ("v2", "v2")):
        savings = np.array([e[key]["saving"] for e in items.values()])
        if len(savings):
            out[variant] = bootstrap_mean(savings, np.random.default_rng(seed_of(f"{SEED}:{variant}")), RESAMPLES)
    if not complete:
        out["decision"] = "incomplete"
        return out
    v1 = out["v1"]
    out["decision"] = ("passes" if v1["low"] > DECISION["passes_ci_low_above"] else
                       "fails" if v1["mean"] <= DECISION["fails_mean_at_most"] else "unclear")
    if out["decision"] == "fails" and out["v2"]["mean"] > DECISION["fails_mean_at_most"]:
        out["decision"] = "unclear"
        out["note"] = "only V2 (granted outline) clears 50%: to the user"
    return out


def command_report(args: argparse.Namespace) -> int:
    units = load_units(args.judge, "judge")
    pixel_units = load_units(args.h3c2, "code")
    h1 = {it["id"]: it for it in json.loads(Path(args.h1_report).read_text())["items"]}
    items = {meta["item"]: item_report(meta, a, pixel_units, h1) for meta, a in units}
    report = {"items": items, "decision": decide(items), "realism": realism(units) if units else {},
              "inputs": {"judge": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.judge],
                         "h3c2": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.h3c2],
                         "h1_report": {"path": args.h1_report, "sha256": file_sha256(Path(args.h1_report))}}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "h3e-report.json", report)
    print(json.dumps({"decision": report["decision"],
                      "items": {k: {"v1_best": v["v1_best"], "v2": v["v2"], "keypoints_pose": v["keypoints_pose"],
                                    "generator_vs_its_input": v["generator_vs_its_input"]} for k, v in items.items()}},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3e")
    commands = parser.add_subparsers(dest="command", required=True)

    def window(p: argparse.ArgumentParser) -> None:
        p.add_argument("--eval-set", required=True)
        p.add_argument("--archive", required=True)
        p.add_argument("--video", action="append", default=[])
        p.add_argument("--items", required=True)
        p.add_argument("--frames", default="240")
        p.add_argument("--poses", action="append", default=[], help="H2's VISOR published.tar (both jobs)")
        p.add_argument("--pose-code", required=True)
        p.add_argument("--mano-left", required=True)
        p.add_argument("--mano-right", required=True)

    prepare = commands.add_parser("prepare")
    window(prepare)
    judge = commands.add_parser("judge")
    window(judge)
    judge.add_argument("--masks", required=True)
    judge.add_argument("--mask-record", required=True)
    judge.add_argument("--generated", required=True, help="the generate job's published tree (and prepare's, for the records)")
    judge.add_argument("--order", default="openpose")
    judge.add_argument("--image-ckpt", required=True)
    judge.add_argument("--video-ckpt", required=True)
    judge.add_argument("--lpips-backbone", required=True)
    judge.add_argument("--inception", required=True)
    judge.add_argument("--qps", default=",".join(map(str, QPS)))
    judge.add_argument("--scales", default=",".join(map(str, SCALES)))
    judge.add_argument("--wilor-checkpoint", required=True)
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--judge", action="append", required=True)
    report.add_argument("--h3c2", action="append", required=True, help="H3c2's code published.tar (one per pilot item)")
    report.add_argument("--h1-report", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"prepare": command_prepare, "judge": command_judge, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
