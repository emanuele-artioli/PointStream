"""PLAN step H4a: grip pose on HOT3D motion capture (docs/experiments.md, 2026-10-13 H4a).

    python -m experiments.visor.h4a run --clip TAR ... --models TAR --clips N
    python -m experiments.visor.h4a validate
    python -m experiments.visor.h4a report --published TAR ... --out DIR

Is a held object still in the hand's frame? Every object of every clip is replayed through three
coders on its *moving* frames. A coder sends the true pose at a *refresh* and predicts the following
frames from it, until the prediction's error in the RGB fisheye image ``214-1`` exceeds τ px (the
mean over the object's mesh vertices) or its hand disappears; a refresh is sent there:

* ``world``: the object stays where it was in the world;
* ``grip``: it moves with the nearer hand's motion-capture wrist (``W(t) · W(r)⁻¹ · P(r)``);
* ``hybrid``: at each refresh, whichever of world, camera, grip-left and grip-right lasts longest
  (an oracle's look-ahead choice).

Each clip is published as ``publish/clips/<clip>.npz`` with a ``.json`` beside it, and checkpointed.
``report`` applies ``DECISION``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import restore_units, save_unit

FPS = 30.0
STREAM = "214-1"
TAUS = (1.0, 2.0, 4.0)
MOVE_MM = 0.5  # mean vertex displacement per frame (central difference over ±2 frames)
DILATE = 5
REST_MM = 2.0  # an object whose vertices never leave its first pose by more than this is at rest
VERTICES = 512
PREDICTORS = ("world", "grip-left", "grip-right", "camera")  # ties go to the earlier one
CODERS = ("world", "grip", "hybrid")
BITS_PER_REFRESH = 6 * 16 + 2
H1_HANDLED_KBPS_CRF62 = 41.3  # SVT-AV1 on handled objects, H1's outcome
DECISION: dict[str, Any] = {
    "tau_px": 2.0,
    "held_share_min": 0.70,
    "median_life_s_min": 0.5,
    "grip_over_world_min": 2.0,
    "rate_share_max": 0.10,
    "unit": "clip; means over clips with moving frames of per-clip values",
}


# ----------------------------------------------------------------- geometry

def quat_matrix(wxyz: list[float], xyz: list[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = w * w + x * x + y * y + z * z
    if abs(n - 1.0) > 1e-3:
        raise ValueError(f"quaternion norm {n}")
    s = 2.0 / n
    out = np.eye(4)
    out[:3, :3] = [[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                   [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                   [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]]
    out[:3, 3] = xyz
    return out


def pose(entry: dict[str, Any]) -> np.ndarray:
    return quat_matrix(entry["quaternion_wxyz"], entry["translation_xyz"])


def rigid_inverse(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def apply(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ m[:3, :3].T + m[:3, 3]


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking ``a``'s rotation to ``b``'s."""
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def glb_vertices(data: bytes) -> np.ndarray:
    """The first mesh's POSITION vertices of a binary glTF (HOT3D's eval meshes have no node transforms)."""
    if data[:4] != b"glTF":
        raise ValueError("not a binary glTF")
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    binary = data[20 + length + 8:]
    if any(k in node for node in doc["nodes"] for k in ("matrix", "rotation", "translation", "scale")):
        raise ValueError("node transforms are not supported")
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    if accessor["componentType"] != 5126 or accessor["type"] != "VEC3":
        raise ValueError("POSITION is not float VEC3")
    view = doc["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    raw = np.frombuffer(binary, np.uint8, count=stride * (accessor["count"] - 1) + 12, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, (accessor["count"], 12), (stride, 1))
    return np.ascontiguousarray(rows).view("<f4").reshape(-1, 3).astype(np.float64)


def subsample(vertices: np.ndarray, count: int = VERTICES) -> np.ndarray:
    """Every k-th vertex, so at most ``count`` (deterministic)."""
    step = max(1, int(np.ceil(len(vertices) / count)))
    return vertices[::step]


def read_models(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """Vertices per BOP id from the eval meshes' tar, models_info, and each mesh's box error against it (m)."""
    with tarfile.open(path) as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}

        def read(name: str) -> bytes:
            handle = tar.extractfile(members[name])
            assert handle is not None
            return handle.read()

        info = json.loads(read("models_info.json"))
        vertices, box_error = {}, {}
        for bop, row in info.items():
            v = glb_vertices(read(f"obj_{int(bop):06d}.glb"))
            lo = np.array([row["min_x"], row["min_y"], row["min_z"]])
            hi = lo + np.array([row["size_x"], row["size_y"], row["size_z"]])
            box_error[bop] = float(max(np.abs(v.min(0) - lo).max(), np.abs(v.max(0) - hi).max()))
            vertices[bop] = subsample(v)
    return vertices, info, box_error


# ----------------------------------------------------------------- clips

def read_clip(path: str) -> list[dict[str, Any]]:
    """Per frame, in order: the RGB camera's JSON, each object's (BOP id, pose, visibility in ``STREAM``),
    and each hand's motion-capture wrist pose. One pass over the tar; images are skipped."""
    frames: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, mode="r|") as tar:
        for member in tar:
            name = member.name
            if not name.endswith(".json") or not name[:6].isdigit():
                continue
            key, kind = name.split(".")[0], name.split(".")[1]
            if kind not in ("objects", "hands", "cameras"):
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            doc = json.loads(handle.read())
            row = frames.setdefault(key, {})
            if kind == "cameras":
                row["camera"] = doc[STREAM]
            elif kind == "hands":
                row["hands"] = {side: pose(h["umetrack_pose"]["T_world_from_wrist"])
                                for side, h in doc.items() if "umetrack_pose" in h}
            else:
                row["objects"] = {inst["object_uid"]: (str(inst["object_bop_id"]), pose(inst["T_world_from_object"]),
                                                       float(inst.get("visibilities_modeled", {}).get(STREAM, 0.0)))
                                  for insts in doc.values() for inst in insts}
    keys = sorted(frames)
    if [int(k) for k in keys] != list(range(len(keys))):
        raise ValueError(f"{path}: frames are not consecutive")
    missing = [k for k in keys if {"camera", "hands", "objects"} - set(frames[k])]
    if missing:
        raise ValueError(f"{path}: frames {missing[:3]} lack a record")
    return [frames[k] for k in keys]


Projector = Callable[[int, np.ndarray], np.ndarray]


class Track:
    """One object over a clip: true poses, the camera, both wrists, and the error of a predicted pose."""

    def __init__(self, vertices: np.ndarray, poses: np.ndarray, visible: np.ndarray, cameras: np.ndarray,
                 wrists: dict[str, np.ndarray], project: Projector) -> None:
        self.v = vertices
        self.P = poses  # (T, 4, 4), NaN where absent
        self.visible = visible
        self.C = cameras
        self.W = wrists  # side -> (T, 4, 4), NaN where absent
        self.project = project
        self._truth: dict[int, np.ndarray] = {}

    def truth_px(self, t: int) -> np.ndarray:
        if t not in self._truth:
            self._truth[t] = self.project(t, apply(self.P[t], self.v))
        return self._truth[t]

    def error_px(self, t: int, predicted: np.ndarray) -> float:
        return float(np.linalg.norm(self.project(t, apply(predicted, self.v)) - self.truth_px(t), axis=1).mean())

    def error_mm(self, t: int, predicted: np.ndarray) -> float:
        return float(1000 * np.linalg.norm(apply(predicted, self.v) - apply(self.P[t], self.v), axis=1).mean())

    def predicted(self, kind: str, r: int, t: int) -> np.ndarray | None:
        if kind == "world":
            return self.P[r]
        if kind == "camera":
            return self.C[t] @ rigid_inverse(self.C[r]) @ self.P[r]
        w = self.W[kind.split("-")[1]]
        if np.isnan(w[r]).any() or np.isnan(w[t]).any():
            return None
        return w[t] @ rigid_inverse(w[r]) @ self.P[r]

    def nearer_hand(self, r: int) -> str | None:
        centre = apply(self.P[r], self.v).mean(0)
        best = None
        for side, w in sorted(self.W.items()):
            if not np.isnan(w[r]).any():
                d = float(np.linalg.norm(w[r][:3, 3] - centre))
                if best is None or d < best[0]:
                    best = (d, side)
        return None if best is None else best[1]

    def displacement_mm(self) -> np.ndarray:
        """Mean vertex displacement per frame, central difference over ±2 frames (NaN where a pose is absent)."""
        n = len(self.P)
        out = np.full(n, np.nan)
        for t in range(n):
            a, b = max(0, t - 2), min(n - 1, t + 2)
            if b > a and not (np.isnan(self.P[a]).any() or np.isnan(self.P[b]).any()):
                out[t] = 1000 * np.linalg.norm(apply(self.P[b], self.v) - apply(self.P[a], self.v), axis=1).mean() / (b - a)
        return out


def moving_segments(displacement: np.ndarray, move_mm: float = MOVE_MM, dilate: int = DILATE) -> list[tuple[int, int]]:
    """Runs of frames whose displacement exceeds ``move_mm``, dilated by ``dilate`` frames and merged, as
    (first, last) inclusive."""
    moving = np.nan_to_num(displacement, nan=0.0) > move_mm
    grown = moving.copy()
    for t in np.flatnonzero(moving):
        grown[max(0, t - dilate):t + dilate + 1] = True
    out, start = [], None
    for t, m in enumerate(grown):
        if m and start is None:
            start = t
        if not m and start is not None:
            out.append((start, t - 1))
            start = None
    if start is not None:
        out.append((start, len(grown) - 1))
    return out


def held_run(track: Track, kind: str, frames: list[int], i: int, tau: float) -> int:
    """How many of ``frames[i + 1:]`` ``kind`` predicts within ``tau`` px from a refresh at ``frames[i]``."""
    r = frames[i]
    n = 0
    for t in frames[i + 1:]:
        p = track.predicted(kind, r, t)
        if p is None or track.error_px(t, p) > tau:
            break
        n += 1
    return n


def replay(track: Track, frames: list[int], coder: str, tau: float) -> list[dict[str, Any]]:
    """Refreshes of one coder over one segment's scored frames: per refresh its frame, predictor, how many
    frames it held, its life (s) and whether the segment's end cut it."""
    out = []
    i = 0
    while i < len(frames):
        r = frames[i]
        if coder == "world":
            options = ["world"]
        elif coder == "grip":
            side = track.nearer_hand(r)
            options = [f"grip-{side}"] if side else ["world"]
        else:
            options = list(PREDICTORS)
        runs = [(held_run(track, k, frames, i, tau), k) for k in options]
        held, kind = max(runs, key=lambda x: (x[0], -options.index(x[1])))
        j = i + 1 + held
        cut = j >= len(frames)
        end = frames[-1] + 1 if cut else frames[j]
        out.append({"frame": r, "kind": kind, "held": held, "life_s": (end - r) / FPS, "cut": cut})
        i = j
    return out


def grip_spread(track: Track, first: int, last: int, displacement: np.ndarray) -> dict[str, Any] | None:
    """How constant the grip is over a segment's undilated moving frames: G(t) = W(t)⁻¹ P(t) for the hand
    nearest most often, against G at the middle such frame; the world pose's spread beside it."""
    frames = [t for t in range(first, last + 1) if displacement[t] > MOVE_MM]
    if len(frames) < 3:
        return None
    sides = [s for s in (track.nearer_hand(t) for t in frames) if s]
    if not sides:
        return None
    side = max(set(sides), key=sides.count)
    w = track.W[side]
    frames = [t for t in frames if not np.isnan(w[t]).any()]
    if len(frames) < 3:
        return None
    mid = frames[len(frames) // 2]
    g = {t: rigid_inverse(w[t]) @ track.P[t] for t in frames}
    rot = [rotation_deg(g[mid], g[t]) for t in frames]
    trans = [1000 * float(np.linalg.norm(g[t][:3, 3] - g[mid][:3, 3])) for t in frames]
    world_rot = [rotation_deg(track.P[mid], track.P[t]) for t in frames]
    world_trans = [1000 * float(np.linalg.norm(track.P[t][:3, 3] - track.P[mid][:3, 3])) for t in frames]
    return {"frames": len(frames), "hand": side, "rot_median_deg": float(np.median(rot)), "rot_max_deg": float(np.max(rot)),
            "trans_median_mm": float(np.median(trans)), "trans_max_mm": float(np.max(trans)),
            "world_rot_median_deg": float(np.median(world_rot)), "world_trans_median_mm": float(np.median(world_trans))}


# ----------------------------------------------------------------- one clip

def toolkit_projector(frames: list[dict[str, Any]]) -> tuple[Projector, np.ndarray, tuple[int, int]]:
    from hand_tracking_toolkit import camera

    cams = [camera.from_json(f["camera"]) for f in frames]
    size = (cams[0].width, cams[0].height)

    def project(t: int, points: np.ndarray) -> np.ndarray:
        return np.asarray(cams[t].world_to_window(points), np.float64)

    return project, np.stack([c.T_world_from_eye for c in cams]), size


def tracks_of(frames: list[dict[str, Any]], vertices: dict[str, np.ndarray], project: Projector,
              cameras: np.ndarray) -> dict[str, tuple[str, Track]]:
    n = len(frames)
    wrists = {side: np.full((n, 4, 4), np.nan) for side in ("left", "right")}
    for t, f in enumerate(frames):
        for side, w in f["hands"].items():
            wrists[side][t] = w
    out = {}
    for uid in sorted({u for f in frames for u in f["objects"]}):
        bop = next(f["objects"][uid][0] for f in frames if uid in f["objects"])
        poses = np.full((n, 4, 4), np.nan)
        visible = np.zeros(n, bool)
        for t, f in enumerate(frames):
            if uid in f["objects"]:
                poses[t] = f["objects"][uid][1]
                visible[t] = f["objects"][uid][2] > 0
        out[uid] = (bop, Track(vertices[bop], poses, visible, cameras, wrists, project))
    return out


def score_clip(frames: list[dict[str, Any]], vertices: dict[str, np.ndarray], project: Projector,
               cameras: np.ndarray, size: tuple[int, int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tracks = tracks_of(frames, vertices, project, cameras)
    refreshes: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    rest_px: list[float] = []
    objects = []
    inside = []
    for uid, (bop, track) in tracks.items():
        disp = track.displacement_mm()
        segments = moving_segments(disp)
        present = [t for t in range(len(frames)) if not np.isnan(track.P[t]).any()]
        for t in present[::15]:
            if track.visible[t]:
                px = track.truth_px(t).mean(0)
                inside.append(bool(0 <= px[0] < size[0] and 0 <= px[1] < size[1]))
        first = present[0] if present else None
        travel = max((track.error_mm(t, track.P[first]) for t in present), default=0.0) if first is not None else 0.0
        if not segments and travel < REST_MM:
            rest_px += [track.error_px(t, track.P[first]) for t in present[1:] if track.visible[t] and track.visible[first]]
        scored = 0
        for s, (a, b) in enumerate(segments):
            frames_s = [t for t in range(a, b + 1) if track.visible[t] and not np.isnan(track.P[t]).any()]
            scored += len(frames_s)
            if not frames_s:
                continue
            for tau in TAUS:
                for coder in CODERS:
                    for row in replay(track, frames_s, coder, tau):
                        refreshes.append({"object": uid, "segment": s, "tau": tau, "coder": coder, "scored": len(frames_s), **row})
            spread = grip_spread(track, a, b, disp)
            if spread:
                spreads.append({"object": uid, "segment": s, **spread})
        objects.append({"uid": uid, "bop": bop, "segments": [list(x) for x in segments], "scored_frames": scored,
                        "travel_mm": travel, "visible_frames": int(track.visible.sum())})
    arrays: dict[str, np.ndarray] = {
        "refresh_tau": np.array([r["tau"] for r in refreshes], np.float32),
        "refresh_coder": np.array([CODERS.index(r["coder"]) for r in refreshes], np.int8),
        "refresh_kind": np.array([PREDICTORS.index(r["kind"]) for r in refreshes], np.int8),
        "refresh_held": np.array([r["held"] for r in refreshes], np.int32),
        "refresh_life_s": np.array([r["life_s"] for r in refreshes], np.float32),
        "refresh_cut": np.array([r["cut"] for r in refreshes], bool),
        "refresh_frame": np.array([r["frame"] for r in refreshes], np.int32),
        "refresh_object": np.array([r["object"] for r in refreshes]),
        "refresh_segment": np.array([r["segment"] for r in refreshes], np.int32),
        "rest_px": np.array(rest_px, np.float32),
    }
    meta: dict[str, Any] = {"frames": len(frames), "objects": objects, "spreads": spreads, "visible_centroid_inside": inside,
            "moving_objects": sum(1 for o in objects if o["segments"]),
            "scored_frames": sum(o["scored_frames"] for o in objects)}
    return arrays, meta


def run_clip(path: str, models: str) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    vertices, _, _ = read_models(models)
    frames = read_clip(path)
    project, cameras, size = toolkit_projector(frames)
    arrays, meta = score_clip(frames, vertices, project, cameras, size)
    meta.update({"clip": Path(path).name, "clip_sha256": file_sha256(Path(path))})
    return Path(path).stem, arrays, meta


def command_run(args: argparse.Namespace) -> int:
    clips = list(args.clip)[:int(args.clips)] if int(args.clips) else list(args.clip)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    restored = restore_units(publish, "clips")
    _, info, box_error = read_models(args.models)
    todo = [c for c in clips if Path(c).stem not in restored]
    done = len(clips) - len(todo)
    progress(done)
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for unit, arrays, meta in pool.map(run_clip, todo, [args.models] * len(todo)):
            save_unit(publish, "clips", unit, arrays, meta)
            done += 1
            progress(done)
            print(f"{unit}: {meta['moving_objects']} moving objects, {meta['scored_frames']} scored frames", flush=True)
    write_json(stage_dir() / "h4a.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "clips": [Path(c).stem for c in clips],
        "restored_units": sorted(restored), "models_sha256": file_sha256(Path(args.models)),
        "mesh_box_error_m": box_error, "objects_in_models": len(info),
        "settings": {"fps": FPS, "stream": STREAM, "taus": TAUS, "move_mm": MOVE_MM, "dilate": DILATE, "rest_mm": REST_MM,
                     "vertices": VERTICES, "predictors": PREDICTORS, "coders": CODERS},
    })
    return 0


# ----------------------------------------------------------------- validate

def load_units(directory: Path) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    out = []
    for meta_path in sorted(directory.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        data = (directory / f"{meta_path.stem}.npz").read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["npz_sha256"]:
            raise ValueError(f"{meta_path.stem}: npz does not match its record")
        with np.load(io.BytesIO(data)) as z:
            out.append((meta, {k: z[k] for k in z.files}))
    return out


def units_from_tar(path: Path) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    out = []
    with tarfile.open(path) as tar:
        members = {m.name: m for m in tar.getmembers()}
        for name in sorted(n for n in members if n.startswith("publish/clips/") and n.endswith(".json")):
            handle = tar.extractfile(members[name])
            assert handle is not None
            meta = json.loads(handle.read())
            blob = tar.extractfile(members[name[:-5] + ".npz"])
            assert blob is not None
            data = blob.read()
            if hashlib.sha256(data).hexdigest() != meta["npz_sha256"]:
                raise ValueError(f"{name}: npz does not match its record")
            with np.load(io.BytesIO(data)) as z:
                out.append((meta, {k: z[k] for k in z.files}))
    return out


def unit_checks(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, bool]:
    inside = [x for meta, _ in units for x in meta["visible_centroid_inside"]]
    rest = np.concatenate([a["rest_px"] for _, a in units]) if units else np.zeros(0)
    accounted = True
    for meta, a in units:
        for tau in TAUS:
            for c in range(len(CODERS)):
                sel = (a["refresh_tau"] == np.float32(tau)) & (a["refresh_coder"] == c)
                if int((a["refresh_held"][sel] + 1).sum()) != meta["scored_frames"]:
                    accounted = False
    return {
        "every_clip_has_frames": all(meta["frames"] >= 30 for meta, _ in units),
        "true_poses_project_inside_the_image_90pct": bool(inside) and float(np.mean(inside)) >= 0.9,
        "objects_at_rest_hold_within_1px": rest.size > 0 and float(np.median(rest)) < 1.0,
        "every_scored_frame_is_held_or_a_refresh_once": accounted,
        "some_clip_has_moving_frames": any(meta["scored_frames"] > 0 for meta, _ in units),
        "lives_positive_and_bounded": all(bool(((a["refresh_life_s"] > 0) & (a["refresh_life_s"] <= meta["frames"] / FPS)).all())
                                          for meta, a in units),
    }


def validate_result(stage: Path) -> dict[str, bool]:
    record = json.loads((stage / "h4a.json").read_text())
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch") / "publish" / "clips"
    units = load_units(publish) if publish.is_dir() else units_from_tar(stage / "published.tar")
    checks = {
        "every_clip_published": sorted(m["clip"][:-4] for m, _ in units) == sorted(record["clips"]),
        "meshes_match_models_info_2mm": all(e < 2e-3 for e in record["mesh_box_error_m"].values()),
    }
    checks.update(unit_checks(units))
    return checks


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report

def clip_values(meta: dict[str, Any], a: dict[str, np.ndarray], coder: str, tau: float) -> dict[str, Any] | None:
    """One clip's held share, median life, frame-weighted life, cut share and refresh rate for one coder."""
    if meta["scored_frames"] == 0:
        return None
    sel = (a["refresh_tau"] == np.float32(tau)) & (a["refresh_coder"] == CODERS.index(coder))
    lives = a["refresh_life_s"][sel]
    held = a["refresh_held"][sel]
    refreshes = int(sel.sum())
    kinds = a["refresh_kind"][sel]
    return {"held_share": 1.0 - refreshes / meta["scored_frames"], "median_life_s": float(np.median(lives)),
            "frame_weighted_life_s": float(np.sum(lives * (held + 1)) / np.sum(held + 1)),
            "cut_share": float(np.mean(a["refresh_cut"][sel])), "refreshes": refreshes,
            "rate_kbps": refreshes * BITS_PER_REFRESH / (meta["frames"] / FPS) / 1000,
            "frames_by_kind": {k: int(np.sum((held + 1)[kinds == i])) for i, k in enumerate(PREDICTORS)}}


def summarize(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tau in TAUS:
        for coder in CODERS:
            rows = [v for meta, a in units if (v := clip_values(meta, a, coder, tau))]
            if not rows:
                continue
            kinds = {k: sum(r["frames_by_kind"][k] for r in rows) for k in PREDICTORS}
            total = sum(kinds.values())
            out[f"{coder}@{tau:g}"] = {
                "clips": len(rows),
                **{k: float(np.mean([r[k] for r in rows])) for k in ("held_share", "median_life_s", "frame_weighted_life_s",
                                                                       "cut_share", "rate_kbps")},
                "held_share_p10_p50_p90": [float(x) for x in np.percentile([r["held_share"] for r in rows], [10, 50, 90])],
                "frames_by_kind_share": {k: v / total for k, v in kinds.items()},
            }
    spreads = [s for meta, _ in units for s in meta["spreads"]]
    if spreads:
        out["grip_spread"] = {"segments": len(spreads), **{k: [float(x) for x in np.percentile([s[k] for s in spreads], [25, 50, 75])]
                                                           for k in ("rot_median_deg", "rot_max_deg", "trans_median_mm",
                                                                     "trans_max_mm", "world_rot_median_deg",
                                                                     "world_trans_median_mm")}}
    rest = np.concatenate([a["rest_px"] for _, a in units])
    out["noise_floor_px"] = {"object_frames": int(rest.size), "median": float(np.median(rest)),
                             "p95": float(np.percentile(rest, 95))}
    out["clips"] = len(units)
    out["clips_with_moving_frames"] = sum(1 for meta, _ in units if meta["scored_frames"] > 0)
    out["moving_objects"] = sum(meta["moving_objects"] for meta, _ in units)
    out["scored_frames"] = sum(meta["scored_frames"] for meta, _ in units)
    return out


def decide(summary: dict[str, Any], tau: float | None = None) -> dict[str, Any]:
    tau = DECISION["tau_px"] if tau is None else tau
    hybrid, world = summary[f"hybrid@{tau:g}"], summary[f"world@{tau:g}"]
    rate_bar = DECISION["rate_share_max"] * H1_HANDLED_KBPS_CRF62
    checks = {
        "a_held_share": hybrid["held_share"] >= DECISION["held_share_min"],
        "b_median_life": hybrid["median_life_s"] >= DECISION["median_life_s_min"],
        "c_grip_explains": hybrid["median_life_s"] >= DECISION["grip_over_world_min"] * world["median_life_s"],
        "rate": hybrid["rate_kbps"] <= rate_bar,
    }
    return {"tau_px": tau, "checks": checks, "passes": all(checks.values()), "rate_bar_kbps": rate_bar}


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    at2 = decide(summary)
    at4 = decide(summary, 4.0)
    if at2["passes"]:
        word = "passes"
    elif at4["passes"] or (at2["checks"]["a_held_share"] and at2["checks"]["b_median_life"]):
        word = "unclear"
    else:
        word = "fails"
    return {"verdict": word, "at_2px": at2, "at_4px": at4}


def command_report(args: argparse.Namespace) -> int:
    units = [u for p in args.published for u in units_from_tar(Path(p))]
    names = [m["clip"] for m, _ in units]
    if len(set(names)) != len(names):
        raise SystemExit("a clip appears twice")
    summary = summarize(units)
    out = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published},
           "summary": summary, "verdict": verdict(summary)}
    out_dir = Path(args.out)
    write_json(out_dir / "h4a-report.json", out)
    print(json.dumps(out["verdict"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--clip", action="append", required=True)
    run.add_argument("--models", required=True, help="tar of HOT3D's object_models_eval")
    run.add_argument("--clips", default="0", help="the first N clips (0: all)")
    run.add_argument("--workers", default="8")
    sub.add_parser("validate")
    report = sub.add_parser("report")
    report.add_argument("--published", action="append", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
