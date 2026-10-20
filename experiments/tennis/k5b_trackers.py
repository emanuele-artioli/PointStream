"""PLAN step K5b (2), stage A: off-the-shelf ball trackers compared on held-out ball labels (no training).

    python -m experiments.tennis.k5b_trackers run --set NAME --select SPEC --wasb-code DIR --wasb-weights DIR \\
        --balltrack PT --balltrack-code DIR [--tracknet ZIP] [--racketvision TAR]
    python -m experiments.tennis.k5b_trackers validate
    python -m experiments.tennis.k5b_trackers report --published TAR ... --out DIR

Candidates (docs/experiments.md, 2026-10-20 K5b 2): WASB's tennis model zoo (`nttcom/WASB-SBDT` at 923462c), each
with the inference its README evaluates it with, and RacketVision's BallTrack (TrackNetV3 with a median frame).
Sets: ``tracknet`` (TrackNet games, every frame labelled), ``svt`` (the same clips through SVT-AV1 at `SVT_CRF`,
labels of the source) and ``racketvision`` (RacketVision tennis test rallies, sparse labels). Every frame is scored
at 1280x720. Per clip the published JSON holds each candidate's per-frame prediction and its counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import time
import types
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

SIZE = (1280, 720)
HIT_PX = (10.0, 5.0)
SVT_CRF = 62
SVT_PRESET = 8
MEDIAN_FRAMES = 100
BATCH = 16
#: Each WASB-zoo setting as WASB-SBDT's GET_STARTED evaluates it on tennis.
ZOO: dict[str, dict[str, Any]] = {
    "wasb": {"model": "wasb", "weights": "wasb_tennis_best.pth.tar", "step": 3, "hm_weight": True, "tracker": "online"},
    "wasb-step1": {"model": "wasb", "weights": "wasb_tennis_best.pth.tar", "step": 1, "hm_weight": True, "tracker": "online"},
    "monotrack": {"model": "monotrack", "weights": "monotrack_tennis_best.pth.tar", "step": 3, "hm_weight": False, "tracker": "intra_frame_peak"},
    "restracknetv2": {"model": "restracknetv2", "weights": "restracknetv2_tennis_best.pth.tar", "step": 3, "hm_weight": False,
                      "tracker": "intra_frame_peak"},
    "tracknetv2": {"model": "tracknetv2", "weights": "tracknetv2_tennis_best.pth.tar", "step": 3, "hm_weight": False, "tracker": "intra_frame_peak"},
    "ballseg": {"model": "ballseg", "weights": "ballseg_tennis_best.pth.tar", "step": 1, "hm_weight": False, "tracker": "intra_frame_peak"},
    "deepball": {"model": "deepball", "weights": "deepball_tennis_best.pth.tar", "step": 1, "detector": "deepball", "tracker": "intra_frame_peak"},
    "deepball-large": {"model": "deepball_large", "weights": "deepball-large_tennis_best.pth.tar", "step": 1, "detector": "deepball",
                       "tracker": "intra_frame_peak"},
}
CANDIDATES = (*ZOO, "balltrack")
#: The two sets held out for every candidate; the choice reads these (docs/experiments.md, the rule).
HELD_OUT_FOR_ALL = ("tracknet:8,9,10", "racketvision:test")

Prediction = tuple[bool, float, float]


# ----------------------------------------------------------------- data


def tracknet_clips(zip_path: Path, games: list[int]) -> list[dict[str, Any]]:
    """Clips of the listed games: id, frame names and labels (visibility, x, y) from Label.csv."""
    z = zipfile.ZipFile(zip_path)
    out = []
    for name in sorted(n for n in z.namelist() if n.endswith("Label.csv")):
        _, game, clip, _ = name.split("/")
        if int(game[4:]) not in games:
            continue
        rows = []
        for line in z.read(name).decode().splitlines()[1:]:
            f = line.split(",")
            if not f[0]:
                continue
            vis = int(f[1]) if f[1] else 0
            rows.append((f[0], vis, float(f[2]) if f[2] else 0.0, float(f[3]) if f[3] else 0.0))
        out.append({"id": f"tracknet/{game}/{clip}", "dir": f"Dataset/{game}/{clip}", "rows": rows})
    return out


def tracknet_frames(zip_path: Path, clip: dict[str, Any]) -> np.ndarray:
    import cv2

    z = zipfile.ZipFile(zip_path)
    frames = []
    for name, *_ in clip["rows"]:
        img = cv2.imdecode(np.frombuffer(z.read(f"{clip['dir']}/{name}"), np.uint8), cv2.IMREAD_COLOR)
        if img.shape[1::-1] != SIZE:
            raise RuntimeError(f"{clip['id']}: {img.shape} is not {SIZE}")
        frames.append(img[:, :, ::-1])
    return np.stack(frames)


def tracknet_labels(clip: dict[str, Any]) -> list[tuple[int, float, float]]:
    """(kind, x, y): kind 1 ball visible (TrackNet 1, 2), 0 no ball (0), -1 excluded (3, occluded)."""
    return [(1 if v in (1, 2) else 0 if v == 0 else -1, x, y) for _, v, x, y in clip["rows"]]


def racketvision_clips(tar_path: Path, split: str) -> list[dict[str, Any]]:
    with tarfile.open(tar_path) as tar:
        handle = tar.extractfile(f"tennis/info/{split}.json")
        assert handle is not None
        pairs = json.loads(handle.read())
    return [{"id": f"racketvision/{m}_{r}", "match": m, "rally": r} for m, r in pairs]


def racketvision_load(tar_path: Path, clip: dict[str, Any], scratch: Path) -> tuple[np.ndarray, list[tuple[int, float, float]], dict[str, Any]]:
    """The rally's frames area-downscaled to `SIZE` and its labels scaled to it (-1 on unlabelled frames)."""
    import cv2

    video = scratch / "rv.mp4"
    with tarfile.open(tar_path) as tar:
        handle = tar.extractfile(f"tennis/videos/{clip['match']}_{clip['rally']}.mp4")
        assert handle is not None
        video.write_bytes(handle.read())
        handle = tar.extractfile(f"tennis/all/{clip['match']}/csv/{clip['rally']}_ball.csv")
        assert handle is not None
        csv_text = handle.read().decode()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    native = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        native = img.shape[1::-1]
        frames.append(cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1] if native != SIZE else img[:, :, ::-1])
    cap.release()
    video.unlink()
    assert native is not None
    sx, sy = SIZE[0] / native[0], SIZE[1] / native[1]
    labels: list[tuple[int, float, float]] = [(-1, 0.0, 0.0)] * len(frames)
    head, *lines = csv_text.splitlines()
    cols = head.split(",")
    for line in lines:
        f = dict(zip(cols, line.split(",")))
        t = int(f["Frame"])
        if 0 <= t < len(frames):
            labels[t] = (1 if f["Visibility"] == "1" else 0, float(f["X"]) * sx, float(f["Y"]) * sy)
    return np.stack(frames), labels, {"native": list(native), "fps": fps, "frames": len(frames)}


def svt_roundtrip(args: argparse.Namespace, frames: np.ndarray, scratch: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """The frames through SVT-AV1 (4:2:0 limited range, K5's conversion), decoded back to RGB."""
    import subprocess

    from experiments.tennis import windows as tw
    from src.codecs import svtav1

    width, height = SIZE
    work = scratch / "svt"
    work.mkdir(parents=True, exist_ok=True)
    yuv = work / "source.yuv"
    proc = subprocess.run([args.ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                           "-i", "-", "-vf", "scale=out_range=tv:out_color_matrix=bt709,format=yuv420p", "-f", "rawvideo", "-y", str(yuv)],
                          input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode()[-2000:])
    rec = svtav1.code(yuv, work, width=width, height=height, fps=Fraction(30), frames=len(frames), crf=SVT_CRF, preset=SVT_PRESET,
                      threads=int(os.environ.get("PS_CPU_ALLOWANCE") or 8), full_range=False, encoder=args.encoder, decoder=args.decoder)
    decoded = np.array(tw.rgb_frames(Path(rec["decoded"]), width, height, len(frames), work / "decoded.npy"))
    info = {"crf": SVT_CRF, "preset": SVT_PRESET, "kbps": rec["payload_bytes"] * 8 / (len(frames) / 30.0) / 1000.0,
            "stream_sha256": rec["stream_sha256"]}
    import shutil

    shutil.rmtree(work)
    return decoded, info


# ----------------------------------------------------------------- candidates


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text())


class AttrDict(dict):
    """A config read by key or attribute, as WASB's code reads Hydra's (HRNet uses ``cfg.MODEL.EXTRA``); the packed
    environment has neither hydra nor omegaconf."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @staticmethod
    def of(value: Any) -> Any:
        if isinstance(value, dict):
            return AttrDict({k: AttrDict.of(v) for k, v in value.items()})
        if isinstance(value, list):
            return [AttrDict.of(v) for v in value]
        return value


class Zoo:
    """WASB-SBDT's tennis models, imported from its code tree, with its preprocessing and postprocessors."""

    def __init__(self, code: Path, weights: Path):
        import torch

        root = code / "src"
        sys.path.insert(0, str(root))
        import importlib.util

        import models
        from trackers import build_tracker
        from utils.image import get_affine_transform

        def by_path(name: str) -> Any:
            # detectors/__init__ imports hydra through the detector class, which this wrapper does not use.
            spec = importlib.util.spec_from_file_location(f"wasb_{name}", root / "detectors" / f"{name}.py")
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        DeepBallPostprocessor = by_path("deepball_postprocessor").DeepBallPostprocessor
        TracknetV2Postprocessor = by_path("postprocessor").TracknetV2Postprocessor

        if not str(Path(models.__file__).resolve()).startswith(str(root.resolve())):
            raise RuntimeError(f"'models' resolved to {models.__file__}, not WASB's tree")
        self.get_affine_transform = get_affine_transform
        self.build_tracker = build_tracker
        self.settings: dict[str, dict[str, Any]] = {}
        dataloader = load_yaml(root / "configs/dataloader/default.yaml")
        loaded: dict[str, Any] = {}
        for name, s in ZOO.items():
            model_cfg = load_yaml(root / f"configs/model/{s['model']}.yaml")
            detector_cfg = load_yaml(root / f"configs/detector/{s.get('detector', 'tracknetv2')}.yaml")
            detector_cfg["postprocessor"].setdefault("use_hm_weight", s.get("hm_weight", False))
            if "hm_weight" in s:
                detector_cfg["postprocessor"]["use_hm_weight"] = s["hm_weight"]
            detector_cfg["step"] = s["step"]
            cfg = AttrDict.of({"model": model_cfg, "detector": detector_cfg, "dataloader": dataloader,
                               "tracker": load_yaml(root / f"configs/tracker/{s['tracker']}.yaml"), "runner": {"device": "cuda", "gpus": [0]}})
            key = s["weights"]
            if key not in loaded:
                net = models.build_model(cfg)
                state = torch.load(weights / key, map_location="cpu", weights_only=True)
                net.load_state_dict(state["model_state_dict"])
                loaded[key] = net.cuda().eval()
            post = (DeepBallPostprocessor if detector_cfg["postprocessor"]["name"] == "deepball" else TracknetV2Postprocessor)(cfg)
            self.settings[name] = {"cfg": cfg, "net": loaded[key], "post": post}

    def transform(self, wh: tuple[int, int], inv: int) -> np.ndarray:
        w, h = SIZE
        return self.get_affine_transform(np.array([w / 2.0, h / 2.0], np.float32), max(h, w) * 1.0, 0, list(wh), inv=inv)

    def run(self, name: str, frames: np.ndarray) -> list[Prediction]:
        """Per frame (visible, x, y) as WASB's eval runner produces it: windows of frames_in at the setting's step,
        each window's outputs on its last frames_out frames (a later window overwrites), then the tracker in order.
        A final window aligned to the clip's end covers frames the step leaves out."""
        import cv2
        import torch

        st = self.settings[name]
        m = st["cfg"]["model"]
        fin, fout, step = m["frames_in"], m["frames_out"], st["cfg"]["detector"]["step"]
        inp = (m["inp_width"], m["inp_height"])
        trans_in = self.transform(inp, 0)
        trans_out_inv = torch.from_numpy(np.stack([self.transform((m["out_width"], m["out_height"]), 1)])).float()
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        n = len(frames)
        starts = list(range(0, n - fin + 1, step))
        if starts and starts[-1] != n - fin:
            starts.append(n - fin)
        cache: dict[int, np.ndarray] = {}

        def prep(t: int) -> np.ndarray:
            if t not in cache:
                img = cv2.warpAffine(np.asarray(frames[t]), trans_in, inp, flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                cache[t] = np.moveaxis((img - mean) / std, -1, 0)
            return cache[t]

        dets: dict[int, list[dict[str, Any]]] = {}
        with torch.no_grad():
            for b in range(0, len(starts), BATCH):
                batch = []
                for s in starts[b:b + BATCH]:
                    x = [prep(t) for t in range(s, s + fin)]
                    if m.get("rgb_diff"):
                        x[0] = np.abs(x[1] - x[0])
                    batch.append(np.concatenate(x, 0))
                for t in [t for t in cache if t < starts[min(b + BATCH, len(starts)) - 1]]:
                    del cache[t]
                preds = st["net"](torch.from_numpy(np.stack(batch)).cuda())
                res = st["post"].run(preds, {0: trans_out_inv.repeat(len(batch), 1, 1)})
                for i, s in enumerate(starts[b:b + BATCH]):
                    for j in range(fout):
                        t = s + fin - fout + j
                        dets[t] = [{"xy": np.asarray(xy), "score": float(sc)} for xy, sc in zip(res[i][j][0]["xys"], res[i][j][0]["scores"])]
        tracker = self.build_tracker(st["cfg"])
        tracker.refresh()
        out: list[Prediction] = []
        for t in range(n):
            r = tracker.update(dets.get(t, []))
            out.append((bool(r["visi"]), float(r["x"]) if r["visi"] else 0.0, float(r["y"]) if r["visi"] else 0.0))
        return out


class BallTrack:
    """RacketVision's TrackNetV3 (seq_len 4, median frame concatenated), loaded without mmengine."""

    SEQ = 4
    WH = (512, 288)

    def __init__(self, code: Path, weights: Path):
        import importlib.util

        import torch
        from torch import nn

        registry = types.ModuleType("mmengine.registry")
        registry.MODELS = types.SimpleNamespace(register_module=lambda name=None, **k: (lambda cls: cls))  # type: ignore[attr-defined]
        model_mod = types.ModuleType("mmengine.model")
        model_mod.BaseModel = nn.Module  # type: ignore[attr-defined]
        root_mod = types.ModuleType("mmengine")
        for name, mod in (("mmengine", root_mod), ("mmengine.registry", registry), ("mmengine.model", model_mod)):
            sys.modules.setdefault(name, mod)
        base = code / "source" / "BallTrack" / "model"
        pkg = types.ModuleType("model")
        pkg.__path__ = [str(base)]
        sys.modules["model"] = pkg
        for sub in ("loss_utils", "tracknet_v3"):
            spec = importlib.util.spec_from_file_location(f"model.{sub}", base / f"{sub}.py")
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"model.{sub}"] = mod
            spec.loader.exec_module(mod)
        net = sys.modules["model.tracknet_v3"].TrackNet(in_dim=15, out_dim=4, last_only=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        net.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state["state_dict"].items()})
        self.net = net.cuda().eval()
        self.meta = {"epoch": state["meta"]["epoch"], "iter": state["meta"]["iter"]}

    def run(self, frames: np.ndarray) -> list[Prediction]:
        """BGR in [0, 1] at 512x288; per frame t the median frame then frames t-3..t (clamped at 0), as its
        training samples are built; the largest blob over 0.5, its bounding box's centre."""
        import cv2
        import torch

        n = len(frames)
        pick = np.linspace(0, n - 1, min(n, MEDIAN_FRAMES)).astype(int)
        median = np.median(np.asarray(frames)[pick], axis=0).astype(np.uint8)
        small = np.stack([cv2.resize(np.ascontiguousarray(f[:, :, ::-1]), self.WH) for f in frames]).astype(np.float32) / 255.0
        med = np.moveaxis(cv2.resize(np.ascontiguousarray(median[:, :, ::-1]), self.WH).astype(np.float32) / 255.0, -1, 0)
        sx, sy = SIZE[0] / self.WH[0], SIZE[1] / self.WH[1]
        out: list[Prediction] = []
        with torch.no_grad():
            for b in range(0, n, BATCH):
                batch = []
                for t in range(b, min(n, b + BATCH)):
                    seq = [np.moveaxis(small[max(0, j)], -1, 0) for j in range(t - self.SEQ + 1, t + 1)]
                    batch.append(np.concatenate([med, *seq], 0))
                with torch.autocast("cuda"):
                    hm, _, _ = self.net(frames=torch.from_numpy(np.stack(batch)).cuda())
                for h in hm.float().cpu().numpy()[:, 0]:
                    mask = (h > 0.5).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        out.append((False, 0.0, 0.0))
                        continue
                    x, y, w, hh = max((cv2.boundingRect(c) for c in cnts), key=lambda r: r[2] * r[3])
                    out.append((True, (x + w / 2) * sx, (y + hh / 2) * sy))
        return out


# ----------------------------------------------------------------- scoring


def counts(preds: list[Prediction], labels: list[tuple[int, float, float]]) -> dict[str, int]:
    """Positives, hits within each HIT_PX, detections far from the ball on positives, negatives and false
    detections on them; excluded frames (kind -1) count nowhere."""
    c = {"positives": 0, "negatives": 0, "false_on_negative": 0, "far_on_positive": 0, **{f"hit_{px:g}px": 0 for px in HIT_PX}}
    for (vis, x, y), (kind, lx, ly) in zip(preds, labels):
        if kind == 1:
            c["positives"] += 1
            d = math.hypot(x - lx, y - ly) if vis else math.inf
            for px in HIT_PX:
                c[f"hit_{px:g}px"] += d <= px
            c["far_on_positive"] += vis and d > HIT_PX[0]
        elif kind == 0:
            c["negatives"] += 1
            c["false_on_negative"] += vis
    return c


def rates(c: dict[str, int]) -> dict[str, float | None]:
    p, n = c["positives"], c["negatives"]
    tp = c[f"hit_{HIT_PX[0]:g}px"]
    fp = c["false_on_negative"] + c["far_on_positive"]
    return {"detection_rate": tp / p if p else None, "detection_rate_5px": c[f"hit_{HIT_PX[1]:g}px"] / p if p else None,
            "false_per_negative": c["false_on_negative"] / n if n else None,
            "f1": 2 * tp / (2 * tp + fp + (p - tp)) if p else None}


# ----------------------------------------------------------------- run


def parse_set(name: str, select: str) -> list[int] | str:
    if name in ("tracknet", "svt"):
        return [int(g) for g in select.split(",")]
    if name == "racketvision":
        return select
    raise SystemExit(f"unknown set {name}")


def command_run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("stage A needs the claimed GPU")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish" / "clips"
    publish.mkdir(parents=True, exist_ok=True)
    restored = set()
    checkpoints = os.environ.get("PS_CHECKPOINT_DIR")
    if checkpoints and Path(checkpoints).is_dir():
        for path in Path(checkpoints).glob("clip--*.json"):
            (publish / path.name[len("clip--"):]).write_bytes(path.read_bytes())
            restored.add(path.name[len("clip--"):-5])
    which = parse_set(args.set, args.select)
    if args.set == "racketvision":
        clips = racketvision_clips(Path(args.racketvision), str(which))
    else:
        clips = tracknet_clips(Path(args.tracknet), list(which))  # type: ignore[arg-type]
    clips = part_of(clips, args.part)
    if int(args.limit):
        clips = clips[: int(args.limit)]
    if not clips:
        raise SystemExit("no clip selected")
    zoo = Zoo(Path(args.wasb_code), Path(args.wasb_weights))
    balltrack = BallTrack(Path(args.balltrack_code), Path(args.balltrack))
    done = len([c for c in clips if safe(args.set, c["id"]) in restored])
    progress(done)
    seconds: dict[str, float] = {k: 0.0 for k in CANDIDATES}
    for clip in clips:
        key = safe(args.set, clip["id"])
        if key in restored:
            continue
        began = time.time()
        extra: dict[str, Any] = {}
        if args.set == "racketvision":
            frames, labels, extra = racketvision_load(Path(args.racketvision), clip, scratch)
        else:
            frames = tracknet_frames(Path(args.tracknet), clip)
            labels = tracknet_labels(clip)
            if args.set == "svt":
                frames, extra = svt_roundtrip(args, frames, scratch)
        row: dict[str, Any] = {"set": args.set, "id": clip["id"], "frames": len(frames), **extra, "candidates": {}}
        for name in CANDIDATES:
            t0 = time.time()
            preds = balltrack.run(frames) if name == "balltrack" else zoo.run(name, frames)
            dt = time.time() - t0
            seconds[name] += dt
            c = counts(preds, labels)
            row["candidates"][name] = {"counts": c, **rates(c), "seconds": round(dt, 2),
                                       "predictions": [[int(v), round(x, 1), round(y, 1)] for v, x, y in preds]}
        row["labels"] = [[k, round(x, 1), round(y, 1)] for k, x, y in labels]
        row["seconds"] = round(time.time() - began, 2)
        body = json.dumps(row, sort_keys=True).encode()
        (publish / f"{key}.json").write_bytes(body)
        if checkpoints:
            from experiments.jobs.monitor import save_checkpoint

            save_checkpoint(f"clip--{key}.json", body)
        done += 1
        progress(done)
        print(f"{clip['id']}: {row['seconds']} s " + " ".join(f"{n}={row['candidates'][n]['detection_rate']}" for n in CANDIDATES), flush=True)
    write_json(stage_dir() / "k5b-trackers.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "set": args.set, "select": args.select, "part": args.part, "clips": [c["id"] for c in clips],
        "restored": sorted(restored), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "on_cuda": {n: str(next(s["net"].parameters()).device) for n, s in zoo.settings.items()}
        | {"balltrack": str(next(balltrack.net.parameters()).device)},
        "balltrack_meta": balltrack.meta, "settings": ZOO, "wasb_code": args.wasb_code, "balltrack_code": args.balltrack_code, "size": SIZE, "hit_px": HIT_PX, "svt": {"crf": SVT_CRF, "preset": SVT_PRESET},
        "inputs": {k: {"path": v, "sha256": file_sha256(Path(v))} for k, v in
                   (("balltrack", args.balltrack), *((("tracknet", args.tracknet),) if args.tracknet else ()),
                    *((("racketvision", args.racketvision),) if args.racketvision else ()))},
        "weights": {w: file_sha256(Path(args.wasb_weights) / w) for w in sorted({s["weights"] for s in ZOO.values()})},
        "seconds_per_candidate": {k: round(v, 1) for k, v in seconds.items()},
    })
    return 0


def part_of(clips: list[dict[str, Any]], part: str) -> list[dict[str, Any]]:
    """Part ``i/n`` (1-based): the i-th of n contiguous, nearly equal chunks of the set's clips."""
    i, n = (int(v) for v in part.split("/"))
    if not 1 <= i <= n:
        raise SystemExit(f"bad part {part}")
    bounds = np.linspace(0, len(clips), n + 1).round().astype(int)
    return clips[bounds[i - 1]:bounds[i]]


def safe(set_name: str, clip_id: str) -> str:
    return f"{set_name}--{clip_id.replace('/', '__')}"


# ----------------------------------------------------------------- validate and report


def rows_from_tar(path: Path) -> list[dict[str, Any]]:
    out = []
    with tarfile.open(path) as tar:
        for m in sorted((m for m in tar.getmembers() if m.name.startswith("publish/clips/") and m.name.endswith(".json")), key=lambda m: m.name):
            handle = tar.extractfile(m)
            assert handle is not None
            out.append(json.loads(handle.read()))
    return out


def validate_rows(rows: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, bool]:
    return {
        "clips_scored": len(rows) > 0 and len(record["clips"]) > 0,
        "every_clip_published": sorted(r["id"] for r in rows) == sorted(record["clips"]),
        "every_candidate_scored": all(set(r["candidates"]) == set(CANDIDATES) for r in rows),
        "a_prediction_per_frame": all(len(c["predictions"]) == r["frames"] == len(r["labels"]) for r in rows for c in r["candidates"].values()),
        "labels_present": sum(sum(k == 1 for k, _, _ in r["labels"]) for r in rows) > 0,
        "models_on_cuda": all(v.startswith("cuda") for v in record["on_cuda"].values()),
        "some_candidate_finds_the_ball": all(any((c["detection_rate"] or 0) > 0.2 for c in r["candidates"].values())
                                             for r in rows if sum(k == 1 for k, _, _ in r["labels"]) >= 10),
        "predictions_inside_the_frame": all(not v or (-1 <= x <= SIZE[0] + 1 and -1 <= y <= SIZE[1] + 1)
                                            for r in rows for c in r["candidates"].values() for v, x, y in c["predictions"]),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    record = json.loads((stage / "k5b-trackers.json").read_text())
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch") / "publish" / "clips"
    rows = [json.loads(p.read_text()) for p in sorted(publish.glob("*.json"))] if publish.is_dir() else rows_from_tar(stage / "published.tar")
    checks = validate_rows(rows, record)
    report = {"passed": all(checks.values()), "checks": checks}
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def pooled(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    total: dict[str, int] = {}
    for r in rows:
        for k, v in r["candidates"][name]["counts"].items():
            total[k] = total.get(k, 0) + v
    return {**total, **rates(total)}


def group(row: dict[str, Any]) -> str:
    if row["set"] == "racketvision":
        return "racketvision:test"
    game = int(row["id"].split("/")[1][4:])
    return f"{row['set']}:{'8,9,10' if game >= 8 else '1-7'}"


def choose(table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The rule: highest mean detection rate over the sets held out for all; within 0.02, the SVT probe's, then
    fewer false detections."""
    score = {n: float(np.mean([table[s][n]["detection_rate"] for s in HELD_OUT_FOR_ALL])) for n in CANDIDATES}
    best = max(score.values())
    near = [n for n in CANDIDATES if score[n] >= best - 0.02]
    probe = table.get("svt:8,9,10")

    def key(n: str) -> tuple[float, float]:
        rate = probe[n]["detection_rate"] if probe else 0.0
        false = float(np.mean([table[s][n]["false_per_negative"] or 0.0 for s in HELD_OUT_FOR_ALL]))
        return (rate, -false)

    return {"mean_held_out": score, "within_0.02": near, "chosen": max(near, key=key)}


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in rows_from_tar(Path(p))]
    keys = [(r["set"], r["id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("a clip appears twice")
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for g in sorted({group(r) for r in rows}):
        sel = [r for r in rows if group(r) == g]
        table[g] = {n: pooled(sel, n) for n in CANDIDATES}
    report: dict[str, Any] = {"published": {p: file_sha256(Path(p)) for p in args.published}, "clips": len(rows), "table": table}
    if all(s in table for s in HELD_OUT_FOR_ALL):
        report["choice"] = choose(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "k5b-trackers-report.json", report)
    for g, t in table.items():
        print(g, {n: (round(v["detection_rate"] or 0, 3), round(v["false_per_negative"] or 0, 3)) for n, v in t.items()})
    print(json.dumps(report.get("choice"), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--set", required=True, choices=["tracknet", "svt", "racketvision"])
    run.add_argument("--select", required=True)
    run.add_argument("--limit", default="0")
    run.add_argument("--part", default="1/1")
    run.add_argument("--wasb-code", required=True)
    run.add_argument("--wasb-weights", required=True)
    run.add_argument("--balltrack", required=True)
    run.add_argument("--balltrack-code", required=True)
    run.add_argument("--tracknet")
    run.add_argument("--racketvision")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--ffmpeg", default="ffmpeg")
    sub.add_parser("validate")
    rep = sub.add_parser("report")
    rep.add_argument("--published", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
