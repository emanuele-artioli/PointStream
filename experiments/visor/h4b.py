"""PLAN step H4b: a bank of reference views for handled objects, as an oracle (docs/experiments.md, 2026-10-13 H4b).

    python -m experiments.visor.h4b run --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --h1-chains JSON --video V.MP4 ... --items all|ID,ID --frames N
    python -m experiments.visor.h4b validate
    python -m experiments.visor.h4b report --published TAR ... --h1-chains JSON --out DIR

Every object track of an item (H1's: each object label's union of fill-mask instances per frame) is replayed
through four coders at H1's two hold thresholds. A coder keeps views (frames it has sent); a view predicts frame t
through a homography when H1's hold test passes (``h1.ReferenceChain``'s: at most 20% of the mask uncovered, PSNR
on the covered pixels at or above the threshold). Views are reached *chained* (H1's frame-to-frame homographies)
or *direct* (SIFT inside the masks, RANSAC). ``single`` is H1's chain and must reproduce H1's recorded counts;
``single+direct``, ``bank-4`` and ``bank`` (every view so far) add direct registration and more views.

Each item is published as ``publish/items/<id>.json`` (and checkpointed); ``report`` applies ``DECISION``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h1 import FILL, HOLD_PSNR, MAX_UNCOVERED, apply_h, fit_models, psnr, sample_map

DIRECT_RATIO = 0.75
DIRECT_RANSAC_PX = 3.0
DIRECT_MIN_INLIERS = 15
DIRECT_TRIES = 4
DILATE_PX = 8
SIFT_FEATURES = 500
SCREEN_STRIDE = 4  # bank candidates are screened on every 4th pixel of the mask in each direction
SCREEN_KEEP = 4
CODERS = {"single": (1, False), "single+direct": (1, True), "bank-4": (4, True), "bank": (None, True)}
DECISION: dict[str, Any] = {
    "threshold_db": 30.0,
    "held_share_min": 0.70,
    "median_interval_s_min": 0.5,
    "bank_over_single_direct_max": 0.5,
    "unit": "item; means over the 34 items of per-item values pooled over the item's handled tracks",
}


# ----------------------------------------------------------------- coders

@dataclass
class View:
    frame: int
    to_current: np.ndarray | None  # chained homography view -> previous frame
    last_used: int


@dataclass
class Coder:
    """Views kept up to ``capacity`` (None: all), reached by chain and, with ``direct``, by registration."""

    name: str
    threshold: float
    capacity: int | None
    direct: bool
    views: list[View] = field(default_factory=list)
    held: int = 0
    frames: int = 0
    new_views: list[int] = field(default_factory=list)
    reached: dict[str, int] = field(default_factory=lambda: {"chained": 0, "direct": 0})

    def _advance_chains(self, h_step: np.ndarray | None, keep: View | None = None) -> None:
        for v in self.views:
            if v is keep:
                continue
            v.to_current = None if (v.to_current is None or h_step is None) else h_step @ v.to_current

    def _add(self, t: int, h_step: np.ndarray | None) -> None:
        self._advance_chains(h_step)
        if self.capacity is not None and len(self.views) >= self.capacity:
            self.views.remove(min(self.views, key=lambda v: (v.last_used, v.frame)))
        self.views.append(View(t, np.eye(3), t))
        self.new_views.append(t)
        self.frames += 1

    def step(self, t: int, h_step: np.ndarray | None, scene: "TrackScene") -> bool:
        """Code frame ``t`` (``h_step``: the homography t-1 -> t, None where there is none); True when held."""
        candidates: list[tuple[View, np.ndarray, str]] = []
        for v in self.views:
            if v.to_current is not None and h_step is not None:
                candidates.append((v, h_step @ v.to_current, "chained"))
        if self.direct and self.views:
            ranked = sorted(((scene.direct(v.frame, t), v) for v in self.views), key=lambda x: -x[0][1])
            for (h, inliers), v in ranked[:DIRECT_TRIES]:
                if h is not None:
                    candidates.append((v, h, "direct"))
        if len(candidates) > SCREEN_KEEP:
            scored = [(scene.evaluate(v.frame, h, t, SCREEN_STRIDE), i) for i, (v, h, _) in enumerate(candidates)]
            keep = sorted(scored, key=lambda x: -(x[0][0] if x[0][0] is not None and x[0][1] <= MAX_UNCOVERED + 0.05 else -1e9))
            candidates = [candidates[i] for _, i in keep[:SCREEN_KEEP]]
        best: tuple[float, View, np.ndarray, str] | None = None
        for v, h, way in candidates:
            value, uncovered = scene.evaluate(v.frame, h, t)
            if value is not None and value >= self.threshold and uncovered <= MAX_UNCOVERED:
                if best is None or value > best[0]:
                    best = (value, v, h, way)
        if best is None:
            self._add(t, h_step)
            return False
        _, view, h, way = best
        self._advance_chains(h_step, keep=view)
        view.to_current, view.last_used = h, t
        self.held += 1
        self.frames += 1
        self.reached[way] += 1
        return True

    def summary(self, last: int, fps: float) -> dict[str, Any]:
        bounds = self.new_views + [last + 1]
        return {"frames": self.frames, "held": self.held, "new_views": list(self.new_views),
                "intervals_s": [round((b - a) / fps, 4) for a, b in zip(bounds, bounds[1:])], "reached": dict(self.reached)}


class TrackScene:
    """One object track's frames and masks, with H1's hold test and cached direct registrations."""

    def __init__(self, frames: Any, grays: list[np.ndarray], masks: dict[int, np.ndarray]) -> None:
        self.frames = frames
        self.grays = grays
        self.masks = masks
        self._features: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._direct: dict[tuple[int, int], tuple[np.ndarray | None, int]] = {}
        self._pixels: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def pixels(self, t: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
        key = (t, stride)
        if key not in self._pixels:
            ys, xs = np.nonzero(self.masks[t])
            if stride > 1:
                keep = (ys % stride == 0) & (xs % stride == 0)
                ys, xs = ys[keep], xs[keep]
            self._pixels[key] = (ys, xs)
        return self._pixels[key]

    def evaluate(self, r: int, h: np.ndarray, t: int, stride: int = 1) -> tuple[float | None, float]:
        """H1's hold test of view ``r`` warped by ``h`` (r -> t) onto the mask at ``t``: (PSNR, uncovered share)."""
        ys, xs = self.pixels(t, stride)
        if len(ys) == 0:
            return None, 1.0
        src = apply_h(np.linalg.inv(h), np.c_[xs, ys].astype(np.float64))
        covered = sample_map(src, self.masks[r], nearest=True)
        uncovered = 1.0 - float(covered.mean())
        if not covered.any():
            return None, uncovered
        pred = sample_map(src[covered], self.frames[r]).astype(np.float64)
        sse = float(((pred - self.frames[t][ys[covered], xs[covered]].astype(np.float64)) ** 2).sum())
        return psnr(sse, int(covered.sum())), uncovered

    def features(self, t: int) -> tuple[np.ndarray, np.ndarray | None]:
        import cv2

        if t not in self._features:
            mask = cv2.dilate(self.masks[t].astype(np.uint8), np.ones((2 * DILATE_PX + 1,) * 2, np.uint8))
            sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)  # type: ignore[attr-defined]
            keypoints, descriptors = sift.detectAndCompute(self.grays[t], mask)
            self._features[t] = (np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2), descriptors)
        return self._features[t]

    def direct(self, r: int, t: int) -> tuple[np.ndarray | None, int]:
        """A homography r -> t from SIFT matches inside the two masks (None below ``DIRECT_MIN_INLIERS``), and its inliers."""
        import cv2

        if (r, t) not in self._direct:
            pr, dr = self.features(r)
            pt, dt = self.features(t)
            out: tuple[np.ndarray | None, int] = (None, 0)
            if dr is not None and dt is not None and len(dr) >= 2 and len(dt) >= 2:
                matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dr, dt, k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < DIRECT_RATIO * m[1].distance]
                if len(good) >= DIRECT_MIN_INLIERS:
                    src = pr[[m.queryIdx for m in good]]
                    dst = pt[[m.trainIdx for m in good]]
                    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, DIRECT_RANSAC_PX)
                    count = int(inl.sum()) if inl is not None else 0
                    if h is not None and count >= DIRECT_MIN_INLIERS and abs(np.linalg.det(h)) > 1e-8:
                        out = (h, count)
                    else:
                        out = (None, count)
            self._direct[(r, t)] = out
        return self._direct[(r, t)]


# ----------------------------------------------------------------- one item

def object_tracks(clip: Any, count: int) -> dict[str, dict[int, np.ndarray]]:
    """H1's object tracks: per frame, each object label's union of fill-mask instances (hands excluded)."""
    from src.segmentation import visor

    tracks: dict[str, dict[int, np.ndarray]] = {}
    for t in range(count):
        objects: dict[str, np.ndarray] = {}
        for inst in clip.frames[t]:
            if inst.class_name in visor.HANDS:
                continue
            key = visor.object_key(inst.class_name, inst.label)
            objects[key] = objects.get(key, np.zeros(visor.FRAME_SIZE, bool)) | inst.mask()
        for key, mask in objects.items():  # empty masks stay in the track, as H1's (they break its chain)
            tracks.setdefault(key, {})[t] = mask
    return tracks


def code_tracks(frames: Any, tracks: dict[str, dict[int, np.ndarray]], handled: set[str], fps: float,
                count: int) -> dict[str, Any]:
    """Every coder at both thresholds over every track (flow and fits as H1's)."""
    import cv2

    from src.segmentation import visor

    shape = visor.FRAME_SIZE
    grays = [cv2.cvtColor(np.ascontiguousarray(frames[t]), cv2.COLOR_RGB2GRAY) for t in range(count)]
    halves = [cv2.resize(g, (shape[1] // 2, shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    scenes = {key: TrackScene(frames, grays, masks) for key, masks in tracks.items()}
    coders = {key: [Coder(name, th, cap, direct) for th in HOLD_PSNR for name, (cap, direct) in CODERS.items()]
              for key in tracks}
    for t in range(count):
        flow = None
        if t > 0:
            small = dis.calc(halves[t], halves[t - 1], None)
            flow = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
        for key, masks in tracks.items():
            if t not in masks or not masks[t].any():
                continue
            h_step = None
            if flow is not None and t - 1 in masks:
                h_step = fit_models(flow, masks[t]).get("homography")
            for coder in coders[key]:
                if not coder.views:
                    coder._add(t, None)
                else:
                    coder.step(t, h_step, scenes[key])
    return {key: {"kind": "handled_object" if key in handled else "other_object", "frames": sum(bool(m.any()) for m in masks.values()),
                  "frames_in_track": len(masks), "first": min(masks), "last": max(masks),
                  "coders": {f"{c.name}@{c.threshold:g}": c.summary(max(masks), fps) for c in coders[key]}}
            for key, masks in tracks.items()}


def process_item(args: dict[str, Any]) -> dict[str, Any]:
    """Decode, load masks and contacts, code every track; runs in its own process."""
    import cv2

    from experiments.visor.b1b import load_inputs
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h1 import contact_objects, decode_window

    cv2.setNumThreads(int(args["threads"]))
    item = args["item"]
    count = int(args["frames"])
    began = time.time()
    target = Path(args["scratch"]) / "frames" / f"{item['id']}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    decoded = decode_window(args["video"], item, count, args["archive"], str(target), int(args["threads"]))
    clips, mask_info = load_mask_sets(item, count, {FILL: args["masks"]}, {FILL: args["mask_record"]}, Path(args["archive"]))
    _, sparse, mapping = load_inputs(item, Path(args["archive"]))
    contact = contact_objects(item, sparse, mapping)
    frames = np.load(target, mmap_mode="r")
    tracks = object_tracks(clips[FILL], count)
    coded = code_tracks(frames, tracks, set(contact["handled"]), float(item["fps"]), count)
    target.unlink()
    return {"id": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": count, "decode": decoded,
            "mask_sets": mask_info, "contact": contact, "tracks": coded, "seconds": round(time.time() - began, 1)}


def command_run(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import select_items

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    videos = {Path(v).stem: v for v in args.video}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish" / "items"
    publish.mkdir(parents=True, exist_ok=True)
    restored = []
    checkpoints = Path(os.environ.get("PS_CHECKPOINT_DIR") or "")
    if os.environ.get("PS_CHECKPOINT_DIR") and checkpoints.is_dir():
        for path in sorted(checkpoints.glob("item--*.json")):
            (publish / path.name[6:]).write_bytes(path.read_bytes())
            restored.append(path.name[6:-5])
    todo = [i for i in items if i["id"] not in restored]
    workers = max(1, min(int(args.workers), len(todo)))
    threads = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1) // workers)
    jobs = [{"item": i, "frames": args.frames, "video": videos[i["video"]], "archive": args.archive, "masks": args.masks,
             "mask_record": args.mask_record, "scratch": str(scratch), "threads": threads} for i in todo]
    done = len(items) - len(todo)
    progress(done)
    # Decode in fresh processes (B2: PyAV stalled in a process that had loaded torchvision).
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(process_item, jobs):
            body = json.dumps(row, sort_keys=True, default=str).encode()
            (publish / f"{row['id']}.json").write_bytes(body)
            if os.environ.get("PS_CHECKPOINT_DIR"):
                from experiments.jobs.monitor import save_checkpoint

                save_checkpoint(f"item--{row['id']}.json", body)
            done += 1
            progress(done)
            print(f"{row['id']}: {len(row['tracks'])} tracks, {row['seconds']} s", flush=True)
    write_json(stage_dir() / "h4b.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "items": [i["id"] for i in items], "restored": restored,
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set)), "name": eval_set["name"]},
        "h1_chains_sha256": file_sha256(Path(args.h1_chains)),
        "settings": {"frames": int(args.frames), "hold_psnr": HOLD_PSNR, "max_uncovered": MAX_UNCOVERED, "coders": CODERS,
                     "direct": {"ratio": DIRECT_RATIO, "ransac_px": DIRECT_RANSAC_PX, "min_inliers": DIRECT_MIN_INLIERS,
                                "tries": DIRECT_TRIES, "dilate_px": DILATE_PX, "sift_features": SIFT_FEATURES},
                     "screen": {"stride": SCREEN_STRIDE, "keep": SCREEN_KEEP}, "workers": workers, "threads": threads},
    })
    return 0


# ----------------------------------------------------------------- validate

def load_items(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def items_from_tar(path: Path) -> list[dict[str, Any]]:
    with tarfile.open(path) as tar:
        out = []
        for member in sorted((m for m in tar.getmembers() if m.name.startswith("publish/items/") and m.name.endswith(".json")),
                             key=lambda m: m.name):
            handle = tar.extractfile(member)
            assert handle is not None
            out.append(json.loads(handle.read()))
    return out


def h1_agreement(rows: list[dict[str, Any]], chains: dict[str, Any]) -> dict[str, Any]:
    """``single`` against H1's recorded chains, per object track and threshold: held counts and references."""
    tracks = exact = close = 0
    worst = 0.0
    for row in rows:
        recorded = chains["items"].get(row["id"], {})
        for key, info in row["tracks"].items():
            if key not in recorded:
                continue
            for th in HOLD_PSNR:
                mine = info["coders"][f"single@{th:g}"]
                theirs = recorded[key][f"{th:g}"]
                tracks += 1
                gap = abs(mine["held"] - theirs["held"]) / max(info["frames"], 1)
                worst = max(worst, gap)
                exact += mine["held"] == theirs["held"] and len(mine["new_views"]) == theirs["references"]
                close += gap <= 0.02
    return {"track_thresholds": tracks, "exact_share": exact / tracks if tracks else 0.0,
            "within_2pct_share": close / tracks if tracks else 0.0, "worst_gap": worst}


def validate_rows(rows: list[dict[str, Any]], record: dict[str, Any], chains: dict[str, Any] | None) -> dict[str, bool]:
    accounted = all(c["held"] + len(c["new_views"]) == c["frames"] == info["frames"]
                    for row in rows for info in row["tracks"].values() for c in info["coders"].values())
    checks = {
        "every_item_published": sorted(r["id"] for r in rows) == sorted(record["items"]),
        "jpeg_gate_holds": all(c["holds"] for r in rows for c in r["decode"]["jpeg_gate"]),
        "fill_masks_match_their_record": all(r["mask_sets"][FILL]["mask_sha256"] == r["mask_sets"][FILL]["record_mask_sha256"]
                                             and r["mask_sets"][FILL]["video_indices_match"] for r in rows),
        "every_frame_held_or_a_new_view_once": accounted,
        "some_handled_track": any(i["kind"] == "handled_object" for r in rows for i in r["tracks"].values()),
    }
    if chains is not None and int(record["settings"]["frames"]) == 240:  # H1's chains cover whole windows
        agreement = h1_agreement(rows, chains)
        checks["single_reproduces_h1_within_2pct_on_95pct"] = agreement["track_thresholds"] > 0 and agreement["within_2pct_share"] >= 0.95
    return checks


def validate_result(stage: Path, chains_path: str | None = None) -> dict[str, bool]:
    record = json.loads((stage / "h4b.json").read_text())
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch") / "publish" / "items"
    rows = load_items(publish) if publish.is_dir() else items_from_tar(stage / "published.tar")
    chains = json.loads(Path(chains_path).read_text()) if chains_path else None
    return validate_rows(rows, record, chains)


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir(), args.h1_chains)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report

def item_values(row: dict[str, Any], coder: str, kind: str) -> dict[str, float] | None:
    infos = [i for i in row["tracks"].values() if i["kind"] == kind]
    if not infos:
        return None
    cs = [i["coders"][coder] for i in infos]
    frames = sum(c["frames"] for c in cs)
    intervals = [v for c in cs for v in c["intervals_s"]]
    views = sum(len(c["new_views"]) for c in cs)
    return {"held_share": sum(c["held"] for c in cs) / frames, "median_interval_s": float(np.median(intervals)),
            "new_views_per_s": views / (row["frames"] / row["fps"]), "new_views": views, "frames": frames,
            "direct_share": sum(c["reached"]["direct"] for c in cs) / max(sum(c["held"] for c in cs), 1)}


def summarize(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for th in HOLD_PSNR:
        for name in CODERS:
            coder = f"{name}@{th:g}"
            values = [v for r in rows if (v := item_values(r, coder, kind))]
            if values:
                out[coder] = {"items": len(values), **{k: float(np.mean([v[k] for v in values]))
                                                       for k in ("held_share", "median_interval_s", "new_views_per_s",
                                                                 "direct_share")}}
        ratios = []
        for r in rows:
            bank, single = item_values(r, f"bank@{th:g}", kind), item_values(r, f"single+direct@{th:g}", kind)
            if bank and single and single["new_views"]:
                ratios.append(bank["new_views"] / single["new_views"])
        out[f"bank_over_single_direct@{th:g}"] = float(np.mean(ratios)) if ratios else None
    return out


def decide(summary: dict[str, Any], threshold: float = 30.0) -> dict[str, Any]:
    bank = summary[f"bank@{threshold:g}"]
    ratio = summary[f"bank_over_single_direct@{threshold:g}"]
    checks = {"a_held_share": bank["held_share"] >= DECISION["held_share_min"],
              "b_median_interval": bank["median_interval_s"] >= DECISION["median_interval_s_min"],
              "c_bank_helps": ratio is not None and ratio <= DECISION["bank_over_single_direct_max"]}
    return {"threshold_db": threshold, "checks": checks, "passes": all(checks.values())}


def verdict(summary: dict[str, Any]) -> dict[str, Any]:
    at30, at3375 = decide(summary, 30.0), decide(summary, 33.75)
    if at30["passes"] and at3375["passes"]:
        word = "passes"
    elif at30["passes"] or (at30["checks"]["a_held_share"] and at30["checks"]["b_median_interval"]):
        word = "unclear"
    else:
        word = "fails"
    return {"verdict": word, "at_30db": at30, "at_33.75db": at3375}


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in items_from_tar(Path(p))]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("an item appears twice")
    chains = json.loads(Path(args.h1_chains).read_text())
    handled = summarize(rows, "handled_object")
    out = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published}, "items": len(rows),
           "handled_object": handled, "other_object": summarize(rows, "other_object"),
           "h1_agreement": h1_agreement(rows, chains), "verdict": verdict(handled)}
    write_json(Path(args.out) / "h4b-report.json", out)
    print(json.dumps(out["verdict"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--eval-set", required=True)
    run.add_argument("--archive", required=True)
    run.add_argument("--masks", required=True, help="the fill mask set's directory")
    run.add_argument("--mask-record", required=True)
    run.add_argument("--h1-chains", required=True)
    run.add_argument("--video", action="append", required=True)
    run.add_argument("--items", required=True)
    run.add_argument("--frames", default="240")
    run.add_argument("--workers", default="8")
    val = sub.add_parser("validate")
    val.add_argument("--h1-chains")
    report = sub.add_parser("report")
    report.add_argument("--published", action="append", required=True)
    report.add_argument("--h1-chains", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
