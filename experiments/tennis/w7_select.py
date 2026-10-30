"""W7 CPU stage: Tier 1 inventory, mechanical window selection and the overlap screen.

Rule: docs/experiments/w7-untouched.md. CPU only; no model, no quality score. Subcommands:

* ``select``  inventory of the Alcaraz and Medvedev archive clips, exclusions, the seeded Medvedev draw and the
  first 240-frame interval the frozen registration rule of the 2026-10-25 windows admits (>= 228 of 240 frames
  with K8b ``fits`` and a finite, nonsingular homography; k6b_confirm.gate).
* ``audit``   the 2026-10-24 overlap screen (exact native yuv420p SHA256 matches, 64x36 grayscale RMSE <= 3) of
  the selected windows against every old clip, fit clip and confirmation clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.k6b_confirm import FROZEN

SEED = 20261030
WINDOW = 240
NEED = 228
MATCHES = ("alcaraz_hurkacz", "medvedev_struff")
TAKE = 3
ARCHIVE = "archive/pre-reset-2026-10-05/tennis_games"
#: Step-5 fit/evaluation clips and the unused-manifest clips (k6b-players.md, "what unused does and does not establish").
FIT = {"alcaraz_hurkacz": ("002", "006", "011", "036", "045"), "medvedev_struff": ("006", "008", "033")}
UNUSED_MANIFEST = {"alcaraz_hurkacz": (), "medvedev_struff": ("007", "016", "023", "028", "034")}
CONFIRM = tuple(sorted(f[0] for f in FROZEN))


def exclusions(old_clips: set[str]) -> dict[str, list[str]]:
    """Clip id -> reasons it is out. `old_clips` are the clips of the 57 old manifest windows (scope.json)."""
    out: dict[str, list[str]] = {}
    for clip in old_clips:
        out.setdefault(clip, []).append("old manifest window")
    for match, ids in FIT.items():
        for i in ids:
            out.setdefault(f"{match}/{i}", []).append("step-5 fit/evaluation clip")
    for match, ids in UNUSED_MANIFEST.items():
        for i in ids:
            out.setdefault(f"{match}/{i}", []).append("unused old-manifest clip")
    for clip in CONFIRM:
        out.setdefault(clip, []).append("2026-10-25 confirmation window")
    return out


def valid_frame(row: Any) -> bool:
    """k6b_confirm.gate's per-frame test: K8b fits is true and the homography is finite and nonsingular."""
    if not row or row.get("fits") is not True:
        return False
    try:
        h = np.asarray(row.get("homography"), dtype=float)
        return bool(h.shape == (3, 3) and np.isfinite(h).all() and np.linalg.matrix_rank(h) == 3)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return False


def first_window(rows: list[Any], length: int = WINDOW, need: int = NEED) -> dict[str, Any]:
    """Earliest start s (stride 1) whose [s, s+length) has at least `need` valid frames; None when no start does.

    Rows are the clip's K8b registration rows indexed by native decoded frame (K8b decodes each clip from its
    start, no seeks). The best interval's count is returned too, so a clip that fails is reported with its margin."""
    flags = np.array([valid_frame(r) for r in rows], dtype=np.int64)
    n = len(flags)
    if n < length:
        return {"start": None, "valid": None, "best_valid": None, "best_start": None, "frames": n}
    csum = np.concatenate([[0], np.cumsum(flags)])
    counts = csum[length:] - csum[:-length]  # counts[s] = valid frames in [s, s+length)
    ok = np.flatnonzero(counts >= need)
    best = int(counts.argmax())
    return {"start": int(ok[0]) if len(ok) else None, "valid": int(counts[ok[0]]) if len(ok) else None,
            "best_valid": int(counts[best]), "best_start": best, "frames": n, "valid_frames_total": int(flags.sum())}


def draw(ids: list[str], seed: int = SEED, size: int = TAKE) -> list[str]:
    """Seeded draw without replacement over the sorted list (numpy default_rng, as written in the W7 entry)."""
    ordered = sorted(ids)
    if len(ordered) <= size:
        return ordered
    return [str(x) for x in np.random.default_rng(seed).choice(ordered, size=size, replace=False)]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def select(clips: dict[str, dict[str, Any]], excluded: dict[str, list[str]], rows: dict[str, list[Any]]) -> dict[str, Any]:
    """Pure selection. `clips`: id -> {frames, fps, sha256}; `rows`: id -> K8b registration rows."""
    result: dict[str, Any] = {"seed": SEED, "rule": f">= {NEED} of {WINDOW} frames fits and finite nonsingular H, earliest start",
                              "matches": {}}
    for match in MATCHES:
        ids = sorted(c for c in clips if c.startswith(match + "/"))
        admissible = [c for c in ids if clips[c]["frames"] >= WINDOW and c not in excluded]
        chosen = admissible[:TAKE] if match == "alcaraz_hurkacz" else draw(admissible)
        picks: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for c in chosen:
            w = first_window(rows[c])
            (picks if w["start"] is not None else failed).append({"clip": c, **w, **clips[c]})
        result["matches"][match] = {
            "clips_total": len(ids), "admissible": admissible,
            "excluded": {c: excluded[c] for c in ids if c in excluded},
            "short": {c: clips[c]["frames"] for c in ids if clips[c]["frames"] < WINDOW and c not in excluded},
            "method": "first three in ID order" if match == "alcaraz_hurkacz"
                      else f"numpy.random.default_rng({SEED}).choice(sorted(admissible), size={TAKE}, replace=False)",
            "chosen_order": chosen, "selected": picks, "failed_registration": failed}
    return result


def _inventory(root: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Archive clips of the two matches: frames/fps from the K8b report, checked against ffprobe and the file hash."""
    out: dict[str, dict[str, Any]] = {}
    for clip in report["clips"]:
        if clip["id"].split("/")[0] not in MATCHES:
            continue
        path = root / ARCHIVE / clip["path"]
        probe = json.loads(subprocess.check_output(
            ["/opt/local/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
             "stream=width,height,nb_frames,nb_read_frames,r_frame_rate", "-of", "json", str(path)], timeout=300))["streams"][0]
        digest = file_sha256(path)
        if digest != clip["sha256"]:
            raise ValueError(f"{clip['id']}: file differs from the K8b report")
        n = int(probe["nb_read_frames"])
        if n != clip["frames"] or int(probe["nb_frames"]) != n:
            raise ValueError(f"{clip['id']}: {n} decoded frames, report has {clip['frames']}")
        out[clip["id"]] = {"frames": n, "fps": clip["fps"], "sha256": digest,
                           "size": f"{probe['width']}x{probe['height']}", "r_frame_rate": probe["r_frame_rate"]}
    return out


def cmd_select(args: argparse.Namespace) -> None:
    from tools.datasets.k6b_recovery import SCOPE, sha

    root = Path(args.datasets)
    scope = root / "pointstream-data/jobs/fleet/inbox/20261024T083117Z-2e644b08/full/scope.json"
    if sha(scope) != SCOPE:
        raise ValueError("changed old input inventory")
    inventory = json.loads(scope.read_text())["inventory"]
    old = {v["clip"] for v in inventory.values()}
    report_path = root / "pointstream-data/tennis/k8b-2026-10-20/review/k8b-report.json"
    report = json.loads(report_path.read_text())
    clips = _inventory(root, report)
    rows = {c["id"]: c["registration"] for c in report["clips"] if c["id"] in clips}
    out = select(clips, exclusions(old), rows)
    out["inputs"] = {"scope_sha256": SCOPE, "k8b_report": str(report_path), "k8b_report_sha256": sha(report_path),
                     "old_manifest_windows": len(inventory), "old_manifest_clips": len(old)}
    out["inventory"] = {c: clips[c] for c in sorted(clips)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for match, m in out["matches"].items():
        print(match, "admissible", m["admissible"], "selected", [(s["clip"], s["start"], s["valid"]) for s in m["selected"]],
              "failed", [(s["clip"], s["best_valid"]) for s in m["failed_registration"]])


# ----------------------------------------------------------------------------------------------- overlap screen


def _decode(ffmpeg: str, path: Path, start: int, frames: int, tmp: Path) -> tuple[list[str], bytes, list[str]]:
    """Native yuv420p frame hashes and 64x36 gray descriptors of [start, start+frames), the 2026-10-24 command."""
    from tools.datasets.k6b_recovery import parse_hashes

    desc = tmp / "descriptor.bin"
    filt = (f"trim=start_frame={start}:end_frame={start + frames},format=yuv420p,split=2[full][small];"
            "[small]scale=64:36:flags=area,format=gray[desc]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-filter_complex_threads", "1", "-i", str(path),
               "-filter_complex", filt, "-map", "[full]", "-an", "-fps_mode", "passthrough", "-f", "framehash", "-hash",
               "sha256", "pipe:1", "-map", "[desc]", "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-y", str(desc)]
    hashes = parse_hashes(subprocess.check_output(command, text=True, timeout=600))
    data = desc.read_bytes()
    if len(hashes) != frames or len(data) != frames * 64 * 36:
        raise ValueError(f"decoded frame-count mismatch: {path}")
    return hashes, data, command


def cmd_audit(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tools.datasets.k6b_recovery import screen, sha

    root = Path(args.datasets)
    selection = json.loads(Path(args.selection).read_text())
    ffmpeg = "/opt/local/bin/ffmpeg"
    tmp = Path(args.scratch)
    tmp.mkdir(parents=True, exist_ok=True)
    # Old collections of 2026-10-24 (read-only): the 49 old clips, decoded whole.
    base = root / "pointstream-data/tennis"
    folders = ["k6b-recovery-20261024-v2-full"] + [f"k6b-recovery-20261024-v3-part{i}" for i in range(4)]
    records: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    receipts = []
    offset = 0
    for name in folders:
        folder = base / name
        report = folder / ("audit.json" if (folder / "audit.json").exists() else "partial.json")
        x = json.loads(report.read_text())
        a = np.fromfile(folder / "descriptors.bin", np.uint8).reshape(-1, 64 * 36)
        if len(a) != sum(r["frames"] for r in x["records"]):
            raise ValueError(f"incomplete descriptors: {name}")
        receipts.append({"folder": name, "report_sha256": sha(report), "descriptors_sha256": sha(folder / "descriptors.bin")})
        for r in x["records"]:
            if r["role"] == "old":  # the 12 confirmation windows of that audit are replaced by their whole clips below
                records.append({**r, "offset": offset})
                arrays.append(a[r["offset"]:r["offset"] + r["frames"]])
                offset += r["frames"]
    old_clips = {r["clip"] for r in records}
    if len(old_clips) != 49:
        raise ValueError(f"expected 49 old clips, got {len(old_clips)}")
    # Whole confirmation clips and the selected windows are decoded now.
    jobs: list[tuple[str, str, int, int, str]] = []  # clip, role, start, frames, sha
    meta = selection["inventory"]
    for clip in CONFIRM:
        jobs.append((clip, "old", 0, meta[clip]["frames"], meta[clip]["sha256"]))
    selected = [s for m in selection["matches"].values() for s in m["selected"]]
    for s in selected:
        if s["clip"] in old_clips or s["clip"] in CONFIRM:
            raise ValueError("selected clip is in an old collection")
        jobs.append((s["clip"], "candidate", s["start"], WINDOW, s["sha256"]))

    def run(job: tuple[str, str, int, int, str]) -> dict[str, Any]:
        clip, role, start, frames, digest = job
        path = root / ARCHIVE / (clip + ".mp4")
        if file_sha256(path) != digest:
            raise ValueError(f"changed source {clip}")
        work = tmp / clip.replace("/", "__")
        work.mkdir(exist_ok=True)
        hashes, data, command = _decode(ffmpeg, path, start, frames, work)
        return {"clip": clip, "role": role, "start": start, "frames": frames, "sha256": digest, "size": "1920x1080",
                "hash_mode": "native", "fps": meta[clip]["fps"], "hashes": hashes, "command": command, "_data": data}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        new = list(pool.map(run, jobs))
    for r in new:
        data = np.frombuffer(r.pop("_data"), np.uint8).reshape(-1, 64 * 36)
        records.append({**r, "offset": offset})
        arrays.append(data)
        offset += len(data)
    merged = tmp / "merged"
    merged.mkdir(exist_ok=True)
    np.concatenate(arrays).tofile(merged / "descriptors.bin")
    (merged / "audit.json").write_text(json.dumps({"records": records, "complete": True, "smoke": False,
                                                    "descriptors_sha256": sha(merged / "descriptors.bin")}))
    result = screen(merged)
    result["old_collections"] = receipts
    result["selected"] = [{"clip": s["clip"], "start": s["start"]} for s in selected]
    result["screened_against"] = {"old_clips": sorted(old_clips), "confirmation_clips_whole": list(CONFIRM)}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({"candidates": [{k: c[k] for k in ("clip", "start", "flagged_frames", "minimum_rmse")}
                                     for c in result["candidates"]],
                      "exact_pairs": len(result["exact_pairs"]), "clips": result["clips"],
                      "decoded_frames": result["decoded_frames"]}, indent=1))


def cmd_manifest(args: argparse.Namespace) -> None:
    """windows.json: the selected windows with the sha256 of K4's conversion of each (k4b_features.convert_command, 240 frames)."""
    from concurrent.futures import ThreadPoolExecutor

    from experiments.tennis import window_manifest as wm
    from experiments.tennis.k4b_features import convert_command

    root = Path(args.datasets)
    selection = json.loads(Path(args.selection).read_text())
    windows = wm.from_selection(selection)
    tmp = Path(args.scratch)
    tmp.mkdir(parents=True, exist_ok=True)

    def one(w: Any) -> tuple[str, str]:
        raw = tmp / f"{w.name}.yuv"
        subprocess.run(convert_command("/opt/local/bin/ffmpeg", root / ARCHIVE / (w.clip_id + ".mp4"), raw, w.start, w.frames),
                       check=True, capture_output=True, timeout=1800)
        if raw.stat().st_size != w.frames * 1920 * 1080 * 3 // 2:
            raise ValueError(f"{w.name}: wrong converted size")
        digest = file_sha256(raw)
        raw.unlink()
        return w.clip_id, digest

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        yuv = dict(pool.map(one, windows))
    wm.dump(wm.from_selection(selection, yuv), Path(args.out), selection_sha256=file_sha256(Path(args.selection)), seed=SEED)
    print(json.dumps(yuv, indent=1))


def cmd_stage(args: argparse.Namespace) -> None:
    """Copy the selected clips to `<dest>/sources/` (the fleet data root; sha256 checked after the copy). Not run by the W7 CPU stage."""
    import shutil

    from experiments.tennis import window_manifest as wm

    root = Path(args.datasets)
    dest = Path(args.dest) / "sources"
    dest.mkdir(parents=True, exist_ok=False)
    for w in wm.load(Path(args.windows)):
        target = dest / w.source_file
        shutil.copyfile(root / ARCHIVE / (w.clip_id + ".mp4"), target)
        if file_sha256(target) != w.source_sha256:
            raise ValueError(f"{w.name}: copy differs from the archive")
    print("staged", dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--datasets", default="/home/itec/emanuele/Datasets")
    s.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--datasets", default="/home/itec/emanuele/Datasets")
    a.add_argument("--selection", required=True)
    a.add_argument("--scratch", required=True)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--datasets", default="/home/itec/emanuele/Datasets")
    m.add_argument("--selection", required=True)
    m.add_argument("--scratch", required=True)
    m.add_argument("--workers", type=int, default=6)
    m.add_argument("--out", required=True)
    g = sub.add_parser("stage")
    g.add_argument("--datasets", default="/home/itec/emanuele/Datasets")
    g.add_argument("--windows", required=True)
    g.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    {"select": cmd_select, "audit": cmd_audit, "manifest": cmd_manifest, "stage": cmd_stage}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
