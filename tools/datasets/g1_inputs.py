"""Clips and staged inputs for PLAN step G1 (camera motion and coverage audit).

    python tools/datasets/g1_inputs.py select --datasets ROOT --eval-set JSON --out CLIPS.JSON
    python tools/datasets/g1_inputs.py stage --datasets ROOT --clips CLIPS.JSON --dest DIR \\
        --visor-fill TAR --visor-dense TAR --hand-objects TAR --checkpoint PT

``select`` fixes the clips content-blind (``RULE``) from metadata only: frame
counts, splits and names, never pixels. ``stage`` hard-links every source file
under ``DEST`` (the fleet data root; same NFS filesystem, so no bytes move),
packs the selected RacketVision videos and TrackNet frames into one tar each
(each member checked against the dataset's recorded sha256 or the zip's CRC),
and writes ``inputs.json`` with every file's sha256.

Run with the PointStream environment's interpreter (PyAV reads frame counts).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

SEED = "pointstream-g1"
STRETCH_S = 120.0
ANALYSIS_FPS = 10.0
RACKETVISION_PER_SPORT = 8
TRACKNET_CLIPS = 8
MARGIN = 2  # frames kept clear of a video's declared end
VISOR_PROMPTS = {"hand": "hand", "arm": "arm"}
RACKET_PROMPTS = {"person": "person", "racket": "racket"}
RULE = {
    "visor-window": "every item of VISOR evaluation set v2 (34 windows of 240 frames), every frame analysed; "
                    "foreground: dense masks with B1b's object fill",
    "visor-long": f"per evaluation-set video, {STRETCH_S:.0f} s of decoded frames starting at the window's first "
                  "frame, moved earlier to end at the video's last frame if needed, or the whole video if shorter; "
                  f"analysed at {ANALYSIS_FPS:g} frames/s; foreground: SAM 3.1 text prompts {sorted(VISOR_PROMPTS)} "
                  "plus EPIC-KITCHENS hand-object detector boxes (score >= 0.5)",
    "ott": f"every OpenTTGames test video (the 7 of 12 small enough to stage whole); {STRETCH_S:.0f} s, or the whole "
           f"video if shorter, starting at sha256('{SEED}:ott:<video>') mod (frames - stretch + 1); "
           f"{ANALYSIS_FPS:g} frames/s; foreground: SAM 3.1 text prompts {sorted(RACKET_PROMPTS)}",
    "racketvision": f"per sport, the {RACKETVISION_PER_SPORT} test-split clips with the smallest "
                    f"sha256('{SEED}:racketvision:<sport>/<match>_<rally>'); whole clip at {ANALYSIS_FPS:g} frames/s; "
                    f"foreground: SAM 3.1 text prompts {sorted(RACKET_PROMPTS)}",
    "tracknet": f"the {TRACKNET_CLIPS} clips with the smallest sha256('{SEED}:tracknet:<game>/<clip>'); whole clip at "
                f"{ANALYSIS_FPS:g} frames/s; foreground: SAM 3.1 text prompts {sorted(RACKET_PROMPTS)}",
}


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def video_counts(path: Path | str) -> tuple[float, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.base_rate or stream.average_rate)
        frames = int(stream.frames)
        if frames <= 0:
            frames = sum(1 for _ in container.demux(stream) if _.size)
    return fps, frames


def rv_listing(root: Path) -> dict[str, list[str]]:
    out = {}
    for sport in ("badminton", "tabletennis", "tennis"):
        with tarfile.open(root / f"{sport}.tar") as tar:
            member = tar.extractfile(f"{sport}/info/test.json")
            assert member is not None
            out[sport] = [f"{m}_{r}" for m, r in json.load(member)]
    return out


def command_select(args: argparse.Namespace) -> int:
    root = Path(args.datasets)
    eval_set = json.loads(Path(args.eval_set).read_text())
    clips: list[dict[str, Any]] = []
    videos = root / "EPIC-KITCHENS-VISOR" / "epic_kitchens_videos"
    for item in eval_set["items"]:
        fps, frames = video_counts(videos / f"{item['video']}.MP4")
        length = min(frames - MARGIN, int(round(STRETCH_S * fps)))
        first = max(0, min(item["first_video_index"], frames - MARGIN - length))
        window = {k: item[k] for k in ("id", "video", "first_video_index", "frames", "fps", "sparse_jpegs")}
        long_id = f"visor-long/{item['video']}"
        clips.append({
            "id": long_id, "dataset": "visor", "group": "visor-long", "video": item["video"],
            "source": {"kind": "video", "name": f"video_{item['video']}"}, "fps": item["fps"], "video_frames": frames,
            "first_index": first, "last_index": first + length - 1, "analysis": {"mode": "rate", "fps": ANALYSIS_FPS},
            "masks": {"kind": "sam_text", "prompts": VISOR_PROMPTS, "boxes": "epic_hand_objects", "window_item": window},
        })
        clips.append({
            "id": f"visor-window/{item['video']}", "dataset": "visor", "group": "visor-window", "video": item["video"],
            "source": {"kind": "video", "name": f"video_{item['video']}"}, "fps": item["fps"], "video_frames": frames,
            "first_index": item["first_video_index"], "last_index": item["first_video_index"] + item["frames"] - 1,
            "analysis": {"mode": "native"}, "masks": {"kind": "visor_fill"}, "item": {**window, "id": item["id"]},
            "lens_from": long_id,
        })
    manifest = json.loads((root / "manifests" / "OpenTTGames.json").read_text())
    for name, info in sorted(manifest["per_video"].items()):
        if info["split"] != "test":
            continue
        fps, frames = video_counts(root / "OpenTTGames" / "raw" / f"{name}.mp4")
        length = min(frames - MARGIN, int(round(STRETCH_S * fps)))
        first = int(rank(f"ott:{name}"), 16) % (frames - MARGIN - length + 1)
        clips.append({
            "id": f"ott/{name}", "dataset": "openttgames", "group": "ott", "video": name,
            "source": {"kind": "video", "name": f"ott_{name}"}, "fps": fps, "video_frames": frames,
            "first_index": first, "last_index": first + length - 1, "analysis": {"mode": "rate", "fps": ANALYSIS_FPS},
            "masks": {"kind": "sam_text", "prompts": RACKET_PROMPTS},
        })
    listing = rv_listing(root / "RacketVision")
    picks = {sport: sorted(names, key=lambda n, s=sport: rank(f"racketvision:{s}/{n}"))[:RACKETVISION_PER_SPORT]  # type: ignore[misc]
             for sport, names in listing.items()}
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        for sport, names in picks.items():
            with tarfile.open(root / "RacketVision" / f"{sport}.tar") as tar:
                for name in names:
                    member = f"{sport}/videos/{name}.mp4"
                    tar.extract(member, tmp)
                    fps, frames = video_counts(Path(tmp) / member)
                    clips.append({
                        "id": f"racketvision/{sport}/{name}", "dataset": "racketvision", "group": "racketvision",
                        "sport": sport, "source": {"kind": "video_in_tree", "name": "racketvision", "member": member},
                        "fps": fps, "video_frames": frames, "first_index": 0, "last_index": frames - 1 - MARGIN,
                        "analysis": {"mode": "rate", "fps": ANALYSIS_FPS},
                        "masks": {"kind": "sam_text", "prompts": RACKET_PROMPTS},
                    })
    with zipfile.ZipFile(root / "TrackNet" / "Dataset.zip") as archive:
        counts: dict[str, int] = {}
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[3].endswith(".jpg"):
                counts[f"{parts[1]}/{parts[2]}"] = counts.get(f"{parts[1]}/{parts[2]}", 0) + 1
    for key in sorted(counts, key=lambda k: rank(f"tracknet:{k}"))[:TRACKNET_CLIPS]:
        clips.append({
            "id": f"tracknet/{key}", "dataset": "tracknet", "group": "tracknet",
            "source": {"kind": "jpeg_dir", "name": "tracknet", "member_dir": f"Dataset/{key}", "frames": counts[key]},
            "fps": 30.0, "video_frames": counts[key], "first_index": 0, "last_index": counts[key] - 1,
            "analysis": {"mode": "rate", "fps": ANALYSIS_FPS}, "masks": {"kind": "sam_text", "prompts": RACKET_PROMPTS},
        })
    out = {"name": "g1-clips-v1", "rule": RULE, "seed": SEED, "eval_set_sha256": sha256(Path(args.eval_set)),
           "clips": clips}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({g: sum(1 for c in clips if c["group"] == g) for g in RULE}))
    return 0


def link(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    if not os.path.samefile(source, target):
        raise SystemExit(f"{target} is not {source}")
    return {"path": str(target), "source": str(source), "bytes": target.stat().st_size, "sha256": sha256(target)}


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o444
    tar.addfile(info, io.BytesIO(data))


def command_stage(args: argparse.Namespace) -> int:
    root, dest = Path(args.datasets), Path(args.dest)
    clips = json.loads(Path(args.clips).read_text())["clips"]
    files: dict[str, Any] = {}
    for clip in clips:
        name = clip["source"]["name"]
        if clip["group"] in ("visor-long", "visor-window") and name not in files:
            files[name] = link(root / "EPIC-KITCHENS-VISOR" / "epic_kitchens_videos" / f"{clip['video']}.MP4",
                               dest / "videos" / f"{clip['video']}.MP4")
        if clip["group"] == "ott" and name not in files:
            files[name] = link(root / "OpenTTGames" / "raw" / f"{clip['video']}.mp4", dest / "videos" / f"ott_{clip['video']}.mp4")
    manifests = {m: json.loads((root / "manifests" / f"{m}.json").read_text()) for m in ("OpenTTGames", "EPIC-KITCHENS-VISOR")}
    ott_sha = {f["path"]: f["sha256"] for f in manifests["OpenTTGames"]["files"]}
    for clip in clips:
        if clip["group"] == "ott":
            assert files[clip["source"]["name"]]["sha256"] == ott_sha[f"raw/{clip['video']}.mp4"], clip["id"]
    for name, path in (("visor_fill", args.visor_fill), ("visor_dense", args.visor_dense),
                       ("hand_objects", args.hand_objects), ("sam_checkpoint", args.checkpoint),
                       ("eval_set", args.eval_set)):
        files[name] = link(Path(path), dest / "inputs" / Path(path).name if name != "visor_fill" else dest / "inputs" / "visor-fill-published.tar")
    sidecar = {}
    for line in (root / "manifests" / "RacketVision.members.sha256").read_text().splitlines():
        digest, member = line.split(maxsplit=1)
        sidecar[member] = digest
    rv = [c for c in clips if c["group"] == "racketvision"]
    target = dest / "inputs" / "racketvision-g1.tar"
    members = []
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=args.scratch, delete=False) as tmp_handle:
            tmp = Path(tmp_handle.name)
        with tarfile.open(tmp, "w") as out:
            for sport in sorted({c["sport"] for c in rv}):
                wanted = {c["source"]["member"] for c in rv if c["sport"] == sport}
                with tarfile.open(root / "RacketVision" / f"{sport}.tar") as tar:
                    for info in tar:
                        if info.name in wanted:
                            handle = tar.extractfile(info)
                            assert handle is not None
                            data = handle.read()
                            if hashlib.sha256(data).hexdigest() != sidecar[info.name]:
                                raise SystemExit(f"{info.name} does not match RacketVision.members.sha256")
                            add_bytes(out, info.name, data)
                            members.append({"member": info.name, "sha256": sidecar[info.name]})
        os.replace(tmp, target)
    files["racketvision"] = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target),
                             "members": members, "source": str(root / "RacketVision")}
    tn = [c for c in clips if c["group"] == "tracknet"]
    target = dest / "inputs" / "tracknet-g1.tar"
    members = []
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=args.scratch, delete=False) as tmp_handle:
            tmp = Path(tmp_handle.name)
        with zipfile.ZipFile(root / "TrackNet" / "Dataset.zip") as archive, tarfile.open(tmp, "w") as out:
            for clip in tn:
                prefix = clip["source"]["member_dir"] + "/"
                for name in sorted(n for n in archive.namelist()
                                   if n.startswith(prefix) and n.endswith(".jpg") and n.count("/") == 3):
                    data = archive.read(name)  # zipfile checks each member's CRC-32
                    add_bytes(out, name, data)
                members.append({"member_dir": clip["source"]["member_dir"]})
        os.replace(tmp, target)
    files["tracknet"] = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target),
                         "members": members, "source": str(root / "TrackNet" / "Dataset.zip"),
                         "source_sha256": json.loads((root / "manifests" / "TrackNet.json").read_text())["files"][0]["sha256"]}
    files["clips"] = link(Path(args.clips), dest / "inputs" / Path(args.clips).name) if Path(args.clips).parent != dest / "inputs" \
        else {"path": args.clips, "sha256": sha256(Path(args.clips))}
    (dest / "inputs.json").write_text(json.dumps(files, indent=1) + "\n")
    print(json.dumps({k: v["sha256"][:12] for k, v in files.items()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--datasets", required=True)
    select.add_argument("--eval-set", required=True)
    select.add_argument("--out", required=True)
    select.add_argument("--scratch", default="/dev/shm")
    stage = sub.add_parser("stage")
    stage.add_argument("--datasets", required=True)
    stage.add_argument("--clips", required=True)
    stage.add_argument("--dest", required=True)
    stage.add_argument("--eval-set", required=True)
    stage.add_argument("--visor-fill", required=True)
    stage.add_argument("--visor-dense", required=True)
    stage.add_argument("--hand-objects", required=True)
    stage.add_argument("--checkpoint", required=True)
    stage.add_argument("--scratch", default="/dev/shm")
    args = parser.parse_args(argv)
    return {"select": command_select, "stage": command_stage}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
