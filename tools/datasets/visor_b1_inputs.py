"""Cut the sha256-identified inputs of PLAN step B1 from the VISOR release.

    python tools/datasets/visor_b1_inputs.py mapping OUT_DIR --video P32_07 ... [--frames 40]
    python tools/datasets/visor_b1_inputs.py evalset OUT_DIR --mapping RESULT.json [--length 240]

``mapping`` writes ``visor-mapping-check.tar``: ``frame_mapping.json``, the sparse
annotation JSON of each listed video and up to ``--frames`` of its released
sparse JPEGs, evenly spaced over the video and always including the last (where
any drift is largest). Each video is hard-linked to
``OUT_DIR/inputs/video_<id>/<id>.MP4`` so a fleet job can stage it from the data
root without copying it.

``evalset`` fixes the evaluation set from the validation split and writes
``visor-val-dense.tar`` with the dense interpolation zips, sparse annotation
JSONs and the released JPEG of each item's first frame, plus
``eval_set.json``. The rule, decided before any item was scored:

1. every validation video with a dense file in a video class (codec, size,
   rate) where the frame-mapping check (``--mapping``) found both VISOR frame
   n = decoded frame n - 1 and the reader's EPIC rule to hold on every checked frame;
2. its runs: maximal stretches of consecutive labelled frames;
3. eligible runs: at least ``--length`` frames, and the first ``--length`` lie
   exactly on the video (`visor.frame_alignment`: no drift between keyframes);
4. among them, the one indexed by sha256("pointstream-b1:<video>") modulo their
   count, content-blind;
5. the item is that run's first ``--length`` frames, starting on a keyframe.

Both commands read the VISOR zip in place and never extract it; they write a
few large files, never many small ones. Run them on a fleet host with the
checkout on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.segmentation import visor

ROOT = Path("/home/itec/emanuele/Datasets/EPIC-KITCHENS-VISOR")
MANIFEST = Path("/home/itec/emanuele/Datasets/manifests/EPIC-KITCHENS-VISOR.json")
ZIP_NAME = "2v6cgv1x04ol22qp9rm9x2j6a7.zip"
SEED = "pointstream-b1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def video_class(video: Path) -> dict[str, Any]:
    """Codec, size and nominal rate: the classes the frame-mapping rules are checked per."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    rate, average = Fraction(stream["r_frame_rate"]), Fraction(stream["avg_frame_rate"])
    return {
        "key": f"{stream['codec_name']} {stream['width']}x{stream['height']} {float(rate):.3f}",
        "codec": stream["codec_name"], "width": stream["width"], "height": stream["height"],
        "r_frame_rate": stream["r_frame_rate"], "avg_frame_rate": stream["avg_frame_rate"],
        "constant_rate": abs(float(rate - average)) < 1e-3, "frames": int(stream.get("nb_frames") or 0),
    }


class Release:
    """The VISOR zip, read in place."""

    def __init__(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        self.files = {f["path"]: f for f in manifest["files"]}
        self.manifest_sha256 = file_sha256(MANIFEST)
        self.zip = zipfile.ZipFile(ROOT / ZIP_NAME)
        self.names = self.zip.namelist()
        self.sparse = {
            m.group(2): (m.group(1), n) for n in self.names
            if (m := re.search(r"/GroundTruth-SparseAnnotations/annotations/(train|val)/(\w+)\.json$", n))
        }
        self.dense = {
            m.group(2): (m.group(1), n) for n in self.names
            if (m := re.search(r"/Interpolations-DenseAnnotations/(train|val)/(\w+)_interpolations\.zip$", n))
        }
        self.mapping_name = next(n for n in self.names if n.endswith("/frame_mapping.json"))

    def member(self, name: str) -> tuple[bytes, dict[str, Any]]:
        data = self.zip.read(name)
        return data, {"member": name, "crc32": self.zip.getinfo(name).CRC, "sha256": sha256(data)}

    def jpegs(self, video: str, names: list[str]) -> dict[str, bytes]:
        split = self.sparse[video][0]
        rgb = next(n for n in self.names if f"/rgb_frames/{split}/" in n and n.endswith(f"/{video}.zip"))
        with zipfile.ZipFile(io.BytesIO(self.zip.read(rgb))) as frames_zip:
            return {name: frames_zip.read(name) for name in names}

    def video(self, video: str) -> dict[str, Any]:
        entry = self.files[f"epic_kitchens_videos/{video}.MP4"]
        return {"path": str(ROOT / entry["path"]), "bytes": entry["bytes"], "sha256": entry["sha256"]}


def write_tar(target: Path, members: dict[str, bytes]) -> str:
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o444
            tar.addfile(info, io.BytesIO(data))
    target.write_bytes(buffer.getvalue())
    target.chmod(0o444)
    return sha256(buffer.getvalue())


def link_video(release: Release, video: str, out_dir: Path) -> dict[str, Any]:
    """Hard link (same filesystem, no bytes copied) so the fleet can stage it from the data root."""
    record = release.video(video)
    link = out_dir / "inputs" / f"video_{video}" / f"{video}.MP4"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        os.link(record["path"], link)
    if os.stat(link).st_ino != os.stat(record["path"]).st_ino:
        raise SystemExit(f"{link} is not a hard link of {record['path']}")
    return {"name": f"video_{video}", "path": str(link), "sha256": record["sha256"], "source": record["path"], "bytes": record["bytes"]}


def spaced(items: list[str], count: int) -> list[str]:
    """``count`` items evenly spaced over ``items``, first and last included."""
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[-1]]
    picks = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in picks]


def command_mapping(args: argparse.Namespace) -> int:
    release = Release()
    mapping_bytes, mapping_record = release.member(release.mapping_name)
    members: dict[str, bytes] = {"frame_mapping.json": mapping_bytes}
    sources: dict[str, Any] = {"frame_mapping.json": mapping_record}
    selection: dict[str, Any] = {}
    staged = []
    for video in args.video:
        split, ann_name = release.sparse[video]
        ann_bytes, ann_record = release.member(ann_name)
        entries = json.loads(ann_bytes)["video_annotations"]
        names = spaced(sorted(e["image"]["name"] for e in entries), args.frames)
        members[f"annotations/{video}.json"] = ann_bytes
        sources[f"annotations/{video}.json"] = ann_record
        for name, data in release.jpegs(video, names).items():
            members[f"rgb_frames/{name}"] = data
        staged.append(link_video(release, video, args.out_dir))
        selection[video] = {"split": split, "sparse_frames": len(entries), "checked": names,
                            "video": release.video(video), "class": video_class(Path(release.video(video)["path"]))}
    members["selection.json"] = json.dumps(selection, indent=1, sort_keys=True).encode()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_sha = write_tar(args.out_dir / "visor-mapping-check.tar", members)
    record = {
        "tool": "tools/datasets/visor_b1_inputs.py mapping", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": {"path": str(ROOT / ZIP_NAME), "sha256": release.files[ZIP_NAME]["sha256"]},
        "dataset_manifest": {"path": str(MANIFEST), "sha256": release.manifest_sha256},
        "tar": {"name": "visor-mapping-check.tar", "sha256": tar_sha, "sources": sources,
                "members_sha256": {name: sha256(data) for name, data in members.items()}},
        "staged_videos": staged, "selection": selection,
    }
    (args.out_dir / "visor-mapping-check.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"tar": tar_sha, "videos": len(staged), "jpegs": sum(len(s["checked"]) for s in selection.values())}))
    return 0


def pick_run(video: str, eligible: list[tuple[int, int]]) -> tuple[int, int]:
    index = int(sha256(f"{SEED}:{video}".encode()), 16) % len(eligible)
    return eligible[index]


def command_evalset(args: argparse.Namespace) -> int:
    release = Release()
    holds = json.loads(args.mapping.read_text())["rule_holds_for_classes"]
    verified = set(holds["visor_minus_one"]) & set(holds["epic_reader"])
    mapping = json.loads(release.zip.read(release.mapping_name))
    mapping_bytes, mapping_record = release.member(release.mapping_name)
    members: dict[str, bytes] = {"frame_mapping.json": mapping_bytes}
    sources: dict[str, Any] = {"frame_mapping.json": mapping_record}
    items, excluded = [], []
    for video in sorted(v for v, (split, _) in release.dense.items() if split == "val"):
        info = video_class(Path(release.video(video)["path"]))
        if info["key"] not in verified or not info["constant_rate"]:
            excluded.append({"video": video, "reason": f"class {info['key']} is not exactly aligned", "class": info})
            continue
        dense_bytes, dense_record = release.member(release.dense[video][1])
        doc = visor.load_annotations(dense_bytes)
        labelled = visor.frames(doc)
        all_runs = visor.runs(labelled)
        fps = float(Fraction(info["r_frame_rate"]))
        alignment = visor.frame_alignment(
            doc, visor.keyframe_anchors(doc, mapping[video], fps), fps / visor.extraction_rate(fps)
        )

        def exact(first: int) -> bool:
            placed = [alignment.get(n) for n in range(first, first + args.length)]
            return all(p is not None and p["exact"] for p in placed) and all(
                b["index"] == a["index"] + 1 for a, b in zip(placed, placed[1:])  # type: ignore[index]
            )

        eligible = [r for r in all_runs if r[1] - r[0] + 1 >= args.length and exact(r[0])]
        if not eligible:
            excluded.append({"video": video, "reason": f"no run of {args.length} frames", "runs": len(all_runs)})
            continue
        run = pick_run(video, eligible)
        first, last = run[0], run[0] + args.length - 1
        if visor.visor_frame_to_video_index(last) >= info["frames"]:
            excluded.append({"video": video, "reason": "window ends after the video", "class": info})
            continue
        sparse_bytes, sparse_record = release.member(release.sparse[video][1])
        start_jpeg = next(
            e["image"]["name"] for e in json.loads(sparse_bytes)["video_annotations"]
            if visor.frame_number(e["image"]["name"]) == first
        )
        members[f"dense/{video}_interpolations.zip"] = dense_bytes
        sources[f"dense/{video}_interpolations.zip"] = dense_record
        members[f"annotations/{video}.json"] = sparse_bytes
        sources[f"annotations/{video}.json"] = sparse_record
        members[f"rgb_frames/{start_jpeg}"] = release.jpegs(video, [start_jpeg])[start_jpeg]
        window = [labelled[n] for n in range(first, last + 1)]
        items.append({
            "id": f"{video}_{first:010d}", "video": video, "split": "val",
            "first_visor_frame": first, "last_visor_frame": last, "frames": args.length,
            "first_video_index": alignment[first]["index"],
            "fps": fps, "video_class": info["key"],
            "run": {"first": run[0], "last": run[1], "eligible_runs": len(eligible), "runs": len(all_runs)},
            "keyframes": [i for i, frame in enumerate(window) if frame.keyframe],
            "labels": sorted({a["name"] for frame in window for a in frame.annotations}),
            "start_jpeg": f"rgb_frames/{start_jpeg}",
            "dense_member": f"dense/{video}_interpolations.zip", "dense_sha256": dense_record["sha256"],
            "video_file": release.video(video),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_sha = write_tar(args.out_dir / "visor-val-dense.tar", members)
    eval_set = {
        "name": "visor-val-b1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": {"split": "val", "length": args.length, "seed": SEED, "verified_classes": sorted(verified),
                 "mapping_result": {"path": str(args.mapping), "sha256": file_sha256(args.mapping)},
                 "summary": "one exactly aligned run per validation video of a verified class, its first frames from a keyframe; tools/datasets/visor_b1_inputs.py"},
        "source_zip": {"path": str(ROOT / ZIP_NAME), "sha256": release.files[ZIP_NAME]["sha256"]},
        "dataset_manifest": {"path": str(MANIFEST), "sha256": release.manifest_sha256},
        "archive": {"path": str(args.out_dir / "visor-val-dense.tar"), "sha256": tar_sha, "sources": sources},
        "items": items, "excluded": excluded,
    }
    (args.out_dir / "eval_set.json").write_text(json.dumps(eval_set, indent=1) + "\n")
    print(json.dumps({"tar": tar_sha, "items": len(items), "excluded": len(excluded)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mapping = sub.add_parser("mapping")
    mapping.add_argument("out_dir", type=Path)
    mapping.add_argument("--video", action="append", required=True)
    mapping.add_argument("--frames", type=int, default=40)
    evalset = sub.add_parser("evalset")
    evalset.add_argument("out_dir", type=Path)
    evalset.add_argument("--mapping", type=Path, required=True)
    evalset.add_argument("--length", type=int, default=240)
    args = parser.parse_args(argv)
    return command_mapping(args) if args.command == "mapping" else command_evalset(args)


if __name__ == "__main__":
    raise SystemExit(main())
