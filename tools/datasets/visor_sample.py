"""Cut a small, sha256-identified VISOR sample for smoke runs.

    python3 tools/datasets/visor_sample.py OUT_DIR [--video P01_03] [--frames 4]

Without --video it picks the smallest downloaded EPIC-KITCHENS video that has
sparse annotations, preferring the validation split. The tar holds that
video's sparse annotation JSON, its ``frame_mapping.json`` entry, and the first
``--frames`` released sparse JPEGs showing both hands. The video itself is not
copied: jobs stage it by its manifest sha256. ``<tar>.json`` records the
source zip, every member's CRC-32 and sha256, and the video identity.
Standard library only; reads the VISOR zip in place, never extracts it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/itec/emanuele/Datasets/EPIC-KITCHENS-VISOR")
MANIFEST = Path("/home/itec/emanuele/Datasets/manifests/EPIC-KITCHENS-VISOR.json")
ZIP_NAME = "2v6cgv1x04ol22qp9rm9x2j6a7.zip"
HANDS = {"left hand", "right hand"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--video")
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    files = {f["path"]: f for f in manifest["files"]}
    with zipfile.ZipFile(ROOT / ZIP_NAME) as z:
        names = z.namelist()
        annotations = {
            m.group(2): (m.group(1), n)
            for n in names
            if (m := re.search(r"/GroundTruth-SparseAnnotations/annotations/(train|val)/(\w+)\.json$", n))
        }
        if args.video:
            video = args.video
        else:
            candidates = [
                (split != "val", files[f"epic_kitchens_videos/{v}.MP4"]["bytes"], v)
                for v, (split, _) in annotations.items()
                if f"epic_kitchens_videos/{v}.MP4" in files
            ]
            video = min(candidates)[2]
        split, ann_name = annotations[video]
        ann_bytes = z.read(ann_name)
        entries = json.loads(ann_bytes)["video_annotations"]
        chosen = [e for e in entries if HANDS <= {a["name"] for a in e["annotations"]}][: args.frames]
        if len(chosen) < args.frames:
            raise SystemExit(f"{video}: only {len(chosen)} sparse frames show both hands")
        mapping_name = next(n for n in names if n.endswith("frame_mapping.json"))
        mapping = json.loads(z.read(mapping_name))[video]
        rgb_name = next(n for n in names if f"/rgb_frames/{split}/" in n and n.endswith(f"/{video}.zip"))
        with z.open(rgb_name) as handle, zipfile.ZipFile(io.BytesIO(handle.read())) as frames_zip:
            jpegs = {e["image"]["name"]: frames_zip.read(e["image"]["name"]) for e in chosen}
        crc = {n: z.getinfo(n).CRC for n in (ann_name, mapping_name, rgb_name)}
    members = {
        f"annotations/{split}/{video}.json": ann_bytes,
        "frame_mapping.json": json.dumps({video: mapping}, sort_keys=True).encode(),
        **{f"rgb_frames/{name}": data for name, data in jpegs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f"visor-sample-{video}.tar"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o444
            tar.addfile(info, io.BytesIO(data))
    target.write_bytes(buffer.getvalue())
    target.chmod(0o444)
    video_entry = files[f"epic_kitchens_videos/{video}.MP4"]
    record = {
        "tool": "tools/datasets/visor_sample.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tar": target.name, "tar_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "video": video, "split": split,
        "video_file": {"path": str(ROOT / video_entry["path"]), "bytes": video_entry["bytes"], "sha256": video_entry["sha256"]},
        "source_zip": {"path": str(ROOT / ZIP_NAME), "sha256": files[ZIP_NAME]["sha256"], "member_crc32": crc},
        "frames": [{"sparse_name": e["image"]["name"], "epic_frame": mapping[e["image"]["name"]],
                    "objects": sorted({a["name"] for a in e["annotations"]})} for e in chosen],
        "members_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
    }
    (args.out_dir / f"{target.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"tar": str(target), "sha256": record["tar_sha256"], "video": video, "split": split}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
