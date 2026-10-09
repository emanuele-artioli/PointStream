"""Staged inputs for PLAN step G1d (depth-aware warps on VISOR's background).

    python tools/datasets/g1d_inputs.py --g1-inputs DIR --job JOBDIR ... --dest DIR

Packs, from G1's recorded jobs, each selected clip's published ``result.json``
(G1's per-frame records and lens) and, for stretches, G1's SAM 3.1
``masks.rle`` into one tar (``g1-visor-published.tar``, members
``clips/<id>/...``), and writes ``inputs.json`` with the sha256 of every input
G1d stages: that tar, G1's clips.json, the videos, the B1 dense archive, the
B1b fill and the detections (hard links already under G1's input directory,
reused in place). Each source ``published.tar`` is named with its job and
sha256. Run with the PointStream environment's interpreter.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.background import g1d  # noqa: E402
from experiments.background.g1 import safe  # noqa: E402
from experiments.visor.b1 import file_sha256  # noqa: E402

KEEP = ("result.json", "masks.rle")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-inputs", required=True, help="G1's input directory (inputs.json, inputs/, videos/)")
    parser.add_argument("--job", action="append", required=True, help="a G1 job directory with full/published.tar")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    g1_dir = Path(args.g1_inputs)
    g1_inputs = json.loads((g1_dir / "inputs.json").read_text())
    clips_path = g1_dir / "inputs" / "clips.json"
    spec = json.loads(clips_path.read_text())
    chosen = g1d.select_clips(spec, "all")
    wanted = {}
    for clip in chosen:
        names = ["result.json"] + (["masks.rle"] if clip["group"] == "visor-long" else [])
        for name in names:
            wanted[f"publish/clips/{safe(clip['id'])}/{name}"] = f"clips/{safe(clip['id'])}/{name}"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "g1-visor-published.tar"
    found: dict[str, str] = {}
    sources = []
    with tarfile.open(target, "w") as out:
        for job in args.job:
            published = Path(job) / "full" / "published.tar"
            sources.append({"job": Path(job).name, "published_tar": str(published), "sha256": file_sha256(published)})
            with tarfile.open(published) as tar:
                for member in tar.getmembers():
                    if member.name in wanted and member.isfile():
                        if member.name in found:
                            raise SystemExit(f"{member.name} in two jobs: {found[member.name]} and {Path(job).name}")
                        data = tar.extractfile(member)
                        assert data is not None
                        payload = data.read()
                        info = tarfile.TarInfo(wanted[member.name])
                        info.size, info.mode, info.mtime = len(payload), 0o444, int(member.mtime)
                        out.addfile(info, io.BytesIO(payload))
                        found[member.name] = Path(job).name
    missing = sorted(set(wanted) - set(found))
    if missing:
        target.unlink()
        raise SystemExit(f"missing from the G1 jobs: {missing}")
    target.chmod(0o444)
    videos = sorted({c["source"]["name"] for c in chosen})
    record = {
        "rule": {"seed": g1d.SEED, "stretches": g1d.STRETCHES, "pilot": g1d.PILOT},
        "clips": [c["id"] for c in chosen],
        "g1_published": {"path": str(target), "sha256": file_sha256(target), "members": len(found), "sources": sources,
                         "from_job": {wanted[k]: v for k, v in sorted(found.items())}},
        "clips_file": {"path": str(clips_path), "sha256": file_sha256(clips_path)},
        **{name: g1_inputs[name] for name in videos},
        **{name: g1_inputs[name] for name in ("visor_fill", "visor_dense", "hand_objects")},
    }
    # Every staged byte is re-hashed by fleet staging against these identities.
    (dest / "inputs.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in ("g1_published", "clips_file")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
