"""Staged inputs for PLAN step G1e (reference refresh on VISOR).

    python tools/datasets/g1e_inputs.py --g1-inputs DIR --job JOBDIR ... --g1d-inputs DIR \\
        --g1d-prepared TAR --dest DIR

Packs, from G1's recorded jobs, each OpenTTGames clip's published
``result.json`` and SAM 3.1 ``masks.rle`` into one tar
(``g1-ott-published.tar``, members ``clips/<id>/...``), and writes
``inputs.json`` with the sha256 of every input G1e stages: that tar, G1's
clips.json, the OpenTTGames videos, B1b's fill (from G1d's inputs) and G1d's
prepared archive (its job's ``published.tar``). Run with the PointStream
environment's interpreter.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.background import g1e  # noqa: E402
from experiments.background.g1 import safe  # noqa: E402
from experiments.visor.b1 import file_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-inputs", required=True, help="G1's input directory (inputs.json, inputs/, videos/)")
    parser.add_argument("--job", action="append", required=True, help="a G1 job directory with full/published.tar")
    parser.add_argument("--g1d-inputs", required=True, help="G1d's input directory (inputs.json)")
    parser.add_argument("--g1d-prepared", required=True, help="G1d's prepare job's published.tar")
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    g1_dir = Path(args.g1_inputs)
    g1_inputs = json.loads((g1_dir / "inputs.json").read_text())
    g1d_inputs = json.loads((Path(args.g1d_inputs) / "inputs.json").read_text())
    clips_path = g1_dir / "inputs" / "clips.json"
    spec = json.loads(clips_path.read_text())
    if file_sha256(clips_path) != g1d_inputs["clips_file"]["sha256"]:
        raise SystemExit("G1's clips.json differs from the one G1d used")
    chosen = g1e.select_ott(spec, "all")
    wanted = {f"publish/clips/{safe(c['id'])}/{name}": f"clips/{safe(c['id'])}/{name}"
              for c in chosen for name in ("result.json", "masks.rle")}
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "g1-ott-published.tar"
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
    prepared = Path(args.g1d_prepared)
    videos = sorted({c["source"]["name"] for c in chosen})
    record = {
        "rule": {"seed": g1e.SEED, "pilot_ott": g1e.PILOT_OTT, "ages": list(g1e.AGES)},
        "ott_clips": [c["id"] for c in chosen],
        "g1_published": {"path": str(target), "sha256": file_sha256(target), "members": len(found), "sources": sources,
                         "from_job": {wanted[k]: v for k, v in sorted(found.items())}},
        "clips_file": {"path": str(clips_path), "sha256": file_sha256(clips_path)},
        "g1d_prepared": {"path": str(prepared), "sha256": file_sha256(prepared), "job": prepared.parent.parent.name},
        "visor_fill": g1d_inputs["visor_fill"],
        **{name: g1_inputs[name] for name in videos},
    }
    # Every staged byte is re-hashed by fleet staging against these identities.
    (dest / "inputs.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k in ("g1_published", "g1d_prepared", "clips_file")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
