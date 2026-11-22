"""Stage a source/label pair per preselected RacketVision TEST ID, verifying upstream hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def stage(manifest: Path, output: Path, root: Path) -> dict:
    records = json.loads(manifest.read_text())
    index = {
        p: h
        for h, p in (
            s.strip().split(maxsplit=1)
            for s in (root / "manifests/RacketVision.members.sha256").read_text().splitlines()
        )
    }
    entries = []
    output.mkdir(exist_ok=True, parents=True)
    with tarfile.open(root / "RacketVision/tennis.tar") as archive:
        official = json.load(archive.extractfile("tennis/info/test.json"))
        ranked = sorted(
            official,
            key=lambda p: hashlib.sha256(
                f"pointstream-20261122-prospective/{p[0]}/{p[1]}".encode()
            ).hexdigest(),
        )
        assert [[r["match"], r["rally"]] for r in records["selection"]] == ranked[:12]
        for r in records["selection"]:
            m, v = r["match"], r["rally"]
            fields = {}
            for kind, name in [
                ("source", f"tennis/videos/{m}_{v}.mp4"),
                ("labels", f"tennis/all/{m}/csv/{v}_ball.csv"),
            ]:
                blob = archive.extractfile(name).read()
                sha = hashlib.sha256(blob).hexdigest()
                if sha != index[name]:
                    raise ValueError(name)
                dest = output / f"{m}-{v}-{sha[:12]}{Path(name).suffix}"
                if dest.exists():
                    assert hashlib.sha256(dest.read_bytes()).hexdigest() == sha
                else:
                    with dest.open("xb") as writer:
                        writer.write(blob)
                fields[kind] = {"path": str(dest), "sha256": sha, "member": name}
            info = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
                    "-of",
                    "json",
                    fields["source"]["path"],
                ]
            )
            fields["video"] = json.loads(info)["streams"][0]
            entries.append(dict(**r, **fields))
    return {"selection": entries, "source_ids": len(official), "metadata_only": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--root", default="/home/itec/emanuele/Datasets", type=Path)
    args = p.parse_args()
    print(json.dumps(stage(args.manifest, args.output, args.root), indent=2))
