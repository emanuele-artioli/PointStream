"""Inventory official racket objects on the previously frozen native source windows."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile

ARCHIVE = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/tennis/k5b-2026-10-20/inputs/racketvision-tennis.tar"
)
LIMIT = {"match209": 64, "match269": 64, "match235": 154}
SOURCE = {
    "match209": "48eb20bf0353fea6d7663332ec31cadbc89939df596ed801ea128e0a777427d0",
    "match269": "c714c168c5334cc324e6821082c74023a96b2c1478ba3840fbb5c8fb58bdf5ed",
    "match235": "d3918629801618f6f6a3cea88302901c577ca1852dc3c786fe0ec2f9cb318fdd",
}


def main():
    by_match = {}
    with tarfile.open(ARCHIVE, "r:") as archive:
        for match, limit in LIMIT.items():
            path = f"tennis/videos/{match}_000.mp4"
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == SOURCE[match]
            prefix = f"tennis/all/{match}/racket/000/"
            rows = []
            for member in archive.getmembers():
                if not member.name.startswith(prefix) or not member.name.endswith(".json"):
                    continue
                tick = int(Path(member.name).stem)
                if tick >= limit:
                    continue
                raw = archive.extractfile(member).read()
                labels = json.loads(raw)
                for obj in labels:
                    assert len(obj["bbox_xywh"]) == 4 and len(obj["keypoints"]) == 5
                    assert obj["bbox_xywh"][2] > 0 and obj["bbox_xywh"][3] > 0
                rows.append(
                    dict(
                        tick=tick,
                        member=member.name,
                        sha256=hashlib.sha256(raw).hexdigest(),
                        objects=labels,
                    )
                )
            by_match[match] = dict(
                source_sha256=SOURCE[match],
                frames=limit,
                rows=sorted(rows, key=lambda row: row["tick"]),
            )
    json.dump(dict(archive=str(ARCHIVE), sources=by_match), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
