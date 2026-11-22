"""Freeze a figure cohort from original source paths, independently of pixels or labels."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

EVENT_DIRS = {
    "alcaraz_hurkacz": "tennis_games/alcaraz_hurkacz",
    "djokovic_federer": "tennis_games/djokovic_federer/videos",
    "djokovic_zverev": "tennis_games/djokovic_zverev",
    "medvedev_struff": "tennis_games/medvedev_struff",
    "rublev_auger-aliassime": "tennis_games/rublev_auger-aliassime/videos",
}
PREFIX = "pointstream-foreground-panel-20261122/"


def freeze(path):
    blob = Path(path).read_bytes()
    archive = json.loads(blob)
    rows = []
    for event, folder in EVENT_DIRS.items():
        pool = [
            r for r in archive["entries"] if r["type"] == "file"
            and str(PurePosixPath(r["path"]).parent) == folder and r["path"].endswith(".mp4")
        ]
        ranked = sorted(pool, key=lambda r: hashlib.sha256((PREFIX + r["path"]).encode()).hexdigest())
        for rank, r in enumerate(ranked[:3], 1):
            rows.append(dict(event=event, rank=rank, pool=len(pool), path=r["path"], bytes=r["bytes"], sha256=r["sha256"]))
    assert len(rows) == 15 and sum(r["pool"] for r in rows if r["rank"] == 1) == 159
    return dict(source_manifest_sha256=hashlib.sha256(blob).hexdigest(), archive=archive["archive"], prefix=PREFIX, original_files=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_manifest")
    print(json.dumps(freeze(parser.parse_args().archive_manifest), indent=2))
