"""Build a separate offline source review without altering any original labels or images."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def make(old, new):
    new.mkdir(exist_ok=True)
    here = Path(__file__).parent
    foot = json.loads((here / "source-marks.json").read_text())
    routes = {(r["rally_id"], r["tick"]): r for r in foot["frames"]}
    frames = []
    for row in foot["pages"]:
        rid = row["rally_id"]
        meta = json.loads((old / rid / "review.json").read_text())
        sender = json.loads((here / "tenniset-ball" / (rid + ".json")).read_text())
        assert (
            meta["source_video_sha256"] == sender["source_sha256"]
            and meta["review_sha256"] == row["review_sha256"]
        )
        for f in meta["frames"]:
            t = f["frame_index"]
            assert sha(old / rid / f["source"]) == f["source_sha256"]
            assert routes[rid, t]["rgb_sha256"] == f["source_rgb_sha256"]
            xy = sender["ball_positions"][t]
            # Original sender and complete packet use a 960 by 544 reference canvas.
            point = None if xy is None else [round(xy[0] * 1280 / 960), round(xy[1] * 720 / 544)]
            assert point is None or (0 <= point[0] < 1280 and 0 <= point[1] < 720)
            frames.append(
                dict(
                    id=rid + ":" + str(t),
                    rally=rid,
                    tick=t,
                    absolute_tick=f["absolute_tick"],
                    route=routes[rid, t]["route"],
                    src=f"../{old.name}/{rid}/{f['source']}",
                    source_sha256=f["source_sha256"],
                    review_sha256=meta["review_sha256"],
                    ball_candidate=point,
                    players=meta["athletes"],
                )
            )
    assert len(frames) == 48 and len({r["id"] for r in frames}) == 48
    manifest = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
    (new / "frames.js").write_text(
        "window.FRAME_PACKET="
        + json.dumps({"manifest": manifest, "frames": frames}, separators=(",", ":"))
        + ";\n"
    )
    (new / "proposals").mkdir(exist_ok=True)
    optional = new / "proposals/person-proposals.js"
    if not optional.exists():
        optional.write_text("window.PERSON_PROPOSALS={};\n")
    shutil.copyfile(here / "ball-first.html", new / "index.html")
    return dict(
        page=str(new / "index.html"),
        frames=len(frames),
        ball_candidates=sum(x["ball_candidate"] is not None for x in frames),
        manifest=manifest,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("new", type=Path)
    a = p.parse_args()
    print(json.dumps(make(a.original, a.new)))
