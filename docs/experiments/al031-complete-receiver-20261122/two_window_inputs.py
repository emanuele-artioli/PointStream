"""Bind the two published source/region grids and their checked AV1 calibration displays."""

import argparse
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from experiments.tennis.player_requalification import array_sha, sha256
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
TICKS = list(range(10, 240, 20))


def load(p):
    return json.loads(p.read_text())


def prepare(a):
    a.out.mkdir(exist_ok=False)
    grid = load(HERE / "al031-campus-evaluation-inputs.json")
    legacy = HERE.parent / "al028-neural-display-20261122"
    rev = load(a.review / "review.json")
    assert rev["review_sha256"] == load(legacy / "player-joint.json")["review_sha256"]
    assert (
        rev["review_sha256"] == "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
    )
    assert [x["frame_index"] for x in rev["frames"]] == TICKS
    first = next(
        r
        for r in load(legacy / "ordinary-three-size-curves.json")["points"]
        if r["width"] == 640 and r["crf"] == 40
    )
    second = load(a.return_score)
    expected = [
        {(r["tick"], r["person"]): r["dists"] for r in first["rows"]},
        {(r["tick"], r["player_index"]): r["dists"] for r in second["body_rows"]},
    ]
    supports = load(a.return_supports)
    assert sha256(a.return_supports) == second["supports_sha256"]
    assert sha256(a.source) == grid["source_sha256"] == second["source_sha256"]
    models = {(r["tick"], r["object_index"]): r for r in supports["rows"] if r["tick"] in TICKS}
    earlier_raw = a.first / "640-crf40" / "decoded.yuv"
    later_raw = a.second / "640-crf40" / "decoded.yuv"
    assert (
        sha256(earlier_raw) == first["raw_sha256"] and sha256(later_raw) == second["arm"]["sha256"]
    )
    frames = [
        np.memmap(p, np.uint8, mode="r", shape=(240, 576, 640)) for p in [earlier_raw, later_raw]
    ]
    rows = []
    cap = cv2.VideoCapture(str(a.source))
    for tick in range(240):
        ok, bgr = cap.read()
        assert ok
        if tick not in TICKS:
            continue
        assert array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) == grid["source_rgb_hashes"][tick]
        old = next(r for r in rev["frames"] if r["frame_index"] == tick)
        pre = a.review / old["source"]
        assert sha256(pre) == old["source_sha256"]
        originals = [cv2.imread(str(pre)), bgr]
        for shot in (0, 1):
            rgb = cv2.resize(
                yuv_to_rgb(frames[shot][tick], 640, 384, False),
                (1920, 1080),
                interpolation=cv2.INTER_CUBIC,
            )
            positive = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for who in (0, 1):
                if shot == 0:
                    role = old["players"][who]
                    mp = a.review / role["proposal"]
                    assert sha256(mp) == role["sha256"]
                    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    ys, xs = np.where(mask > 0)
                    assert len(xs) == role["pixels"]
                    xy = [
                        max(0, int(xs.min()) - 32),
                        max(0, int(ys.min()) - 32),
                        min(1920, int(xs.max()) + 33),
                        min(1080, int(ys.max()) + 33),
                    ]
                    mask = mask[xy[1] : xy[3], xy[0] : xy[2]]
                    value = expected[0][tick, role["id"]]
                else:
                    r = models[tick, who]
                    mp = a.return_supports.parent / r["mask"]
                    assert sha256(mp) == r["sha256"]
                    xy = r["xyxy"]
                    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    value = expected[1][tick, who]
                x0, y0, x1, y1 = xy
                box = np.s_[y0:y1, x0:x1]
                crops = {"source": originals[shot][box], "mask": mask, "positive": positive[box]}
                assert all(m.shape[:2] == (y1 - y0, x1 - x0) for m in crops.values())
                entries = {}
                for kind, pix in crops.items():
                    dest = a.out / f"{shot}-{tick}-{who}-{kind}.png"
                    assert cv2.imwrite(str(dest), pix)
                    entries[kind] = [dest.name, sha256(dest)]
                rows.append(
                    dict(
                        shot=shot, tick=tick, role=who, xy=xy, positive_dists=value, assets=entries
                    )
                )
    cap.release()
    assert len(rows) == 48
    segments = [a.first / "640.yuv", a.second / "640.yuv"]
    digests = [sha256(p) for p in segments]
    assert digests[0] == load(HERE / "first-shot-extra-controls.json")["source_yuv_sha256"]
    assert all(p.stat().st_size == 240 * 640 * 384 * 3 // 2 for p in segments)
    receipt = dict(
        components=["al028", "al031"],
        input_sha256=digests,
        source_files=[
            load(legacy / "source-rgb-audit.json")["source_sha256"],
            grid["source_sha256"],
        ],
        comparison_review=rev["review_sha256"],
        return_supports=sha256(a.return_supports),
        physical_caps={
            k: v["total_bytes"]
            for k, v in load(HERE / "two-shot-price.json").items()
            if k in ("send_new", "reuse")
        },
        rows=rows,
    )
    (a.out / "grid.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with tarfile.open(a.archive, "x:gz") as t:
        for f in sorted(a.out.iterdir()):
            t.add(f, arcname=f.name)
    print("Source masks", len(rows), "archive", a.archive.stat().st_size, sha256(a.archive))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "source",
        "review",
        "return-supports",
        "return-score",
        "first",
        "second",
        "out",
        "archive",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    prepare(p.parse_args())
