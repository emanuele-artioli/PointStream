"""Install an authenticated small scoring result and its twelve PNGs without copying native arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

TICKS = (9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def extract(archive, validation, out):
    receipt = json.loads(validation.read_text())
    assert receipt["passed"] and receipt["checks"]["complete"]
    assert 0 < archive.stat().st_size < 70 * 1024**2
    assert not out.exists()
    with tarfile.open(archive, "r:") as tar:
        files = {f.name: f for f in tar.getmembers() if f.isfile()}
        expected = {"publish/score.json"}
        have = receipt["checks"]["review_samples"]
        if have:
            assert have == len(TICKS)
            expected |= {
                "publish/review/manifest.json",
                *(f"publish/review/tick-{i:03}.png" for i in TICKS),
            }
        assert set(files) == expected and sum(f.size for f in files.values()) < 70 * 1024**2
        data = {k.removeprefix("publish/"): tar.extractfile(k).read() for k in expected}
    score = json.loads(data["score.json"])
    assert score["source_raw_rgb_crosschecked"] in (24, 240)
    assert (
        score["source_raw_rgb_crosschecked"]
        == score["frames"]
        == receipt["checks"]["source_frames_verified"]
    )
    if have:
        review = json.loads(data["review/manifest.json"])
        assert review["finish_job"] == score["producer_job"] == receipt["checks"]["producer"]
        assert review["joint_rally_bytes"] == score["bytes"]
        assert sorted(map(int, review["ticks"])) == list(TICKS)
        for t in TICKS:
            r = review["ticks"][str(t)]
            assert sha(data["review/" + r["file"]]) == r["png_sha256"]
    out.mkdir(parents=True)
    for k, contents in data.items():
        f = out / k
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(contents)
    summary = dict(
        archive_sha256=sha(archive.read_bytes()),
        name=score["name"],
        bytes=score["bytes"],
        frames=score["frames"],
        body_dists=score["pooled"]["dists"],
        review_frames=have,
        producer=score["producer_job"],
        validation_sha256=sha(validation.read_bytes()),
    )
    (out / "retrieval.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "validation", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    extract(a.archive, a.validation, a.out)
