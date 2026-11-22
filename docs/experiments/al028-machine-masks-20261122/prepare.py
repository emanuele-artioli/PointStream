"""Bind the independent source labels to the original four displays without revealing modes to inference."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

DISPLAY = "2eb230df276f52780fd34a11331743c2895c3fe94bde05c6645dc0165ee33559"
TRUTHS = "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"
ARMS = ("source", "av1_320_q52_75", "dcvc_q24", "joint-20")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(review, truths, display, destination):
    destination.mkdir()
    assert digest(truths.read_bytes()) == TRUTHS and digest((display / "result.json").read_bytes()) == DISPLAY
    labels = json.loads(truths.read_text())["frames"]
    packet = json.loads((review / "review.json").read_text())
    shots = {r["frame_index"]: r for r in packet["frames"]}
    original = json.loads((display / "result.json").read_text())["private_review_png_sha256"]
    keys = {}
    direct = {}
    with ZipFile(destination / "frames.zip", "x") as archive:
        for label in labels:
            tick = label["tick"]
            for arm in ARMS:
                if arm == "source":
                    ref = shots[tick]
                    pixel = (review / ref["source"]).read_bytes()
                    assert digest(pixel) == ref["source_sha256"] == label["source_png_sha256"]
                else:
                    key = f"{tick:03d}/{arm}"
                    pixel = (display / "private-review" / (key + ".png")).read_bytes()
                    assert digest(pixel) == original[key]
                identity = digest(pixel)
                assert identity not in keys
                archive.writestr(ZipInfo(identity + ".png"), pixel)
                keys[identity] = [arm, tick]
            direct[tick] = []
            for i in (1, 2):
                key = f"{tick:03d}/receiver-mask-{i}"
                assert digest((display / "private-review" / (key + ".png")).read_bytes()) == original[key]
                direct[tick].append(original[key])
    assert len(keys) == 48 and len(direct) == 12
    smoke = {arm: key for key, (arm, t) in keys.items() if t == 10}
    report = dict(display_sha256=DISPLAY, normalized_labels_sha256=TRUTHS, archive_sha256=digest((destination / "frames.zip").read_bytes()), smoke=smoke, keys=keys, receiver_masks=direct)
    (destination / "binding.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "truths", "display", "destination"):
        parser.add_argument(name,type=Path)
    a=parser.parse_args()
    prepare(a.review,a.truths,a.display,a.destination)
