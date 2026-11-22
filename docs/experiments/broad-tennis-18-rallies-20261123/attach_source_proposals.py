"""Attach only the supplied, frame-bound native source mask PNGs to an offline author packet."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image


def read_assignment(path, name):
    text = path.read_text().strip()
    lhs, rhs = text.split("=", 1)
    if lhs.strip() != f"window.{name}":
        raise ValueError("unexpected assignment")
    return json.loads(rhs.rstrip(";"))


def run(selected, page):
    packet = read_assignment(page / "frames.js", "FRAME_PACKET")
    frames = {f["id"]: f for f in packet["frames"]}
    items = read_assignment(selected / "person-proposals.js", "PERSON_PROPOSALS")
    target = page / "proposals"
    target.mkdir(exist_ok=True)
    copied = {}
    for fid, proposals in items.items():
        if fid not in frames or len({r["id"] for r in proposals}) != len(proposals):
            raise ValueError("unexpected frame or duplicate candidate")
        output = []
        for prop in proposals:
            rel = Path(prop["mask"])
            if len(rel.parts) != 2 or rel.parts[0] != "proposals" or rel.suffix != ".png":
                raise ValueError("unexpected mask path")
            source = selected / rel
            with Image.open(source) as image:
                if image.size != (1280, 720) or image.convert("L").getextrema() != (0, 255):
                    raise ValueError("source mask must cover a non-empty native region")
            dest = target / rel.name
            contents = source.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if dest.exists() and dest.read_bytes() != contents:
                raise ValueError("existing mask belongs to a different proposal revision")
            shutil.copyfile(source, dest)
            output.append({**prop, "mask_sha256": digest})
        copied[fid] = output
    input_sha = hashlib.sha256((selected / "person-proposals.js").read_bytes()).hexdigest()
    inputs = {"selected_proposals_sha256": input_sha, "source_frame_manifest": packet["manifest"]}
    for name in ("proposal-provenance.json", "provenance.json", "selection.json", "receipt.json"):
        f = selected / name
        if f.exists():
            inputs[name] = hashlib.sha256(f.read_bytes()).hexdigest()
            shutil.copyfile(f, target / name)
    js = (
        "window.PERSON_PROPOSALS="
        + json.dumps(copied, separators=(",", ":"))
        + ";\nwindow.PERSON_PROPOSAL_SOURCE="
        + json.dumps(inputs)
        + ";\n"
    )
    (target / "person-proposals.js").write_text(js)
    return dict(
        marked_frames=sum(bool(v) for v in copied.values()),
        masks=sum(map(len, copied.values())),
        **inputs,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selected", type=Path)
    p.add_argument("private_page", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.selected, a.private_page)))
