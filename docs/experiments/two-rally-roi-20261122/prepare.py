"""Recover original encoder-side regions and authenticate the selected native inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire
from experiments.tennis.k6b_rate_streams import decode_masks
from experiments.tennis.w1_roi import roi_text
from src.pointstream import ball

TEMP = Path("/private/tmp")
HEIGHT = {160: 96, 240: 136, 320: 192, 480: 272, 640: 384}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def maps(out):
    out.mkdir(exist_ok=False)
    matte = TEMP / "ps-al028-review-mattes-20261122.tar"
    assert digest(matte) == "8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607"
    with tarfile.open(matte) as archive:
        member = next(n for n in archive.getnames() if n.endswith("/exact-packed.matte"))
        _, masks = decode_masks(archive.extractfile(member).read())
    regions = np.any(masks, axis=1)
    assert len(regions) == 240
    p = TEMP / "ps-nov22-al028-exact-budget/joint-53.5/rally.psj"
    assert digest(p) == "ab12294fec2bba5bb5b21bf373a3a31c620b80cf9463cd92ab0383647d1a8aa9"
    b = p.read_bytes()
    profile = packet._embedded_profile(b)
    content, _ = wire.unpack(b, profile)
    positions, _ = ball.decode_frames(content["trajectory"], 240, 960, 544)
    for t, pos in enumerate(positions):
        if pos is None:
            continue
        h, w = regions[t].shape
        x = round(pos[0] * w / 960)
        y = round(pos[1] * h / 544)
        # Eight pixels on the 960-wide encoder-side mask grid.
        dx = max(1, round(8 * w / 960))
        dy = max(1, round(8 * h / 540))
        regions[t, max(0, y - dy) : min(h, y + dy + 1), max(0, x - dx) : min(w, x + dx + 1)] = True
    ledger = {}
    for event, widths in (("nadal", (160, 240, 480)), ("al028", (160, 320, 640))):
        ledger[event] = {}
        for w in widths:
            h = HEIGHT[w]
            if event == "nadal":
                directory = TEMP / f"ps-joint-trajectory-controls-20261122/20-{w}"
                src = directory / f"{w}.yuv"
                m = directory / "roi.txt"
                record = json.loads((directory / "result.json").read_text())
                assert (
                    digest(src) == record["scaled_source_sha256"]
                    and digest(m) == record["roi_map_sha256"]
                )
                base = m.read_text()
                assert set(x for line in base.splitlines() for x in line.split()[1:]) == {"0", "20"}
            else:
                src = TEMP / f"ps-al028-av1-ladder-20261122/{w}.yuv"
                base = roi_text(regions, w, h, -20, outside=True)
            assert src.stat().st_size == 240 * h * w * 3 // 2 and len(base.splitlines()) == 240
            rows = {}
            for offset in (8, 20, 40):
                text = "".join(
                    " ".join([words[0], *[str(offset) if x == "20" else x for x in words[1:]]])
                    + "\n"
                    for line in base.splitlines()
                    if (words := line.split())
                )
                path = out / f"{event}-{w}-{offset}.txt"
                path.write_text(text)
                rows[str(offset)] = {"path": str(path), "sha256": digest(path)}
            ledger[event][str(w)] = dict(
                source=str(src),
                sha256=digest(src),
                fps=50 if event == "nadal" else 25,
                height=h,
                cap=12974 if event == "nadal" else 24494,
                maps=rows,
            )
    (out / "sources.json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    maps(p.parse_args().out)
