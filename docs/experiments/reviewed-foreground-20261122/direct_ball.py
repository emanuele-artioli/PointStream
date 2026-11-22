"""Compare direct readout of the paid Al028 packet with the author marks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.tennis import common_player_packet, w8_payload
from src.pointstream import ball

HERE = Path(__file__).resolve().parent


def run(packet, labels, output):
    raw = packet.read_bytes()
    bill = json.loads(
        (HERE.parent / "al028-neural-display-20261122/common-budget.json").read_text()
    )["selected"]["joint"]
    assert hashlib.sha256(raw).hexdigest() == bill["stream_sha256"] and len(raw) == bill["bytes"]
    data = labels.read_bytes()
    human = json.loads(data)
    assert (
        hashlib.sha256(data).hexdigest()
        == "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"
    )
    profile = common_player_packet._embedded_profile(raw)
    sections, _ = w8_payload.unpack(raw, profile)
    w, h = profile["ball_canvas"]
    assert (w, h) == (960, 544)
    centers, looks = ball.decode_frames(sections["trajectory"], 240, w, h)
    assert ball.encode_frames(centers, looks, w, h) == sections["trajectory"]
    rows = []
    for frame in human["frames"]:
        t = frame["tick"]
        position = centers[t]
        status = frame["ball"]["status"]
        error = None
        if status == "visible" and position is not None:
            x, y = frame["ball"]["center_xy"]
            error = math.hypot(x - position[0] * 1920 / w, y - position[1] * 1080 / h)
        rows.append(
            dict(tick=t, mark=status, present=position is not None, error_native_pixels=error)
        )
    summary = dict(
        packet_sha256=bill["stream_sha256"],
        labels_sha256=hashlib.sha256(data).hexdigest(),
        kbps=bill["kbps"],
        bytes=len(raw),
        trajectory_sha256=hashlib.sha256(sections["trajectory"]).hexdigest(),
        trajectory_bytes=len(sections["trajectory"]),
        frames=len(centers),
        rows=rows,
        visible=sum(r["mark"] == "visible" for r in rows),
        missing=sum(r["mark"] == "missing" for r in rows),
        present_on_missing=sum(r["mark"] == "missing" and r["present"] for r in rows),
        hits={
            str(tol): sum(
                r["error_native_pixels"] is not None and r["error_native_pixels"] <= tol
                for r in rows
            )
            for tol in (5, 10)
        },
    )
    with output.open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "labels", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.packet, a.labels, a.output)
