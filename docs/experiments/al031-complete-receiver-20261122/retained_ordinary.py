"""Select existing non-primary SVT comparison packets by their delivered sizes."""

import argparse
import json
from pathlib import Path

from experiments.tennis.common_player_packet import _ivf
from experiments.tennis.player_requalification import sha256

HERE = Path(__file__).resolve().parent


def select(a):
    paid = json.loads((HERE / "fresh-prices.json").read_text())
    tried = json.loads((a.search / "tried.json").read_text())
    first = json.loads((a.original / "result.json").read_text())
    assert first["original_source_sha256"] == paid["source_sha256"]
    data = {}
    for point in sorted(paid["rates"], key=lambda r: r["joint_bytes"])[:2]:
        cap = point["joint_bytes"]
        for w in (160, 320, 480, 640):
            available = [r for r in tried if r["width"] == w and r["bytes"] <= cap]
            if not available:
                continue
            picked = max(available, key=lambda r: r["bytes"])
            step = int(picked["crf"] * 4)
            key = f"w{w}-q4{step}"
            if key in data:
                data[key]["caps"].append(cap)
                continue
            path = a.search / key / "stream.ivf"
            from_grid = not path.exists()
            if from_grid:
                original = next(
                    q for q in first["grid"] if q["stream_sha256"] == picked["stream_sha256"]
                )
                path = a.original / f"{w}-crf{original['crf']}" / "stream.ivf"
            _ivf(path.read_bytes(), w, picked["height"], (25, 1))
            assert (
                path.stat().st_size == picked["bytes"] and sha256(path) == picked["stream_sha256"]
            )
            data[key] = dict(
                caps=[cap],
                grid_already_scored=from_grid,
                width=w,
                height=picked["height"],
                crf=picked["crf"],
                bytes=picked["bytes"],
                kbps=picked["kbps"],
                source_sha256=paid["source_sha256"],
                ivf_sha256=picked["stream_sha256"],
                decoded_sha256=picked["decoded_sha256"],
                path=str(path),
            )
    assert sum(not row["grid_already_scored"] for row in data.values()) == 4
    a.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("search", "original", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    select(cli.parse_args())
