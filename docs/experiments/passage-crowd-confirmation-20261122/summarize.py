"""Join both complete spectator intervals to the frozen camera labels and motion routing."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EARLIER = HERE.parent / "passage-routing-20261122"


def read(path):
    return json.loads(path.read_text())


def summarize():
    labels = read(EARLIER / "routing.json")
    seen = set()
    result = []
    for interval, receipt in (
        ("crowd_point", EARLIER / "crowd_point-result.json"),
        ("crowd_interlude", HERE / "native-result.json"),
    ):
        row = read(receipt)["records"][interval]
        samples = []
        for part in row["parts"]:
            for tick in range(part["start"], part["end"]):
                assert tick not in seen
                seen.add(tick)
                (human,) = [r["label"] for r in labels["human"] if r["start"] <= tick < r["end"]]
                (motion,) = [r["route"] for r in labels["motion"] if r["start"] <= tick < r["end"]]
                samples.append((human, motion, part["predicted"]))
        assert len(samples) == len(row["rows"]) == row["frames"]
        assert all(h == "crowd" for h, _, _ in samples)
        result.append(
            dict(
                interval=interval,
                frames=len(samples),
                seconds=len(samples) / 50,
                motion_point=sum(m == "point" for _, m, _ in samples),
                court=sum(c == "court" for _, _, c in samples),
            )
        )
    return dict(
        fps=50,
        intervals=result,
        total_frames=len(seen),
        correctly_non_court=sum(r["frames"] - r["court"] for r in result),
    )


if __name__ == "__main__":
    with (HERE / "geometry-confirmed.json").open("x") as f:
        json.dump(summarize(), f, indent=2)
        f.write("\n")
