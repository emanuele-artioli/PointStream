"""Index every verified Sydney calibration packet at its own actual native price."""

from argparse import ArgumentParser
import json
from pathlib import Path


def index(physical, records, caps):
    rows = []
    for receipt in sorted(physical.glob("match127-anchor*.json")):
        key = receipt.stem
        state = json.loads((records / (key + "-terminal.json")).read_text())
        r = json.loads(receipt.read_text())
        assert (
            state["status"] == "complete"
            and state["gate"]["passed"]
            and r["match"] == "match127"
            and r["fps"] == "60"
            and r["frames"] == 240
        )
        sha = (records / (key + "-archive-sha.txt")).read_text().strip()
        assert len(sha) == 64
        mode = "roi" if r["mode"] == "foreground-roi" else "ordinary"
        if mode == "roi" and r.get("roi_offset", 40) != 40:
            mode = f"roi{r['roi_offset']}"
        if r.get("cadence_stride", 1) > 1:
            mode += f"-{r['coded_fps']}fps-hold"
        selected = {}
        for label, item in r["selected"].items():
            assert item["ceiling"] == caps[label]
            s = dict(item)
            if s["affordable"]:
                assert s["bytes"] <= caps[label]
                s.update(
                    file="publish/" + s["file"],
                    decoded_file="publish/" + s["decoded_file"],
                    kbps=s["bytes"] * 8 * 60 / 240000,
                )
            selected[label] = s
        rows.append(
            dict(
                job_id=state["job_id"],
                phase=key,
                state="complete",
                mode=mode,
                archive=state["directory"] + "/full/published.tar",
                archive_sha256=sha,
                width=r["width"],
                height=r["height"],
                selected=selected,
            )
        )
    return dict(match="match127", frames=240, fps=[60, 1], ceilings_bytes=caps, rows=rows)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("physical", type=Path)
    p.add_argument("records", type=Path)
    p.add_argument("caps", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(
            index(a.physical, a.records, json.loads(a.caps.read_text())["ceilings_bytes"]), indent=2
        )
        + "\n"
    )
