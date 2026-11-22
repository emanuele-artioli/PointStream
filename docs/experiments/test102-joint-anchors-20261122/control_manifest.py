"""Index the byte-selected native ordinary and ROI renders for independent scoring."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path


def manifest(receipts, terminal, allow_partial):
    methods = {
        "ordinary": [
            "anchor" + str(width) + ("-fixed" if width == 480 else "")
            for width in (320, 480, 640, 960, 1920)
        ],
        "roi": ["anchorROI" + str(width) for width in (320, 480, 640)],
    }
    result = {
        "source": "RacketVision TEST match102, start 0",
        "frames": 240,
        "fps": "30000/1001",
        "decoder_format": "planar 8-bit YUV 4:2:0, zlib transport",
        "ceilings_bytes": {"warm": 17794, "cold": 155274},
        "rows": [],
    }
    for mode, phases in methods.items():
        for phase in phases:
            key = "match102-" + phase
            report = receipts / (key + ".json")
            status_file = terminal / (key + "-terminal.json")
            if not status_file.exists():
                if allow_partial:
                    continue
                raise ValueError("Missing terminal receipt: " + key)
            status = json.loads(status_file.read_text())
            row = {
                "mode": mode,
                "phase": phase,
                "job_id": status["job_id"],
                "state": status["status"],
                "spec_sha256": status["gate"]["spec_sha256"],
            }
            result["rows"].append(row)
            if row["state"] != "complete":
                continue
            r = json.loads(report.read_text())
            assert r["frames"] == result["frames"] and r["fps"] == result["fps"]
            sha_file = terminal / (key + "-archive-sha.txt")
            row.update(
                width=r["width"],
                height=r["height"],
                report=report.name,
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                archive=status["directory"] + "/full/published.tar",
                archive_sha256=sha_file.read_text().strip(),
                selected={},
            )
            for use, s in r["selected"].items():
                assert s["ceiling"] == result["ceilings_bytes"][use]
                if not s["affordable"]:
                    row["selected"][use] = {"affordable": False}
                    continue
                assert s["bytes"] <= s["ceiling"]
                out = dict(s)
                out["file"] = "publish/" + s["file"]
                out["decoded_file"] = "publish/" + s["decoded_file"]
                out["kbps"] = 8 * s["bytes"] * 30000 / (1001 * 240 * 1000)
                row["selected"][use] = out
    return result


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("receipts", type=Path)
    p.add_argument("terminal", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    data = manifest(a.receipts, a.terminal, a.allow_partial)
    a.out.write_text(json.dumps(data, indent=2) + "\n")
