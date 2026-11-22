"""Import only the price-selected physical ordinary and ROI IVF files and their independent receiver hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / "cohort-expansion-20261122/test-physical"
SOURCE = "fc5d3ee431f84417d2de3412402f4e8efb0546418ce96f1f0a1e2f1887a88b1a"


def main(a):
    a.out.mkdir(exist_ok=True, parents=True)
    rows = []
    for label in a.banks:
        assert re.fullmatch(r"anchor(?:ROI)?(?:320|480|640|960|1920)(?:-fixed)?", label)
        rec = json.loads((BANK / ("match102-" + label + ".json")).read_text())
        assert (
            rec["source_sha256"] == SOURCE and rec["frames"] == 240 and rec["fps"] == "30000/1001"
        )
        ledger = [
            json.loads(line) for line in (BANK / (label + "-jobs.jsonl")).read_text().splitlines()
        ]
        job = next(
            r["id"] for r in reversed(ledger) if r["phase"] == label and r["match"] == "match102"
        )
        assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", job)
        prefix = f"/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/{job}/full/published.tar"
        for cost, r in rec["selected"].items():
            if not r["affordable"]:
                continue
            assert cost in ("warm", "cold") and re.fullmatch(r"q\d+\.ivf", r["file"])
            item = ("roi-" if "ROI" in label else "av1-") + str(rec["width"]) + "-" + cost
            p = a.out / (item + ".ivf")
            if not p.exists():
                cmd = [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    a.host,
                    "tar",
                    "-xOf",
                    prefix,
                    "publish/" + r["file"],
                ]
                data = subprocess.run(cmd, check=True, capture_output=True, timeout=80).stdout
                assert len(data) == r["bytes"] and hashlib.sha256(data).hexdigest() == r["sha256"]
                with p.open("xb") as f:
                    f.write(data)
            assert (
                p.stat().st_size == r["bytes"]
                and hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
            )
            rows.append(
                dict(
                    name=item,
                    kind="ivf",
                    path=str(p),
                    width=rec["width"],
                    height=rec["height"],
                    bytes=r["bytes"],
                    sha256=r["sha256"],
                    decoded_sha256=r["decoded_sha256"],
                    job=job,
                )
            )
    rows.sort(key=lambda r: (not r["name"].endswith("warm"), "roi" not in r["name"], r["width"]))
    a.manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print([(r["name"], r["bytes"]) for r in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--host", default="gpu5", choices=("gpu5", "gpu6"))
    main(p.parse_args())
