"""Losslessly transport every paid archive member and compare all native controls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zlib

INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
JOBS = [
    "20261121T215627Z-77ce917a",
    "20261121T220316Z-7cbf21fc",
    "20261121T220950Z-53f30f05",
    "20261121T221317Z-fa6e8d74",
    "20261121T221842Z-cd1a8b24",
    "20261121T221746Z-7f377ae9",
    "20261121T222531Z-0269f5cd",
    "20261121T222531Z-6cb6570f",
    "20261121T222531Z-e7aae2b8",
    "20261121T223136Z-e9219942",
    "20261121T223136Z-d6a41fc2",
    "20261121T223514Z-9be902b0",
    "20261121T223514Z-8e4a7930",
]
CHOSEN = frozenset(
    ["20261121T222531Z-e7aae2b8", "20261121T223136Z-e9219942", "20261121T223136Z-d6a41fc2"]
)
RECEIVER = """
import hashlib,sys,zlib
data=sys.stdin.buffer.read()
if data[0]==1:
    raw=zlib.decompress(data[1:])
elif data[0]==0:
    raw=data[1:]
else:
    raise ValueError("invalid transport mode")
print(hashlib.sha256(raw).hexdigest())
"""


def serialize(blob):
    compressed = zlib.compress(blob, 6)
    return b"\x01" + compressed if len(compressed) < len(blob) else b"\x00" + blob


def main():
    import resource

    def cpu():
        return (
            time.process_time()
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        )

    start = cpu()
    rows = []
    for index, job in enumerate(JOBS):
        with tarfile.open(INBOX / job / "full/published.tar") as archive:
            report = json.load(archive.extractfile("publish/result.json"))
            for row in report["results"]:
                blob = archive.extractfile("publish/" + row["file"]).read()
                assert (
                    len(blob) == row["bytes"] and hashlib.sha256(blob).hexdigest() == row["sha256"]
                )
                wire = serialize(blob)
                recovered = subprocess.check_output([sys.executable, "-c", RECEIVER], input=wire)
                assert recovered.decode().strip() == row["sha256"]
                duration = report["frames"] / report["fps"]
                assert abs(len(blob) * 0.008 / duration - row["kbps"]) < 1e-6
                scores = [entry["dists"] for entry in row["crop"]]
                rows.append(
                    dict(
                        job=job,
                        source=report["source"],
                        chosen=job in CHOSEN and row["mode"] == "tracked",
                        mode=row["mode"],
                        input_sha256=row["sha256"],
                        wire_sha256=hashlib.sha256(wire).hexdigest(),
                        wire_mode=wire[0],
                        bytes=len(blob),
                        wire_bytes=len(wire),
                        frames=report["frames"],
                        fps=report["fps"],
                        original_kbps=row["kbps"],
                        wire_kbps=len(wire) * 0.008 / duration,
                        ball_dists=sum(scores) / len(scores),
                        labels=len(scores),
                    )
                )
        if index == 0:
            assert {r["mode"] for r in rows} >= {"plain", "tracked"}
            forecast = 1.5 * (cpu() - start) * len(JOBS) + 15
            assert forecast < 120
        assert cpu() - start < 120
    comparisons = []
    for r in (v for v in rows if v["chosen"]):
        for cost in ("original_kbps", "wire_kbps"):
            affordable = [
                a
                for a in rows
                if a["source"] == r["source"]
                and a["mode"] in ("plain", "roi")
                and a[cost] <= r[cost]
            ]
            best = min(affordable, key=lambda a: (a["ball_dists"], a[cost]))
            comparisons.append(
                dict(
                    source=r["source"],
                    cost=cost,
                    tile_rate=r[cost],
                    tile_dists=r["ball_dists"],
                    best_rate=best[cost],
                    best_dists=best["ball_dists"],
                    best_mode=best["mode"],
                    best_original_sha256=best["input_sha256"],
                )
            )
    version = sys.version
    print(
        json.dumps(
            dict(
                rows=rows,
                comparisons=comparisons,
                zlib=zlib.ZLIB_RUNTIME_VERSION,
                python=version,
                host=os.uname().nodename,
                cpu_seconds=cpu() - start,
                forecast=forecast,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
