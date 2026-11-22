"""Bind both externally generated player crops and their verified receiver receipts."""

import argparse
import hashlib
import json
from pathlib import Path

from receiver_spec import INBOX, save


def seal(event, role, sender, folder, job):
    receipt_path = folder / "generation.json"
    receipt = json.loads(receipt_path.read_text())
    timing = json.loads((folder / "worker_timing.json").read_text())
    (row,) = receipt["outputs"]
    selected = next(r for r in sender["rates"] if r["crf"] == sender["selected_crf"])
    assert receipt["frames"] == row["distinct_frames"] == 240
    assert receipt["carrier_sha256"] == selected["carrier_sha256"]
    assert receipt["conditioning_only"] is False and receipt["gpu_peak_mib"] > 0
    assert row["unit"] == event + "_" + role and row["shape"] == [240, 784, 512, 3]
    iso = receipt["isolation"]
    assert iso["interfaces"] == ["lo"] and iso["landlock_abi"] >= 3 and iso["mutation_denied"]
    assert iso["denials"] and set(iso["denials"].values()) <= {1, 13}
    assert row["kernels"]["kernel_launches"] > 0 and isinstance(row["seed"], int)
    assert len(receipt["weight_hashes"]) == 8 and len(timing["archive_sha256"]) == 64
    assert 0 < row["render_and_save_seconds"] < timing["worker_wall_seconds"] < 2680
    return {
        "path": str(INBOX / job / "full/published.tar"),
        "archive_sha256": timing["archive_sha256"],
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "shape": row["shape"],
        "crops_sha256": row["crops_file_sha256"],
        "producer": job,
        "render_seconds": row["render_and_save_seconds"],
        "worker_s": timing["worker_wall_seconds"],
        "device": receipt["device"],
        "seed": row["seed"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", required=True)
    p.add_argument("--sender", type=Path, required=True)
    p.add_argument("--far", required=True, nargs=2, metavar=("JOB", "FOLDER"))
    p.add_argument("--near", required=True, nargs=2, metavar=("JOB", "FOLDER"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sender = json.loads(a.sender.read_text())
    seals = {}
    for role, (job, folder) in (("far", a.far), ("near", a.near)):
        seals[a.event + "_" + role] = seal(a.event, role, sender, Path(folder), job)
    assert len({v["crops_sha256"] for v in seals.values()}) == 2
    assert len({v["seed"] for v in seals.values()}) == 2
    save(a.out, seals)
