"""Bind a second authentic native receiver bank to the first committed Shanghai seal."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def main(a):
    receipt = json.loads(a.receipt.read_text())
    timing = json.loads(a.timing.read_text())
    reference = json.loads((HERE / "match102-generation-seals-partial.json").read_text())
    source = json.loads((HERE / "match102-sender.json").read_text())
    selected = next(r for r in source["rates"] if r["crf"] == source["selected_crf"])
    (row,) = receipt["outputs"]
    assert receipt["frames"] == row["distinct_frames"] == 240
    assert (
        receipt["carrier_sha256"] == selected["carrier_sha256"]
        and receipt["conditioning_only"] is False
    )
    assert row["unit"] == "match102_far" and row["shape"] == [240, 784, 512, 3]
    assert row["kernels"]["kernel_launches"] > 0 and receipt["isolation"]["interfaces"] == ["lo"]
    assert receipt["isolation"]["landlock_abi"] >= 3 and set(
        receipt["isolation"]["denials"].values()
    ) <= {1, 13}
    assert len(a.archive_sha) == 64 and timing["archive_sha256"] == a.archive_sha
    assert 0 < row["render_and_save_seconds"] < timing["worker_wall_seconds"] < 2680
    reference[row["unit"]] = dict(
        path=str(INBOX / a.job / "full/published.tar"),
        archive_sha256=a.archive_sha,
        receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),
        shape=row["shape"],
        crops_sha256=row["crops_file_sha256"],
        producer=a.job,
        render_seconds=row["render_and_save_seconds"],
        worker_s=timing["worker_wall_seconds"],
    )
    a.out.write_text(json.dumps(reference, indent=2) + "\n")
    print(a.job, round(row["render_and_save_seconds"], 2), round(timing["worker_wall_seconds"], 2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("receipt", "timing", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--archive-sha", required=True)
    main(p.parse_args())
