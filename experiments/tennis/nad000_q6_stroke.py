"""Same-host full decode-only replay, preserving three verified stroke frames."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from experiments.tennis import starved_dcvc_regions as prior

PACKET = "1d5c6f5e19e360b038e73ad8018075854ecea98c0a9ed05d3801f492bf02a30b"
TICKS = (78, 86, 94)
SIZE = 960 * 544 * 3 // 2


def run(a):
    import torch

    start = time.monotonic()
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4 or torch.cuda.device_count() != 1:
        raise ValueError(f"insufficient isolation: devices={torch.cuda.device_count()} allowed_cpus={allowed}")
    os.sched_setaffinity(0, set(allowed[:4]))
    if len(os.sched_getaffinity(0)) != 4:
        raise ValueError("four-CPU affinity was not enforced")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    shared = os.environ.get("PS_GPU_SHARING") == "shared"
    if shared:
        if os.environ.get("PS_GPU_MEMORY_LIMIT_MIB") != "8192":
            raise ValueError("shared GPU claim mismatch")
        torch.cuda.set_per_process_memory_fraction(2048 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        resource = root / "shared-resource-receipts"
        resource.mkdir()
        os.environ["PS_K5D_SHARED_NUMERICAL"] = "pointstream-parent2-child6-v1"
        os.environ["PS_K5D_RESOURCE_RECEIPTS"] = str(resource)
    work = root / "stroke-replay"
    work.mkdir()
    publish = root / "publish"
    publish.mkdir()
    prior.PACKETS[6] = PACKET
    with tarfile.open(a.archive) as archive:
        frames, receipt = prior.replay(a, archive, 6, work)
    if shared:
        receipts = list(resource.glob("child-*.json"))
        if len(receipts) != 1:
            raise ValueError("missing DCVC child allocator receipt")
        receipt_data = json.loads(receipts[0].read_text())
        if receipt_data["budget_mib"] != 6144 or receipt_data["action"] != "decode":
            raise ValueError("DCVC child allocator receipt mismatch")
        (publish / "resource-receipt.json").write_text(json.dumps(receipt_data) + "\n")
    shutil.copyfile(work / "dcvc-s2-q6.yuv", publish / "full.yuv")
    if prior.sha(publish / "full.yuv") != receipt["raw_sha256"]:
        raise ValueError("full decoded YUV identity")
    raw = b"".join(frames[t].tobytes() for t in TICKS)
    if len(raw) != 3 * SIZE:
        raise ValueError("stroke size")
    (publish / "stroke.yuv").write_bytes(raw)
    result = {
        "ticks": list(TICKS),
        "frames_verified": 240,
        "fps": 50,
        "size": [960, 544],
        "kbps": receipt["kbps"],
        "packet_sha256": PACKET,
        "hashes": [hashlib.sha256(frames[t].tobytes()).hexdigest() for t in TICKS],
        "stroke_sha256": hashlib.sha256(raw).hexdigest(),
        "full_yuv_sha256": receipt["raw_sha256"],
        "replay": receipt,
        "shared_allocator": shared,
        "stage": os.environ["PS_STAGE"],
        "elapsed_seconds": time.monotonic() - start,
    }
    output = json.dumps(result, indent=2) + "\n"
    (publish / "stroke.json").write_text(output)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(output)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    checks = {
        "ticks": r["ticks"] == list(TICKS),
        "full_decode": r["frames_verified"] == 240
        and r["replay"]["historical_frame_hashes_matched"] == 240,
        "packet": r["packet_sha256"] == PACKET,
    }
    if r["stage"] == "smoke":
        checks["forecast"] = 1.5 * r["elapsed_seconds"] + 30 < 180
    with tarfile.open(stage / "published.tar") as tar:
        if r["shared_allocator"]:
            resource = json.load(tar.extractfile("publish/resource-receipt.json"))
            checks["allocator"] = resource["budget_mib"] == 6144 and resource["action"] == "decode"
        data = tar.extractfile("publish/stroke.yuv").read()
        full = tar.extractfile("publish/full.yuv")
        checks["full_yuv"] = hashlib.file_digest(full, "sha256").hexdigest() == r["full_yuv_sha256"]
        checks["bytes"] = (
            len(data) == 3 * SIZE and hashlib.sha256(data).hexdigest() == r["stroke_sha256"]
        )
        checks["frames"] = all(
            hashlib.sha256(data[i * SIZE : (i + 1) * SIZE]).hexdigest() == h
            for i, h in enumerate(r["hashes"])
        )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n"
    )
    if not all(checks.values()):
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "image", "video"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)


if __name__ == "__main__":
    main()
