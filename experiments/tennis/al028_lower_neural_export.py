"""Recover selected original Al028 DCVC Q36 frames with all-frame identity gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from src.codecs import dcvc_uf_worker

PACKETS = {6: "60353bf942a12e9da7e7dfd0f66e02a348a288bddaac1543b31b47a311d98325", 18: "d3909f4c4bd56d0b3a8b6ef2f5110ed46e8f7ddae1b52cef51fda357367cb7f8", 30: "74e89cd836432e1b8d29d0f373bf17c382f4504a57c757f2c9d382dc09faca31"}
RATES = {6: 7.654166666666667, 18: 14.571666666666665, 30: 28.82916666666667}
TICKS = list(range(10, 240, 20))
SIZE = 960 * 544 * 3 // 2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(a):
    import torch

    start = time.monotonic()
    cpus = sorted(os.sched_getaffinity(0))
    if (
        len(cpus) < 4
        or torch.cuda.device_count() != 1
        or os.environ.get("PS_GPU_MEMORY_LIMIT_MIB") != "8192"
    ):
        raise ValueError("resource allocation")
    os.sched_setaffinity(0, set(cpus[:4]))
    torch.cuda.set_per_process_memory_fraction(
        2048 * 2**20 / torch.cuda.get_device_properties(0).total_memory
    )
    root = Path(os.environ["PS_SCRATCH_DIR"])
    receipts = root / "shared-resource-receipts"
    receipts.mkdir()
    os.environ["PS_K5D_SHARED_NUMERICAL"] = "pointstream-parent2-child6-v1"
    os.environ["PS_K5D_RESOURCE_RECEIPTS"] = str(receipts)
    work = root / "decode"
    work.mkdir()
    publish = root / "publish"
    publish.mkdir()
    with tarfile.open(a.archive) as tar:
        packet = tar.extractfile(f"publish/al028/dcvc-s2-q{a.qp}.psdc").read()
        prior = json.load(tar.extractfile(f"publish/al028/dcvc-s2-q{a.qp}-reports.json"))
    if digest(packet) != PACKETS[a.qp]:
        raise ValueError("packet identity")
    expected = [
        prior["decode"][tag]["passes"][i]["frame_sha256"]
        for tag in ("decode-a", "decode-b")
        for i in (0, 1)
    ]
    if any(len(x) != 240 or x != expected[0] for x in expected):
        raise ValueError("producer four-pass mismatch")
    container = work / "stream.psdc"
    container.write_bytes(packet)
    out = work / "decoded.yuv"
    plan = work / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "structure": "htl",
                "width": 960,
                "height": 544,
                "frame_count": 240,
                "src_type": "yuv420",
                "image_ckpt": str(a.image.resolve()),
                "video_ckpt": str(a.video.resolve()),
                "image_sha256": "b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302",
                "video_sha256": "934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3",
                "container": str(container),
                "out_file": str(out),
                "profile": False,
                "deterministic": True,
                "stream_policy": "legacy",
            }
        )
    )
    report = work / "decode.json"
    command, env, cwd = dcvc_uf_worker.dcvc_command(Path(sys.prefix), "decode", plan, report)
    r = subprocess.run(
        command, env={**os.environ, **env}, cwd=cwd, capture_output=True, timeout=180
    )
    if r.returncode:
        raise ValueError(r.stderr.decode(errors="replace")[-2000:])
    decoded = json.loads(report.read_text())
    arrays = [p["frame_sha256"] for p in decoded["passes"]]
    if not decoded["deterministic"] or len(arrays) != 2 or any(v != expected[0] for v in arrays):
        raise ValueError("all-frame producer replay failed")
    res = list(receipts.glob("child-*.json"))
    if len(res) != 1:
        raise ValueError("allocator receipt")
    resource = json.loads(res[0].read_text())
    if resource["budget_mib"] != 6144 or resource["action"] != "decode":
        raise ValueError("allocator mismatch")
    raw = out.read_bytes()
    if len(raw) != 240 * SIZE:
        raise ValueError("native dimensions")
    selected = b"".join(raw[t * SIZE : (t + 1) * SIZE] for t in TICKS)
    hashes = [digest(selected[i * SIZE : (i + 1) * SIZE]) for i in range(12)]
    if any(h != expected[0][t] for h, t in zip(hashes, TICKS, strict=True)):
        raise ValueError("selected frame mismatch")
    (publish / "review12.yuv").write_bytes(selected)
    record = {
        "source_sha256": "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934",
        "packet_sha256": PACKETS[a.qp],
        "qp": a.qp,
        "ticks": TICKS,
        "fps": 25,
        "size": [960, 544],
        "display_size": [1920, 1080],
        "kbps": RATES[a.qp],
        "frames_verified": 240,
        "frame_sha256": hashes,
        "review12_sha256": digest(selected),
        "allocator": resource,
        "stage": os.environ["PS_STAGE"],
        "elapsed_seconds": time.monotonic() - start,
    }
    text = json.dumps(record, indent=2) + "\n"
    (publish / "review12.json").write_text(text)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(text)


def validate(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    checks = {
        "ticks": r["ticks"] == TICKS,
        "native_frames": r["frames_verified"] == 240,
        "packet": r["packet_sha256"] == PACKETS[a.qp],
        "allocator": r["allocator"]["budget_mib"] == 6144,
    }
    if r["stage"] == "smoke":
        checks["forecast"] = 1.5 * r["elapsed_seconds"] + 30 < 180
    with tarfile.open(stage / "published.tar") as tar:
        raw = tar.extractfile("publish/review12.yuv").read()
        checks["bytes"] = len(raw) == 12 * SIZE and digest(raw) == r["review12_sha256"]
        checks["frames"] = all(
            digest(raw[i * SIZE : (i + 1) * SIZE]) == h for i, h in enumerate(r["frame_sha256"])
        )
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": ok, "checks": checks}) + "\n"
    )
    if not ok:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "image", "video"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--qp", type=int, choices=sorted(PACKETS), required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate(a) if a.validate else run(a)


if __name__ == "__main__":
    main()
