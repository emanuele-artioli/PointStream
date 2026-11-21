import hashlib
import json
import tarfile
import sys
import tempfile
from pathlib import Path
scratch = Path(tempfile.mkdtemp(prefix="paid-ball-track-", dir="/local/users/emanuele/pointstream/metadata-prep"))
source = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T220124Z-1481aee0/source.tar")
assert hashlib.sha256(source.read_bytes()).hexdigest() == "c5edc1d35add0c0e05e00768bc6616f8239a482057abf92023b52aa5a5c64d48"
with tarfile.open(source) as source_tar:
    selected = [m for m in source_tar if m.name.startswith("src/pointstream/") and m.isfile()]
    source_tar.extractall(scratch, members=selected, filter="data")
sys.path.insert(0, str(scratch))
from src.pointstream import ball, ball_stream  # noqa: E402

root = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
items = [
    ("fixed", root / "20261121T215556Z-5d39ce00/full/published.tar", "publish/report.json", None),
    ("refitted", root / "20261121T220124Z-1481aee0/full/published.tar", "publish/report.json", "publish/refitted.psbl"),
]
result = {}
for name, archive_path, receipt_name, payload_name in items:
    expected_archive = {
        "fixed": "5519ff97990c0e4aa17b2785f10340aa30acda3568aa2dfb90e38e81dc5b2667",
        "refitted": "4e1bf9df5fc7ff697ec44475bcaba9103319f332c7d7c1b3f2ea885f63480f22",
    }[name]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_archive
    with tarfile.open(archive_path) as archive:
        receipt = json.load(archive.extractfile(receipt_name))
        if name == "fixed":
            best = receipt["best"]["packet_sha256"]
            index = next(i for i, row in enumerate(receipt["trials"]) if row["packet_sha256"] == best)
            payload_name = f"publish/candidate-{index}.psbl"
        else:
            best = receipt["refitted_sha256"]
        data = archive.extractfile(payload_name).read()
    assert hashlib.sha256(data).hexdigest() == best
    packet = ball_stream.unpack(data)
    points, appearances = ball.decode_frames(packet.ball_payload, packet.frames, packet.width, packet.height)
    assert len(points) == 292
    result[name] = {"packet_sha256": best, "packet_bytes": len(data),
                    "positions": [None if p is None else list(p) for p in points]}
print(json.dumps(result))
