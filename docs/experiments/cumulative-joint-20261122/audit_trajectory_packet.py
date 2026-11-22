"""Independently count and hash every real trajectory PSB section."""

import hashlib
import json
import struct
from pathlib import Path

path = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/tennis/cumulative-joint-20261122/three-segment-trajectory-v1/delivery.psb"
)
raw = path.read_bytes()
assert raw[:4] == b"PSB1"
size = struct.unpack_from(">I", raw, 4)[0]
header = json.loads(raw[8 : 8 + size])
offset = 8 + size
for row in header["entries"]:
    data = raw[offset : offset + row["bytes"]]
    offset += len(data)
    assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
assert offset == len(raw)
result = {
    "packet_path": str(path),
    "packet_sha256": hashlib.sha256(raw).hexdigest(),
    "total_bytes": len(raw),
    "index_bytes": 8 + size,
    "entries": header["entries"],
    "segments": header["segments"],
    "trajectory": header["trajectory"],
    "parsed_to_eof": True,
}
print(json.dumps(result, indent=2))
