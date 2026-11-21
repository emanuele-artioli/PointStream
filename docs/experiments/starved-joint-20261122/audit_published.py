"""Read-only audit of the three native IVF members and their full-stage receipt."""
import hashlib
import json
import struct
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "native-report.json").read_text())
assert report["source_sha256"] == "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
assert report["source_fps"] == 25 and report["window_frames"] == 240
assert report["converted_dimensions"] == [480, 272] and report["stage"] == "full"
assert [row["crf"] for row in report["rows"]] == [36, 48, 60]
summaries = []
with tarfile.open(root / "published.tar") as archive:
    for row in report["rows"]:
        member = archive.extractfile("publish/" + row["published_stream"])
        assert member is not None
        raw = member.read()
        assert hashlib.sha256(raw).hexdigest() == row["stream_sha256"]
        assert len(raw) == row["file_bytes"] and raw[:4] == b"DKIF"
        assert struct.unpack_from("<HHIIII", raw, 12) == (480, 272, 25, 1, 240, 0)
        offset = 32
        payload = 0
        for tick in range(240):
            size, timestamp = struct.unpack_from("<IQ", raw, offset)
            assert timestamp == tick and offset + 12 + size <= len(raw)
            payload += size
            offset += 12 + size
        assert offset == len(raw) and payload == row["payload_bytes"]
        rate = len(raw) * 8 * 25 / 240 / 1000
        assert abs(rate - row["complete_kbps"]) < 1e-9
        assert row["decoded_frames"] == 240
        summaries.append(dict(crf=row["crf"], kbps=rate, bytes=len(raw),
                              stream_sha256=row["stream_sha256"], decoded_sha256=row["decoded_sha256"]))
print(json.dumps({"passed": True, "streams": summaries,
                  "native_report_sha256": hashlib.sha256((root / "native-report.json").read_bytes()).hexdigest(),
                  "published_tar_sha256": hashlib.sha256((root / "published.tar").read_bytes()).hexdigest()}))
