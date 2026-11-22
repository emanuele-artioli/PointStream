"""Extract fixed native instants while verifying entire archived decoded pictures."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import zlib


def add(target, name, blob):
    item = tarfile.TarInfo(name)
    item.size = len(blob)
    target.addfile(item, io.BytesIO(blob))


def selected_yuv(stream, width, height, ticks):
    stride = width * height * 3 // 2
    digest = hashlib.sha256()
    picked = {t: bytearray() for t in ticks}
    length = 0
    decompressor = zlib.decompressobj()
    while data := stream.read(32768):
        raw = decompressor.decompress(data)
        digest.update(raw)
        finish = length + len(raw)
        for t in ticks:
            start = max(length, t * stride)
            end = min(finish, (t + 1) * stride)
            if start < end:
                picked[t].extend(raw[start - length : end - length])
        length = finish
    assert decompressor.eof and not decompressor.unused_data and not decompressor.unconsumed_tail
    assert length == 240 * stride and all(len(v) == stride for v in picked.values())
    return digest.hexdigest(), picked


def main():
    config = json.load(sys.stdin)
    root = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox").resolve()
    start = time.monotonic()
    seen = set()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as target:
        for row in config["rows"]:
            assert row["state"] == "complete"
            path = Path(row["archive"]).resolve()
            assert path == root / row["job_id"] / "full" / "published.tar"
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(4 << 20), b""):
                    h.update(block)
            assert h.hexdigest() == row["archive_sha256"]
            with tarfile.open(path) as a:
                for label, rec in row["selected"].items():
                    if not rec["affordable"]:
                        continue
                    if config.get("deduplicate") and rec["sha256"] in seen:
                        continue
                    seen.add(rec["sha256"])
                    prefix = f"{row['mode']}-{row['width']}-{label}"
                    packet = a.extractfile(rec["file"]).read()
                    assert (
                        len(packet) == rec["bytes"]
                        and hashlib.sha256(packet).hexdigest() == rec["sha256"]
                    )
                    yuv, frames = selected_yuv(
                        a.extractfile(rec["decoded_file"]),
                        row["width"],
                        row["height"],
                        config["ticks"],
                    )
                    assert yuv == rec["decoded_sha256"]
                    meta = dict(
                        width=row["width"],
                        height=row["height"],
                        band=label,
                        mode=row["mode"],
                        job=row["job_id"],
                        archive_sha256=row["archive_sha256"],
                        bitstream_sha256=rec["sha256"],
                        bytes=rec["bytes"],
                        decoded_sha256=yuv,
                        frames={str(t): hashlib.sha256(v).hexdigest() for t, v in frames.items()},
                    )
                    for tick, data in frames.items():
                        add(target, f"{prefix}/{tick}.z", zlib.compress(data, 1))
                    add(target, f"{prefix}/meta.json", json.dumps(meta).encode())
        add(
            target,
            "sampling.json",
            json.dumps({"seconds": time.monotonic() - start, "ticks": config["ticks"]}).encode(),
        )


if __name__ == "__main__":
    main()
