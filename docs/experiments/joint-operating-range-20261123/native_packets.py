"""Replay transferred IVF packets and require complete decoded YUV identity."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
import zlib


def write_member(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, BytesIO(data))


def sample(packets, receipt, output, decoder):
    plan = json.loads(receipt.read_text())
    assert len(plan["rows"]) == 1 and plan["rows"][0]["state"] == "complete"
    row = plan["rows"][0]
    width, height = row["width"], row["height"]
    stride = width * height * 3 // 2
    ticks = plan["ticks"]
    content = packets.read_bytes()
    offset = 0
    by_digest = {}
    while offset < len(content):
        start = offset
        assert content[start:start + 4] == b"DKIF"
        assert struct.unpack_from("<H", content, start + 6)[0] == 32
        offset += 32
        for _ in range(240):
            size = struct.unpack_from("<I", content, offset)[0]
            offset += 12 + size
            assert offset <= len(content)
        packet = content[start:offset]
        by_digest[sha256(packet).hexdigest()] = packet
    assert len(by_digest) == len(row["selected"])
    with tempfile.TemporaryDirectory(prefix="pointstream-av1-") as scratch:
        root = Path(scratch)
        with tarfile.open(output, "x") as tar:
            for label, point in row["selected"].items():
                assert point["affordable"]
                size = point["bytes"]
                packet = by_digest.pop(point["sha256"])
                assert len(packet) == size
                source = root / f"{label}.ivf"
                decoded = root / f"{label}.yuv"
                source.write_bytes(packet)
                subprocess.run(
                    [decoder, "-i", str(source), "-o", str(decoded),
                     "--muxer", "yuv", "--threads", "1"],
                    check=True, capture_output=True, timeout=120,
                )
                assert decoded.stat().st_size == 240 * stride
                with decoded.open("rb") as data:
                    digest = sha256()
                    frames = {}
                    picked = {}
                    for tick in range(240):
                        frame = data.read(stride)
                        digest.update(frame)
                        if tick in ticks:
                            frames[str(tick)] = sha256(frame).hexdigest()
                            picked[tick] = frame
                assert digest.hexdigest() == point["decoded_sha256"]
                prefix = f"{row['mode']}-{width}-{label}"
                for tick, frame in picked.items():
                    write_member(tar, f"{prefix}/{tick}.z", zlib.compress(frame, 1))
                meta = dict(
                    width=width, height=height, band=label, mode=row["mode"], job=row["job_id"],
                    archive_sha256=row["archive_sha256"], bitstream_sha256=point["sha256"],
                    bytes=size, decoded_sha256=digest.hexdigest(), frames=frames,
                )
                write_member(tar, f"{prefix}/meta.json", json.dumps(meta).encode())
            assert not by_digest
            write_member(tar, "sampling.json", json.dumps(dict(ticks=ticks)).encode())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("packets", "receipt", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--decoder", default="dav1d")
    args = parser.parse_args()
    sample(args.packets, args.receipt, args.output, args.decoder)
