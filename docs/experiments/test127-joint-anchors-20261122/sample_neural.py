"""Extract registered test-rally ticks while authenticating the complete four-pass neural stream."""

from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile
import zlib


def add(archive, name, data):
    record = tarfile.TarInfo(name)
    record.size = len(data)
    archive.addfile(record, io.BytesIO(data))


def collect(request, sink):
    path = Path(request["archive"]).resolve()
    root = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox").resolve()
    assert path == root / request["job_id"] / "full" / "published.tar"
    archive_hash = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            archive_hash.update(chunk)
    assert archive_hash.hexdigest() == request["archive_sha256"]
    ticks = request["ticks"]
    profiles = {
        "8ebca8b693202d8bf7afaee515dde4d723a9fb83250115e6eff080038b0bbf50": (
            [60, 1],
            [9, 32, 44, 68, 92, 115, 127, 151, 175, 198, 210, 234],
        ),
        "fc5d3ee431f84417d2de3412402f4e8efb0546418ce96f1f0a1e2f1887a88b1a": (
            [30000, 1001],
            [9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238],
        ),
    }
    fps, frozen_ticks = profiles[request["source_sha256"]]
    assert ticks == frozen_ticks
    with tarfile.open(path) as archive, tarfile.open(fileobj=sink, mode="w|") as out:
        report = json.load(archive.extractfile("publish/result.json"))
        r = report["native"]
        assert report["source_sha256"] == request["source_sha256"] and report["native_fps"] == fps
        assert report["frames"] == 240 and r["accepted"] and r["identical_frame_hashes"]
        packet = archive.extractfile("publish/stream.psdc").read()
        assert (
            len(packet) == r["bytes"]
            and hashlib.sha256(packet).hexdigest() == r["container_sha256"]
        )
        w, h = r["size"]
        stream = archive.extractfile("publish/decoded.yuv")
        marks = {}
        total = hashlib.sha256()
        assert len(report["yuv_frames"]) == 240
        prefix = f"dcvc-{w}-q{report['qp']}"
        for t in range(240):
            frame = stream.read(w * h * 3 // 2)
            assert (
                len(frame) == w * h * 3 // 2
                and hashlib.sha256(frame).hexdigest() == report["yuv_frames"][t]
            )
            total.update(frame)
            if t in ticks:
                marks[str(t)] = hashlib.sha256(frame).hexdigest()
                add(out, f"{prefix}/{t}.z", zlib.compress(frame, 1))
        assert not stream.read(1) and total.hexdigest() == report["raw_sha256"]
        meta = dict(
            width=w,
            height=h,
            band="native",
            mode=f"DCVC-UF Q{report['qp']}",
            job=request["job_id"],
            archive_sha256=request["archive_sha256"],
            bitstream_sha256=r["container_sha256"],
            bytes=r["bytes"],
            decoded_sha256=report["raw_sha256"],
            frames=marks,
            neural_qp=report["qp"],
        )
        add(out, prefix + "/meta.json", json.dumps(meta).encode())
        add(out, "sampling.json", json.dumps({"ticks": ticks}).encode())


if __name__ == "__main__":
    collect(json.load(sys.stdin), sys.stdout.buffer)
