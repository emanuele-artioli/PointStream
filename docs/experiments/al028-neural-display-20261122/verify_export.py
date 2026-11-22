"""Verify and extract the second-match native neural display for joint comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(producer, recovery, folder):
    if sha(producer) != "63104b7832ab9d8f8e6125bc2acf852feffd4954ff3950ce33d08f6523270e49":
        raise ValueError("original producer identity")
    with tarfile.open(producer) as archive:
        reports = json.load(archive.extractfile("publish/al028/dcvc-s2-q24-reports.json"))
        packet = archive.extractfile("publish/al028/dcvc-s2-q24.psdc").read()
        source = json.load(archive.extractfile("publish/al028/result.json"))
    if (source["window"], source["wid"], source["fps"], source["frames"]) != (
        "al028",
        "alcaraz_hurkacz/028@120",
        25,
        240,
    ):
        raise ValueError("producer source window or native timing")
    passes = [p for d in reports["decode"].values() for p in d["passes"]]
    assert len(passes) == 4 and len(packet) == 24494
    folder.mkdir(exist_ok=False)
    output = folder / "full.yuv"
    with tarfile.open(recovery) as archive:
        r = json.load(archive.extractfile("publish/stroke.json"))
        raw = archive.extractfile("publish/full.yuv")
        assert r["stage"] == "full" and r["window"] == "al028" and r["size"] == [960, 544]
        assert r["packet_sha256"] == hashlib.sha256(packet).hexdigest()
        checks = []
        with output.open("xb") as f:
            for _ in range(240):
                frame = raw.read(960 * 544 * 3 // 2)
                assert len(frame) == 960 * 544 * 3 // 2
                checks.append(hashlib.sha256(frame).hexdigest())
                f.write(frame)
        assert raw.read(1) == b"" and all(
            p["frame_sha256"] == p["yuv420_sha256"] == checks for p in passes
        )
    assert (
        sha(output) == r["full_yuv_sha256"]
        and r["fps"] == 25
        and r["historical_passes_verified"] == 4
    )
    receipt = dict(
        producer="20261030T221148Z-4973d2f4",
        recovery="20261122T035934Z-8f78ba52",
        producer_archive_sha256=sha(producer),
        recovery_full_archive_sha256=sha(recovery),
        packet_sha256=r["packet_sha256"],
        complete_bytes=len(packet),
        native_size=[960, 544],
        fps=25,
        frames=240,
        kbps=len(packet) * 8 / (1000 * 9.6),
        raw_sha256=r["full_yuv_sha256"],
        original_decoder_processes=2,
        original_passes_per_process=2,
        original_hashes_reproduced=True,
        recovery_report=r,
    )
    (folder / "verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(checks)} native frames at {receipt['kbps']:.3f} kbps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "recovery", "folder"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    verify(a.producer, a.recovery, a.folder)
