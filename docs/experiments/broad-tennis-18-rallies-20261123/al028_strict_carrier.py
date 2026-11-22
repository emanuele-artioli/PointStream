"""Reproduce the measured CRF54 packet from four original archived source members."""

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from experiments.tennis import common_player_packet, w8_payload
from src.codecs import svtav1


def main(archive_members, output):
    here = Path(__file__).resolve().parent.parent
    manifest = json.loads((here / "al028-neural-display-20261122/common-budget.json").read_text())
    target = next(r for r in manifest["rows"]["joint"] if r["crf"] == 54)
    spec = importlib.util.spec_from_file_location(
        "donor", here / "joint-replication-feasibility-20261122/al028_donor.py"
    )
    donor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(donor)
    output.mkdir(exist_ok=False)
    donor.CARRIER = output / "portable.pscp"
    donor.SOURCE = output / "source.mp4"
    donor.BALL = output / "ball.payload"
    donor.OUT = output / "clean"
    previous = output / "rally-53.5.psj"
    with archive_members.open("rb") as f:
        for path, n in [
            (donor.CARRIER, 168267),
            (donor.SOURCE, 3017138),
            (donor.BALL, 996),
            (previous, 24489),
        ]:
            path.write_bytes(f.read(n))
        assert not f.read()
    assert donor.sha(previous) == manifest["selected"]["joint"]["stream_sha256"]
    donor.main(portable=True)
    cleaned = donor.OUT / "clean.yuv"
    assert donor.sha(cleaned) == "5895cb05c85ee72f57d919eb15278ad6043d842eb1ca30cd4844ad1486ada7af"
    scaled = output / "scaled.yuv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "960x544",
            "-r",
            "25",
            "-i",
            str(cleaned),
            "-vf",
            "scale=320:192:flags=area",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            str(scaled),
        ],
        check=True,
        timeout=45,
    )
    assert donor.sha(scaled) == manifest["input_sha256"]["joint"]
    native = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    assert all(native[k]["sha256"] == manifest["native_tools"][k]["sha256"] for k in native)
    pkt = previous.read_bytes()
    prof = common_player_packet._embedded_profile(pkt)
    parts, _ = w8_payload.unpack(pkt, prof)
    for q in (53.5, 54):
        root = output / f"q{q}"
        root.mkdir()
        r = svtav1.code(
            scaled,
            root,
            width=320,
            height=192,
            fps=Fraction(25),
            frames=240,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=native["SvtAv1EncApp"]["path"],
            decoder=native["dav1d"]["path"],
            timeout=90,
        )
        row = next(i for i in manifest["rows"]["joint"] if i["crf"] == q)
        stream = Path(r["stream"]).read_bytes()
        assert (
            hashlib.sha256(stream).hexdigest() == row["ivf_sha256"]
            and donor.sha(Path(r["decoded"])) == row["raw_sha256"]
        )
        full = w8_payload.pack(
            prof,
            [
                (k, "frame", stream if k == "base" else parts[k])
                for k in ("base", "poses", "trajectory")
            ],
        )
        assert (
            len(full) == row["bytes"] and hashlib.sha256(full).hexdigest() == row["stream_sha256"]
        )
        (root / "rally.psj").write_bytes(full)
    print("Recovered", target["bytes"], target["stream_sha256"], flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
