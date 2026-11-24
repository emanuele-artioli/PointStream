"""Measure physical AV1 and PSC1 cost for a source-derived split at tick 93."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

from src.codecs import svtav1
from src.pointstream import codec

ROOT = Path("/private/tmp/ps-tenniset-av1-first-20261123")
SRC = Path("/private/tmp/ps-tenniset-native-points-20261123/v006_p0014.mkv")
SOURCE = "8ab59163aa86f3cb0f26b1981e6e546f148cca916e40731c0cdee5c6621e4b78"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    assert digest(SRC) == SOURCE
    tool = {name: svtav1.tool(name) for name in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    began = time.monotonic()
    with TemporaryDirectory(prefix="ps-london-split-") as temporary:
        root = Path(temporary)
        raw = root / "source.yuv"
        command = [tool["ffmpeg"]["path"], "-nostdin", "-v", "error", "-threads", "1", "-i",
                   str(SRC), "-an", "-vf", "scale=640:360:flags=area:out_range=tv,format=yuv420p",
                   "-f", "rawvideo", str(raw)]
        subprocess.run(command, check=True, timeout=30)
        assert digest(raw) == digest(ROOT / "input-640.yuv")
        data = raw.read_bytes()
        frame_size = 640 * 360 * 3 // 2
        assert len(data) == 106 * frame_size
        outputs = []
        for start, end in ((0, 93), (93, 106)):
            piece = root / f"{start}.yuv"
            piece.write_bytes(data[start * frame_size:end * frame_size])
            stream = root / f"{start}.ivf"
            encoded = svtav1.encode_command(tool["SvtAv1EncApp"]["path"], piece, stream, width=640,
                height=360, fps=Fraction(25), frames=end-start, crf=70, preset=4, full_range=False)
            svtav1.run(encoded, timeout=45)
            native = stream.read_bytes()
            meta = {"version": 1, "profile": "native-av1", "width": 640, "height": 360,
                    "fps": [25, 1], "frames": end-start, "full_range": False,
                    "decoder_sha256": tool["dav1d"]["sha256"]}
            packet = codec.pack(meta, native)
            assert codec.unpack(packet) == (meta, native)
            decoded = root / f"{start}-decoded.yuv"
            svtav1.run(svtav1.decode_command(tool["dav1d"]["path"], stream, decoded, threads=1), timeout=30)
            assert decoded.stat().st_size == frame_size * (end-start)
            outputs.append({"start": start, "stop": end, "ivf_bytes": len(native),
                "packet_bytes": len(packet), "ivf_sha256": digest(stream),
                "decoded_sha256": digest(decoded), "encoder_arguments": encoded[5:]})
        result = {"scope": "ordinary AV1 on both sides, no player or ball, 640x360 crf70",
            "source_sha256": SOURCE, "input_yuv_sha256": digest(raw),
            "script_sha256": digest(Path(__file__)), "tools": tool, "ceiling_bytes": 10812,
            "unsplit_av1_bytes": (ROOT / "640-crf70.ivf").stat().st_size,
            "packets": outputs, "total_packet_bytes": sum(v["packet_bytes"] for v in outputs),
            "elapsed_seconds": time.monotonic() - began}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
