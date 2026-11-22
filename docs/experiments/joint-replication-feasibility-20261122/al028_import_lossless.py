import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

src = Path("/private/tmp/ps-al028-frames20-validated.mkv")
rpath = Path("/private/tmp/ps-al028-linux-score-result/full/result.json")
r = json.loads(rpath.read_text())
out = Path("/private/tmp/ps-al028-joint-primary-validated")
out.mkdir(exist_ok=False)
(out / "result.json").write_bytes(rpath.read_bytes())
p = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(src),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
arr = np.lib.format.open_memmap(
    out / "frames-20.npy", mode="w+", dtype=np.uint8, shape=(240, 1080, 1920, 3)
)
h = hashlib.sha256()
n = 1080 * 1920 * 3
for i in range(240):
    b = bytearray()
    while len(b) < n:
        part = p.stdout.read(n - len(b))
        if not part:
            raise ValueError("premature EOF")
        b.extend(part)
    h.update(b)
    arr[i] = np.frombuffer(b, np.uint8).reshape(1080, 1920, 3)
if p.stdout.read(1):
    raise ValueError("unexpected extra RGB data")
err = p.stderr.read()
code = p.wait(timeout=60)
if code or h.hexdigest() != r["rgb_stream_sha256"]["20"]:
    raise ValueError(f"lossless validation failed: {err[-1000:]!r}")
arr.flush()
print(
    "frames",
    240,
    "rgb_sha256",
    h.hexdigest(),
    "result_sha256",
    hashlib.sha256(rpath.read_bytes()).hexdigest(),
)
