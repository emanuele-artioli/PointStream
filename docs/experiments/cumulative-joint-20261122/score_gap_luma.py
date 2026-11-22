"""Score the actual 240x136 gap against the full-passage 480x272 luma support."""

import hashlib
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np

root = Path("/local/users/emanuele/pointstream/audits/cumulative-gap-independent-20261122")
source = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/tennis/cumulative-joint-20261122/nadal-kyrgios-broadcast.mp4"
)
ff = "/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg"
decoded = root / "decoded.yuv"
assert (
    hashlib.sha256(decoded.read_bytes()).hexdigest()
    == "a8f847be912988f9a7c75535d9f23802120f86566530d8a3426c894cb54b19a2"
)
assert len(os.sched_getaffinity(0)) == 1
n = 10825
started = time.monotonic()
cmd = [
    ff,
    "-v",
    "error",
    "-nostdin",
    "-ss",
    "1537.34",
    "-i",
    str(source),
    "-frames:v",
    str(n),
    "-vf",
    "scale=480:272:flags=area:out_range=tv,format=yuv420p",
    "-filter_threads",
    "1",
    "-fps_mode",
    "passthrough",
    "-threads",
    "1",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "yuv420p",
    "-",
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
frames = np.memmap(decoded, np.uint8, "r", shape=(n, 204, 240))
sse = 0
count = 480 * 272 * 3 // 2
try:
    for t in range(n):
        raw = p.stdout.read(count)
        if len(raw) != count:
            raise ValueError(f"short source at {t}")
        ref = np.frombuffer(raw, np.uint8).reshape(408, 480)[:272]
        y = cv2.resize(frames[t, :136], (480, 272), interpolation=cv2.INTER_CUBIC)
        delta = ref.astype(np.int32) - y
        sse += int(np.sum(np.square(delta), dtype=np.int64))
    if p.wait(timeout=30) != 0:
        raise RuntimeError(p.stderr.read().decode())
finally:
    if p.poll() is None:
        p.kill()
        p.wait()
r = {
    "frames": n,
    "support": [480, 272],
    "sse": sse,
    "luma_psnr": 10 * math.log10(255**2 * n * 480 * 272 / sse),
    "gap_stream_sha256": "0b4960e92ce07db9dabee33daa98a705c215191ba042c6291b1c546a6a87c1d3",
    "decoded_sha256": "a8f847be912988f9a7c75535d9f23802120f86566530d8a3426c894cb54b19a2",
    "source_sha256": "676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66",
    "source_start": 76866,
    "elapsed_seconds": time.monotonic() - started,
    "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
    + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
    "method": "source area TV limited yuv420p; decoded Y resized INTER_CUBIC; integer SSE, peak 255",
}
path = root / "luma-score.json"
with path.open("x") as f:
    json.dump(r, f, indent=2)
print(json.dumps(r))
