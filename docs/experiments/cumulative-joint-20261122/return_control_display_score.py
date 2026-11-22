"""Score same-gap /002 AV1 through the common RGB display-to-luma path."""

import hashlib
import json
import math
import subprocess
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np
from src.codecs import svtav1
from src.pointstream import tracked_patch

root = Path("/home/itec/emanuele/Datasets/pointstream-data")
out = Path("/local/users/emanuele/pointstream/audits/return-control-display-20261122")
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
job = root / "jobs/fleet/inbox/20261121T234041Z-ad3f14db/full/published.tar"
with tarfile.open(job) as tar:
    ivf = tar.extractfile("publish/960-q63.5.ivf").read()
assert (
    hashlib.sha256(ivf).hexdigest()
    == "66d3af87818b76c12435a9b2b4f5e9797a219a3d32faf4aface4568af74b8bf8"
)
dec = svtav1.tool("dav1d")
tracked_patch.decode_native(
    ivf,
    out / "control.yuv",
    {"width": 960, "height": 544, "fps": [50, 1], "frames": 240},
    dec["real_path"],
)
control = np.memmap(out / "control.yuv", np.uint8, "r", shape=(240, 816, 960))
ff = svtav1.tool("ffmpeg")["path"]
source = root / "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4"
subprocess.run(
    [
        ff,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-frames:v",
        "240",
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
        str(out / "source.yuv"),
    ],
    check=True,
    timeout=120,
)
truth = np.memmap(out / "source.yuv", np.uint8, "r", shape=(240, 408, 480))
sse = 0
for t, frame in enumerate(control):
    rgb = cv2.resize(
        tracked_patch.yuv_to_rgb(frame, 960, 544, False),
        (1920, 1080),
        interpolation=cv2.INTER_CUBIC,
    )
    small = cv2.resize(rgb, (480, 272), interpolation=cv2.INTER_AREA).astype(np.float64)
    y = np.clip(
        np.rint(
            16
            + 219
            * (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2])
            / 255
        ),
        0,
        255,
    ).astype(np.int32)
    d = truth[t, :272].astype(np.int32) - y
    sse += int(np.square(d).sum(dtype=np.int64))
r = {
    "frames": 240,
    "sse": sse,
    "display_luma_psnr": 10 * math.log10(255**2 * 480 * 272 * 240 / sse),
    "native_y_luma_psnr": 37.024792124667776,
    "stream_sha256": hashlib.sha256(ivf).hexdigest(),
    "elapsed_seconds": time.monotonic() - started,
}
(out / "receipt.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
