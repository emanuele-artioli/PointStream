"""Independently assemble decoded base and tiles and compare all receiver hashes."""

import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from experiments.tennis import common_player_ball as ball, common_player_packet as player
from src.codecs import svtav1
from src.pointstream import tracked_patch

j = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T233145Z-06141332/full"
)
carrier = (j / "native.pscp").read_bytes()
side = (j / "ball.sidecar").read_bytes()
expected = json.loads((j / "binding.json").read_text())
dec = svtav1.tool("dav1d")
positions, tile, size = ball._open(carrier, side, dec["sha256"])
parts, _, _ = player.unpack(carrier)
beg = time.monotonic()
root = Path("/local/users/emanuele/pointstream/audits/return-ball-independent-20261122")
root.mkdir(parents=True, exist_ok=False)
tracked_patch.decode_native(
    parts["base"],
    root / "base.yuv",
    {"width": 240, "height": 136, "fps": [50, 1], "frames": 240},
    dec["real_path"],
)
tracked_patch.decode_native(
    tile,
    root / "tile.yuv",
    {"width": 64, "height": 64, "fps": [50, 1], "frames": 240},
    dec["real_path"],
)
b = np.memmap(root / "base.yuv", np.uint8, "r", shape=(240, 204, 240))
p = np.memmap(root / "tile.yuv", np.uint8, "r", shape=(240, 96, 64))
matches = 0
for t in range(240):
    image = cv2.resize(
        tracked_patch.yuv_to_rgb(b[t], 240, 136, False), size, interpolation=cv2.INTER_CUBIC
    )
    if positions[t] is not None:
        x, y = positions[t]
        crop = tracked_patch.yuv_to_rgb(p[t], 64, 64, False)[16:48, 16:48]
        image[y : y + 32, x : x + 32] = crop
    matches += hashlib.sha256(image.tobytes()).hexdigest() == expected["rendered_hashes"][t]
assert matches == 240
result = {
    "matched_frames": matches,
    "total_frames": 240,
    "carrier_sha256": hashlib.sha256(carrier).hexdigest(),
    "sidecar_sha256": hashlib.sha256(side).hexdigest(),
    "elapsed_seconds": time.monotonic() - beg,
    "independent_assembly": True,
}
(root / "audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
