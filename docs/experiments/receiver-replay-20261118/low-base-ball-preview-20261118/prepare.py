"""Rebind the checked ball channel to the 480x272 receiver carrier."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from experiments.tennis import common_player_ball as ball
from src.pointstream.tracked_patch import yuv_to_rgb

OUT = Path('/private/tmp/ps-low-ball-stage-20261118')
CARRIER = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
PACKET = Path('/Users/manu/Datasets/pointstream-submission-20261118/selected-rate-support-linux-png/packets/48')
DAV1D_SHA = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'

def sha(data):
    return hashlib.sha256(data).hexdigest()

old = CARRIER.read_bytes()
new = (PACKET / 'players.pscp').read_bytes()
side = (PACKET / 'ball.sidecar').read_bytes()
assert sha(old) == '3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd'
assert sha(side) == '616f63b0f3f2ca84e3eb87645d1a0877d29a6aa7130d986c31a3a4f0a7b3b2d2'
positions, tile, size = ball._open(new, side, DAV1D_SHA)
assert size == (1920, 1080)
rebound = ball.pack(old, positions, tile, decoder_sha256=DAV1D_SHA)
assert ball._open(old, rebound, DAV1D_SHA) == (positions, tile, size)
raw = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-c:v', 'libdav1d', '-f', 'ivf', '-i', 'pipe:0', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1'], input=tile, capture_output=True, check=True, timeout=35).stdout
assert len(raw) == 240 * 96 * 64 and sha(raw) == '9c997a21482fdc2fac4e36d7511a86158ed2eddcaf6aca57594e06ddd067bb1a'
planes = np.frombuffer(raw, np.uint8).reshape(240, 96, 64)
patches = np.stack([yuv_to_rgb(frame, 64, 64, False)[16:48, 16:48] for frame in planes])
coords = np.array([p if p is not None else (-1, -1) for p in positions], dtype=np.int32)
assert len(positions) == 240 and sum(p is not None for p in positions) == 195
np.savez(OUT / 'ball-input.npz', patches=patches, coords=coords)
(OUT / 'ball.sidecar').write_bytes(rebound)
meta = {'source_sha256': 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d', 'carrier_sha256': sha(old), 'carrier_bytes': len(old), 'manifest_bytes': 625, 'sidecar_sha256': sha(rebound), 'sidecar_bytes': len(rebound), 'ball_input_sha256': sha((OUT / 'ball-input.npz').read_bytes()), 'tile_yuv_sha256': sha(raw), 'placements': 195}
(OUT / 'input.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta))
