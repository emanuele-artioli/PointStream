"""Validate finished 24-tick AVIF crops and encode small local review clips."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.tennis.player_requalification import sha256, validate_output
from experiments.tennis.player_reference_review import UNITS, resolved
from experiments.tennis.player_requalification_preview import encode

p = argparse.ArgumentParser()
for name in ('config', 'conditioning', 'base', 'smoke', 'full', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--config-sha256', required=True)
a = p.parse_args()
config = resolved(a.config, a.config_sha256, a.conditioning, a.base)
assert [u['unit'] for u in config['units']] == UNITS
s = validate_output(a.smoke, config, a.config_sha256, 'tennis_only', 1)
f = validate_output(a.full, config, a.config_sha256, 'tennis_only', 2)
assert s['plan']['units'][0] == f['plan']['units'][0]
repeated = UNITS[0] + '@tennis_only.npz'
with np.load(a.smoke / repeated, allow_pickle=False) as x, np.load(a.full / repeated, allow_pickle=False) as y:
    assert set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
a.output.mkdir(parents=True, exist_ok=False)
items = {}
for unit in UNITS:
    with np.load(a.full / f'{unit}@tennis_only.npz', allow_pickle=False) as inp:
        crops = inp['crops']
        assert crops.shape == (24, 784, 512, 3) and crops.dtype == np.uint8
        for fps in (10, 50):
            encode(crops, a.output / f'{unit}-{fps}.mp4', fps)
        Image.fromarray(crops[12]).save(a.output / f'{unit}.png')
    items[unit] = {path.name: dict(sha256=sha256(path), bytes=path.stat().st_size)
                   for path in a.output.glob(unit + '*')}
for path in a.output.glob('*.mp4'):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)],
                           check=True, text=True, capture_output=True)
    assert json.loads(probe.stdout)['streams'][0]['nb_read_frames'] == '24'
result = dict(config_sha256=a.config_sha256, smoke_receipt_sha256=sha256(a.smoke/'receipt.json'),
              full_receipt_sha256=sha256(a.full/'receipt.json'), repeated_player_equal=True,
              units=items, source='AVIF-conditioned production worker, 24 ticks',
              script_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest())
(a.output/'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
