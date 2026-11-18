"""Check a finished foreground reference export against the registered source-only bundle."""

import argparse
import base64
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


PLAYER_IDS = ('person:000000000000', 'person:000000000001')
STATUSES = {'visible', 'absent', 'uncertain'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_mask(data, size):
    with Image.open(BytesIO(data)) as image:
        require(image.format == 'PNG' and image.size == size, 'expected native-size PNG mask')
        rgb = np.asarray(image.convert('RGBA'))
    mask = rgb[:, :, 0]
    require(np.all(rgb[:, :, 3] == 255) and np.all(rgb[:, :, :3] == mask[:, :, None]), 'expected opaque grayscale mask')
    require(np.all((mask == 0) | (mask == 255)), 'expected binary mask')
    return mask


def validate(export_path, bundle_path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    size = tuple(receipt['canvas'])
    data = json.loads(export_path.read_text())
    require(data['schema'] == 'pointstream.foreground-reference.v1', 'expected finished label export')
    require(isinstance(data['reviewer'], str) and data['reviewer'].strip(), 'missing reviewer')
    require((data['width'], data['height']) == size, 'wrong native size')
    require(data['source_sha256'] == receipt['source_sha256'], 'wrong source video')
    require(data['bundle_sha256'] == receipt['zip_sha256'] == hashlib.sha256(bundle_path.read_bytes()).hexdigest(), 'wrong source bundle')
    require([f['tick'] for f in data['frames']] == receipt['ticks'], 'missing, duplicated or out-of-order source frames')
    counts = {key: Counter() for key in [*PLAYER_IDS, 'ball']}
    with zipfile.ZipFile(bundle_path) as bundle:
        for label, expected in zip(data['frames'], receipt['frames'], strict=True):
            tick = expected['tick']
            root = f'frame-{tick:03d}/'
            require(label['source_png_sha256'] == hashlib.sha256(bundle.read(root + 'source.png')).hexdigest(), f'wrong source frame at {tick}')
            provenance = 'human_without_proposal' if expected['blind'] else 'human_reviewed_model_assisted'
            require(label['provenance'] == provenance, f'wrong proposal provenance at {tick}')
            require([p['id'] for p in label['players']] == list(PLAYER_IDS), f'wrong player identities at {tick}')
            for player in label['players']:
                player_id = player['id']
                status = player['status']
                name = root + player_id.replace(':', '-') + '.png'
                proposal = bundle.read(name) if player_id in expected.get('player_proposals', []) else None
                require(player['proposal_sha256'] == (hashlib.sha256(proposal).hexdigest() if proposal else None), f'wrong proposal at {tick}')
                require(status in STATUSES, f'missing player decision at {tick}')
                counts[player_id][status] += 1
                if status != 'visible':
                    require(player['mask'] is None and player['method'] is None, f'{status} player carries a mask at {tick}')
                    continue
                mask = decode_mask(base64.b64decode(player['mask']['png_base64'], validate=True), size)
                area = np.count_nonzero(mask)
                require(area > 0 and area == player['mask']['foreground_pixels'], f'wrong mask area at {tick}')
                if player['method'] == 'human_accepted_proposal':
                    require(proposal is not None and np.array_equal(mask, decode_mask(proposal, size)), f'accepted proposal differs at {tick}')
                else:
                    require(player['method'] == 'human_drawn_or_corrected', f'unknown player review method at {tick}')
            ball = label['ball']
            require(ball['proposal_xy'] == expected.get('ball_center_proposal'), f'wrong ball proposal at {tick}')
            require(ball['status'] in STATUSES, f'missing ball decision at {tick}')
            counts['ball'][ball['status']] += 1
            center = ball['center_xy']
            if ball['status'] == 'visible':
                require(isinstance(center, list) and len(center) == 2 and all(type(n) is int and 0 <= n < limit for n, limit in zip(center, size)), f'invalid ball center at {tick}')
            else:
                require(center is None, f'{ball["status"]} ball carries a center at {tick}')
    return {'export_sha256': hashlib.sha256(export_path.read_bytes()).hexdigest(), 'bundle_sha256': data['bundle_sha256'], 'frames': len(data['frames']), 'decisions': counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('finished_labels', type=Path)
    parser.add_argument('source_bundle', type=Path)
    args = parser.parse_args()
    result = validate(args.finished_labels, args.source_bundle, Path(__file__).with_name('receipt.json'))
    print(json.dumps(result, indent=2))
