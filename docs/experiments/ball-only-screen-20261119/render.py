"""Render the 12 reviewed ball-only frames and authenticate the complete streams."""
import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball
from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

DATA = Path('/Users/manu/Datasets/pointstream-submission-20261118/ps-3rate-ball-20261118')
ROOT = Path('docs/experiments/ball-on-development-rally-20261118')
ANCHORS = Path('/Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/report.json')
LABELS = Path('docs/experiments/foreground-references-20261118/finished-review-1621utc.json')
DEST = Path('/private/tmp/ps-regional-ball-only-frames-20261119')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    cv2.setNumThreads(1)
    assert cv2.__version__ == '4.10.0' and np.__version__ == '1.26.4'
    original = json.loads((ROOT / 'three-rate.json').read_text())
    report = json.loads(ANCHORS.read_text())
    labels = json.loads(LABELS.read_text())
    ticks = labels['ticks']
    require(ticks == list(range(10, 231, 20)) and labels['source_sha256'] == original['source_sha256'] == report['source_sha256'], 'source or tick identity')
    require(sha(ANCHORS) == original['anchor_report_sha256'], 'anchor ledger identity')
    decoder = svtav1.tool('dav1d')
    require(decoder['sha256'] == original['decoder']['sha256'], 'decoder binary identity')
    DEST.mkdir(exist_ok=False)
    results = []
    for row in original['rows']:
        rate = row['crf']
        inp = DATA / str(rate)
        carrier_path, sidecar_path, base_path = inp/'native.pscp', inp/'ball.sidecar', inp/'decoded.yuv'
        require(sha(carrier_path) == row['native_sha256'] and sha(sidecar_path) == row['sidecar_sha256'], 'packet identity')
        base_entry = next(x for x in report['rows'] if x['mode'] == 'plain' and x['crf'] == rate)
        require(sha(base_path) == base_entry['decoded_sha256'], 'base decoded YUV identity')
        require(base_path.stat().st_size == 240 * 816 * 960, 'base size')
        carrier, sidecar = carrier_path.read_bytes(), sidecar_path.read_bytes()
        positions, _, size = ball._open(carrier, sidecar, decoder['sha256'])
        require(size == (1920, 1080) and len(positions) == 240, 'canvas geometry')
        frames = np.memmap(base_path, np.uint8, 'r', shape=(240, 816, 960))
        folder = DEST / str(rate)
        folder.mkdir()
        def canvases():
            for t, frame in enumerate(frames):
                yield t, cv2.resize(yuv_to_rgb(frame, 960, 544, False), size, interpolation=cv2.INTER_CUBIC)
        whole = hashlib.sha256()
        saved = {}
        count = 0
        for t, frame in ball.render(carrier, sidecar, canvases(), scratch=folder, decoder=decoder['path']):
            require(t == count and frame.shape == (1080, 1920, 3), 'frame coverage')
            whole.update(frame)
            count += 1
            if t in ticks:
                path = folder/f'{t:03d}.png'
                require(cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)), 'PNG encoding')
                check = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                require(np.array_equal(frame, check), 'PNG roundtrip')
                saved[str(t)] = {'sha256': sha(path), 'ball_origin': positions[t]}
        require(count == 240 and whole.hexdigest() == row['all_frame_rgb_sha256'], 'full RGB stream identity')
        require(len(saved) == 12, 'review coverage')
        results.append(dict(crf=rate, native_sha256=row['native_sha256'], sidecar_sha256=row['sidecar_sha256'],
                            base_yuv_sha256=base_entry['decoded_sha256'], stream_rgb_sha256=whole.hexdigest(),
                            total_bytes=len(carrier)+len(sidecar), total_kbps=(len(carrier)+len(sidecar))/600,
                            fps=50, seconds=4.8, saved=saved))
    elapsed = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(source_sha256=original['source_sha256'], label_export_sha256=labels['export_sha256'],
                  ledger_sha256=sha(ROOT / 'three-rate.json'), anchor_sha256=sha(ANCHORS), decoder=decoder,
                  opencv=cv2.__version__, numpy=np.__version__, rows=results,
                  cpu_seconds=elapsed.ru_utime+elapsed.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime,
                  wall_seconds=time.monotonic()-started)
    require(result['cpu_seconds'] < 180, 'local CPU limit')
    (DEST/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'rows': [{'crf': x['crf'], 'total_kbps': x['total_kbps'], 'total_bytes': x['total_bytes']} for x in results],
                      'cpu_seconds': result['cpu_seconds'], 'wall_seconds': result['wall_seconds'], 'output': str(DEST)}, indent=2))


if __name__ == '__main__':
    main()
