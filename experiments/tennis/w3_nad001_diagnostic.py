"""One exclusive 25-frame NAD001 exact decoder diagnostic with a W3 AV1 control."""
import json
import os
from pathlib import Path
import sys
import tarfile

from experiments.tennis import w3_neural_baseline as w3


def main():
    control = Path(sys.argv[sys.argv.index('--diagnostic-control') + 1]) if '--diagnostic-control' in sys.argv else None
    if control is not None:
        i = sys.argv.index('--diagnostic-control')
        del sys.argv[i:i+2]
    stage = Path(os.environ['PS_STAGE_DIR'])
    if '--validate' in sys.argv:
        x = json.loads((stage/'w3-summary.json').read_text())
        c = json.loads((stage/'diagnostic-control.json').read_text())
        p = x['windows'][0]['dcvc'][0]
        checks = {'25_native_frames': x['frames'] == 25, 'accepted': p['accepted'], 'four_pass': p['identical_frame_hashes'], 'historical_av1_bytes': c['stream_match'], 'historical_av1_scores': c['max_error'] < 0.001, 'smoke_cost': x['seconds'] * 1.5 + 60 < 300}
        output = {'passed': all(checks.values()), 'checks': checks}
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(output)+'\n')
        if not output['passed']:
            raise SystemExit(str(checks))
        return
    if control is None:
        raise ValueError('historical W3 control required')
    w3.main()
    x = json.loads((stage/'w3-summary.json').read_text())
    with tarfile.open(control) as tar:
        old = json.load(tar.extractfile('publish/nad001/result.json'))
    now = next(p for p in x['windows'][0]['svt'] if p['scale'] == 8)
    prior = next(p for p in old['svt'] if p['scale'] == 8)
    errors = [abs(a-b) for a,b in zip(now['scores']['foreground_dists'],prior['scores']['foreground_dists'])]
    if len(errors) != 9:
        raise ValueError('historical W3 score control length')
    (stage/'diagnostic-control.json').write_text(json.dumps({'stream_match': now['stream_sha256'] == prior['stream_sha256'], 'max_error': max(errors)})+'\n')


if __name__ == '__main__':
    main()
