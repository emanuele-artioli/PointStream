"""Check exact campus JPEG RGB parity against the source-authenticated pilot arrays."""
import argparse
import base64
import hashlib
import io
import json
import os
import platform
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def worker():
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    request = json.load(sys.stdin)
    rows = []
    cpu = time.process_time()
    for quality, entry in request['qualities'].items():
        archive_path = Path('/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-pilot-20261117') / f'q{int(quality)}' / 'conditioning.tar'
        archive = archive_path.read_bytes()
        check(sha(archive) == entry['archive_sha256'], 'published pilot archive identity')
        for unit in entry['units']:
            jpeg = base64.b64decode(unit['jpeg'], validate=True)
            check(sha(jpeg) == unit['jpeg_sha256'], 'compressed JPEG identity')
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                member = tar.extractfile('publish/' + unit['name'] + '.npz').read()
            check(sha(member) == unit['expected_npz_sha256'], 'published NPZ identity')
            with np.load(io.BytesIO(member), allow_pickle=False) as data:
                expected = data['reference']
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            check(image.shape == expected.shape == (784, 512, 3) and image.dtype == expected.dtype == np.uint8, 'reference shape')

            def rgb_sha(array):
                return sha(f'{array.dtype.str}:{array.shape}:'.encode() + array.tobytes())

            check(rgb_sha(expected) == unit['expected_rgb_sha256'], 'archived training-independent reference identity')
            delta = np.abs(image.astype(np.int16) - expected.astype(np.int16))
            rows.append({'quality': int(quality), 'unit': unit['name'], 'jpeg_sha256': sha(jpeg),
                         'ref1_sha256': unit['ref1_sha256'], 'expected_npz_sha256': sha(member),
                         'expected_rgb_sha256': rgb_sha(expected), 'linux_rgb_sha256': rgb_sha(image),
                         'pixel_equality': np.array_equal(image, expected), 'different_rgb_channels': int(np.count_nonzero(delta)),
                         'channels': int(delta.size), 'max_abs_delta': int(delta.max()), 'mean_abs_delta': float(delta.mean())})
    cvfile = next((Path(cv2.__file__).parent).glob('cv2*.so'))
    info = cv2.getBuildInformation()
    record = {'host': platform.node(), 'platform': platform.platform(), 'python': sys.executable, 'numpy': np.__version__,
              'opencv': cv2.__version__, 'opencv_extension': str(cvfile), 'opencv_extension_sha256': sha(cvfile.read_bytes()),
              'opencv_build_sha256': sha(info.encode()), 'jpeg_build_line': [line.strip() for line in info.splitlines() if 'JPEG:' in line],
              'mac_build_sha256': request['mac_build_sha256'], 'carrier_hashes': request['carrier_hashes'],
              'pilot_archive_hashes': {q: v['archive_sha256'] for q, v in request['qualities'].items()}, 'references': rows,
              'mac_build_gate_satisfied': request['mac_build_sha256'] == sha(info.encode()),
              'worker_cpu_seconds': time.process_time() - cpu, 'gpu_visibility': os.environ.get('CUDA_VISIBLE_DEVICES')}
    print(json.dumps(record))


def controller():
    from experiments.tennis import common_player_packet as player

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    records = Path('docs/experiments/player-reference-pilot-20261117')
    manifest = json.loads((records/'prepared-manifest.json').read_text())
    durable = {r['path']: r['sha256'] for r in json.loads((records/'packet-durable-manifest.json').read_text())['verified_files']}
    request = {'qualities': {}, 'carrier_hashes': {}, 'mac_build_sha256': 'bade1ed132afc491c4cadda9ea0590a3d0745e16807f765450d3d3d4152a780b'}
    for q in ('70', '40'):
        name = f'q{q}/packet.pscp'
        blob = (args.prepared/name).read_bytes()
        check(sha(blob) == durable[name], 'carrier identity')
        request['carrier_hashes'][q] = sha(blob)
        sections, _, _ = player.unpack(blob)
        profile = player.inspect(blob)['profile']
        entries = {entry['unit']: entry for entry in manifest[q]['units']}
        units = []
        for run in profile['runs']:
            frozen = entries[run['unit']]
            body = sections[run['reference']]
            check(sha(body) == frozen['encoded_ref1_sha256'], 'frozen REF1 identity')
            meta, jpg = player._bounded_transport(body, b'REF1', player.MAX_REF)
            check(meta['unit'] == run['unit'] and int(q) == meta['quality'], 'unit and quality')
            units.append({'name': run['unit'], 'jpeg': base64.b64encode(jpg).decode(), 'jpeg_sha256': sha(jpg),
                          'ref1_sha256': sha(body), 'expected_npz_sha256': frozen['new_npz_sha256'],
                          'expected_rgb_sha256': frozen['decoded_rgb_sha256']})
        request['qualities'][q] = {'archive_sha256': manifest[q]['archive_sha256'], 'units': units}
    env_sha = 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e'
    python = f'/local/users/emanuele/pointstream/cache/{env_sha}/tree/bin/python'
    command = 'env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 90s ' + shlex.join([python, '-I', '-B', '-c', Path(__file__).read_text(), 'worker'])
    start = time.monotonic()
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'gpu5', command],
                              input=json.dumps(request), capture_output=True, text=True, timeout=100)
    if response.returncode:
        raise ValueError(f'campus worker failed ({response.returncode}): {response.stderr[-1200:]}')
    report = json.loads(response.stdout)
    report.update(environment_archive_sha256=env_sha, local_total_wall_seconds=time.monotonic()-start,
                  script_sha256=sha(Path(__file__).read_bytes()), revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print('Build gate:', report['mac_build_gate_satisfied'], 'unequal RGB channel counts:', [x['different_rgb_channels'] for x in report['references']])


if __name__ == '__main__':
    if sys.argv[1:] == ['worker']:
        worker()
    else:
        controller()
