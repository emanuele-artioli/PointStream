"""Encode a frozen-rate AV1 candidate and count the bound ball sidecar."""
import argparse
import hashlib
import json
import resource
import time
from fractions import Fraction
from pathlib import Path

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from src.codecs import svtav1

ROOT = Path('/Users/manu/Datasets/pointstream-submission-20261118')
SOURCE = ROOT / 'ps-native-ctl-20261118-full/source.yuv'
PRIOR = ROOT / 'ps-3rate-ball-20261118/48'
SOURCE_SHA = '6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c'
ENCODER_SHA = 'db044194faf94d553c2509c8e033ecd05a373ba54d919d0fb49a3a13149d1687'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cpu():
    usage = [resource.getrusage(k) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(r.ru_utime + r.ru_stime for r in usage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', choices=(24, 240), type=int, required=True)
    parser.add_argument('--crf', choices=(49, 50, 51), type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    start, cpu_start = time.monotonic(), cpu()
    assert sha(SOURCE) == SOURCE_SHA and SOURCE.stat().st_size == 240 * 816 * 960
    tools = {key: svtav1.tool(key) for key in ('SvtAv1EncApp', 'dav1d', 'ffmpeg')}
    assert tools['SvtAv1EncApp']['sha256'] == ENCODER_SHA
    a.output.mkdir(exist_ok=False, parents=True)
    row = svtav1.code(SOURCE, a.output, width=960, height=544, fps=Fraction(50), frames=a.frames,
                       crf=a.crf, preset=4, threads=2, full_range=False,
                       encoder=tools['SvtAv1EncApp']['path'], decoder=tools['dav1d']['path'], timeout=90)
    assert row['decoded_frames'] == a.frames
    assert Path(row['decoded']).stat().st_size == a.frames * 816 * 960
    row['decoded_sha256'] = sha(Path(row['decoded']))
    if a.frames == 240:
        prior_carrier = (PRIOR / 'native.pscp').read_bytes()
        prior_sidecar = (PRIOR / 'ball.sidecar').read_bytes()
        assert hashlib.sha256(prior_carrier).hexdigest() == 'b60466f988bdd4cc852134371f208d77bf7a53adc5d153551c761d64612da6eb'
        assert hashlib.sha256(prior_sidecar).hexdigest() == '205faa1fa276a8a43f431bc28636483673258e0a0797d445c01efa72b0bec233'
        origins, tile, size = ball._open(prior_carrier, prior_sidecar, tools['dav1d']['sha256'])
        assert size == (1920, 1080) and len(origins) == 240
        profile = {'mode': 'native_only', 'recipe': player.NATIVE_RECIPE,
                   'resources': {'native_decoder': tools['ffmpeg']['sha256']}, 'coded_size': [960, 544],
                   'canvas_size': [1920, 1080], 'cuts': [], 'runs': [], 'order': []}
        carrier = player.build(profile, Path(row['stream']).read_bytes(), None, {})
        sidecar = ball.pack(carrier, origins, tile, decoder_sha256=tools['dav1d']['sha256'])
        assert ball._open(carrier, sidecar, tools['dav1d']['sha256']) == (origins, tile, size)
        (a.output / 'native.pscp').write_bytes(carrier)
        (a.output / 'ball.sidecar').write_bytes(sidecar)
        row.update(carrier_bytes=len(carrier), sidecar_bytes=len(sidecar), total_bytes=len(carrier) + len(sidecar),
                   total_kbps=(len(carrier) + len(sidecar)) / 600,
                   carrier_sha256=hashlib.sha256(carrier).hexdigest(), sidecar_sha256=hashlib.sha256(sidecar).hexdigest())
    result = {'frames': a.frames, 'crf': a.crf, 'source_yuv_sha256': SOURCE_SHA, 'tools': tools,
              'script_sha256': sha(Path(__file__)), 'elapsed_seconds': time.monotonic() - start,
              'cpu_seconds_including_children': cpu() - cpu_start, 'row': row}
    (a.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'frames': a.frames, 'crf': a.crf, 'cpu_seconds': result['cpu_seconds_including_children'],
                      'elapsed_seconds': result['elapsed_seconds'], 'total_kbps': row.get('total_kbps')}))


if __name__ == '__main__':
    main()
