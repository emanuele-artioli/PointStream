"""Actual-input CPU witness for the canvas sidecar using a bound sender track or prescribed artificial locations."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball
from experiments.tennis import common_player_packet as player
from experiments.tennis import k4b_features
from src.codecs import svtav1
from src.pointstream import tracked_patch


def file_sha(path):
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def check(value, label):
    if not value:
        raise ValueError(label)


def ffraw(ffmpeg, ivf):
    argv = [ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-c:v', 'libdav1d', '-f', 'ivf',
            '-i', 'pipe:0', '-map', '0:v:0', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    return subprocess.run(argv, input=ivf, check=True, capture_output=True, timeout=35).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ('source', 'carrier', 'storage', 'report'):
        parser.add_argument('--' + opt, type=Path, required=True)
    parser.add_argument('--track-json', type=Path, help='source-aligned model detections, independent of labels')
    parser.add_argument('--track-sha256', help='previously recorded SHA256 of the sender track')
    args = parser.parse_args()
    if (args.track_json is None) != (args.track_sha256 is None):
        parser.error('--track-json and --track-sha256 must be supplied together')
    cv2.setNumThreads(1)
    started = time.monotonic()
    cv = Path('docs/experiments/w7-specs/tier2-eval/windows-final.json')
    rows = json.loads(cv.read_text())['windows']
    window, = (r for r in rows if r['name'] == 'nad000')
    source_sha = file_sha(args.source)
    check(source_sha == 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d', 'source identity')
    carrier = args.carrier.read_bytes()
    carrier_sha = hashlib.sha256(carrier).hexdigest()
    check(carrier_sha == 'df2214dcc052bfc5d80305c0c26e5745b00ad344877de22f3a53ec3c9d5d4717', 'Q70 carrier')
    sections, _, _ = player.unpack(carrier)
    info = player.inspect(carrier)['profile']
    ffmpeg = str(Path(shutil.which('ffmpeg')).resolve())
    check(file_sha(Path(ffmpeg)) == info['resources']['native_decoder'], 'native decoder')
    decoder = svtav1.tool('dav1d')
    encoder = svtav1.tool('SvtAv1EncApp')
    if args.track_json is not None:
        raw_track = args.track_json.read_bytes()
        check(hashlib.sha256(raw_track).hexdigest() == args.track_sha256, 'frozen tracker output hash')
        track = json.loads(raw_track)
        check(track['source_sha256'] == source_sha and track['window'] == 'nadal_kyrgios/000@120'
              and track['fps'] == [50, 1] and track['canvas_size'] == [1920, 1080], 'sender scope')
        check(isinstance(track['centers'], list) and len(track['centers']) == 240, 'sender tick coverage')
        positions = [tracked_patch.origin(pos, 1920, 1080) for pos in track['centers']]
        placement = {'kind': 'sender predictions; no independent reference labels',
                     'path': str(args.track_json), 'sha256': args.track_sha256}
        sidecar_name = 'sender-placement.sidecar'
    else:
        positions = [None, (0, 0), (1888, 1048), (944, 524)] * 60
        placement = {'kind': 'synthetic test placements', 'positions_cycle': positions[:4]}
        sidecar_name = 'development-placement.sidecar'
    args.storage.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='source-', dir=args.storage) as tmp:
        root = Path(tmp)
        raw, patches = root / 'source.yuv', root / 'patch.yuv'
        conversion = k4b_features.convert_command(ffmpeg, args.source, raw, 0, 240)
        subprocess.run(conversion, check=True, capture_output=True, timeout=60)
        source_yuv_sha = file_sha(raw)
        check(source_yuv_sha == window['source_yuv_sha256'], 'canonical source pixels')
        enc, tile, tiles = [], b'', None
        if any(pos is not None for pos in positions):
            tracked_patch._patch_video(raw, patches, {'width': 1920, 'height': 1080, 'full_range': False}, positions)
            coded = root / 'tile.ivf'
            enc = svtav1.encode_command(encoder['real_path'], patches, coded, width=64, height=64,
                                        fps=Fraction(50), frames=240, crf=36, preset=6, full_range=False)
            svtav1.run(enc, 60)
            tile = coded.read_bytes()
            reference = ffraw(ffmpeg, tile)
            check(len(reference) == 240*64*96, 'reference tile count')
            tiles = np.frombuffer(reference, np.uint8).reshape((240, 96, 64))
        sidecar = ball.pack(carrier, positions, tile, decoder_sha256=decoder['sha256'])
        native = ffraw(ffmpeg, sections['base'])
        check(len(native) == 240*480*408, 'reference base count')
        planes = np.frombuffer(native, np.uint8).reshape((240, 408, 480))
        latest = []

        def canvases():
            for tick, frame in enumerate(planes):
                base = cv2.resize(tracked_patch.yuv_to_rgb(frame, 480, 272, False), (1920, 1080), interpolation=cv2.INTER_CUBIC)
                latest[:] = [base]
                yield tick, base

        changed = 0
        frame_hash = hashlib.sha256()
        emit_count = 0
        for tick, actual in ball.render(carrier, sidecar, canvases(), scratch=root, decoder=decoder['real_path']):
            expected = latest[0].copy()
            pos = positions[tick]
            if pos is not None:
                x, y = pos
                rgb = tracked_patch.yuv_to_rgb(tiles[tick], 64, 64, False)
                expected[y:y + 32, x:x + 32] = rgb[16:48, 16:48]
                changed += int(np.any(expected[y:y+32, x:x+32] != latest[0][y:y+32, x:x+32]))
            check(np.array_equal(expected, actual), f'pixel parity at tick {tick}')
            frame_hash.update(actual)
            emit_count += 1
        check(emit_count == 240, 'output count')
        if args.track_json is None:
            check(changed > 0, 'source-dependent writes at artificial placements')
        with (args.storage / sidecar_name).open('xb') as fh:
            fh.write(sidecar)
    stats = ball.inspect(carrier, sidecar, decoder_sha256=decoder['sha256'])
    report = {'scope': 'actual native fallback and paid source tiles; receiver component correctness only',
              'source_sha256': source_sha, 'source_yuv_sha256': source_yuv_sha, 'carrier_sha256': carrier_sha,
              'encoder': encoder, 'decoder': decoder, 'ffmpeg': {'path': ffmpeg, 'sha256': file_sha(Path(ffmpeg))},
              'placement': placement, 'frames_equal_to_offline_reference': emit_count,
              'visible_test_frames_with_changed_pixels': changed, 'decoded_rgb_sha256': frame_hash.hexdigest(),
              'tile_sha256': hashlib.sha256(tile).hexdigest(), 'sidecar_sha256': hashlib.sha256(sidecar).hexdigest(),
              'stored_sidecar': str(args.storage / sidecar_name), 'components': stats,
              'source_conversion': conversion, 'encode_command': enc, 'numpy': np.__version__, 'cv2': cv2.__version__,
              'wall_seconds': time.monotonic()-started, 'revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'source_files_sha256': {str(f): file_sha(f) for f in (Path(__file__), Path(ball.__file__))}}
    with args.report.open('x') as fh:
        json.dump(report, fh, indent=2)
        fh.write('\n')
    print('PASS', emit_count, 'matched ticks;', stats)


if __name__ == '__main__':
    main()
