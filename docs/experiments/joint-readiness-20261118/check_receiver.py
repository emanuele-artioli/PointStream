"""Replay the bounded CPU component test on the paid development packets."""
import argparse
import hashlib
import json
import resource
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b, k6b_real_base_compose, k6b_receiver_sam
from src.pointstream.tracked_patch import yuv_to_rgb


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, what):
    if not ok:
        raise ValueError(what)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', type=Path, required=True)
    parser.add_argument('--jpeg-root', type=Path, required=True)
    parser.add_argument('--motion-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    wall, cpu = time.monotonic(), time.process_time()
    proof = Path('docs/experiments/common-player-packet-20261117/retained-input-final-audit.json')
    png_record = json.loads(proof.read_text())
    durable = Path('docs/experiments/player-reference-pilot-20261117/packet-durable-manifest.json')
    members = {r['path']: r['sha256'] for r in json.loads(durable.read_text())['verified_files']}
    blob = args.png.read_bytes()
    require(sha(blob) == png_record['new_packet']['sha256'], 'PNG carrier')
    profile = json.loads(Path('docs/experiments/common-player-packet-20261117/cpu-profile.json').read_text())
    sections, _, units = packet.unpack(blob, profile)
    carriers = {'PNG': sha(blob)}
    for q in (70, 40):
        name = f'q{q}/packet.pscp'
        jpeg = (args.jpeg_root / name).read_bytes()
        require(sha(jpeg) == members[name], name)
        content, _, _ = packet.unpack(jpeg)
        for key in ('base', 'keypoints'):
            require(sections[key] == content[key], key)
        carriers[f'Q{q}'] = sha(jpeg)
    require(packet.coverage(units, profile['order']) == [profile['order']] * 240, 'ordered tick coverage')
    players = []
    for i in (1, 2):
        record = json.loads(Path(f'docs/experiments/player-motion-amendment-20261117/player-{i}-audit.json').read_text())
        full = record['stages']['full']
        name, = (f for f in full['files'] if f.endswith('.npz'))
        archive = args.motion_root / f'amended-player{i}/full/publish' / name
        digest = file_sha(archive)
        require(digest == full['files'][name]['sha256'], 'actual published crops')
        unit = name.split('@')[0]
        run, = units[unit]
        with np.load(archive, allow_pickle=False) as data:
            require(np.array_equal(data['t'], np.arange(240)), 'crop tick order')
            require(np.array_equal(data['t'], run['t']), 'paid tick join')
            require(np.array_equal(data['box'], run['boxes']), 'paid box join')
        with zipfile.ZipFile(archive) as arrays, arrays.open('crops.npy') as values:
            require(np.lib.format.read_magic(values) == (1, 0), 'crop format')
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(values)
            require(shape == (240, 784, 512, 3) and not fortran and dtype == np.uint8, 'crop array layout')
        count, prompts = [], hashlib.sha256()
        for kp, box in zip(run['kp'], run['boxes'], strict=True):
            pts, labels = k6b_receiver_sam.pose_prompt(k6b.to_crop(kp, box))
            require(np.isfinite(pts).all() and (pts >= 0).all() and (pts <= 1).all(), 'prompt bounds')
            require(list(labels[:2]) == [2, 3] and len(labels) >= 3 and (labels[2:] == 1).all(), 'prompt labels')
            require(pts.shape == (len(labels), 2), 'prompt shape')
            count.append(len(labels) - 2)
            prompts.update(len(labels).to_bytes(1, 'little') + pts.tobytes() + labels.tobytes())
        players.append({'unit': unit, 'archive_sha256': digest, 'generation_job': record['job_id'],
                        'matching_boxes_ticks': len(count), 'positive_points_min_max': [min(count), max(count)],
                        'prompt_sequence_sha256': prompts.hexdigest()})
    clock_path = args.jpeg_root / 'base-decoded.framemd5'
    require(file_sha(clock_path) == members['base-decoded.framemd5'], 'native reference timestamps')
    clock = clock_path.read_text().splitlines()
    require('#tb 0: 1/50' in clock, 'native timebase')
    ticks = [[v.strip() for v in row.split(',')] for row in clock if not row.startswith('#')]
    require(len(ticks) == 240, 'native timeline')
    ffmpeg = Path(png_record['native_decode']['binary'])
    require(file_sha(ffmpeg) == profile['resources']['native_decoder'], 'native receiver identity')
    command = [str(ffmpeg), '-nostdin', '-v', 'error', '-threads', '1', '-c:v', 'libdav1d', '-f', 'ivf',
               '-i', 'pipe:0', '-map', '0:v:0', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    raw = subprocess.run(command, input=sections['base'], capture_output=True, check=True, timeout=35).stdout
    width, height = profile['coded_size']
    size = width * height * 3 // 2
    require(len(raw) == 240 * size, 'decoded frame count')
    output = hashlib.sha256()
    empty = np.zeros((1080, 1920), dtype=bool)
    masks = {unit: empty for unit in profile['order']}
    for t, row in enumerate(ticks):
        yuv = raw[t*size:(t+1)*size]
        require([int(v) for v in row[:5]] == [0, t, t, 1, size], 'native presentation timestamp')
        require(hashlib.md5(yuv, usedforsecurity=False).hexdigest() == row[5], 'native frame mismatch')
        planes = np.frombuffer(yuv, dtype=np.uint8).reshape(-1, width)
        canvas = cv2.resize(yuv_to_rgb(planes, width, height, False), tuple(profile['canvas_size']),
                            interpolation=cv2.INTER_CUBIC)
        result, written = k6b_real_base_compose.compose(canvas, {}, {}, masks, profile['order'])
        require(not written.any() and np.array_equal(canvas, result), 'native fallback')
        output.update(result)
    report = {'status': 'passed', 'scope': 'CPU carrier prompt and missing-player fallback diagnostic',
              'source_window': 'nadal_kyrgios/000@120', 'carriers_sha256': carriers, 'players': players,
              'native': {'argv': command, 'ffmpeg_sha256': file_sha(ffmpeg), 'ivf_sha256': sha(sections['base']),
                         'reference_framemd5_sha256': file_sha(clock_path), 'frames': len(ticks), 'fps': [50, 1],
                         'fallback_pixels_all_equal': True, 'canvas_rgb24_sha256': output.hexdigest(),
                         'color_path': 'tracked_patch.yuv_to_rgb BT709 limited then INTER_CUBIC'},
              'numpy': np.__version__, 'cv2': cv2.__version__, 'opencv_build_sha256': sha(cv2.getBuildInformation().encode()),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': file_sha(Path(__file__)), 'wall_seconds': time.monotonic() - wall,
              'parent_cpu_seconds': time.process_time() - cpu,
              'child_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2])}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(f'PASS {sum(p["matching_boxes_ticks"] for p in players)} paid prompts, {len(ticks)} native fallbacks; {args.output}')


if __name__ == '__main__':
    main()
