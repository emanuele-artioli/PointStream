"""Recompose an authenticated joint preview on the priced CRF 48 native base."""
import argparse
import hashlib
import io
import json
import os
import resource
import shlex
import subprocess
import sys
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

CAMPUS = os.environ.get('PS_PREVIEW_CAMPUS') == '1'
ROOT = Path(os.environ['PS_PREVIEW_STAGE']) if CAMPUS else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import CARRIER, MANIFEST, UNITS, bind_generation, overlay  # noqa: E402

from experiments.tennis import common_player_ball as ball  # noqa: E402
from experiments.tennis import common_player_packet as player  # noqa: E402
from experiments.tennis import common_receiver_manifest as manifests  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

PACKETS = Path('/local/users/emanuele/pointstream/selected-rate-native-check-20261118/48' if CAMPUS else '/Users/manu/Datasets/pointstream-submission-20261118/selected-rate-support-linux-png/packets/48')
BASE = Path('/local/users/emanuele/pointstream/selected-rate-native-check-20261118/48/base.yuv' if CAMPUS else '/Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/plain-48/decoded.yuv')
DETAILS = Path('/local/users/emanuele/pointstream/selected-rate-native-check-20261118/48/native-frames.json' if CAMPUS else '/Users/manu/Datasets/pointstream-submission-20261118/selected-rate-support-linux-png/native-plane-receipts/crf48.json')
ORIGINAL = Path('/home/itec/emanuele/Datasets/pointstream-data/tennis/receiver-replay-20261118-3dd4aa753dfa/qpng.pscp' if CAMPUS else '/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
DAV1D_SHA = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
BASE_SHA = '4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b'
FFMPEG = '/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg' if CAMPUS else '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffprobe' if CAMPUS else '/opt/homebrew/bin/ffprobe'
REMOTE_FFMPEG = '/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg'
SMOKE = {0, 1, 2, 3, 41, 42, 43, 44}
FINISH_ARCHIVE_SHA = '48542980357c28e5e2857e2b0ecb52759e6b78ea5da2f343f148cfc771d6be13'
FINISH_REPORT_SHA = '645d719da006b9df133849753ed45b2da00a821958b716c1b4debf5e625ce131'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def cpu_seconds():
    return sum(getattr(resource.getrusage(k), f) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
               for f in ('ru_utime', 'ru_stime'))


def input_data():
    ledger = json.loads((ROOT / 'selected-rate-support/rates.json').read_text())
    native_report = json.loads((ROOT / 'selected-rate-support/native-report.json').read_text())
    row = next(row for row in ledger['rows'] if row['crf'] == 48)
    qualified = next(row for row in native_report['rows'] if row['crf'] == 48)
    require(ledger['source_sha256'] == native_report['source_sha256'] == 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d', 'source identity')
    inputs = {name: (PACKETS / name).read_bytes() for name in ('players.pscp', 'receiver.manifest', 'ball.sidecar')}
    require({name: sha(data) for name, data in inputs.items()} == row['sha256'], 'paid inputs')
    require(sum(len(x) for x in inputs.values()) == row['joint_total_bytes'] == 571630, 'total charged bytes')
    manifests.verify(PACKETS / 'receiver.manifest', sha(inputs['players.pscp']))
    require(file_sha(ORIGINAL) == CARRIER and file_sha(BASE) == BASE_SHA, 'original carrier or decoded base identity')
    new_sections, _, _ = player.unpack(inputs['players.pscp'])
    old_sections, _, _ = player.unpack(ORIGINAL.read_bytes())
    require(all(new_sections[n] == old_sections[n] for n in ('keypoints', 'ref0', 'ref1')), 'foreground sections')
    new_profile, old_profile = player.inspect(inputs['players.pscp'])['profile'], player.inspect(ORIGINAL.read_bytes())['profile']
    require(new_profile['coded_size'] == [960, 544], 'new coded size')
    new_profile['coded_size'] = old_profile['coded_size']
    require(new_profile == old_profile, 'profile other than coded size')
    require(sha(new_sections['base']) == row['base_ivf_sha256'] == qualified['native_ivf_sha256'], 'qualified coded base')
    require(qualified['yuv420p_sha256'] == BASE_SHA and file_sha(DETAILS) == qualified['native_frame_details_sha256'], 'qualified decoded base receipt')
    positions, tile, size = ball._open(inputs['players.pscp'], inputs['ball.sidecar'], DAV1D_SHA)
    require(size == (1920, 1080) and sha(tile) == row['ball_tile_sha256'], 'paid ball identity')
    return inputs, positions, tile


def decode_tile(tile):
    common = ['-v', 'error', '-nostdin', '-threads', '1', '-c:v', 'libdav1d', '-f', 'ivf', '-i', 'pipe:0', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1']
    local = subprocess.run([FFMPEG, *common], input=tile, capture_output=True, check=True, timeout=35).stdout
    if CAMPUS:
        require(len(local) == 240 * 96 * 64 and sha(local) == '9c997a21482fdc2fac4e36d7511a86158ed2eddcaf6aca57594e06ddd067bb1a', 'pinned Linux tile decode')
        return np.frombuffer(local, np.uint8).reshape(240, 96, 64), sha(local)
    remote_code = ('import hashlib,subprocess,sys; r=subprocess.run([' + repr(REMOTE_FFMPEG) + ',' + repr(common)[1:-1] + '],input=sys.stdin.buffer.read(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True); print(len(r.stdout),hashlib.sha256(r.stdout).hexdigest())')
    remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'gpu5', 'python3', '-c', shlex.quote(remote_code)], input=tile, capture_output=True, check=True, timeout=90).stdout.decode().split()
    require(len(local) == 240 * 96 * 64 and remote == [str(len(local)), sha(local)], 'cross-platform paid tile decode')
    return np.frombuffer(local, np.uint8).reshape(240, 96, 64), sha(local)


def run(args):
    start, start_cpu = time.monotonic(), cpu_seconds()
    cv2.setNumThreads(1)
    require(np.__version__ == '1.26.4' and cv2.__version__ == '4.10.0', 'registered native conversion runtime')
    inputs, positions, tile = input_data()
    tiles, tile_hash = decode_tile(tile)
    details = json.loads(DETAILS.read_text())
    native = np.memmap(BASE, np.uint8, 'r', shape=(240, 816, 960))
    target = list(range(240)) if args.full else sorted(SMOKE)
    audit = json.loads(args.finish_audit.read_text())
    require(audit['stage'] == 'full' and audit['all_frames_exact'] and audit['ticks'] == 240 and audit['archive_sha256'] == FINISH_ARCHIVE_SHA and file_sha(args.finish_dir / 'finish.json') == FINISH_REPORT_SHA, 'independent finish seal')
    require(audit['carrier_sha256'] == CARRIER and audit['manifest_sha256'] == MANIFEST, 'source-denied witness')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / ('joint-full.mp4' if args.full else 'joint-smoke.mp4')
    command = [FFMPEG, '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '50', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'ultrafast', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(output)]
    rows = []
    with ExitStack() as stack:
        report = json.loads((args.finish_dir / 'finish.json').read_text())
        require(report['ticks'] == list(range(240)) and report['order'] == UNITS, 'finish timeline and order')
        generations = dict(bind_generation(stack, Path(p), Path(a), report) for p, a in args.generation)
        require(set(generations) == set(UNITS), 'two players')
        masks = {}
        for unit in UNITS:
            data = (args.finish_dir / (unit + '-masks.npz')).read_bytes()
            require(sha(data) == report['mask_files'][unit], 'mask digest')
            zipped = stack.enter_context(zipfile.ZipFile(io.BytesIO(data)))
            require(np.array_equal(read_small(zipped, 't'), np.arange(240)), 'mask timeline')
            masks[unit] = stack.enter_context(array_frames(zipped, 'masks', (240, 784, 512), np.bool_))
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for t in range(target[-1] + 1):
                crops = {unit: next(generations[unit][1]) for unit in UNITS}
                current_masks = {unit: next(masks[unit]) for unit in UNITS}
                if t not in target:
                    continue
                base = cv2.resize(yuv_to_rgb(native[t], 960, 544, False), (1920, 1080), interpolation=cv2.INTER_CUBIC)
                require(sha(base) == details[t]['rgb24_sha256'] and details[t]['pts'] == t, 'qualified native frame')
                canvas = base.copy()
                support = np.zeros((1080, 1920), bool)
                projected, status = {}, {}
                for unit in UNITS:
                    projected[unit], status[unit] = overlay(canvas, support, crops[unit], generations[unit][0][t], current_masks[unit])
                require(status == report['rows'][t]['status'] and projected == report['rows'][t]['projected_pixels'], 'player mask projection')
                frame_before_ball = sha(canvas)
                pos = positions[t]
                if pos is not None:
                    x, y = pos
                    before_ball = canvas.copy()
                    pixels = yuv_to_rgb(tiles[t], 64, 64, False)
                    canvas[y:y + 32, x:x + 32] = pixels[16:48, 16:48]
                    unchanged = (np.array_equal(canvas[:y], before_ball[:y])
                                 and np.array_equal(canvas[y + 32:], before_ball[y + 32:])
                                 and np.array_equal(canvas[y:y + 32, :x], before_ball[y:y + 32, :x])
                                 and np.array_equal(canvas[y:y + 32, x + 32:], before_ball[y:y + 32, x + 32:]))
                    require(unchanged, 'ball changed pixels outside its paid patch')
                rows.append({'tick': t, 'base_rgb_sha256': details[t]['rgb24_sha256'], 'player_rgb_sha256': frame_before_ball, 'joint_rgb_sha256': sha(canvas), 'ball_position': pos, 'projected_pixels': projected, 'status': status})
                process.stdin.write(canvas.tobytes())
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode()
        require(process.wait(timeout=45) == 0, 'preview encoding: ' + stderr[-500:])
    probe = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height', '-of', 'json', str(output)], capture_output=True, text=True, check=True, timeout=15)
    stream = json.loads(probe.stdout)['streams'][0]
    require(int(stream['nb_read_frames']) == len(target) and stream['r_frame_rate'] == '50/1', 'preview cadence')
    receipt = {'scope': 'offline composition of independently audited generated players and masks over the qualified CRF 48 base, with paid ball tiles last', 'frames': len(target), 'ticks': target, 'fps': 50, 'joint_bytes': sum(len(x) for x in inputs.values()), 'joint_kbps': 571630 / 600, 'native_yuv_sha256': BASE_SHA, 'decoded_tile_sha256': tile_hash, 'finish_archive_sha256': audit['archive_sha256'], 'finish_audit_sha256': file_sha(args.finish_audit), 'wire_sha256': {k: sha(v) for k, v in inputs.items()}, 'video_sha256': file_sha(output), 'video_bytes': output.stat().st_size, 'rows': rows, 'cpu_seconds': cpu_seconds() - start_cpu, 'wall_seconds': time.monotonic() - start, 'encoding': command, 'ffprobe': stream}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('frames', 'joint_kbps', 'cpu_seconds', 'wall_seconds', 'video_sha256')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--finish-dir', type=Path, required=True)
    p.add_argument('--finish-audit', type=Path, required=True)
    p.add_argument('--generation', type=Path, nargs=2, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    run(p.parse_args())
