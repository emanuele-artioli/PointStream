"""Apply an authenticated ball patch to audited original receiver frames."""
import hashlib
import json
import resource
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path('/local/users/emanuele/pointstream/low-base-ball-preview-20261118')
ARCHIVE = Path('/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261118T171044Z-cb2e385d/full/published.tar')
FFMPEG = '/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffmpeg'
FFPROBE = '/local/users/emanuele/pointstream/cache/32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661/tree/bin/ffprobe'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def cpu():
    return sum(getattr(resource.getrusage(k), f) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN) for f in ('ru_utime', 'ru_stime'))

start, start_cpu = time.monotonic(), cpu()
meta = json.loads((ROOT / 'input.json').read_text())
audit = json.loads((ROOT / 'finish-full-audit.json').read_text())
assert meta['source_sha256'] == 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d'
assert audit['all_frames_exact'] and audit['ticks'] == 240 and audit['carrier_sha256'] == meta['carrier_sha256']
assert audit['archive_sha256'] == '48542980357c28e5e2857e2b0ecb52759e6b78ea5da2f343f148cfc771d6be13'
assert sha((ROOT / 'ball-input.npz').read_bytes()) == meta['ball_input_sha256']
assert sha((ROOT / 'ball.sidecar').read_bytes()) == meta['sidecar_sha256']
with np.load(ROOT / 'ball-input.npz', allow_pickle=False) as data:
    patches, coords = data['patches'], data['coords']
assert patches.shape == (240, 32, 32, 3) and coords.shape == (240, 2)
output = ROOT / 'low-base-with-ball-v2.mp4'
command = [FFMPEG, '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '50', '-i', 'pipe:0', '-frames:v', '240', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '1', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(output)]
rows = []
with tarfile.open(ARCHIVE) as tar, tar.extractfile('publish/frames.npy') as data:
    assert np.lib.format.read_magic(data) == (1, 0)
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(data)
    assert shape == (240, 1080, 1920, 3) and not fortran and dtype == np.uint8
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for tick in range(240):
            raw = data.read(1080 * 1920 * 3)
            assert len(raw) == 1080 * 1920 * 3
            before_hash = hashlib.sha256(b'|u1:(1080, 1920, 3):')
            before_hash.update(raw)
            before = before_hash.hexdigest()
            assert audit['rows'][tick]['tick'] == tick and audit['rows'][tick]['frame_sha256'] == before
            frame = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).copy()
            x, y = (int(v) for v in coords[tick])
            present = x != -1
            assert present or (x == y == -1)
            if present:
                assert x % 2 == y % 2 == 0 and 0 <= x <= 1888 and 0 <= y <= 1048
                frame[y:y+32, x:x+32] = patches[tick]
                assert np.array_equal(frame[:y], np.frombuffer(raw, np.uint8).reshape(1080,1920,3)[:y])
                assert np.array_equal(frame[y+32:], np.frombuffer(raw, np.uint8).reshape(1080,1920,3)[y+32:])
                assert np.array_equal(frame[y:y+32,:x], np.frombuffer(raw, np.uint8).reshape(1080,1920,3)[y:y+32,:x])
                assert np.array_equal(frame[y:y+32,x+32:], np.frombuffer(raw, np.uint8).reshape(1080,1920,3)[y:y+32,x+32:])
            after_hash = hashlib.sha256(b'|u1:(1080, 1920, 3):')
            after_hash.update(frame)
            after = after_hash.hexdigest()
            assert present or before == after
            rows.append({'tick': tick, 'pre_ball_sha256': before, 'post_ball_sha256': after, 'position': [x,y] if present else None})
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    stderr = proc.stderr.read().decode()
    assert proc.wait(timeout=60) == 0, stderr[-1000:]
assert sum(r['position'] is not None for r in rows) == 195
probe = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', str(output)], capture_output=True, text=True, check=True, timeout=20)
stream = json.loads(probe.stdout)['streams'][0]
assert (stream['width'], stream['height'], stream['r_frame_rate'], stream['nb_read_frames']) == (1920, 1080, '50/1', '240')
total = meta['carrier_bytes'] + meta['manifest_bytes'] + meta['sidecar_bytes']
receipt = {'source_sha256': meta['source_sha256'], 'input': meta, 'finish_archive_sha256': audit['archive_sha256'], 'checked_finisher_frames': len(rows), 'placements': 195, 'absent_frames': 45, 'total_bytes': total, 'kbps': total / 600, 'video_sha256': hashlib.file_digest(output.open('rb'), 'sha256').hexdigest(), 'video_bytes': output.stat().st_size, 'stream': stream, 'cpu_seconds': cpu()-start_cpu, 'wall_seconds': time.monotonic()-start, 'rows': rows}
(ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'rows'}))
