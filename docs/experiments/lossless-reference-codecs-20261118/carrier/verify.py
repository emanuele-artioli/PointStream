"""Price lossless player carriers, then open the actual bytes in a separate CPU receiver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import cv2
import numpy as np
from experiments.tennis import common_player_ball as ball
from experiments.tennis import common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import w8_payload as wire
from experiments.tennis.player_requalification import array_sha
from experiments.tennis.w8_receiver_native_bootstrap import _landlock

HERE = Path(__file__).resolve().parent
LOW = Path('/home/itec/emanuele/Datasets/pointstream-data/tennis/receiver-replay-20261118-3dd4aa753dfa')
MID = Path('/local/users/emanuele/pointstream/selected-rate-native-check-20261118/48')
LOW_SHA = '3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd'
DAV1D = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def prepare(out, middle):
    old = MID / 'players.pscp' if middle else LOW / 'qpng.pscp'
    ledger = json.loads((HERE / 'rates.json').read_text())
    row = next(r for r in ledger['rows'] if r['crf'] == 48)
    expected = row['sha256']['players.pscp'] if middle else LOW_SHA
    manifest.verify(MID / 'receiver.manifest' if middle else LOW / 'qpng.manifest', expected)
    original = old.read_bytes()
    require(sha(original) == expected, 'input carrier identity')
    out.mkdir()
    optimized = player.optimize_lossless_carrier(original)
    path = out / 'players.pscp'
    path.write_bytes(optimized)
    new_sha = sha(optimized)
    (out / 'receiver.manifest').write_bytes(manifest.encode(new_sha))
    a = player.unpack(original)[0]
    b = player.unpack(optimized)[0]
    require(a.keys() == b.keys() and a['base'] == b['base'] and a['keypoints'] == b['keypoints'], 'base or pose changed')
    refs = {}
    for k in ('ref0', 'ref1'):
        old_meta, old_rgb = player.rates.decode_reference(a[k])
        new_meta, new_rgb = player.rates.decode_reference(b[k])
        require(np.array_equal(old_rgb, new_rgb) and {**old_meta, 'codec': new_meta['codec']} == new_meta, 'reference changed')
        refs[k] = dict(old_bytes=len(a[k]), new_bytes=len(b[k]), codec=new_meta['codec'], rgb_sha256=array_sha(new_rgb))
    total_old, total_new = len(original) + 625, len(optimized) + manifest.verify(out / 'receiver.manifest', new_sha)['manifest_bytes']
    result = dict(original_carrier_sha256=expected, carrier_sha256=new_sha, original_player_total=total_old,
                  player_total=total_new, refs=refs, bytes_saved=total_old-total_new, base_sha256=sha(a['base']))
    if middle:
        side = (MID / 'ball.sidecar').read_bytes()
        require(sha(side) == row['sha256']['ball.sidecar'], 'original paid sidecar')
        # The archived sidecar names its NumPy runtime. Open its canonical envelope
        # with the recorded identity; the new sidecar pins this worker's actual runtime.
        old_profile = dict(profile='common-player-ball-sidecar-v1', carrier_sha256=expected,
                           tile_decoder_sha256=DAV1D, numpy_version=ledger['numpy'])
        parts, _ = wire.unpack(side, old_profile)
        coords = [None if p == ball.ABSENT else p for p in ball.PAIR.iter_unpack(parts['positions'])]
        tile = parts['tile']
        require(sha(tile) == row['ball_tile_sha256'] and len(coords) == 240, 'ball streams changed')
        (out / 'ball.sidecar').write_bytes(ball.pack(optimized, coords, tile, decoder_sha256=DAV1D))
        result.update(original_joint_total=row['joint_total_bytes'], joint_total=total_new + (out / 'ball.sidecar').stat().st_size,
                      positions_sha256=sha(parts['positions']), tile_sha256=sha(tile), original_ball_numpy=ledger['numpy'])
    result['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in out.iterdir()}
    return result


def receive(folder, digest):
    inputs = [folder / 'players.pscp', folder / 'receiver.manifest']
    side = folder / 'ball.sidecar'
    has_side = side.is_file()
    if has_side:
        inputs.append(side)
    original = LOW / 'qpng.pscp'
    require(original.exists(), 'denial probe must exist')
    empty = folder / ('receiver-output-' + str(os.getpid()))
    empty.mkdir()
    require(not list(empty.iterdir()), 'fresh receiver directory')
    interfaces = [name for _, name in socket.if_nameindex()]
    require(interfaces == ['lo'] and os.environ['CUDA_VISIBLE_DEVICES'] == '', 'CPU and network isolation')
    np.percentile(np.arange(3), 90)  # Load NumPy's deferred modules before sealing file reads.
    abi = _landlock(inputs, [], [], empty)
    try:
        original.open('rb').close()
    except PermissionError:
        pass
    else:
        raise ValueError('source carrier accessible to receiver')
    carrier = inputs[0].read_bytes()
    require(sha(carrier) == digest, 'received carrier digest')
    manifest.verify(inputs[1], digest)
    decoded = player.unpack(carrier)[2]
    result = dict(landlock_abi=abi, interfaces=interfaces, original_carrier_denied=True, players={})
    for name, runs in decoded.items():
        require(len(runs) == 1 and np.array_equal(runs[0]['t'], np.arange(240)), 'incomplete track')
        run = runs[0]
        # Match the generator's tensor digest while retaining one pose frame at a time.
        pose = hashlib.sha256(b'|u1:(240, 784, 512, 3):')
        for tick in range(240):
            frame = player.pose_for_tick(run, tick)
            require(frame.shape == (784, 512, 3) and frame.dtype == np.uint8, 'pose shape')
            pose.update(frame)
        result['players'][name] = dict(seed=run['seed'], pose_sha256=pose.hexdigest(),
                                      reference_sha256=array_sha(run['reference']), t_sha256=array_sha(run['t']),
                                      box_sha256=array_sha(np.asarray(run['boxes'])))
    if has_side:
        positions, tile, size = ball._open(carrier, side.read_bytes(), DAV1D)
        packed = b''.join(ball.PAIR.pack(*(p if p is not None else ball.ABSENT)) for p in positions)
        result.update(tile_sha256=sha(tile), positions_sha256=sha(packed), placements=sum(p is not None for p in positions), canvas=size)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['low', 'middle', 'receive'])
    p.add_argument('folder', type=Path)
    p.add_argument('digest', nargs='?')
    args = p.parse_args()
    cv2.setNumThreads(1)
    timer = time.process_time()
    record = receive(args.folder, args.digest) if args.mode == 'receive' else prepare(args.folder, args.mode == 'middle')
    record.update(cpu_seconds=time.process_time()-timer, opencv_build_sha256=sha(cv2.getBuildInformation().encode()),
                  opencv=cv2.__version__, numpy=np.__version__, python=sys.executable)
    print(json.dumps(record, sort_keys=True))
