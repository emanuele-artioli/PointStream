"""Bounded local verification of a new packet using retained paid inputs."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import time

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b, k6b_rate_streams as rates, w8_payload as oldwire


def digest(data):
    return hashlib.sha256(data).hexdigest()


start = time.monotonic()
cpu_start = time.process_time()
inputs = Path('/private/tmp/pointstream-requalification-inputs-20261116')
output = Path('/private/tmp/pointstream-common-packet-proof-20261117')
output.mkdir(exist_ok=False)
old_blob = (inputs / 'packet-eff1aca8.psw8').read_bytes()
profile_bytes = (inputs / 'profile-bec98d67.json').read_bytes()
assert digest(old_blob) == 'eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4'
assert digest(profile_bytes) == 'bec98d67280e9f777a0d3cc110baa51ae2bf2c876aefe6b247d9691432f61107'
old = json.loads(profile_bytes)
sections, old_account = oldwire.unpack_window(old_blob, old)
old_kps = sections[old['kps']]
old_meta, units = rates.decode_keypoints(old_kps)
by_name = {u['name']: u for u in units}
order = [k6b.unit_name(old['window'], p) for p in old['recipe']['compositing_order']]
new_kps = rates.encode_keypoints_kps2({'native_frames': 240}, [by_name[n] for n in order])
_, new_units = rates.decode_keypoints(new_kps)
for unit in new_units:
    before = by_name[unit['name']]
    assert np.array_equal(unit['t'], before['t']) and np.array_equal(unit['kp'], before['kp'])
    assert np.array_equal(unit['t'], np.arange(240))
ivf = sections[old['ivf']]
header = struct.unpack_from('<4sHH4sHHIIII', ivf)
native = Path(shutil.which('ffmpeg')).resolve(strict=True)
native_hash = digest(native.read_bytes())
resources = {k: old['models'][k] for k in packet.RESOURCES if k not in ('common_lora', 'generator_code', 'native_decoder')}
resources.update(common_lora=old['models']['tennis_base'],
                 generator_code=digest(Path('experiments/tennis/k6b_generate.py').read_bytes()),
                 native_decoder=native_hash)
profile = {'mode': 'players', 'recipe': copy.deepcopy(packet.RECIPE), 'resources': resources,
           'coded_size': list(header[4:6]), 'canvas_size': [1920, 1080], 'cuts': [],
           'order': order, 'runs': [{'unit': u['id'], 'start': 0, 'end': 240, 'reference': u['reference']}
                                   for u in old['units']]}
refs = {u['reference']: sections[u['reference']] for u in old['units']}
blob = packet.build(profile, ivf, new_kps, refs)
_, accounting, decoded = packet.unpack(blob, profile)
assert accounting['match'] == 0 and accounting['frame'] == len(ivf) + len(new_kps)
assert sum(accounting.values()) == len(blob)
assert packet.coverage(decoded, order) == [order] * 240
comparisons = []
for name in order:
    runs = decoded[name]
    assert len(runs) == 1
    run = runs[0]
    golden_path = inputs / (name + '.npz')
    with np.load(golden_path, allow_pickle=False) as golden:
        expected_pose = golden['pose']
        checks = {'reference': np.array_equal(run['reference'], golden['reference']),
                  'reference_box': np.array_equal(run['reference_box'], golden['reference_box']),
                  'ticks': np.array_equal(run['t'], golden['t']),
                  'boxes': np.array_equal(np.asarray(run['boxes']), golden['box']),
                  'run_ids': np.array_equal(np.zeros(240, np.int64), golden['run']),
                  'poses': all(np.array_equal(packet.pose_for_tick(run, i), expected_pose[i]) for i in range(240)),
                  'seed': run['seed'] == int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) * 1009}
        del expected_pose
    comparisons.append({'unit': name, 'checks': checks, 'golden_sha256': digest(golden_path.read_bytes())})
    assert all(checks.values()), (name, checks)
(output / 'packet.pscp').write_bytes(blob)
(output / 'profile.json').write_text(json.dumps(profile, indent=2, sort_keys=True) + '\n')
(output / 'base.ivf').write_bytes(ivf)
cmd = [str(native), '-v', 'error', '-nostdin', '-threads', '1', '-i', str(output / 'base.ivf'),
       '-map', '0:v:0', '-f', 'framemd5', str(output / 'base-decoded.framemd5')]
native_start = time.monotonic()
subprocess.run(cmd, check=True, timeout=60)
lines = (output / 'base-decoded.framemd5').read_text().splitlines()
frames = [line for line in lines if line and not line.startswith('#')]
assert len(frames) == 240
result = {'status': 'PASS', 'scope': 'Local paid packet/conditioning and native-container decode verification; no learned receiver or quality scoring',
          'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'old_packet_sha256': digest(old_blob), 'old_profile_sha256': digest(profile_bytes),
          'new_packet': {'path': str(output / 'packet.pscp'), 'sha256': digest(blob), 'bytes': len(blob), 'accounted_role_bytes': accounting},
          'new_profile_sha256': digest((output / 'profile.json').read_bytes()),
          'keypoints_repack': {'old_sha256': digest(old_kps), 'new_sha256': digest(new_kps),
                               'explicit_metadata_repack': True, 'decoded_coordinates_visibility_ticks_exact': True,
                               'old_metadata': old_meta, 'new_metadata': rates.decode_keypoints(new_kps)[0]},
          'coded_size': list(header[4:6]), 'canvas_size': [1920, 1080], 'comparisons': comparisons,
          'native_decode': {'binary': str(native), 'sha256': native_hash, 'command': cmd, 'frames': len(frames),
                            'decoded_framemd5_sha256': digest((output / 'base-decoded.framemd5').read_bytes()),
                            'wall_seconds': time.monotonic() - native_start, 'canvas_resample_performed': False},
          'wall_seconds': time.monotonic() - start, 'parent_cpu_seconds': time.process_time() - cpu_start,
          'child_cpu_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
          'limits': ['Original eight model hashes and generator environment are declarations reused from verified native provenance; not locally rehashed or loaded.',
                     'Generator code identity is the measured renderer source file; this does not verify an entire newly installed receiver environment.',
                     'Native decode is at coded480x272 resolution; no canvas projection, source-denied filesystem audit, generation, masks, composition or metric evaluation is claimed.',
                     'The packet reuses lossless paid references. No low-bitrate or rate-quality gain follows from serialization.']}
(output / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': result['status'], 'packet_bytes': len(blob), 'roles': accounting,
                  'compared_units': len(comparisons), 'poses_each': 240, 'native_frames': len(frames),
                  'wall_seconds': result['wall_seconds'], 'parent_cpu_seconds': result['parent_cpu_seconds'],
                  'child_cpu_seconds': result['child_cpu_seconds'], 'evidence': str(output / 'audit.json')}))
