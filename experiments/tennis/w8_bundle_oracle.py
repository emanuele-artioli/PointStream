"""Build W8's frozen-window packet and screen its actual rate; input equivalence remains separate."""
from __future__ import annotations

import argparse
import json
import os
import resource
import tarfile
import time
from pathlib import Path

TRANSPORT_SHA = 'd80fac2ab880743e100a90a0dbb46af6173bc2479389a6d8f75663578b9db017'
MATCH_SHA = 'fef98258469b57ae4518a1ef570579e9918de7435b310737a04cdfaeb15bb810'
BASE_ARCHIVE_SHA = '075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285'
WINDOW = 'nadal_kyrgios/000@120'
UNIT_PREFIX = 'nadal_kyrgios__000_at120__person_'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ivf', 'transport', 'match', 'common-base', 'inbox', 'repairs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ivf-sha256', required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    start = time.monotonic()
    import numpy as np
    from experiments.tennis import k6b_generate, w8_oracle, w8_payload
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_reference
    from experiments.tennis.w8_inputs import archive_members

    def read_verified(path: Path, expected: str) -> bytes:
        blob = path.read_bytes()
        if w8_payload.sha(blob) != expected:
            raise ValueError(f'hash mismatch: {path}')
        return blob

    ivf = read_verified(args.ivf, args.ivf_sha256)
    match = read_verified(args.match, MATCH_SHA)
    read_verified(args.transport, TRANSPORT_SHA)
    read_verified(args.common_base, BASE_ARCHIVE_SHA)
    with tarfile.open(args.common_base) as archive:
        found = [m for m in archive if m.name == 'publish/tennis-base@fit-100.pt' and m.isfile()]
        if len(found) != 1 or found[0].size > 100_000_000:
            raise ValueError('invalid common-base archive')
        stream = archive.extractfile(found[0])
        if stream is None:
            raise ValueError('unreadable base checkpoint')
        tennis_base = stream.read()
    unit_ids = [UNIT_PREFIX + '000000000000', UNIT_PREFIX + '000000000001']
    prefix = 'publish/packets/nadal_kyrgios__000@120/'
    names = [prefix + 'keypoints.kps'] + [prefix + unit + '.png.ref' for unit in unit_ids]
    packets = archive_members(args.transport, names)
    kps = packets[names[0]]
    ref_blobs = {unit: packets[name] for unit, name in zip(unit_ids, names[1:])}
    kmeta, tracks = decode_keypoints(kps)
    if kps[:4] != b'KPS2' or kmeta['window'] != WINDOW or {x['name'] for x in tracks} != set(unit_ids):
        raise ValueError('wrong KPS input')
    units = []
    for i, unit in enumerate(unit_ids):
        meta, _ = decode_reference(ref_blobs[unit])
        if meta.get('unit') != unit or meta.get('codec') != '.png' or meta.get('quality') is not None:
            raise ValueError('reference identity mismatch')
        units.append({'id': unit, 'reference': f'ref{i}', 'reference_tick': meta['reference_frame'], 'box': meta['reference_box']})
    judged = np.packbits(np.isin(np.arange(240), np.arange(0, 240, 3))).tobytes()
    root = Path(__file__).resolve().parents[2]
    specs = root / 'docs/experiments/w7-specs/tier2-eval'
    gen_path, receiver_path, judge_path = (specs / f'{stage}-nad000.json' for stage in ('generate', 'receiver', 'judge'))
    gen, receiver, judge = (json.loads(path.read_text()) for path in (gen_path, receiver_path, judge_path))
    staged = {row['name']: row['sha256'] for row in gen['staged_inputs']}
    sam = next(row['sha256'] for row in receiver['staged_inputs'] if row['name'] == 'sam')
    models = {name: staged[key] for name, key in zip(('vae_config', 'vae_weights', 'image_encoder_config',
              'image_encoder_weights', 'denoising_unet', 'reference_unet', 'pose_guider', 'motion_module'),
              ('w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7'))}
    from experiments.tennis.k6b_native_base import TOOLS
    models.update({'tennis_base': w8_payload.sha(tennis_base), 'sam': sam, 'generator_env': gen['environment']['sha256'],
                   'sam_env': receiver['environment']['sha256'], 'native_env': judge['environment']['sha256'],
                   'native_decoder': TOOLS['dav1d'], 'native_ffmpeg': TOOLS['ffmpeg'],
                   'generator_code': w8_payload.sha(Path(k6b_generate.__file__).read_bytes())})
    for code_name, code_path in {'receiver_code': 'experiments/tennis/k6b_receiver_sam.py',
                                 'geometry_code': 'experiments/tennis/k6b.py',
                                 'packet_code': 'experiments/tennis/k6b_rate_streams.py',
                                 'native_color_code': 'src/codecs/quality.py',
                                 'native_rgb_code': 'experiments/tennis/k6b_rgb_diagnostic.py',
                                 'native_memmap_code': 'experiments/visor/h3.py',
                                 'payload_decoder_code': 'experiments/tennis/w8_payload.py',
                                 'conditioning_decoder_code': 'experiments/tennis/w8_inputs.py'}.items():
        models[code_name] = w8_payload.sha((root / code_path).read_bytes())
    recipe = {'profile': 'match5', 'steps': 30, 'guidance': 2, 'context': [24, 4],
              'seed_rule': 'sha256(unit_name)[:7] then seed*1009+run', 'native_frames': 240, 'fps': [50, 1],
              'saved_ticks': list(range(0, 240, 3)), 'fallback': 'decoded_base', 'scheduler': k6b_generate.SCHEDULER,
              'rank': 4, 'compositing_order': kmeta['compositing_order'],
              'color_path': 'dav1d yuv420; limited-range yuv420_to_rgb cuda; bicubic resize RGB to 1920x1080',
              'generator_input_size': [512, 784], 'run_policy': 'all native run frames; retain judged ticks',
              'guider': 'original plus decoded match delta', 'adapter': 'nk-C@fit-100-fp16.pt 4-bit',
              'native_scale': 4, 'sync_start_tick': 0, 'match_duration_assumption_s': 7200}
    profile = {'window': WINDOW, 'models': models, 'recipe': recipe, 'units': units,
               'ivf': 'ivf', 'kps': 'kps', 'judged': 'judged', 'match': {'section': 'match', 'sha256': MATCH_SHA}}
    bundle = w8_payload.assemble_window(profile, ivf, kps, judged, ref_blobs, match)
    sections, counts = w8_payload.unpack_window(bundle, profile)
    cache = w8_payload.cache_identity(profile, sections)
    rates = w8_oracle.rates(counts, cache_contract=cache, expected_cache=cache)
    repair_bytes = args.repairs.read_bytes()
    hybrid, anchors = w8_oracle.accepted_nad000(args.inbox, json.loads(repair_bytes)['nad000'])
    gate = w8_oracle.gate(rates['conditional_kbps'], hybrid, anchors)
    intervals = w8_oracle.strict_win_intervals(hybrid, anchors)
    args.out.mkdir(parents=False, exist_ok=False)
    (args.out / 'window.psw8').write_bytes(bundle)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {'scope': 'optimistic payload accounting screen only; Gate 1 input equivalence remains unverified',
               'candidate_sha256': w8_payload.sha(bundle), 'candidate_bytes': len(bundle), 'bytes_by_unit': counts,
               'rates_kbps': rates, 'cache_contract': cache, 'cache_assumption': 'exact match and two reference packets retained for 7200 s; no reuse quality demonstrated',
               'input_sha256': {'ivf': args.ivf_sha256, 'transport_archive': TRANSPORT_SHA, 'match': MATCH_SHA,
                                'common_base_archive': BASE_ARCHIVE_SHA, 'repairs_config': w8_payload.sha(repair_bytes),
                                'generate_spec': w8_payload.sha(gen_path.read_bytes()), 'receiver_spec': w8_payload.sha(receiver_path.read_bytes()),
                                'judge_spec': w8_payload.sha(judge_path.read_bytes())},
               'profile': profile, 'strict_rate_gate': gate, 'win_intervals': intervals,
               'rate_headroom_kbps': intervals['maximum_win_rate_supremum_kbps'] - rates['conditional_kbps'] if intervals['maximum_win_rate_supremum_kbps'] is not None else None,
               'cpu_seconds': usage.ru_utime + usage.ru_stime, 'wall_seconds': time.monotonic() - start,
               'cpu_limit_s': 180, 'wall_limit_s': 300}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('candidate_sha256', 'candidate_bytes', 'bytes_by_unit', 'rates_kbps',
                                              'rate_headroom_kbps', 'cpu_seconds', 'wall_seconds')}, sort_keys=True))


if __name__ == '__main__':
    main()
