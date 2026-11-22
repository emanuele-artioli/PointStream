"""Prepare the single shared-quality AVIF references for the matched player pilot."""
import argparse
import hashlib
import json
import resource
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.jobs.inbox import validate_spec
from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire
from experiments.tennis.player_requalification import array_sha
from experiments.tennis.player_requalification_specs import build_specs

ROOT = Path('docs/experiments/common-reference-q78-20261122')
PREV = Path('/Users/manu/Documents/PointStream-prepared/2026-11-17-reference-pilot-v2/q40')
PNG = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
OUT = Path('/private/tmp/ps-common-avif-q78-20261122')
REMOTE = '/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-common-avif-q78-20261122'
ENC = Path('/opt/homebrew/bin/avifenc')
DEC = Path('/opt/homebrew/bin/avifdec')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def main(q=78):
    t = time.monotonic()
    meta = json.loads(Path('docs/experiments/common-reference-20261122/receipt.json').read_text())
    previous = json.loads(Path('docs/experiments/player-reference-pilot-20261117/prepared-manifest.json').read_text())['40']
    config_data = (PREV / 'config.json').read_bytes()
    assert sha(config_data) == previous['config_sha256']
    assert file_sha(PNG) == meta['source_packet_sha256']
    original = PNG.read_bytes()
    sections, _ = wire.unpack(original, packet._embedded_profile(original))
    ladder = json.loads(Path('docs/experiments/reference-ladder-20261122/result.json').read_text())
    if q == 90:
        checks = json.loads(Path('docs/experiments/common-reference-q90-20261122/image-preflight.json').read_text())
        assert checks['passes_30_percent']
        registered = {row['unit']: dict(quality=q, payload_sha256=row['payload_sha256'],
                                         payload_bytes=row['image_bytes'], pixel_sha256=row['decoded_rgb_sha256'])
                      for row in checks['rows']}
    else:
        probe = next(row for row in ladder['probes'] if row['codec'] == 'avif' and row['quality'] == 78)
        registered = {}
        for index, measured in enumerate(probe['players']):
            prior = next(row for row in meta['rows'] if row['section'] == f'ref{index}')
            registered[prior['unit']] = dict(quality=q, payload_sha256=measured['sha256'],
                                             payload_bytes=measured['bytes'],
                                             pixel_sha256=measured['decoded_rgb_sha256'])
    expected = {entry['unit']: entry for entry in previous['units']}
    out = OUT if q == 78 else Path("/private/tmp/ps-common-avif-q90-verified-20261122")
    remote = REMOTE.replace("q78", f"q{q}")
    out.mkdir(parents=True, exist_ok=False)
    publish = out / 'publish'
    publish.mkdir()
    config = json.loads(config_data)
    config['conditioning']['provenance'] = (
        'Authenticated selected PNG packet RGB is encoded with AVIF and decoded on the preparation '
        'host. Only the reference array in the verified JPEG-Q40 conditioning NPZ is replaced. '
        'All pose, box, timestamp, and seed arrays remain byte-identical to the prior pilot. '
        'This is a tensor-conditioning pilot rather than a serialized receiver witness.'
    )
    rows = []
    for unit in config['units']:
        name = unit['unit']
        source_npz = PREV / 'publish' / f'{name}.npz'
        assert file_sha(source_npz) == expected[name]['new_npz_sha256'] == unit['sha256']
        ref_name = next(key for key in ('ref0', 'ref1')
                        if packet._bounded_transport(sections[key], b'REF1', packet.MAX_REF)[0]['unit'] == name)
        ref_meta, png = packet._bounded_transport(sections[ref_name], b'REF1', packet.MAX_REF)
        source = out / f'{name}-source.png'
        source.write_bytes(png)
        original_rgb = np.array(Image.open(source).convert('RGB'))
        assert array_sha(original_rgb) == expected[name]['source_rgb_sha256']
        target = registered[name]
        avif = out / f'{name}.avif'
        quality = target['quality']
        command = [str(ENC), '-q', str(quality), '--speed', '8', '--jobs', '1', str(source), str(avif)]
        subprocess.run(command, capture_output=True, check=True, timeout=60)
        assert file_sha(avif) == target['payload_sha256'] and avif.stat().st_size == target['payload_bytes']
        decoded = out / f'{name}-decoded.png'
        subprocess.run([str(DEC), str(avif), str(decoded)], capture_output=True, check=True, timeout=60)
        rgb = np.array(Image.open(decoded).convert('RGB'))
        assert rgb.shape == (784, 512, 3) and rgb.dtype == np.uint8
        assert (array_sha(rgb) if q == 90 else sha(rgb.tobytes())) == target['pixel_sha256']
        with np.load(source_npz, allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        prior_hashes = {k: array_sha(v) for k, v in arrays.items() if k != 'reference'}
        assert array_sha(arrays['reference']) == expected[name]['decoded_rgb_sha256']
        arrays['reference'] = rgb
        target_npz = publish / source_npz.name
        np.savez_compressed(target_npz, **arrays)
        with np.load(target_npz, allow_pickle=False) as check:
            assert all(array_sha(check[k]) == value for k, value in prior_hashes.items())
            assert np.array_equal(check['reference'], rgb)
        unit['sha256'] = file_sha(target_npz)
        rows.append(dict(unit=name, reference_frame=ref_meta['reference_frame'],
                         reference_label=unit['review_label'], quality=quality,
                         avif_bytes=avif.stat().st_size, avif_sha256=file_sha(avif),
                         decoded_rgb_sha256=array_sha(rgb), original_rgb_sha256=array_sha(original_rgb),
                         unchanged_arrays=prior_hashes, npz_sha256=unit['sha256'],
                         encoder=command[0:7], jpeg_q40_bytes=next(entry['payload_bytes'] for entry in meta['rows'] if entry['unit'] == name and entry['codec'] == 'jpeg' and entry['quality'] == 40)))
    config['reference_pilot'] = dict(codec='avif', units=rows, source_packet_sha256=sha(original),
                                      encoder=dict(path=str(ENC), sha256=file_sha(ENC),
                                      version=subprocess.run([str(ENC), '--version'], capture_output=True, text=True, check=True).stdout.strip()),
                                      decoder=dict(path=str(DEC), sha256=file_sha(DEC),
                                      version=subprocess.run([str(DEC), '--version'], capture_output=True, text=True, check=True).stdout.strip()))
    with tarfile.open(out / 'conditioning.tar', 'w') as tar:
        for f in sorted(publish.iterdir()):
            tar.add(f, arcname=f'publish/{f.name}')
    config_path = out / 'config.json'
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    inventory = json.loads((PREV / 'inputs.json').read_text())
    candidate = next(x for x in inventory['inputs'] if x['name'] == 'conditioning')
    candidate.update(path=f'{remote}/conditioning.tar', sha256=file_sha(out / 'conditioning.tar'))
    inventory_path = out / 'inputs.json'
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    spec = build_specs(config_path, inventory_path, f'{remote}/config.json', '2099-01-01T00:00:00Z')['tennis_only']
    spec['smoke'].update(seconds=300, representative_basis='Player 2 shared-quality AVIF conditioning, one complete 24-tick production context.')
    spec['full']['seconds'] = 450
    spec['budget_seconds'] = 900
    validate_spec(spec)
    spec['deadline'] = 'REPLACE_AFTER_RELEASE'
    (out / 'spec-template.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(status='prepared_no_gpu', config_sha256=file_sha(config_path),
                  inputs_sha256=file_sha(inventory_path), archive_sha256=file_sha(out / 'conditioning.tar'),
                  spec_sha256=file_sha(out / 'spec-template.json'), rows=rows,
                  cpu_seconds=usage.ru_utime + usage.ru_stime + child.ru_utime + child.ru_stime,
                  wall_seconds=time.monotonic()-t, output=str(out))
    assert report['cpu_seconds'] < 120 and report['wall_seconds'] < 240
    (out / 'manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=int, choices=(78, 90), default=78)
    main(parser.parse_args().quality)
