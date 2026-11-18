"""Re-envelope frozen local inputs with measured Linux receiver identities."""
import argparse
import hashlib
import json
import zlib
from pathlib import Path

from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire
from experiments.tennis import common_receiver_manifest as resources
from experiments.tennis.k6b_rate_streams import HEADER
from src.segmentation.masks import decompress
from experiments.tennis.w8_receiver_finish import FFMPEG_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--png', type=Path)
    args = parser.parse_args()
    parity = json.loads(Path('docs/experiments/joint-readiness-20261118/linux-jpeg.json').read_text())
    if len(parity['references']) != 4 or not all(r['pixel_equality'] and not r['different_rgb_channels'] for r in parity['references']):
        raise ValueError('unresolved campus pixel identity')
    args.output.mkdir(parents=True, exist_ok=False)
    registry = {}
    old_hashes = dict(parity['carrier_hashes'])
    paths = {q: args.root / f'q{q}/packet.pscp' for q in old_hashes}
    if args.png:
        paths['png'] = args.png
        old_hashes['png'] = 'f91ce34198771e924a39a5bbf1255cfaedb9207a2f2b91869f342892df3f7836'
    for q, path in paths.items():
        source = path.read_bytes()
        if wire.sha(source) != old_hashes[q]:
            raise ValueError('old carrier identity')
        profile = packet.inspect(source)['profile']
        sections, _, _ = packet.unpack(source, profile)
        if q != 'png':
            profile['resources']['reference_decoder'] = parity['opencv_build_sha256']
        profile['resources']['native_decoder'] = FFMPEG_SHA
        order = [('base', 'frame'), ('keypoints', 'frame'), ('ref0', 'reference'), ('ref1', 'reference')]
        portable = sections.copy()
        raw_hashes = {}
        for name in ('keypoints', 'ref0', 'ref1'):
            part = sections[name]
            magic, size, _, digest = HEADER.unpack_from(part)
            raw = decompress(part[HEADER.size:])
            if len(raw) != size or hashlib.sha256(raw).digest() != digest:
                raise ValueError('source transport mismatch')
            coded = zlib.compress(raw, 6)
            portable[name] = HEADER.pack(magic,size,len(coded),digest)+coded
            limit = packet.MAX_KPS if name == 'keypoints' else packet.MAX_REF
            if packet._bounded_transport(portable[name],magic,limit) != packet._bounded_transport(part,magic,limit):
                raise ValueError('inner coded bytes or metadata differ')
            raw_hashes[name] = digest.hex()
        new = wire.pack(profile, [(name, role, portable[name]) for name, role in order])
        rebound, roles = wire.unpack(new, profile)
        if portable != rebound or sum(roles.values()) != len(new):
            raise ValueError('section bytes changed')
        target = args.output / f'q{q}.pscp'
        target.write_bytes(new)
        manifest_path = args.output / f'q{q}.manifest'
        manifest_path.write_bytes(resources.encode(wire.sha(new)))
        billed = resources.verify(manifest_path, wire.sha(new))
        registry[q] = {'receiver_manifest_sha256': billed['manifest_sha256'], 'receiver_manifest_bytes': billed['manifest_bytes'],
                       'total_player_receiver_bytes': len(new)+billed['manifest_bytes'], 'sha256': wire.sha(new), 'original_sha256': wire.sha(source), 'bytes': len(new),
                       'role_bytes': roles, 'path': str(target), 'sections': {k: wire.sha(v) for k,v in portable.items()}, 'old_sections': {k: wire.sha(v) for k,v in sections.items()}, 'identical_raw_member_sha256':raw_hashes}
    receipt = {'carriers': registry, 'linux_opencv': parity['opencv_build_sha256'], 'linux_native_ffmpeg': FFMPEG_SHA,
               'parity_input_sha256': hashlib.sha256(Path('docs/experiments/joint-readiness-20261118/linux-jpeg.json').read_bytes()).hexdigest()}
    (args.output / 'linux-carriers.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(registry, indent=2))


if __name__ == '__main__':
    main()
