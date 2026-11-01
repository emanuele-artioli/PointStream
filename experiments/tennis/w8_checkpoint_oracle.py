"""Bounded local diagnostic for the frozen W8 match packet, not production evidence."""
from __future__ import annotations

import argparse
import io
import json
import os
import resource
import tarfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', required=True, type=Path)
    parser.add_argument('--guider', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    # Set before importing numerical libraries, and retain a hard per-process bound.
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    start = time.monotonic()
    import torch
    from experiments.tennis.k6b_fit import quantize_adapters
    from experiments.tennis.w8_payload import compare_checkpoint, pack_match, sha, unpack_match

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fit_hash = 'cceb49b3d593c770e5b0d8dcd5d69b5b358bf88d76030cef8034674417907584'
    guider_hash = '1a8b7c1b4db92980fd977b4fd003c1396bbae9a9cdea00c35d452136d5e4f488'
    if sha(args.fits.read_bytes()) != fit_hash or sha(args.guider.read_bytes()) != guider_hash:
        raise ValueError('frozen archive/guider hash mismatch')
    with tarfile.open(args.fits) as archive:
        members = [m for m in archive if m.name == 'publish/nk-C@fit-100-fp16.pt' and m.isfile()]
        if len(members) != 1 or members[0].size > 100_000_000:
            raise ValueError('missing/ambiguous match checkpoint')
        member = archive.extractfile(members[0])
        if member is None:
            raise ValueError('missing match bytes')
        checkpoint = member.read()
    state = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
    base = torch.load(args.guider, map_location='cpu', weights_only=True)
    expected, billing = quantize_adapters(state, 4, base)
    bundle = pack_match(state, base, guider_hash)
    decoded = unpack_match(bundle, sha(bundle), guider_hash, base)
    counts = compare_checkpoint(decoded, expected)
    args.out.mkdir(parents=False, exist_ok=False)
    (args.out / 'match.psw8').write_bytes(bundle)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {'scope': 'local torch diagnostic only; packed production runtime remains unverified',
               'fits_archive_sha256': fit_hash, 'checkpoint_sha256': sha(checkpoint), 'guider_sha256': guider_hash,
               'match_sha256': sha(bundle), 'match_bytes': len(bundle), 'prior_billed_tensor_bytes': billing['bytes'],
               'tensor_counts_exact': counts, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
               'cpu_seconds': usage.ru_utime + usage.ru_stime, 'wall_seconds': time.monotonic() - start,
               'max_rss_platform_units': usage.ru_maxrss, 'cpu_limit_s': 180, 'wall_limit_s': 300}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
