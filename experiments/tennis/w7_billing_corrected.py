"""Recompute W7 at corrected 12-crop bank rates from immutable fleet archives.

Example: python -m experiments.tennis.w7_billing_corrected --inbox /path/to/fleet/inbox
    --out-dir /new/output/directory
The directory must not already contain either result. The hashes and jobs are
fixed in the committed manifest; historical W7 results are never modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import w7_analysis as w
from experiments.tennis import w7_rates

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY / 'docs/experiments/w9-inputs/w7-archive-hashes.json'
TIER2 = {
    'nad000': ('20261031T180054Z-b9927a25', '20261031T192320Z-47fdb78c', '20261031T181055Z-28301343'),
    'nad001': ('20261031T180345Z-aaca7792', '20261031T182114Z-3bfa1a68', '20261031T181305Z-ff4c22b4'),
    'nad002': ('20261031T174602Z-7891a279', '20261031T192632Z-002fd19f', '20261031T175547Z-5f8c559b'),
}


def corrected_bank(row: dict[str, Any]) -> tuple[float, float, float]:
    """Reconstruct the historical charge and the fixed 12-PNG bank charge."""
    png = [f for f in row['reference_files'] if f['codec'] == 'png']
    if len(row['reference_files']) != 8 or len(png) != 2 or len({f['file'] for f in png}) != 2:
        raise ValueError(f"{row['window']}: unexpected PNG crop inventory")
    amount = row['references']['png_bytes']
    if sum(f['bytes'] for f in png) != amount:
        raise ValueError(f"{row['window']}: PNG bytes mismatch")
    fixed = w7_rates.rates({'rows': [row]}, {'nadal_kyrgios': 1_489_876})[row['window']]
    fixed_rate = fixed['match_bank_kbps_7200']
    old_bytes = amount * w7_rates.BANK_CROPS / len(row['reference_files'])
    old_rate = w7_rates.kbps(old_bytes, w7_rates.AMORTISED_SECONDS)
    fixed_bytes = fixed_rate * 1000 * w7_rates.AMORTISED_SECONDS / 8
    return fixed_bytes, old_bytes, fixed_rate - old_rate


def analyse(inbox: Path, out_dir: Path, length_seconds: float = 7200) -> None:
    if not 0 < length_seconds < float('inf'):
        raise ValueError('length must be finite and positive')
    if out_dir.exists():
        raise FileExistsError(f'output directory already exists: {out_dir}')
    manifest = json.loads(MANIFEST.read_text())
    files = {j: inbox / j / 'full/published.tar' for group in manifest.values() for j in group}
    for group in manifest.values():
        for job, digest in group.items():
            w.verify(files[job], digest)
    transports = []
    for job in manifest['transport']:
        with tarfile.open(files[job]) as tar:
            member = next(m for m in tar if m.isfile() and Path(m.name).name == 'transport.json')
            stream = tar.extractfile(member)
            assert stream is not None
            transports.extend(json.load(stream)['rows'])
    transport = {r['window']: r for r in transports}
    jobs = {n: (v['judge']['job'], v['score']['job'],
                v['dcvc'][0].get('job') if v['dcvc'] else None) for n, v in w.W7_INPUTS.items()}
    jobs.update(TIER2)
    results: dict[int, dict[str, Any]] = {}
    for tier, names in ((1, w.WINDOWS), (2, w.TIER2_WINDOWS)):
        windows = []
        for name in names:
            judge, score, dcvc = jobs[name]
            quality = copy.deepcopy(w.read_json(files[judge], 'judge'))
            original = w.hybrid_from_judge(quality, name)
            row = transport[original['id']]
            amount = row['references']['png_bytes']
            bank_bytes, old_bank_bytes, _ = corrected_bank(row)
            adapter = {'alcaraz_hurkacz': 1_488_252, 'medvedev_struff': 1_495_592, 'nadal_kyrgios': 1_489_876}[row['match']]
            correction = w7_rates.kbps(adapter + bank_bytes, length_seconds) - w7_rates.kbps(adapter + old_bank_bytes, 7200)
            judge_row = next(r for r in quality['rows'] if r['id'] == original['id'])
            next(r for r in judge_row['rates'] if r['scale'] == 4)['total_kbps'] += correction
            if name == 'al028':
                path = REPOSITORY / 'docs/experiments/w7-specs/dcvc-points-al028-scale2.json'
                w.verify(path, 'e5d05fe54feedf186bc83b44b153bd2e6ae3247d60648859ef69e73f154408f1')
                sources = [json.loads(path.read_text())]
            else:
                sources = [w.read_json(files[dcvc], 'dcvc')] if dcvc else []
            try:
                result = w.analyse_window(name, quality, w.read_json(files[score], 'score'), sources)
            except ValueError as exc:
                if 'no anchor family resolved' not in str(exc):
                    raise
                result = {'window': name, 'rate_kbps': original['rate_kbps'] + correction,
                          'unresolved': str(exc)}
            result['billing_correction'] = {'png_bytes_two_crops': amount, 'intended_bank_bytes': bank_bytes,
                                            'old_bank_bytes': old_bank_bytes, 'adapter_bytes': adapter, 'length_seconds': length_seconds, 'old_rate_kbps': original['rate_kbps'],
                                            'added_rate_kbps': correction}
            windows.append(result)
        unresolved = [r['window'] for r in windows if 'unresolved' in r]
        results[tier] = {'windows': windows, 'group': None if unresolved else w.summarize_group(windows),
                         'unresolved_windows': unresolved}
    out_dir.mkdir(parents=True, exist_ok=False)
    for tier, result in results.items():
        path = out_dir / f'w7-tier{tier}-billing-corrected.json'
        with path.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inbox', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--length-seconds', type=float, default=7200)
    args = parser.parse_args()
    analyse(args.inbox, args.out_dir, args.length_seconds)


if __name__ == '__main__':
    main()
