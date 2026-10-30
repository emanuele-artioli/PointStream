"""Descriptive fixed-point report; qualified clip scope, full-cohort rates, no extrapolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_confirm import PROVENANCE, check_manifest, cohort_of
from experiments.tennis.k6b_confirm_validate import strict_won


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep all 24 cells in the denominator, including failures/missingness."""
    tally = {}
    for rate in ('billed', 'unbilled'):
        cells = [r[rate][m] for r in rows for m in ('lpips', 'dists')]
        wins = sum(c['won'] is True for c in cells)
        known = sum(c['won'] is not None for c in cells)
        tally[rate] = {'wins': wins, 'resolved': known, 'denominator': 24, 'score': wins / 24,
                       'maximum_possible_wins': wins + 24 - known}
    matches = {match: sum(r['match'] == match for r in rows) for match in ('alcaraz_hurkacz', 'medvedev_struff')}
    complete = (len(rows) == 12 and min(matches.values()) >= 3 and tally['billed']['resolved'] == 24
                and (tally['billed']['wins'] >= 18 or tally['unbilled']['resolved'] == 24))
    if complete:
        decision = ('descriptive billed bar reached' if tally['billed']['wins'] >= 18 else
                    'user judgment: unbilled bar only' if tally['unbilled']['wins'] >= 18 else
                    'descriptive criterion failed')
    elif tally['unbilled']['maximum_possible_wins'] < 18:
        decision = 'futility: even all remaining unbilled wins cannot reach 18/24'
    else:
        decision = 'incomplete; no interim pass'
    return {'decision': decision, 'complete': complete, 'matches': matches, 'scores': tally}


def receipt(stage: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    from experiments.visor.b1 import file_sha256
    def read(name: str) -> Any:
        return json.loads((stage.parent / name).read_text())
    gate = read('gate.json')
    full = json.loads((stage / 'scientific-validation.json').read_text())
    if (gate.get('passed') is not True or full.get('passed') is not True or not full.get('checks')
            or any(v is not True for v in full['checks'].values())):
        raise ValueError(f'{stage}: missing substantive smoke/full validation')
    return {'job': stage.parent.name, 'path': str(stage), 'ready': read('ready.json'),
            'gpu': metadata.get('gpu', metadata.get('device')), 'state': read('state.json'),
            'spec_sha256': file_sha256(stage.parent / 'spec.json'),
            'published_sha256': file_sha256(stage / 'published.tar')}


def report(manifest: dict[str, Any], anchor_stages: list[Path], judge_stages: list[Path],
           generation_stages: list[Path]) -> dict[str, Any]:
    from experiments.tennis.k4 import safe
    from experiments.tennis.k6 import ANCHORS, anchor_name
    from experiments.tennis.k6_report import pooled_bits, window_metrics
    from experiments.tennis.k6b import window_means
    from experiments.visor.b1 import file_sha256
    from src.segmentation.masks import decompress

    check_manifest(manifest)
    ids = [f'{c}@{s+120}' for c,s,_,_ in cohort_of(manifest)]
    sources = []
    anchors = {}
    judges = {}
    billing: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix='k6b-confirm-report-', dir='/tmp') as temp:
        root = Path(temp)
        for i, stage in enumerate(anchor_stages):
            meta = json.loads((stage / 'k6.json').read_text())
            if meta['stage'] != 'full' or not meta.get('complete'):
                raise ValueError('Only complete native anchor stages')
            sources.append(receipt(stage, meta))
            with tarfile.open(stage / 'published.tar') as tar:
                tar.extractall(root / f'a{i}', filter='data')
            for row in meta['windows']:
                wid = row['id']
                if wid in anchors or wid not in ids or row['frames'] != 240:
                    raise ValueError('Anchor cohort/span mismatch')
                path = root / f'a{i}/publish' / safe(wid)
                if any(file_sha256(path/n) != h for n,h in row['files'].items()):
                    raise ValueError('Changed anchor artifact')
                kp = json.loads(decompress((path / 'keypoints.json.zst').read_bytes()))
                anchors[wid] = {'row': row, 'scores': dict(np.load(path / 'scores.npz')), 'symbols': kp['symbols']}
        # Rates are pooled over all 12 source-code symbol sets, even for an early futility report.
        if set(anchors) != set(ids):
            raise ValueError('All 12 source-code sets required to freeze pooled rates')
        costs = dict(zip(ids, pooled_bits([anchors[w]['symbols']['q4-velocity-e4'] for w in ids])))
        for stage in generation_stages:
            meta = json.loads((stage / 'k6b_generate.json').read_text())
            if meta['stage'] != 'full' or meta['max_frames'] != 0:
                raise ValueError('Only complete generation stages')
            sources.append(receipt(stage, meta))
            matches = {u.split('__', 1)[0] for u in meta['units']}
            if len(matches) != 1:
                raise ValueError('One match adapter per generation job')
            match = next(iter(matches))
            nbytes = meta['models']['match5']['fit']['billing']['bytes']
            if match in billing and billing[match] != nbytes:
                raise ValueError('Inconsistent billed adapter/guider bytes')
            billing[match] = nbytes
        for i, stage in enumerate(judge_stages):
            meta = json.loads((stage / 'k6b.json').read_text())
            if meta['stage'] != 'full' or meta['max_judged'] != 0 or meta['profiles'] != ['match5']:
                raise ValueError('Only full frozen-profile judging')
            sources.append(receipt(stage, meta))
            with tarfile.open(stage / 'published.tar') as tar:
                tar.extractall(root / f'j{i}', filter='data')
            for row in meta['windows']:
                wid = row['id']
                if wid in judges or wid not in ids or not row['s_masks_match_k6']:
                    raise ValueError('Judge cohort/mask mismatch')
                path = root / f'j{i}/publish' / safe(wid)
                if any(file_sha256(path/n) != h for n,h in row['files'].items()):
                    raise ValueError('Changed judge artifact')
                arow = anchors[wid]['row']
                if row['judged'] != arow['judged'] or row['scoring_masks'] != arow['scoring_masks']:
                    raise ValueError('Actual scoring indices or masks differ')
                means = window_means(dict(np.load(path / 'scores.npz')), row['judged'], row['variants'])
                judges[wid] = means
        rows = []
        for wid in ids:
            if wid not in judges:
                continue
            item = anchors[wid]
            row = item['row']
            metrics = window_metrics(item)
            match = wid.split('/')[0]
            keypoint_rate = costs[wid] / row['scored_seconds'] / 1000
            rate = keypoint_rate + 8 * billing[match] / 7200 / 1000
            # Verified native tennis-base@fit-100.pt payload; alternative billing only.
            base_bytes = 48878567
            base_rate = rate + 8 * base_bytes / 7200 / 1000
            names = [anchor_name(a) for a in ANCHORS]
            fair = all(not row['anchors'][n].get('failed') for n in names)
            curve = {n: {'kbps': row['anchors'][n]['kbps_r'], **metrics[n]} for n in names if not row['anchors'][n].get('failed')}
            method = judges[wid]['gen-match5']
            out: dict[str, Any] = {'window': wid, 'match': match, 'keypoint_kbps': keypoint_rate,
                    'native_fps': row['fps'], 'whole_window_seconds': 240 / float(row['fps']),
                    'registered_seconds': row['scored_seconds'], 'keypoint_bits': costs[wid],
                    'whole_window_keypoint_kbps': costs[wid] / (240 / float(row['fps'])) / 1000,
                    'anchor_points': curve, 'method': method, 'billed_kbps': rate, 'base_checkpoint_bytes': base_bytes, 'base_charged_kbps': base_rate, 'adapter_guider_bytes': billing[match], 'fair_envelope': fair,
                    'source_height_px': row['median_height'], 'scored_frames': len(row['scored']),
                    'judged_frames': len(row['judged']), 'diagnostics': {n: metrics[n] for n in row['variants'][len(ANCHORS):]}}
            for label, r in (('billed', rate), ('unbilled', keypoint_rate), ('base_charged', base_rate)):
                out[label] = {m: strict_won(curve, m, r, method[m]) if fair else
                              {'resolved': False, 'won': None, 'reason': 'incomplete anchor envelope'} for m in ('lpips','dists')}
            rows.append(out)
    return {**verdict(rows), 'rows': rows, 'inputs': sources, 'provenance': PROVENANCE,
            'limitations': {'reference': 'granted/unbilled', 'exact_target_matte': 'granted/unbilled',
            '115_px': 'exploratory only', 'registration': 'all 12 pass; discrimination untested',
            'original_source_and_replay_overlap': 'unresolved', 'clusters': 2,
            'base': 'shipped common decoder data, bytes/alternative amortization reported beside final result',
            'claim': 'clip-local same-broadcast diagnostic; no independent-source confirmation',
            'historical_step5': 'fail unchanged', 'bd_rate': 'not defined for one method operating point'}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--anchor-stage', required=True, action='append', type=Path)
    parser.add_argument('--judge-stage', required=True, action='append', type=Path)
    parser.add_argument('--generation-stage', required=True, action='append', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = report(json.loads(args.manifest.read_text()), args.anchor_stage, args.judge_stage, args.generation_stage)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k:out[k] for k in ('decision','complete','scores')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
