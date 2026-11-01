"""W7: per-window rate inputs derived from the generalised stages (CPU, no quality).

`rates(transport)` -> the `--rates` file of `k6b_real_base_judge` for manifest windows (keypoint packet, bank and adapter terms);
`window_state(...)` -> the `--window-state` entry of the w1_* anchor stages. Policy (docs/experiments/w7-untouched.md, 2026-10-30):
the reference recipe is `window_frame0` (two crops per window, coded and billed as before). The old bank charge was the PNG bytes
of 12 crops over 7200 s; a window has two crops, so its bank term is its own PNG bytes times `BANK_CROPS / 2` (a 12-crop bank of
crops of the same size) over 7200 s, to keep the charge comparable to the old windows. This scaling is a reading to be confirmed.
"""
from __future__ import annotations

import json
from typing import Any

AMORTISED_SECONDS = 7200
BANK_CROPS = 12
#: adapter + guider bytes billed per match (k6b-players.md, 2026-10-25 accounting), amortised over 7200 s
ADAPTER_BYTES = {'alcaraz_hurkacz': 1_488_252, 'medvedev_struff': 1_495_592}
START_BASE = {4: 60, 2: 67, 1: 70}  # first CRF probes only; never read as a result


def kbps(n_bytes: float, seconds: float) -> float:
    return n_bytes * 8 / seconds / 1000


def rates(transport: dict[str, Any], adapter_bytes: dict[str, int] | None = None) -> dict[str, dict[str, float]]:
    if any(k in ADAPTER_BYTES and v != ADAPTER_BYTES[k] for k, v in (adapter_bytes or {}).items()):
        raise ValueError('a frozen adapter charge changed')
    charges = {**ADAPTER_BYTES, **(adapter_bytes or {})}
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in charges.values()):
        raise ValueError('adapter charges must be positive integer bytes')
    out: dict[str, dict[str, float]] = {}
    for row in transport['rows']:
        fps = row['keypoints']['contract']['fps'][0]
        # Each crop has PNG plus JPEG alternatives. Count only the transmitted
        # lossless PNG entries, not the number of encoded alternatives.
        png = [f for f in row['reference_files'] if f['codec'] == 'png']
        if not png or len({f['file'] for f in png}) != len(png):
            raise ValueError(f"{row['window']}: missing or duplicate PNG references")
        if sum(f['bytes'] for f in png) != row['references']['png_bytes']:
            raise ValueError(f"{row['window']}: PNG reference byte total mismatch")
        bank_bytes = row['references']['png_bytes'] * BANK_CROPS / len(png)
        out[row['window']] = {'keypoint_kbps': kbps(row['keypoints']['bytes'], 240 / fps),
                              'match_bank_kbps_7200': kbps(bank_bytes, AMORTISED_SECONDS),
                              'adapter_guider_kbps_7200': kbps(charges[row['match']], AMORTISED_SECONDS),
                              # worst-case sensitivity: the window's own crops billed over the 240-frame window alone
                              'bank_worst_case_kbps': kbps(row['references']['png_bytes'], 240 / fps)}
    return out


def window_state(base_rows: list[dict[str, Any]], transport_row: dict[str, Any], name: str) -> dict[str, Any]:
    """`w1_roi.window_from_spec` state: plain CRF 63 base bytes per scale (native base manifest), packet sizes (transport)."""
    ivf = {int(r['scale']): int(r['ivf']['bytes']) for r in base_rows if r['window'] == name}
    if sorted(ivf) != [2, 4, 8]:
        raise ValueError(f'{name}: native base needs scales 8, 4 and 2')
    return {'control_ivf_bytes': ivf, 'start_base': dict(START_BASE), 'kps1_bytes': int(transport_row['keypoints']['kps1_bytes']),
            'kps2_bytes': int(transport_row['keypoints']['bytes']), 'hybrid_archive_sha': None}


def base_rows(archive: Any) -> list[dict[str, Any]]:
    """Rows of the native-base manifest inside a `k6b_native_base` stage archive (published.tar -> publish/bundle.tar -> base/manifest.json)."""
    import tarfile

    with tarfile.open(archive) as outer:
        member = outer.extractfile('publish/bundle.tar')
        if member is None:
            raise ValueError('native base archive has no bundle')
        with tarfile.open(fileobj=member) as bundle:
            manifest = bundle.extractfile('base/manifest.json')
            if manifest is None:
                raise ValueError('native base bundle has no manifest')
            return list(json.loads(manifest.read())['rows'])


def compute(transport_json: Any, base_archive: Any, windows_file: Any, names: list[str]) -> dict[str, Any]:
    from pathlib import Path

    from experiments.tennis import window_manifest as wm

    transport = json.loads(Path(transport_json).read_text())
    manifest = json.loads(Path(windows_file).read_text())
    charges = manifest.get('adapter_bytes', {})
    if any(match not in wm.ADAPTERS or manifest.get('adapters', {}).get(match) != wm.ADAPTERS[match] for match in charges):
        raise ValueError('adapter charge has no matching fitted adapter identity')
    specs = {w.name: w for w in wm.select(wm.load(Path(windows_file)), names or None)}
    rows = base_rows(base_archive)
    by_id = {r['window']: r for r in transport['rows']}
    state = {name: window_state(rows, by_id[spec.id], name) for name, spec in specs.items()}
    return {'rates': rates({'rows': [by_id[spec.id] for spec in specs.values()]}, charges), 'window_state': state}


def main(argv: list[str] | None = None) -> int:
    """`--transport transport.json --base published.tar --windows-file windows.json [--window NAME]...` writes rates.json and window-state.json
    to PS_STAGE_DIR (the `--rates` file of the judge and the `--window-state` file of the w1_* stages); `--validate` recomputes both from the
    inputs recorded in dispatch.json and compares them."""
    import argparse
    import os
    from pathlib import Path

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--transport')
    p.add_argument('--base')
    p.add_argument('--windows-file')
    p.add_argument('--window', action='append', default=[])
    p.add_argument('--expect', type=int, default=0, help='number of windows the job is meant to cover (a fleet scale parameter); 0 = unchecked')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args(argv)
    stage = Path(os.environ['PS_STAGE_DIR'])
    if a.validate:
        command = json.loads((stage / 'dispatch.json').read_text())['command']
        names = [command[i + 1] for i, c in enumerate(command) if c == '--window']
        value = lambda flag: command[command.index(flag) + 1]  # noqa: E731
        again = json.loads(json.dumps(compute(value('--transport'), value('--base'), value('--windows-file'), names)))  # as written: string keys
        got = {'rates': json.loads((stage / 'rates.json').read_text()), 'window_state': json.loads((stage / 'window-state.json').read_text())}
        checks = {'rates_recomputed': got['rates'] == again['rates'], 'state_recomputed': got['window_state'] == again['window_state'],
                  'one_per_window': len(got['rates']) == len(names) == len(got['window_state']),
                  'positive': all(v > 0 for r in got['rates'].values() for v in r.values()),
                  'scales_8_4_2': all(sorted(int(k) for k in s['control_ivf_bytes']) == [2, 4, 8] for s in got['window_state'].values())}
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}))
        return int(not all(checks.values()))
    if not (a.transport and a.base and a.windows_file):
        p.error('--transport, --base and --windows-file are required')
    if a.expect and a.expect != len(a.window):
        p.error('--expect does not match the number of --window arguments')
    out = compute(a.transport, a.base, a.windows_file, a.window)
    (stage / 'rates.json').write_text(json.dumps(out['rates'], indent=1, sort_keys=True))
    (stage / 'window-state.json').write_text(json.dumps(out['window_state'], indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
