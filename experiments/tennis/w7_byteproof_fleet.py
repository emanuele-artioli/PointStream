"""Three-point W7 byte-proof pilot; no scoring or automatic 75-role run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from experiments.tennis import w7_byteproof as bp, w7_mkv_profile as profile, w7_remux_job as old

INVENTORY_SHA = '867d9fff8974f4ec5320c9b830138d4aac2ee22c285d7de45aebb04dff3bab11'
INVENTORY = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/mkv-53-inventory-20261105.json'
NAMES = ('med000_control', 'med019_roi', 'med000_new')


def context(stage: Path) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, Any], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (bp.digest(INVENTORY) != INVENTORY_SHA or spec.get('device') != 'cpu'
            or spec.get('cpu_threads') != 1 or spec.get('single_cpu_campaign') is not True
            or spec.get('budget_seconds') != 540 or spec.get('smoke', {}).get('seconds') != 150
            or spec.get('full', {}).get('seconds') != 30 or spec.get('validator_seconds') != 60
            or spec.get('environment', {}).get('sha256') != old.ENV_SHA
            or staging['environment']['sha256'] != old.ENV_SHA
            or len(dispatch.get('cpu_affinity', [])) != 1):
        raise ValueError('pilot campaign, source inventory or environment mismatch')
    rows = bp.representative_three(json.loads(INVENTORY.read_text()))
    staged = {x['name']: x for x in staging['inputs']}
    declared = {x['name']: x for x in spec['staged_inputs']}
    if set(staged) != set(NAMES) or set(declared) != set(NAMES):
        raise ValueError('pilot requires exactly three archived inputs')
    root = Path(staging['local_root']).resolve(strict=True)
    paths: dict[str, Path] = {}
    for name, row in zip(NAMES, rows):
        entry, item = staged[name], declared[name]
        if (entry['sha256'] != row['encode_archive_sha256'] or item['sha256'] != row['encode_archive_sha256']
                or entry.get('extract') is not True or item.get('extract') is not True):
            raise ValueError('staged archive identity differs')
        tree = Path(entry['path']).resolve(strict=True)
        archive = tree.parent / Path(entry['source']).name
        if not tree.is_relative_to(root) or bp.digest(archive) != row['encode_archive_sha256']:
            raise ValueError('staged archive bytes or local root differ')
        member = Path(row['ivf_member'])
        if not member.parts or member.parts[0] != 'publish':
            raise ValueError('invalid historical archive member')
        native = (tree / member).resolve(strict=True)
        if not native.is_relative_to(tree) or bp.digest(native) != row['original_ivf_sha256']:
            raise ValueError('staged original IVF differs')
        paths[name] = native
    bins = Path(staging['environment']['bin']).resolve(strict=True)
    if not bins.is_relative_to(root) or bp.digest(bins / 'ffmpeg') != old.FFMPEG_SHA or bp.digest(bins / 'ffprobe') != old.FFPROBE_SHA:
        raise ValueError('packed tools differ')
    paths['ffmpeg'], paths['ffprobe'] = bins / 'ffmpeg', bins / 'ffprobe'
    return rows, paths, spec, dispatch


def run(stage: Path, scratch: Path) -> None:
    rows, paths, _, _ = context(stage)
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single CPU affinity required')
    out = scratch / 'publish'
    out.mkdir(exist_ok=False)
    if stage.name == 'full':
        (out / 'pilot-only.json').write_text(json.dumps({'scope': 'pilot only; 75-role run requires separate review'}) + '\n')
        return
    if stage.name != 'smoke':
        raise ValueError('unknown pilot stage')
    for name, row in zip(NAMES, rows):
        bp.prove(row, paths[name], out / name, paths['ffmpeg'], paths['ffprobe'])


def validate(stage: Path, output: Path) -> None:
    try:
        rows, paths, spec, dispatch = context(stage)
        staging = json.loads((stage / 'staging.json').read_text())
        import tarfile
        with tempfile.TemporaryDirectory(prefix='w7-byteproof-validation-', dir=os.environ['TMPDIR']) as folder:
            base = Path(folder)
            with tarfile.open(stage / 'published.tar') as tar:
                if bp.digest(stage / 'published.tar') != staging['published']['sha256']:
                    raise ValueError('published smoke digest differs')
                members = tar.getmembers()
                if len(members) > 40 or any(not (x.isfile() or x.isdir()) for x in members):
                    raise ValueError('unexpected proof archive members')
                tar.extractall(base, filter='data')
            for name, row in zip(NAMES, rows):
                work = base / 'publish' / name
                record = bp.validate_proof(row, paths[name], work)
                fresh = base / (name + '.ivf')
                cmd = [str(paths['ffmpeg']), '-nostdin', '-v', 'error', '-threads', '1', '-i',
                       str(work / 'delivered.mkv'), '-map', '0:v:0', '-an', '-c:v', 'copy',
                       '-fps_mode', 'passthrough', '-f', 'ivf', '-n', str(fresh)]
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                checked = bp.compare_ivf(paths[name], fresh)
                if not checked['exact_ivf_bytes'] or bp.digest(fresh) != bp.digest(work / 'decoder-scratch.ivf'):
                    raise ValueError('independent MKV to IVF extraction differs')
                probes = profile.commands(str(paths['ffmpeg']), str(paths['ffprobe']), 'unused', paths[name], work)
                before = json.loads(subprocess.run(probes['probe_original'], capture_output=True, check=True, timeout=30).stdout)
                after = json.loads(subprocess.run(probes['probe_mkv'], capture_output=True, check=True, timeout=30).stdout)
                original_probe = bp.probe(before, row['width'], row['height'])
                delivery_probe = bp.probe(after, row['width'], row['height'])
                if (original_probe['timestamps'] != delivery_probe['timestamps']
                        or original_probe['colors'] != record['original_color_metadata']
                        or delivery_probe['colors'] != record['delivered_color_metadata']):
                    raise ValueError('independent timestamp or color metadata differs')
        execution = json.loads((stage / 'execution.json').read_text())
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
            raise ValueError('invalid pilot execution or clock')
        # Include validator work, use all 75 logical roles without assuming deduplication.
        measured = max(execution['seconds'], now - clock['phase_started'])
        projected = bp.smoke_forecast(measured, clock['budget_end'] - now, 540, 60)
        output.write_text(json.dumps({'passed': True, 'checks': {'three_frozen_roles': True,
                          'independent_mkv_extract': True, 'exact_ivf': True, 'timestamps': True,
                          'cost_forecast': True}, 'projection_seconds': projected,
                          'scope': 'pilot only; separately review remaining budget before full run'}) + '\n')
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'byteproof': False},
                                      'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise


def main() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    if sys.argv[1:] == ['--validate-smoke']:
        validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif sys.argv[1:] == ['--stage', stage.name] and stage.name in ('smoke', 'full'):
        run(stage, Path(os.environ['PS_SCRATCH_DIR']))
    else:
        raise ValueError('unexpected arguments')


if __name__ == '__main__':
    main()
