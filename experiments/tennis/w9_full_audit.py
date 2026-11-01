"""Fresh-process, read-only W7 repair full-output audit.

Input job files are never modified. The output directory must not exist. Run
only after the corresponding fleet smoke gate has passed and the full archive
and normalized spec identities have been independently recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

from experiments.jobs.inbox import digest as spec_digest
from experiments.tennis import w1_score, window_manifest


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_score_inputs(spec: dict, score_inputs: dict) -> None:
    """Bind W1's positional extra names to the frozen staged arguments."""
    staged = spec['staged_inputs']
    names = [item['name'] for item in staged]
    if len(names) != len(set(names)):
        raise ValueError('duplicate staged input names')
    args = spec.get('arguments', [])
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise ValueError('invalid scorer arguments')
    if any(arg.startswith('--encode-extra=') for arg in args) or args.count('--encode-extra') > 1:
        raise ValueError('ambiguous encode-extra binding')
    extras = []
    if '--encode-extra' in args:
        position = args.index('--encode-extra') + 1
        while position < len(args) and not args[position].startswith('--'):
            match = re.fullmatch(r'\{staged:([A-Za-z_][A-Za-z0-9_]*)\}', args[position])
            if match is None:
                raise ValueError('encode-extra must be a whole staged placeholder')
            extras.append(match.group(1))
            position += 1
        if not extras:
            raise ValueError('encode-extra requires staged inputs')
    if (len(extras) != len(set(extras)) or
            any(name not in names or name in ('windows', 'state') or args.count(f'{{staged:{name}}}') != 1
                for name in extras)):
        raise ValueError('duplicate, missing or invalid extra input binding')
    if any(re.fullmatch(r'encode_x[0-9]+', name) and name not in extras for name in names):
        raise ValueError('unbound staged extra input')
    mapping = {name: f'encode_x{index}' for index, name in enumerate(extras)}
    direct = {name for name in names if name not in extras and name not in ('windows', 'state')}
    if direct & set(mapping.values()):
        raise ValueError('extra input summary key collision')
    expected_extra_keys = set(mapping.values())
    if {key for key in score_inputs if re.fullmatch(r'encode_x[0-9]+', key)} != expected_extra_keys:
        raise ValueError('missing or unexpected extra summary keys')
    for item in staged:
        name = item['name']
        if name not in ('windows', 'state') and score_inputs.get(mapping.get(name, name)) != item['sha256']:
            raise ValueError(f'full summary missing or changed staged input: {name}')


def audit(job: Path, manifest: Path, hybrid: Path, windows: Path, output: Path, archive_sha: str, expected_spec_sha: str) -> None:
    if output.exists():
        raise FileExistsError(f'audit output exists: {output}')
    spec = json.loads((job / 'spec.json').read_text())
    if spec_digest(spec) != expected_spec_sha:
        raise ValueError('normalized spec SHA mismatch')
    source = job / 'full' / 'published.tar'
    if sha(source) != archive_sha:
        raise ValueError('published archive SHA mismatch')
    expected = next(x['sha256'] for x in spec['staged_inputs'] if x['name'] == 'repair_manifest')
    if sha(manifest) != expected:
        raise ValueError('selection manifest SHA mismatch')
    inputs = {x['name']: x['sha256'] for x in spec['staged_inputs']}
    if sha(windows) != inputs['windows'] or sha(hybrid) != inputs['hybrid']:
        raise ValueError('windows or original hybrid archive SHA mismatch')
    summary = json.loads((job / 'full' / 'score-summary.json').read_text())
    with tarfile.open(source) as tar:
        stream = tar.extractfile('publish/score.json')
        if stream is None or json.load(stream) != summary:
            raise ValueError('published and external summaries disagree')
    execution_path = job / 'full' / 'execution.json'
    if not execution_path.is_file() or json.loads(execution_path.read_text()).get('exit_code') != 0:
        raise ValueError('successful full-stage execution record required')
    score_inputs = summary.get('inputs', {})
    check_score_inputs(spec, score_inputs)
    if summary.get('repair_manifest') != json.loads(manifest.read_text()):
        raise ValueError('embedded and pinned selection manifests disagree')
    matches = [w for w in window_manifest.load(windows) if w.name == summary['window']]
    if len(matches) != 1 or matches[0].id != summary['window_id']:
        raise ValueError('original window identity mismatch')
    controls = w1_score.audit_repair_controls(source, archive_sha, hybrid, inputs['hybrid'], matches[0].safe_id)
    if len(summary.get('ticks', [])) != len(w1_score.TICKS):
        raise ValueError('full 80-tick output required')
    output.mkdir(parents=True, exist_ok=False)
    stage = output / 'full'
    stage.mkdir()
    shutil.copyfile(job / 'spec.json', output / 'spec.json')
    shutil.copyfile(job / 'full' / 'score-summary.json', stage / 'score-summary.json')
    shutil.copyfile(source, stage / 'published.tar')
    local_manifest = output / 'repair-manifest.json'
    shutil.copyfile(manifest, local_manifest)
    (stage / 'staging.json').write_text(json.dumps({'inputs': [{'name': 'repair_manifest', 'sha256': expected,
                                                               'extract': False, 'path': str(local_manifest.resolve())}]}))
    execution = job / 'full' / 'execution.json'
    if execution.exists():
        shutil.copyfile(execution, stage / 'execution.json')
    before = {name: os.environ.get(name) for name in ('PS_STAGE_DIR', 'PS_VALIDATION_PATH')}
    try:
        os.environ['PS_STAGE_DIR'] = str(stage.resolve())
        os.environ['PS_VALIDATION_PATH'] = str((output / 'validation.json').resolve())
        w1_score.validate()
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    result = json.loads((output / 'validation.json').read_text())
    if result.get('passed') is not True:
        raise ValueError('full validator did not pass')
    (output / 'audit.json').write_text(json.dumps({'passed': True, 'archive_sha256': archive_sha,
                                                    'spec_sha256': expected_spec_sha,
                                                    'manifest_sha256': expected, 'raw_control_errors': controls, 'checks': result['checks']}, indent=2) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--hybrid', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--spec-sha256', required=True)
    a = p.parse_args()
    audit(a.job, a.manifest, a.hybrid, a.windows, a.out_dir, a.archive_sha256, a.spec_sha256)


if __name__ == '__main__':
    main()
