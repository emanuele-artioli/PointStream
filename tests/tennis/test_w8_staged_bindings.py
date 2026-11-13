from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.jobs import inbox
from experiments.tennis import w8_staged_bindings as bindings
from experiments.tennis import w8_witness_cpu_proof_job as proof


def fixture(tmp_path: Path) -> tuple[dict, Path, list[dict]]:
    spec = proof.make_spec((datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat(),
                           '/home/itec/emanuele/Datasets/pointstream-data/test/inventory.json')
    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    cache = tmp_path / 'cache'
    cache.mkdir()
    rows = []
    for row, direct in zip(spec['staged_inputs'], spec['inputs'], strict=True):
        source = tmp_path / ('source-' + row['name'])
        staged = cache / ('cache-' + row['name'])
        source.write_bytes(row['name'].encode())
        staged.write_bytes(source.read_bytes())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        row['path'], row['sha256'] = str(source), digest
        direct['path'], direct['sha256'] = str(source), digest
        rows.append({'name': row['name'], 'source': str(source), 'path': str(staged),
                     'sha256': digest, 'extract': False, 'mode': 'local-hit'})
    bindings.configure(spec)
    inbox.validate_spec(spec)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'local_root': str(tmp_path), 'inputs': rows}))
    command = inbox.phase_command(spec, 'smoke', rows, python='/python')
    (stage / 'dispatch.json').write_text(json.dumps({'command': command, 'staged_inputs': rows, 'input_identity': spec['inputs']}))
    return spec, stage, rows


def test_real_proof_builder_phase_cli_and_verify_skip(tmp_path, monkeypatch):
    spec, stage, rows = fixture(tmp_path)
    command = inbox.phase_command(spec, 'smoke', rows, python='/python')
    seen = []
    monkeypatch.setattr(proof, 'run', lambda path: seen.append(path))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(sys, 'argv', ['w8_witness_cpu_proof_job', *command[3:]])
    proof.main()
    assert seen == [stage]
    hashed = []
    def digest(path):
        hashed.append(str(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setattr(inbox, 'file_digest', digest)
    inbox.verify_inputs(spec, tmp_path)
    assert set(hashed) == {row['path'] for row in spec['staged_inputs'] if row['name'] in bindings.CPU_DIRECT}


def test_cli_rejects_missing_duplicate_wrong_path_or_source_alias(tmp_path):
    spec, stage, rows = fixture(tmp_path)
    supplied = [[r['name'], r['path']] for r in rows if r['name'] in bindings.staged_only_names(spec)]
    bindings.check_cli(stage, supplied, 'run')
    for altered in (supplied[:-1], supplied + supplied[:1],
                    [[supplied[0][0], rows[0]['source']], *supplied[1:]],
                    [[supplied[0][0], supplied[1][1]], *supplied[1:]]):
        with pytest.raises(ValueError):
            bindings.check_cli(stage, altered, 'run')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    dispatch['staged_inputs'][0]['sha256'] = '0' * 64
    (stage / 'dispatch.json').write_text(json.dumps(dispatch))
    with pytest.raises(ValueError):
        bindings.check_cli(stage, supplied, 'run')
    spec['arguments'].pop()
    with pytest.raises(ValueError):
        bindings.check_spec(spec)


def test_cli_rejects_hardlink_to_direct_source(tmp_path):
    import os

    spec, stage, rows = fixture(tmp_path)
    alias = Path(rows[0]['path'])
    alias.unlink()
    os.link(rows[0]['source'], alias)
    with pytest.raises(ValueError, match='identity differs'):
        bindings.check_cli(stage, [[r['name'], r['path']] for r in rows if r['name'] in bindings.staged_only_names(spec)], 'run')


def test_witness_phase_reaches_cli_only_with_real_staged_bindings(tmp_path, monkeypatch):
    from experiments.tennis import w8_witness_job as witness

    spec, stage, rows = fixture(tmp_path)
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_witness_job']
    spec.pop('w8_combined_cpu_proof_profile')
    spec['w8_witness_profile'] = 'w8-context-pilot-v1'
    spec['scale'] = {'phase': {'smoke': 'pilot', 'full': 'noop'}}
    bindings.configure(spec)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    command = inbox.phase_command(spec, 'smoke', rows, python='/python')
    (stage / 'dispatch.json').write_text(json.dumps({'command': command, 'staged_inputs': rows,
                                                      'input_identity': spec['inputs']}))
    seen = []
    monkeypatch.setattr(witness, 'run_pilot', lambda path: seen.append(path))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *command[3:]])
    witness.main()
    assert seen == [stage]
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *command[3:-3]])
    with pytest.raises(ValueError):
        witness.main()
    assert seen == [stage]


def test_archive_extraction_directory_binding_and_legacy_contract(tmp_path):
    spec, stage, rows = fixture(tmp_path)
    rows[0]['extract'] = True
    spec['staged_inputs'][0]['extract'] = True
    extracted = tmp_path / 'cache' / 'extracted-tree'
    extracted.mkdir()
    rows[0]['path'] = str(extracted)
    bindings.configure(spec)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'local_root': str(tmp_path), 'inputs': rows}))
    command = inbox.phase_command(spec, 'smoke', rows, python='/python')
    (stage / 'dispatch.json').write_text(json.dumps({'command': command, 'staged_inputs': rows,
                                                      'input_identity': spec['inputs']}))
    bindings.check_cli(stage, [[r['name'], r['path']] for r in rows if r['name'] in bindings.staged_only_names(spec)], 'run')
    spec.pop('w8_staged_arguments_profile')
    spec['arguments'] = ['{phase}']
    assert bindings.compatible_arguments(spec)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    bindings.check_cli(stage, None, 'run')
    with pytest.raises(ValueError):
        bindings.check_cli(stage, [['packet', str(extracted)]], 'run')


def test_witness_direct_validator_inputs_remain_fleet_verified(tmp_path, monkeypatch):
    spec, _stage, _rows = fixture(tmp_path)
    spec.pop('w8_combined_cpu_proof_profile')
    spec['w8_witness_profile'] = 'w8-context-pilot-v1'
    bindings.configure(spec)
    hashed = []

    def digest(path):
        hashed.append(str(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(inbox, 'file_digest', digest)
    inbox.verify_inputs(spec, tmp_path)
    assert set(hashed) == {row['path'] for row in spec['staged_inputs'] if row['name'] in bindings.WITNESS_DIRECT}
