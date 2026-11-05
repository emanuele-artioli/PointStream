import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.jobs.inbox import validate_spec
from experiments.tennis import w8_folder_real_oracle as job
from experiments.tennis import w8_folder_oracle as synthetic


def test_actual_reference_spec_is_cpu_only_and_binds_public_inputs():
    root = Path(job.__file__).resolve().parents[2]
    ref = json.loads((root / 'docs/experiments/w8-inputs/receiver-20261101/w8-folder-synthetic-66226a15-reference-20261105.json').read_text())
    spec = job.make_spec('2026-11-05T07:37:00Z', ref)
    validate_spec(spec, now=1793862000)
    assert spec['device'] == 'cpu' and spec['hosts'] == ['gpu6'] and not spec['gpu_models']
    assert spec['budget_seconds'] == 180 and spec['smoke']['seconds'] == 120
    assert spec['synthetic_reference_job_id'] == '20261105T064712Z-66226a15'
    assert all('{staged:' + item['name'] + '}' in spec['arguments'] for item in spec['staged_inputs'])
    assert spec['verify_staged_inputs'] is True
    actual = {x['name']: x['sha256'] for x in spec['staged_inputs']}
    assert {k: actual[k] for k in job.INPUT_NAMES} == {k: job.materialize.PINNED[v] for k, v in job.INPUT_NAMES.items()}
    with pytest.raises(ValueError, match='six actual'):
        job.make_spec('2026-11-05T07:37:00Z', {'job_id': ref['job_id'], 'sha256': {}})


def test_no_copy_without_synthetic_gate_and_readiness(monkeypatch):
    now = 100.0
    monkeypatch.setattr(job.time, 'monotonic', lambda: now)
    dispatch = {'campaign_clock': {'started': 90.0, 'phase_started': 95.0, 'budget_end': 270.0}}
    assert job.remaining_gate(dispatch, 110, 140)['phase_seconds_remaining'] == 115
    with pytest.raises(ValueError, match='insufficient'):
        job.remaining_gate(dispatch, 116, 140)
    with pytest.raises(ValueError, match='insufficient'):
        job.remaining_gate(dispatch, 110, 171)


def test_bound_synthetic_gate_rejects_counterfeit_even_with_matching_hashes(tmp_path, monkeypatch):
    root = Path(job.__file__).resolve().parents[2]
    cache = tmp_path / 'cache'
    cache.mkdir()
    stage = tmp_path / 'smoke'
    stage.mkdir()
    job_id = '20261105T064712Z-66226a15'
    names = ['experiments/tennis/w8_folder_oracle.py', 'experiments/tennis/w8_public_folder_materialize.py',
             'experiments/tennis/w8_cuda_system_leaves.py', 'experiments/tennis/w8_receiver_bootstrap.py',
             'experiments/tennis/k6b_generate.py']
    source = cache / 'proof_source'
    with tarfile.open(source, 'w') as archive:
        for name in names:
            archive.add(root / name, arcname=name)
    code = {name: job.sha(root / name) for name in names}
    loader = {'uid': 0, 'mode': 0o100644, 'device': 1, 'inode': 123}
    leaves = [{'name': name, 'resolved': name.replace('/proc/self/', '/proc/101/'), 'uid': 65534,
               'mode': 0o100644, 'device': 1, 'inode': 123} for name in job.policy.APPROVED_LEAVES]
    rows = {}
    for name in job.materialize.SIZES:
        body = synthetic.expected_bytes(name)
        digest = hashlib.sha256(body).hexdigest()
        rows[name] = {'source_sha256': digest, 'copy_sha256': digest, 'bytes': len(body)}
    prior = {'scope': job.SYNTHETIC_SCOPE, 'profile': job.materialize.PROFILE, 'cuda_visible_devices': '',
             'no_torch': True, 'cleanup_complete': True, 'output_write_ok': True, 'symlink_errno': 13,
             'landlock_abi': 3, 'handled_mask': job.policy.handled_rights(3), 'denial_errnos': [13] * 10,
             'mutation_denials': dict.fromkeys(('write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'), 13),
             'namespaces': {'user': 'c', 'net': 'c'}, 'parent_ns': {'user': 'p', 'net': 'p'},
             'approved_leaves': leaves, 'loader': loader,
             'materialization': {'files': rows, 'bytes': sum(x['bytes'] for x in rows.values())}, 'code_sha256': code}
    def save_publication():
        body = json.dumps(prior).encode()
        with tarfile.open(cache / 'proof_published', 'w') as archive:
            member = tarfile.TarInfo('publish/folder-oracle.json')
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    save_publication()
    validation = {'passed': True, 'scope': job.SYNTHETIC_SCOPE, 'checks': dict.fromkeys(
        ('real_folder_helper', 'strict_policy_synthetic_only', 'no_gpu_or_model', 'source_identity'), True)}
    ready = {'spec_sha256': 'a' * 64, 'source_sha256': 'b' * 64, 'snapshot_sha256': job.sha(source)}
    gate = {'passed': True, 'spec_sha256': 'a' * 64, 'source_sha256': 'b' * 64, 'validation': validation}
    for name, value in {'proof_validation': validation, 'proof_gate': gate, 'proof_ready': ready,
                        'proof_state': {'job_id': job_id, 'status': 'complete'}}.items():
        (cache / name).write_text(json.dumps(value))
    identity = cache / 'identity'
    identity.write_text('{}')
    original_sha = job.sha
    monkeypatch.setattr(job, 'sha', lambda path: job.IDENTITY_SHA if Path(path) == identity else original_sha(path))
    for key, name in job.INPUT_NAMES.items():
        (cache / key).write_bytes(b'x')
        monkeypatch.setitem(job.materialize.SIZES, name, 1)
    def freeze_spec():
        ref = {'job_id': job_id, 'sha256': {name: job.sha(cache / name) for name in job.PROOF_NAMES}}
        spec = job.make_spec('2026-11-05T07:37:00Z', ref)
        (tmp_path / 'spec.json').write_text(json.dumps(spec))
        staged = [{'name': x['name'], 'source': x['path'], 'sha256': x['sha256'], 'path': str(cache / x['name']), 'mode': 'local-hit'}
                  for x in spec['staged_inputs']]
        (stage / 'staging.json').write_text(json.dumps({'local_root': str(cache), 'inputs': staged,
                                                        'environment': {'sha256': spec['environment']['sha256']}}))
        (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [1]}))
    freeze_spec()
    job.stage_inputs(stage)
    staging = json.loads((stage / 'staging.json').read_text())
    next(x for x in staging['inputs'] if x['name'] == 'w1')['mode'] = 'local-copy'
    (stage / 'staging.json').write_text(json.dumps(staging))
    with pytest.raises(ValueError, match='public weight identity or size'):
        job.stage_inputs(stage)
    prior['symlink_errno'] = 0
    save_publication()
    freeze_spec()
    with pytest.raises(ValueError, match='prerequisite isolation'):
        job.stage_inputs(stage)


def test_protected_denial_probe_never_reads_even_if_open_succeeds(monkeypatch):
    closed = []
    monkeypatch.setattr(job.os, 'open', lambda *args: 77)
    monkeypatch.setattr(job.os, 'close', lambda fd: closed.append(fd))
    monkeypatch.setattr(job.os, 'read', lambda *args: pytest.fail('protected body read'))
    with pytest.raises(RuntimeError, match='unexpectedly openable'):
        job.check_denied_without_read([Path('/protected')])
    assert closed == [77]
