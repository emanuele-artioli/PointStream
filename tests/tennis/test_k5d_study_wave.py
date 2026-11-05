from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import k5d_study_wave as wave


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    rows = [{'clip': clip, 'frames': data['label_rows'], 'prepare': {'state': 'pending'},
             'score': {'state': 'pending'}} for clip, data in wave.study.selection().items()]
    ledger = {'schema': 1, 'selection_sha256': wave.study.fleet.SOURCE_SHA,
              'execution_identity': identity, 'gpu_model': 'RTX A6000', 'clips': rows}
    now = datetime.now(timezone.utc)
    inspection = {'schema': 1, 'gpu_models': ['RTX A6000'], 'inspected_hosts': [f'gpu{i}' for i in range(1, 7)],
                  'observed_at': (now - timedelta(seconds=1)).isoformat(),
                  'valid_until': (now + timedelta(minutes=2)).isoformat(), 'active_or_uncertain_jobs': []}
    evidence = {'cost': {phase: {'seconds16': 100, 'seconds48': 164, 'margin': 1.5, 'reserve_seconds': 20}
                         for phase in ('prepare', 'score')}}
    monkeypatch.setattr(wave, 'read', lambda path, sha: ledger if path.name == 'ledger' else inspection)
    monkeypatch.setattr(wave.study, 'admission', lambda gate, gs, ep, es, clip, ident:
                        ({}, evidence, wave.study.selection()[clip]['label_rows']))
    monkeypatch.setattr(wave, 'verify_completed', lambda *a: None)
    kwargs: dict[str, Any] = {name + '_path': tmp_path / name for name in ('ledger', 'gate', 'evidence', 'inspection')}
    kwargs.update({name + '_sha': 'b' * 64 for name in ('ledger', 'gate', 'evidence', 'inspection')})
    kwargs.update(identity=identity, gpu_budget_seconds=3000, cpu_budget_seconds=3000, now=now)
    return kwargs, ledger, inspection


def test_wave_requires_process_tree_cpu_proof(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, _, _ = setup
    with pytest.raises(ValueError, match='process-tree CPU'):
        wave.propose(**kwargs)


def test_stop_on_uncertain_job_missing_host_or_budget(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, ledger, inspection = setup
    ledger['clips'][0]['prepare'] = {'state': 'unknown', 'job_id': '20261105T000000Z-12345678'}
    with pytest.raises(ValueError, match='no replay'):
        wave.propose(**kwargs)
    ledger['clips'][0]['prepare'] = {'state': 'pending'}
    inspection['inspected_hosts'].pop()
    with pytest.raises(ValueError, match='all-host'):
        wave.propose(**kwargs)
    inspection['inspected_hosts'].append('gpu6')
    kwargs['cpu_budget_seconds'] = 100
    with pytest.raises(ValueError, match='process-tree CPU'):
        wave.propose(**kwargs)


def test_cannot_omit_historical_failures(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, ledger, _ = setup
    ledger['clips'] = [row for row in ledger['clips'] if row['clip'] != 'tracknet/game10/Clip11']
    with pytest.raises(ValueError, match='30-clip'):
        wave.propose(**kwargs)


def test_completed_receipt_binds_publications_and_fleet_records(tmp_path: Path) -> None:
    import json
    from experiments.jobs import inbox

    job = '20261105T000000Z-12345678'
    root = tmp_path / job
    root.mkdir()
    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    spec = {'gpu_models': ['RTX A6000'], 'entrypoint': ['-m', 'experiments.tennis.k5d_full_study'], 'inputs': []}

    def write(path: Path, value: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return wave.oracle.sha(path)

    records: dict[str, Any] = {'spec_sha256': write(root / 'spec.json', spec)}
    publications = {}
    for stage, frames in (('smoke', 16), ('full', 292)):
        artifact = tmp_path / (stage + '.tar')
        artifact.write_bytes(stage.encode())
        sha = wave.oracle.sha(artifact)
        publications[stage] = {'local_path': str(artifact), 'sha256': sha, 'checkpoint_sha256': 'b' * 64}
        dispatch = {'code': {'spec_sha256': inbox.digest(spec), 'source_sha256': identity['source_sha256'],
                             'snapshot_sha256': identity['snapshot_sha256']},
                    'staged_environment': {'sha256': identity['environment_sha256']},
                    'staged_inputs': [], 'input_identity': [],
                    'environment': {'gpu': {'name': 'NVIDIA RTX A6000'}},
                    'command': ['python', '-m', 'experiments.tennis.k5d_full_study', 'prepare',
                                '--clip', 'tracknet/game9/Clip5', '--frames', str(frames)]}
        staging = {'inputs': [], 'published': {'sha256': sha, 'path': f'/remote/{job}/{stage}/published.tar'}}
        records[stage] = {'dispatch_sha256': write(root / stage / 'dispatch.json', dispatch),
                          'staging_sha256': write(root / stage / 'staging.json', staging),
                          'execution_sha256': write(root / stage / 'execution.json', {'exit_code': 0})}
    receipt = {'job_id': job, 'state': 'succeeded', 'clip': 'tracknet/game9/Clip5', 'phase': 'prepare',
               'frames': 292, 'execution_identity': identity, 'gpu_model': 'RTX A6000',
               'job_directory': str(root), 'records': records, 'publications': publications}
    receipt_path = tmp_path / 'receipt.json'
    entry = {'job_id': job, 'receipt_path': str(receipt_path), 'receipt_sha256': write(receipt_path, receipt),
             'publications': publications}
    import tarfile
    with pytest.raises(tarfile.ReadError):
        wave.verify_completed('tracknet/game9/Clip5', 'prepare', entry, 292, identity)
    Path(publications['full']['local_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='publication SHA'):
        wave.verify_completed('tracknet/game9/Clip5', 'prepare', entry, 292, identity)


def test_wave_rejects_understated_margin(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs, _, _ = setup
    evidence = {'cost': {'prepare': {'seconds16': 100, 'seconds48': 164, 'margin': 1.49, 'reserve_seconds': 20}}}
    monkeypatch.setattr(wave.study, 'admission', lambda gate, gs, ep, es, clip, ident:
                        ({}, evidence, wave.study.selection()[clip]['label_rows']))
    with pytest.raises(ValueError, match='margin'):
        wave.propose(**kwargs)


def test_score_rejects_wrong_producer_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    @contextmanager
    def unpack(*args: Any) -> Any:
        yield tmp_path

    monkeypatch.setattr(wave, 'unpack_publication', unpack)
    producer = {'publications': {'full': {'sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64}}}
    dispatch = {'command': ['experiments.tennis.k5d_control_study', '--checkpoint-sha256', 'c' * 64],
                'staged_inputs': [{'sha256': 'a' * 64}]}
    with pytest.raises(ValueError, match='not bound'):
        wave.validate_full('clip', 'score', 10, {}, {'local_path': 'unused', 'sha256': 'd' * 64}, producer, dispatch)


def test_noncommand_bounds_require_audited_job_records_and_affinity(tmp_path: Path) -> None:
    import json

    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    clips = wave.CONFIRMATIONS
    rows: dict[str, dict[str, Any]] = {}
    jobs = []
    for i, clip in enumerate(clips):
        rows[clip] = {}
        for j, phase in enumerate(('prepare', 'score')):
            job = f'20261105T00000{i * 2 + j}Z-{i * 2 + j:08x}'
            jobs.append(job)
            directory = tmp_path / job
            directory.mkdir()
            receipt = tmp_path / f'{job}.json'
            receipt.write_text(json.dumps({'job_directory': str(directory)}))
            rows[clip][phase] = {'job_id': job, 'receipt_path': str(receipt), 'receipt_sha256': wave.oracle.sha(receipt)}
    for job in jobs:
        directory = tmp_path / job
        (directory / 'run').mkdir()
        (directory / 'smoke').mkdir()
        spec = {'entrypoint': ['-m', 'test'], 'single_cpu_campaign': True, 'cpu_threads': 1}
        (directory / 'spec.json').write_text(json.dumps(spec))
        (directory / 'run/status.json').write_text(json.dumps({'status': 'complete'}))
        dispatch = {'code': {'source_sha256': identity['source_sha256'], 'snapshot_sha256': identity['snapshot_sha256'],
                             'spec_sha256': wave.inbox.digest(spec)},
                    'staged_environment': {'sha256': identity['environment_sha256']}, 'cpu_affinity': [2],
                    'environment': {'gpu': {'uuid': 'GPU-test'}}}
        (directory / 'smoke/dispatch.json').write_text(json.dumps(dispatch))
        (directory / 'before.json').write_text(json.dumps({'children_resource_usage': [0, 0]}))
        (directory / 'after.json').write_text(json.dumps({'children_resource_usage': [1, 0]}))
    groups: dict[str, dict[str, Any]] = {}
    for kind in wave.OVERHEAD_KINDS:
        job = jobs[0]
        directory = tmp_path / job
        raw = {'schema': 1, 'job_id': job, 'kind': kind, 'covers': [kind], 'execution_identity': identity,
               'cpu_method': 'single-cpu-interval-upper-bound-v1', 'complete_descendant_wait': True,
               'observed_at': '2026-01-01T00:00:00Z', 'job_directory': str(directory), 'stage': 'smoke',
               'gpu_uuid': 'GPU-test',
               'status_sha256': wave.oracle.sha(directory / 'run/status.json'),
               'spec_sha256': wave.oracle.sha(directory / 'spec.json'),
               'dispatch_sha256': wave.oracle.sha(directory / 'smoke/dispatch.json'),
               'cpu_before': dict.fromkeys(('self_user', 'self_system', 'children_user', 'children_system'), 0),
               'cpu_after': {'self_user': 0, 'self_system': 0, 'children_user': 1, 'children_system': 0},
               'self_cpu_observed': False,
               'counter_before_path': str(directory / 'before.json'),
               'counter_before_sha256': wave.oracle.sha(directory / 'before.json'),
               'counter_after_path': str(directory / 'after.json'),
               'counter_after_sha256': wave.oracle.sha(directory / 'after.json'),
               'interval_start': 1, 'interval_end': 3, 'gpu_claim_start': 1, 'gpu_claim_end': 2}
        raw_path = tmp_path / f'{kind}.json'
        raw_path.write_text(json.dumps(raw))
        audit = {'schema': 1, 'scope': 'independent noncommand measurement audit', 'audited': True, 'passed': True,
                 'job_id': job, 'kind': kind, 'covers': [kind], 'raw_sha256': wave.oracle.sha(raw_path), 'execution_identity': identity,
                 'checks': dict.fromkeys(('raw_job_and_scope_join', 'complete_process_tree', 'gpu_claim_coverage',
                                          'one_cpu_affinity_and_tree', 'external_costs_covered'), True)}
        audit_path = tmp_path / f'{kind}-audit.json'
        audit_path.write_text(json.dumps(audit))
        groups[kind] = {'raw_path': str(raw_path), 'raw_sha256': wave.oracle.sha(raw_path),
                        'audit_path': str(audit_path), 'audit_sha256': wave.oracle.sha(audit_path),
                        'covers': [kind], 'frequency': 'per_campaign'}
    bundle = {'schema': 1, 'groups': groups, 'command_covered': [],
              'failure_reserve': 'one-bounded-campaign'}
    values = wave.verified_noncommand(bundle, identity, rows)
    assert values['groups']['staging']['cpu'] == 3 and values['groups']['staging']['gpu'] == 1.5
    # One genuinely shared, independently audited interval can cover both
    # staging and publication exactly once.
    raw_path, audit_path = tmp_path / 'staging.json', tmp_path / 'staging-audit.json'
    raw = json.loads(raw_path.read_text())
    raw['covers'] = ['staging', 'publication']
    raw_path.write_text(json.dumps(raw))
    audit = json.loads(audit_path.read_text())
    audit.update(covers=raw['covers'], raw_sha256=wave.oracle.sha(raw_path))
    audit_path.write_text(json.dumps(audit))
    groups['staging'].update(covers=raw['covers'], raw_sha256=wave.oracle.sha(raw_path),
                             audit_sha256=wave.oracle.sha(audit_path))
    del groups['publication']
    values = wave.verified_noncommand(bundle, identity, rows)
    assert len(values['groups']) == 5 and values['groups']['staging']['cpu'] == 3
    (tmp_path / 'staging.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA|sha|identity'):
        wave.verified_noncommand(bundle, identity, rows)


def test_qualified_wave_proposes_only_pending_stage_under_both_budgets(setup, monkeypatch):
    from experiments.tennis import k5d_control_study as control

    kwargs, ledger, _ = setup
    for row in ledger['clips']:
        if row['clip'] in wave.CONFIRMATIONS:
            row['prepare'] = {'state': 'complete', 'job_id': '20261105T000001Z-11111111'}
            row['score'] = {'state': 'complete', 'job_id': '20261105T000002Z-22222222'}
    programme = {resource: {'readiness_seconds': 20, 'stage_staging_seconds': 1,
                            'validator_seconds': 1, 'audit_seconds': 1, 'publication_seconds': 1,
                            'isolation_seconds': 1, 'failed_start_reserve_seconds': 1, 'noncommand_groups': {'whole': 5}}
                 for resource in ('gpu', 'cpu')}
    evidence = {'cost': {phase: {'seconds16': 100} for phase in ('prepare', 'score')},
                'cpu_calibration': {}, 'programme': programme}
    monkeypatch.setattr(control, 'approved', lambda *_args: (evidence, 100))
    monkeypatch.setattr(wave.study, 'forecast', lambda *_args: 100)
    monkeypatch.setattr(wave, 'verified_cpu_calibration', lambda *_args: {
        phase: {'margin': 1.5, 'fixed_seconds': 1, 'marginal_seconds_per_frame': 0.1}
        for phase in ('prepare', 'score')})
    fields = ('bridge_path', 'bridge_sha', 'calibration_source', 'consumer_source', 'budget_ledger',
              'budget_ledger_sha', 'budget_attestation', 'budget_attestation_sha', 'native_cpu_receipt',
              'gpu_parity_receipt', 'noncommand_evidence')
    kwargs['control_assets'] = dict.fromkeys(fields, 'x')
    proposal = wave.propose(**kwargs)
    assert proposal['action'] == 'prepare' and proposal['clip'] not in wave.CONFIRMATIONS
    assert proposal['gpu_seconds_upper'] > 100 and proposal['cpu_seconds_upper'] > 0
    kwargs['cpu_budget_seconds'] = 1
    with pytest.raises(ValueError, match='remaining budgets'):
        wave.propose(**kwargs)


def test_cpu_calibration_requires_four_namespace_logs_and_independent_audits(tmp_path, monkeypatch):
    import json

    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    rows = wave.study.selection()
    samples = {'prepare': [], 'score': []}
    for i, clip in enumerate(wave.CONFIRMATIONS):
        prep = {'job_id': f'20261105T00000{i}Z-11111111', 'receipt_path': str(tmp_path / f'prep{i}'),
                'receipt_sha256': 'a' * 64}
        for phase in ('prepare', 'score'):
            entry = prep if phase == 'prepare' else {'job_id': f'20261105T00000{i}Z-22222222',
                                                     'receipt_path': str(tmp_path / f'score{i}'),
                                                     'receipt_sha256': 'a' * 64}
            samples[phase].append({'clip': clip, 'entry': entry, **({'preparation': prep} if phase == 'score' else {})})
    log = tmp_path / 'full/command.log'
    log.parent.mkdir()
    log.write_text('K5D_CPU_ACCOUNTING ' + json.dumps({'method': 'linux-pidns-init-waited-rusage-v1'}) + '\n')
    spec = {'entrypoint': ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']}
    receipt = {'job_directory': str(tmp_path), 'records': {'spec_sha256': 's',
               'full': {'command_log_sha256': wave.oracle.sha(log)}}}
    monkeypatch.setattr(wave, 'read', lambda path, _sha: spec if path.name == 'spec.json' else receipt)
    monkeypatch.setattr(wave, 'verify_completed', lambda *_args: None)
    audits = []
    monkeypatch.setattr(wave, 'verify_independent_confirmation', lambda *args: audits.append((args[0], args[1])))
    monkeypatch.setattr(wave.accounting, 'read_receipt', lambda _log, *, frames, **_kwargs: float(frames))
    result = wave.verified_cpu_calibration(samples, identity)
    assert len(audits) == 4 and all(x['margin'] == 1.5 for x in result.values())
    assert result['prepare']['marginal_seconds_per_frame'] == 1
    assert rows[wave.CONFIRMATIONS[0]]['label_rows'] != rows[wave.CONFIRMATIONS[1]]['label_rows']
    spec['entrypoint'] = ['-m', 'experiments.tennis.k5d_control_study']
    with pytest.raises(ValueError, match='namespace execution'):
        wave.verified_cpu_calibration(samples, identity)
