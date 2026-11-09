import hashlib
import json

import pytest

from experiments.tennis import w8_witness_assessment as assessment


def manifest(path, data):
    path.write_text(json.dumps(data, sort_keys=True))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def producer(tmp_path):
    digest = 'a' * 64
    return manifest(tmp_path / 'producer.json', {'schema': 1,
        'scope': 'independently reviewed W8 full candidate and finish', 'passed': True,
        'quality_claim': False, 'source_sha256': digest, 'snapshot_sha256': digest,
        'independent_audit_sha256': digest, 'revision': 'b' * 40,
        'jobs': {'unit0': '20261109T000000Z-12345678', 'unit1': '20261109T000001Z-12345678',
                 'finish': '20261109T000002Z-12345678'},
        'candidate_sha256': {k: digest for k in assessment.finish.CANDIDATE_KEYS}})


def test_candidate_authentication_precedes_any_postseal_input(tmp_path, monkeypatch):
    cfg = {'producer_manifest': producer(tmp_path), 'candidate_files': {},
           'asset_manifest': {'path': '/nonexistent', 'sha256': 'x'}}
    called = []
    monkeypatch.setattr(assessment.finish, '_inputs', lambda *args: called.append(True))
    with pytest.raises(ValueError, match='candidate artifacts'):
        assessment.authenticate(cfg)
    assert not called


def test_role_manifest_rejects_unreviewed_or_extra_asset(tmp_path):
    digest = 'a' * 64
    item = manifest(tmp_path / 'assets.json', {'schema': 1,
        'scope': 'independently reviewed W8 post-seal assets', 'historical_source_sha256': digest,
        'role_sha256': {'source': {k: digest for k in assessment.SOURCE_KEYS}}})
    cfg = {'asset_manifest': item, 'role_files': {k: {'path': '/missing', 'sha256': digest}
                                                   for k in assessment.SOURCE_KEYS | {'extra'}}}
    with pytest.raises(ValueError, match='asset manifest'):
        assessment.role_files(cfg, 'source')


def test_producer_manifest_hash_and_scope_are_required(tmp_path):
    item = producer(tmp_path)
    item['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='manifest identity'):
        assessment.authenticate({'producer_manifest': item})


def test_context_qualification_seal_cannot_pass_candidate(tmp_path):
    paths = {}
    for unit in assessment.fixed.UNITS:
        seal = tmp_path / (unit + '.json')
        seal.write_text(json.dumps({'version': 1, 'unit': unit,
            'packet_sha256': assessment.fixed.WINDOW_SHA, 'profile_sha256': assessment.fixed.PROFILE_SHA,
            'scope': 'context_qualification_only'}))
        paths[f'generation_seal:{unit}'] = seal
        paths[f'crop:{unit}'] = tmp_path / (unit + '.npz')
    with pytest.raises(ValueError, match='generation seal mismatch'):
        assessment.finish._generation_seals(paths)


def test_assessment_target_has_fixed_config_cli(tmp_path, monkeypatch):
    import os
    import sys
    from experiments.tennis import w8_witness_namespace as namespace

    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0}, raising=False)
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(namespace.Path(namespace.__file__).resolve().parents[2]))
    config = tmp_path / 'config.json'
    config.write_text('{"phase": "score"}')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    args = ['--phase', 'score', '--config', str(config), '--config-sha256', digest]
    assert namespace.command(namespace.ASSESSMENT_TARGET, args)[3:] == args
    with pytest.raises(ValueError):
        namespace.command(namespace.ASSESSMENT_TARGET, ['--phase', 'source', *args[2:]])
    with pytest.raises(ValueError):
        namespace.command(namespace.ASSESSMENT_TARGET, [*args[:-1], '0' * 64])


def test_prospective_cost_requires_whole_stage_fit():
    plan = assessment.cost_plan(phase='score', observed_wall=10, observed_cpu=4,
        staging=2, validation=3, cleanup=4, campaign_wall_cap=24, cpu_cap=24, gpu_cap=24)
    assert plan['wall_upper_seconds'] == 24 and plan['cpu_upper_seconds'] == 24
    assert plan['gpu_upper_seconds'] == 24 and plan['automatic_release'] is False
    with pytest.raises(ValueError):
        assessment.cost_plan(phase='score', observed_wall=10, observed_cpu=4,
            staging=2, validation=3, cleanup=4, campaign_wall_cap=23, cpu_cap=24, gpu_cap=24)


def test_prior_assessment_requires_real_cuda_and_landlock_receipt(tmp_path):
    item = tmp_path / 'score.json'
    record = {'scope': 'post-seal W8 score', 'producer_manifest_sha256': 'a' * 64,
              'candidate_sha256': {}, 'outputs': {'scores.npz': 'b' * 64},
              'device': {'provider': 'torch CUDA', 'uuid': 'u', 'visible_uuid': 'u', 'major': 8, 'minor': 9},
              'role_isolation': {'landlock_abi': 3, 'read_only_inputs': {'/fixed': 13}}}
    item.write_text(json.dumps(record))
    assert assessment.validate_prior_assessment(item, phase='score', producer_sha='a' * 64,
        candidate={}, output_name='scores.npz', output_sha='b' * 64) == record
    record['role_isolation']['read_only_inputs'] = {}
    item.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        assessment.validate_prior_assessment(item, phase='score', producer_sha='a' * 64,
            candidate={}, output_name='scores.npz', output_sha='b' * 64)


def test_spec_builder_rejects_before_privileged_input_hash(tmp_path, monkeypatch):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'phase': 'score', 'candidate_files': {},
        'producer_manifest': {'sha256': 'b' * 64},
        'role_files': {'source_rgb': {'path': '/forbidden', 'sha256': 'a' * 64}}}))
    events = []
    def reject(_config):
        events.append('producer-check')
        raise ValueError('independent producer not accepted')
    monkeypatch.setattr(assessment, 'authenticate', reject)
    monkeypatch.setattr(assessment.fixed, 'file_sha', lambda path: events.append(str(path)) or 'a' * 64)
    with pytest.raises(ValueError, match='producer not accepted'):
        assessment.make_fleet_spec(template_local=template,
            template_remote='/home/itec/emanuele/Datasets/pointstream-data/template.json',
            deadline='2026-11-10T00:00:00Z', host_name='gpu5', approved_producer_sha256='b' * 64,
            approved_asset_manifest_sha256='a' * 64, approved_promotion_sha256='c' * 64,
            approved_ledger_sha256='d' * 64, plan={}, command_wall=30,
            validator_wall=20, campaign_wall=100)
    assert events == ['producer-check']


def test_real_fleet_spec_dispatches_only_one_postseal_phase(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from experiments.jobs.inbox import phase_command, validate_spec

    root = '/home/itec/emanuele/Datasets/pointstream-data/'
    row = {'path': root + 'registered', 'sha256': 'a' * 64}
    config = {'phase': 'source', 'output': '{assessment_output}', 'cpu_seconds': 10,
              'child_cpu_seconds': 10, 'wall_seconds': 10,
              'producer_manifest': {'path': root + 'producer', 'sha256': 'b' * 64},
              'asset_manifest': row, 'promotion_manifest': row, 'ledger': row,
              'candidate_files': {key: row for key in assessment.finish.CANDIDATE_KEYS},
              'role_files': {key: row for key in assessment.SOURCE_KEYS}}
    template = tmp_path / 'template.json'
    template.write_text(json.dumps(config))
    monkeypatch.setattr(assessment, 'authenticate', lambda _config: ({}, {}))
    monkeypatch.setattr(assessment, 'check_asset_manifest', lambda _config, _phase: None)
    monkeypatch.setattr(assessment, 'check_assessment_promotion', lambda *_args, **_kwargs: None)
    plan = assessment.cost_plan(phase='source', observed_wall=10, observed_cpu=4,
        staging=2, validation=3, cleanup=4, campaign_wall_cap=100, cpu_cap=100, gpu_cap=100)
    spec = assessment.make_fleet_spec(template_local=template, template_remote=root + 'template.json',
        deadline=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(), host_name='gpu5',
        approved_producer_sha256='b' * 64, approved_asset_manifest_sha256='a' * 64,
        approved_promotion_sha256='a' * 64, approved_ledger_sha256='a' * 64,
        plan=plan, command_wall=30, validator_wall=20, campaign_wall=100)
    validate_spec(spec)
    assert phase_command(spec, 'smoke', python='/python')[3] == 'fleet-run'
    assert phase_command(spec, 'full', python='/python')[3] == 'fleet-noop'
    assert spec['gpu_sharing'] == {'mode': 'exclusive'}


def test_policy_probe_never_accepts_visible_gpu(tmp_path, monkeypatch):
    import os
    import sys
    from experiments.tennis import w8_witness_namespace as namespace

    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0}, raising=False)
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(namespace.Path(namespace.__file__).resolve().parents[2]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-visible')
    config = tmp_path / 'config.json'
    config.write_text('{"phase": "policy-probe"}')
    args = ['--phase', 'policy-probe', '--config', str(config), '--config-sha256',
            hashlib.sha256(config.read_bytes()).hexdigest()]
    with pytest.raises(ValueError, match='hide CUDA'):
        namespace.command(namespace.ASSESSMENT_TARGET, args)


def test_same_header_wrong_golden_fails_member_provenance(tmp_path):
    from experiments.tennis import w8_receiver_score as scorer

    digest = 'a' * 64
    role = {key: digest for key in assessment.OFFLINE_KEYS}
    origins = {key: {'artifact_sha256': digest, 'archive_sha256': 'b' * 64, 'member_sha256': 'c' * 64}
               for key in assessment.GOLDEN_KEYS}
    origins['golden_base']['artifact_sha256'] = 'd' * 64
    item = manifest(tmp_path / 'assets.json', {'schema': 1,
        'scope': 'independently reviewed W8 post-seal assets',
        'historical_source_sha256': scorer.RECEIVER_SHA, 'judge_archive_sha256': scorer.JUDGE_ARCHIVE,
        'region_archive_sha256': scorer.REGION_ARCHIVE, 'independent_audit_sha256': 'e' * 64,
        'role_sha256': {'offline': role}, 'golden_origins': origins})
    cfg = {'asset_manifest': item, 'role_files': {key: {'path': '/unread', 'sha256': digest} for key in role}}
    with pytest.raises(ValueError, match='golden archive/member'):
        assessment.check_asset_manifest(cfg, 'offline')
    item['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='manifest identity'):
        assessment.check_asset_manifest(cfg, 'offline')


def test_forecast_rejects_different_host_before_any_release(monkeypatch):
    from experiments.tennis import w8_witness_job as job

    sha = 'a' * 64
    config = {'phase': 'source', 'promotion_manifest': {'path': '/gate', 'sha256': sha},
              'ledger': {'path': '/ledger', 'sha256': sha}, 'producer_manifest': {'sha256': sha}}
    producer = {'source_sha256': 'b' * 64, 'snapshot_sha256': 'c' * 64}
    monkeypatch.setattr(assessment, '_manifest', lambda path, _sha: {'host': 'gpu6'} if str(path) == '/gate' else {})
    monkeypatch.setattr(job, 'promotion_ledger_remaining', lambda _ledger: (10.0, 10.0))
    with pytest.raises(ValueError, match='source or host'):
        assessment.check_assessment_promotion(config, producer=producer, host_name='gpu5', plan={},
            promotion_sha=sha, ledger_sha=sha, previous_sha=None)


def test_spec_rejects_unpinned_asset_before_manifest_read(tmp_path, monkeypatch):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'phase': 'source', 'producer_manifest': {'sha256': 'a' * 64},
                                    'asset_manifest': {'sha256': 'b' * 64}}))
    monkeypatch.setattr(assessment, 'authenticate', lambda _config: ({}, {}))
    monkeypatch.setattr(assessment, 'check_assessment_promotion', lambda *_args, **_kwargs: None)
    touched = []
    monkeypatch.setattr(assessment, 'check_asset_manifest', lambda *_args: touched.append('read'))
    with pytest.raises(ValueError, match='root-pinned asset'):
        assessment.make_fleet_spec(template_local=template,
            template_remote='/home/itec/emanuele/Datasets/pointstream-data/template.json',
            deadline='2026-11-10T00:00:00Z', host_name='gpu5', approved_producer_sha256='a' * 64,
            approved_asset_manifest_sha256='c' * 64, approved_promotion_sha256='d' * 64,
            approved_ledger_sha256='e' * 64, plan={}, command_wall=30, validator_wall=20, campaign_wall=100)
    assert touched == []
