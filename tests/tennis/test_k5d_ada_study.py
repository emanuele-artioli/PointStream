"""Admission regressions, without scientific jobs, GPUs, models or native codecs."""
import argparse
import copy
import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from experiments.jobs import inbox
from experiments.tennis import k5d_ada_study as ada
from experiments.tennis import k5d_full_cost_builder as builder

IDENTITY = {'source_sha256': '1' * 64, 'snapshot_sha256': '2' * 64, 'environment_sha256': '3' * 64}
JOB = '20261113T000000Z-12345678'
SCORER = '20261113T000001Z-12345678'
UUID = 'GPU-dbbb295f-6156-1133-8658-841f3f5b17d0'
GPU = {'name': ada.adapter.GPU_NAME, 'uuid': UUID, 'capability': [8, 9]}
CLIP = 'tracknet/game9/Clip5'


def write(path, value):
    path.write_text(json.dumps(value))
    return ada.oracle.sha(path)


def test_profiles_preserve_hardcoded_a6000_and_reject_unknown():
    assert ada.profile({}) == ada.A6000
    assert ada.hardware({}) == ('RTX A6000', [8, 6])
    assert ada.hardware({'registered_gpu_profile': ada.ADA}) == ('RTX 6000 Ada', [8, 9])
    with pytest.raises(ValueError, match='unknown'):
        ada.hardware({'registered_gpu_profile': 'auto'})


@pytest.mark.parametrize('key', IDENTITY)
def test_ada_calibration_cannot_bridge_source_archive_or_environment(key):
    changed = {**IDENTITY, key: '4' * 64}
    with pytest.raises(ValueError, match='identical frozen'):
        ada.same_source(changed, IDENTITY)
    ada.same_source(IDENTITY, IDENTITY)


def test_identity_bridge_rehashes_both_actual_archives(tmp_path):
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as bundle:
        item = tarfile.TarInfo('a.py')
        item.size = 3
        bundle.addfile(item, io.BytesIO(b'x=1'))
    digest = ada.oracle.sha(archive)
    source, _ = ada.bridge.manifest(archive, digest)
    identity = {**IDENTITY, 'source_sha256': source, 'snapshot_sha256': digest}
    copied = tmp_path / 'copy.tar'
    copied.write_bytes(archive.read_bytes())
    result = ada.source_bridge(archive, identity, copied, identity)
    assert result['changed_paths'] == [] and result['unchanged_files'] == 1
    copied.write_bytes(copied.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='archive SHA'):
        ada.source_bridge(archive, identity, copied, identity)


def device_records():
    return ({'hosts': ['gpu6'], 'cpu_threads': 1, 'single_cpu_campaign': True,
             'gpu_models': ['RTX 6000 Ada'], 'gpu_sharing': {'mode': 'exclusive'}},
            {'environment': {'host': 'gpu6.itec.aau.at', 'alias': 'gpu6', 'gpu': GPU.copy()},
             'gpu_sharing': {'mode': 'exclusive'}})


@pytest.mark.parametrize('field,value', [('hosts', ['gpu5']), ('gpu_models', ['RTX A6000']),
                                         ('cpu_threads', 2), ('gpu_sharing', {'mode': 'shared'})])
def test_ada_hardware_cannot_be_inferred_from_model_filter(field, value):
    spec, dispatch = device_records()
    assert ada.device(spec, dispatch) == GPU
    spec[field] = value
    with pytest.raises(ValueError):
        ada.device(spec, dispatch)


@pytest.mark.parametrize('field,value', [('host', 'gpu5.itec.aau.at'), ('alias', 'gpu5'),
                                         ('gpu', {**GPU, 'name': 'NVIDIA RTX A6000'}),
                                         ('gpu', {**GPU, 'uuid': 'GPU-not-a-uuid'})])
def test_actual_dispatch_hardware_is_required(field, value):
    spec, dispatch = device_records()
    dispatch['environment'][field] = value
    with pytest.raises(ValueError):
        ada.device(spec, dispatch)


def measured_cost():
    audit = {'schema': 1, 'registered_gpu_profile': ada.ADA, 'execution_identity': IDENTITY,
             'calibration_jobs': {'prepare': JOB, 'score': SCORER}, 'config_sha256': 'c' * 64, 'stages': {}}
    for action in ('prepare', 'score'):
        for n in (16, 48):
            audit['stages'][action + str(n)] = {'job': JOB if action == 'prepare' else SCORER,
                'stage': 'smoke' if n == 16 else 'full', 'seconds': n + 10.0, 'gpu': GPU.copy(),
                'published_sha256': 'b' * 64, 'cpu_seconds': 1.0, 'command_log_sha256': 'd' * 64,
                'sha256': {name: 'a' * 64 for name in ('spec', 'dispatch', 'staging', 'execution')}}
    return {'registered_gpu_profile': ada.ADA, 'gpu_model': 'RTX 6000 Ada', 'gpu_capability': [8, 9],
            'calibration_archives': {k: v['published_sha256'] for k, v in audit['stages'].items()},
            'calibration_execution_sha256': {k: v['sha256']['execution'] for k, v in audit['stages'].items()},
            'score_timing_sha256': {'16': 'a' * 64, '48': 'b' * 64},
            'calibration_execution_identity': IDENTITY,
            'consumer_execution_identity': IDENTITY, 'calibration_jobs': audit['calibration_jobs'],
            'ada_calibration': audit, 'calibration_audit_sha256': inbox.digest(audit),
            'budget_reconciled': True, 'cost': {p: builder.model(26, 58, 1.5, 20) for p in ('prepare', 'score')}}


def test_costs_require_actual_measured_jobs_and_identical_affine_model():
    receipt = measured_cost()
    ada.validate_calibration(receipt)
    for mutate in (lambda r: r['cost']['prepare'].update(seconds48=57),
                   lambda r: r['cost']['score'].update(margin=1),
                   lambda r: r['ada_calibration']['stages']['prepare16'].update(job=SCORER),
                   lambda r: r['ada_calibration']['stages']['prepare48']['gpu'].update(uuid='GPU-00000000-0000-0000-0000-000000000000'),
                   lambda r: r['ada_calibration']['stages']['score48'].update(seconds=float('inf'))):
        changed = copy.deepcopy(receipt)
        mutate(changed)
        if not any(v['seconds'] == float('inf') for v in changed['ada_calibration']['stages'].values()):
            changed['calibration_audit_sha256'] = inbox.digest(changed['ada_calibration'])
        with pytest.raises(ValueError):
            ada.validate_calibration(changed)


@pytest.fixture
def producer(tmp_path, monkeypatch):
    from experiments.tennis import k5d_full_study as study
    monkeypatch.setattr(study, 'selection', lambda: {CLIP: {'label_rows': 292}})
    write(tmp_path / 'checkpoint.json', {})
    points = [{'codec': c, 'point': p} for c, p in ada.enc.POINTS]
    write(tmp_path / 'pilot.json', {'points': points})
    write(tmp_path / 'source-qualification.json', {'device': {**GPU, 'uuid': UUID[4:]}})
    config = tmp_path / 'config.json'
    runtime = {'runtime_files': {}}
    digest = write(config, {'execution_identity': IDENTITY, 'profiles': {'gpu6': runtime}})
    for point in points:
        folder = tmp_path / f"{point['codec']}-{point['point']}" / 'receiver-evidence'
        folder.mkdir(parents=True)
        write(folder / 'receiver.json', {'point': point})
    marker = {'producer_job': JOB, 'producer_stage': 'full', 'frames': 292, 'gpu': GPU.copy()}
    def sealed(root, sha, identity, *, require_same_job_receiver):
        assert require_same_job_receiver and identity == IDENTITY
        return {'clip': CLIP, 'frames': 292}
    monkeypatch.setattr(ada.cp, 'verify', sealed)
    monkeypatch.setattr(ada.receiver, 'verify', lambda *a, **kw: marker)
    calls = []
    def verify(receipt, *, expected_context, expected_config_sha256):
        assert expected_context == {'job_id': JOB, 'stage': 'full', 'execution_identity': IDENTITY, 'gpu': GPU}
        assert expected_config_sha256 == digest
        calls.append(receipt['point'])
        return {'profile_sha256': ada.worker.canonical_sha(runtime),
                'runtime_inventory_sha256': ada.worker.canonical_sha({}), 'actual_policy_sha256': 'a' * 64}
    return tmp_path, config, digest, marker, calls, verify


def test_registered292_requires_all_six_actual_denial_verifiers(producer, monkeypatch):
    root, config, digest, marker, calls, verify = producer
    monkeypatch.setattr(ada, 'importlib', SimpleNamespace(import_module=lambda _: SimpleNamespace(verify_receipt=verify)))
    assert ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=292,
                                   config=config, config_sha=digest, dispatch_gpu=GPU) == marker
    assert calls == [{'codec': c, 'point': p} for c, p in ada.enc.POINTS]
    marker['producer_job'] = SCORER
    with pytest.raises(ValueError, match='producer job'):
        ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=292, config=config, config_sha=digest)


def test_registered_producer_rejects_api_only_receipt_and_partial_timeline(producer):
    root, config, digest, _, _, _ = producer
    with pytest.raises(ValueError, match='isolated receiver evidence missing'):
        ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=292, config=config, config_sha=digest)
    with pytest.raises(ValueError, match='complete registered timeline'):
        ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=16, config=config, config_sha=digest)


def test_wrong_config_and_dispatch_uuid_fail_closed(producer):
    root, config, digest, _, _, _ = producer
    with pytest.raises(ValueError, match='config.*checksum|asset checksum'):
        ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=292, config=config, config_sha='f' * 64)
    with pytest.raises(ValueError, match='dispatch'):
        ada.registered_producer(root, IDENTITY, job=JOB, clip=CLIP, frames=292, config=config, config_sha=digest,
                                 dispatch_gpu={**GPU, 'uuid': 'GPU-00000000-0000-0000-0000-000000000000'})


def test_score_binds_root_approved_actual_producer_and_config(tmp_path, monkeypatch):
    stage = tmp_path / 'full'
    stage.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    config = 'c' * 64
    producer = {'job_id': JOB, 'remote': str(ada.cf.DATA_ROOT / 'jobs/fleet/inbox' / JOB / 'full/published.tar'),
                'archive_sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64, 'config_sha256': config}
    gate = {'registered_gpu_profile': ada.ADA, 'release': 'cost-confirmation',
            'receiver_isolation_config_sha256': config, 'ada_producers': {CLIP: producer}}
    write(stage / 'staging.json', {'inputs': [{'name': 'checkpoint', 'source': producer['remote'], 'sha256': 'a' * 64}]})
    args = argparse.Namespace(certificate=None, clip=CLIP, checkpoint_sha256='b' * 64)
    ada.score_binding(gate, args, IDENTITY)
    assert args.receiver_producer_job == JOB
    with pytest.raises(ValueError, match='staged Ada score'):
        ada.score_binding(gate, argparse.Namespace(certificate=None, clip=CLIP, checkpoint_sha256='e' * 64), IDENTITY)
    gate['ada_producers'][CLIP]['config_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='config differs'):
        ada.score_binding(gate, args, IDENTITY)


@pytest.mark.parametrize('frames', [292, 586, 160])
@pytest.mark.parametrize('action', ['prepare', 'score'])
def test_control_run_requires_fresh_producer_beyond_first292(tmp_path, monkeypatch, frames, action):
    from experiments.tennis import k5d_control_study as control
    stage = tmp_path / 'job/full'
    stage.mkdir(parents=True)
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path / 'scratch'))
    monkeypatch.setenv('PS_STAGE', 'full')
    monkeypatch.setattr(control, 'require_namespace_worker', lambda: None)
    monkeypatch.setattr(control.cp, 'execution_identity', lambda: IDENTITY)
    monkeypatch.setattr(control, 'verify_gate', lambda *a: frames)
    monkeypatch.setattr(control.study, 'checked_json', lambda *a: {'registered_gpu_profile': ada.ADA, 'release': 'full-study'})
    args = argparse.Namespace(action=action, mode='full', frames=frames, clip=CLIP,
            full_gate='gate', full_gate_sha256='a' * 64, output=tmp_path / 'output', checkpoint=str(tmp_path / 'checkpoint'),
            checkpoint_sha256='b' * 64, receiver_producer_job=JOB, receiver_isolation_config='config',
            receiver_isolation_config_sha256='c' * 64)
    events = []
    def configure(data, *rest):
        data.same_job_receiver = data.isolated_receiver = True
        data.receiver_producer_context = {'gpu': GPU}
        events.append('configure')
    def prepared(data, identity):
        assert data.same_job_receiver and data.isolated_receiver
        events.append('prepare')
    def actual(root, identity, **kwargs):
        assert kwargs['frames'] == frames and kwargs['config_sha'] == 'c' * 64
        if action == 'score':
            assert kwargs['job'] == JOB
        events.append('verify-fresh')
    monkeypatch.setattr(control.ada, 'configure', configure)
    monkeypatch.setattr(control.benchmark, 'build', lambda _: args)
    monkeypatch.setattr(control.cp, 'prepare', prepared)
    monkeypatch.setattr(control.ada, 'registered_producer', actual)
    monkeypatch.setattr(control.co, 'score_all', lambda *a: events.append('score'))
    monkeypatch.setattr(control.ada, 'full_metrics', lambda *a: events.append('full-metrics'))
    control.run(args)
    assert events == (['configure', 'prepare', 'verify-fresh'] if action == 'prepare'
                      else ['configure', 'verify-fresh', 'score', 'full-metrics'])


@pytest.mark.parametrize('missing', ['full_metrics', 'points', 'frames'])
def test_full_metric_gate_rejects_partial_scores(monkeypatch, tmp_path, missing):
    result = {'full_metrics': True, 'points': 6, 'frames': 292}
    monkeypatch.setattr(ada.enc, 'validate', lambda _: result)
    ada.full_metrics(tmp_path, 292)
    del result[missing]
    with pytest.raises(ValueError, match='full metrics and judges'):
        ada.full_metrics(tmp_path, 292)


def test_independent_hardware_certificate_cannot_be_relabelled():
    from experiments.tennis import k5d_full_admission as admission
    gate = {'registered_gpu_profile': ada.ADA}
    with pytest.raises(ValueError, match='profiles differ'):
        admission.check_certificate({}, gate, IDENTITY, CLIP, 'prepare', '', '', '', None, {})


def test_reference_bridge_and_witness_remain_independently_pinned(monkeypatch):
    gate = {'registered_gpu_profile': ada.ADA, 'confirmation_qualification': {
        'native_cpu_receipt_sha256': 'n' * 64, 'unshare_sha256': 'u' * 64}}
    args = argparse.Namespace(registered_gpu_profile=ada.ADA, native_cpu_receipt='native')
    for index, name in enumerate(ada.ASSETS):
        sha = str(index + 1) * 64
        gate[name + '_sha256'] = sha
        setattr(args, name, name)
        setattr(args, name + '_sha256', sha)
    called = []
    def admitted(actual, identity):
        assert actual.bridge == 'ada_reference_bridge' and actual.bridge_sha256 == gate['ada_reference_bridge_sha256']
        assert identity == IDENTITY
        called.append(True)
    monkeypatch.setattr(ada.worker, 'admission', admitted)
    ada.bindings(gate, args, IDENTITY)
    assert called == [True]
    args.ada_reference_bridge_sha256 = 'f' * 64
    with pytest.raises(ValueError, match='asset binding'):
        ada.bindings(gate, args, IDENTITY)
    args.registered_gpu_profile = ada.A6000
    with pytest.raises(ValueError, match='explicit Ada'):
        ada.bindings(gate, args, IDENTITY)


def test_admission_recognizes_only_exact_complete_calibration_jobs(tmp_path):
    receipt = measured_cost()
    for stage in ('smoke', 'full'):
        folder = tmp_path / stage
        folder.mkdir()
        write(tmp_path / 'spec.json', {'fixed': 'spec'})
        write(folder / 'dispatch.json', {'fixed': 'dispatch'})
        write(folder / 'staging.json', {'fixed': 'staging'})
        write(folder / 'execution.json', {'fixed': 'execution'})
        (folder / 'command.log').write_text('pinned log')
        frame = '16' if stage == 'smoke' else '48'
        record = receipt['ada_calibration']['stages']['prepare' + frame]
        record['command_log_sha256'] = ada.oracle.sha(folder / 'command.log')
        for key in record['sha256']:
            path = tmp_path / 'spec.json' if key == 'spec' else folder / (key + '.json')
            record['sha256'][key] = ada.oracle.sha(path)
    receipt['calibration_execution_sha256'] = {k: v['sha256']['execution'] for k, v in receipt['ada_calibration']['stages'].items()}
    receipt['calibration_audit_sha256'] = inbox.digest(receipt['ada_calibration'])
    assert not ada.known_calibration_job('20261113T000002Z-87654321', 'complete', tmp_path, receipt)
    assert ada.known_calibration_job(JOB, 'complete', tmp_path, receipt)
    with pytest.raises(ValueError, match='not canonically complete'):
        ada.known_calibration_job(JOB, 'failed', tmp_path, receipt)
    write(tmp_path / 'full/execution.json', {'tampered': True})
    with pytest.raises(ValueError, match='physical records changed'):
        ada.known_calibration_job(JOB, 'complete', tmp_path, receipt)


@pytest.mark.parametrize('failure', ['missing', 'failed', 'wait-false', 'paused', 'contended'])
def test_complete_stage_cost_rejects_false_clean_exit(tmp_path, failure):
    execution = {'exit_code': 0, 'timing_contaminated': False, 'contention': [], 'paused_seconds': 0, 'seconds': 10}
    namespace = {'reason': 'complete', 'status': 0, 'complete_descendant_wait': True,
                 'namespace_init': {'namespace_pid': 1}, 'children_cpu_seconds': 2, 'init_cpu_seconds': 0.1}
    write(tmp_path / 'execution.json', execution)
    log = tmp_path / 'command.log'
    log.write_text('K5D_NAMESPACE_CANDIDATE ' + json.dumps(namespace) + '\n')
    assert ada.stage_cost(tmp_path) == (10.0, 2.1, ada.oracle.sha(log))
    if failure == 'missing':
        log.write_text('no namespace receipt\n')
    elif failure == 'failed':
        namespace['status'] = 1
    elif failure == 'wait-false':
        namespace['complete_descendant_wait'] = False
    elif failure == 'paused':
        execution['paused_seconds'] = 1
    elif failure == 'contended':
        execution['contention'] = [{'foreign': True}]
    if failure != 'missing':
        log.write_text('K5D_NAMESPACE_CANDIDATE ' + json.dumps(namespace) + '\n')
    write(tmp_path / 'execution.json', execution)
    with pytest.raises(ValueError, match='complete native namespace'):
        ada.stage_cost(tmp_path)


@pytest.fixture
def qualification_job(tmp_path):
    modes = {'smoke': 'ada-pgid-cpu', 'full': 'ada-pgid-cpu-noop'}
    spec = {'entrypoint': ['-m', 'experiments.tennis.k5d_scoped_namespace'],
            'arguments': ['outer', '--', '{proof_mode}'], 'scale': {'proof_mode': modes}, 'inputs': []}
    ready = {'source_sha256': IDENTITY['source_sha256'], 'snapshot_sha256': IDENTITY['snapshot_sha256'],
             'spec_sha256': inbox.digest(spec)}
    hashes = {'spec': write(tmp_path / 'spec.json', spec), 'ready': write(tmp_path / 'ready.json', ready)}
    for stage in ('smoke', 'full'):
        folder = tmp_path / stage
        folder.mkdir()
        dispatch = {'command': inbox.phase_command(spec, stage, [], '/exact/python'), 'code': {**IDENTITY, 'spec_sha256': inbox.digest(spec)},
                    'staged_environment': {'sha256': IDENTITY['environment_sha256']}, 'input_identity': [], 'staged_inputs': []}
        staging = {'inputs': [], 'environment': {'sha256': IDENTITY['environment_sha256']}}
        execution = {'exit_code': 0, 'timing_contaminated': False, 'contention': [], 'paused_seconds': 0}
        hashes[stage] = {name: write(folder / (name + '.json'), data)
                         for name, data in (('dispatch', dispatch), ('staging', staging), ('execution', execution))}
        (folder / 'command.log').write_text('immutable qualification command output')
        hashes[stage]['command.log'] = ada.oracle.sha(folder / 'command.log')
    record = {'job_id': JOB, 'state': 'complete', 'execution_identity': IDENTITY, 'sha256': hashes}
    proof = {'approved': True, 'consumer_execution_identity': IDENTITY, 'all_six_exact': True,
             'actual_filesystem_denial_passed': True, 'qualification_jobs': {
                 'actual-cpu-denial': record, 'ada-first-witness': {**copy.deepcopy(record), 'job_id': SCORER}}}
    return tmp_path, proof


def test_only_pinned_successful_qualification_classes_are_recognized(qualification_job):
    root, proof = qualification_job
    assert ada.known_qualification_job(JOB, 'complete', root, proof, IDENTITY)
    assert not ada.known_qualification_job('20261113T000002Z-87654321', 'complete', root, proof, IDENTITY)
    with pytest.raises(ValueError, match='status'):
        ada.known_qualification_job(JOB, 'failed', root, proof, IDENTITY)


@pytest.mark.parametrize('tamper', ['missing-map', 'id', 'spec-sha', 'stage-sha', 'mode', 'identity', 'status', 'command'])
def test_qualification_exemption_fails_closed_on_tampering(qualification_job, tamper):
    root, proof = qualification_job
    record = proof['qualification_jobs']['actual-cpu-denial']
    if tamper == 'missing-map':
        del proof['qualification_jobs']
    elif tamper == 'id':
        record['job_id'] = 'not-a-job'
    elif tamper == 'spec-sha':
        record['sha256']['spec'] = 'f' * 64
    elif tamper == 'stage-sha':
        record['sha256']['full']['dispatch'] = 'f' * 64
    elif tamper == 'identity':
        record['execution_identity'] = {**IDENTITY, 'environment_sha256': 'f' * 64}
    elif tamper == 'status':
        record['state'] = 'failed'
    elif tamper == 'mode':
        spec = json.loads((root / 'spec.json').read_text())
        spec['scale']['proof_mode']['full'] = 'ada-scale-prepare'
        record['sha256']['spec'] = write(root / 'spec.json', spec)
    elif tamper == 'command':
        dispatch = json.loads((root / 'full/dispatch.json').read_text())
        dispatch['command'][-1] = 'ada-first-witness'
        record['sha256']['full']['dispatch'] = write(root / 'full/dispatch.json', dispatch)
    with pytest.raises(ValueError):
        ada.known_qualification_job(JOB, 'complete', root, proof, IDENTITY)
