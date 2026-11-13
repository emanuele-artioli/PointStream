from experiments.jobs.claim_observer import observe

CFG = {'nonce':'n', 'spec_sha256':'s', 'source_sha256':'x', 'inputs_sha256':'i', 'inbox':'/inbox', 'boot':'b', 'max_gap_ns':100, 'ready_wall_ns':1}
BIND = {'nonce':'n', 'spec_sha256':'s', 'source_sha256':'x', 'inputs_sha256':'i', 'job_id':'job'}


def sample(n, **kw):
    return {'start_ns':n, 'end_ns':n+1, 'boot':'b', **kw}


def valid():
    return [sample(1, gpu_state='absent'), sample(5, gpu_job_dir='/inbox/job/run', gpu_token_sha256='t', job_join=True, cpu_token_sha256='c', cpu_token_present=True, leader_live=True, supervisor_live=True, leader_pid=1, leader_birth=1, supervisor_pid=2, supervisor_birth=2, pgid=1),
            sample(9, gpu_state='absent', cpu_token_sha256='c', leader_pid=1, leader_birth=1, supervisor_pid=2, supervisor_birth=2, pgid=1, cpu_token_present=False, group_absent=True, supervisor_absent=True,
                   terminal='complete', spec_sha256='s', source_sha256='x', published_ns=2)]


def test_claim_bracket_requires_real_join_and_release():
    result = observe(CFG, BIND, valid())
    assert result['status'] == 'bounded'
    assert result['start_lower_ns'] == 1 and result['release_upper_ns'] == 10


def test_missing_or_unobserved_claim_is_unknown():
    assert observe(CFG, BIND, [sample(1, gpu_state='absent')])['status'] == 'unknown'
    assert observe(CFG, None, valid())['status'] == 'unknown'
    assert observe(CFG, BIND, valid()[1:])['status'] == 'unknown'


def test_fail_closed_on_gap_read_error_and_incomplete_cleanup():
    data = valid()
    data[1]['error'] = 'NFS timed out'
    assert observe(CFG, BIND, data)['status'] == 'unknown'
    data = valid()
    data[2]['start_ns'] = 999
    data[2]['end_ns'] = 1000
    assert observe(CFG, BIND, data)['status'] == 'unknown'
    data = valid()
    data[2]['group_absent'] = False
    assert observe(CFG, BIND, data)['status'] == 'unknown'


def test_mismatched_token_or_binding_never_passes():
    data = valid()
    data[2]['gpu_state'] = 'other'
    data[2]['gpu_token_sha256'] = 't'
    assert observe(CFG, BIND, data)['status'] == 'unknown'
    assert observe(CFG, {**BIND, 'source_sha256':'wrong'}, valid())['status'] == 'unknown'


def test_replay_rejects_missing_final_or_changed_result(tmp_path):
    import json
    from experiments.jobs.claim_observer import validate_journal
    path = tmp_path / 'journal'
    rows = [{'event':'ready', 'config':CFG}, {'event':'ready_complete', 'wall_ns':1}, {'event':'binding', 'binding':BIND}]
    rows += [{'event':'sample', **s} for s in valid()]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert validate_journal(path)['status'] == 'unknown'
    rows.append({'event':'final', 'result':{'status':'bounded'}})
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert validate_journal(path)['status'] == 'unknown'
    rows[-1]['result'] = observe(CFG, BIND, valid())
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert validate_journal(path)['status'] == 'bounded'


def test_partial_lock_is_unknown_not_absence(tmp_path):
    from experiments.jobs.claim_observer import sample_once
    import pytest
    lock = tmp_path / 'gpu'
    lock.mkdir()
    with pytest.raises(ValueError, match='lock without record'):
        sample_once({'gpu_claim':str(lock / 'claim.json')}, None)


def test_claim_host_and_uuid_are_checked(tmp_path):
    import json
    import pytest
    from experiments.jobs.claim_observer import sample_once
    claim = tmp_path / 'claim.json'
    claim.write_text(json.dumps({'host':'gpu6.example', 'device_uuid':'GPU-a', 'sharing':'exclusive',
                                 'job_dir':'/inbox/job/run', 'token':'secret'}))
    cfg = {'gpu_claim':str(claim), 'host_fqdn':'gpu6.example', 'gpu_uuid':'GPU-a'}
    assert sample_once(cfg, None)['gpu_state'] == 'other'
    with pytest.raises(ValueError, match='identity mismatch'):
        sample_once({**cfg, 'gpu_uuid':'GPU-b'}, None)


def test_actual_claim_shapes_join_source_spec_and_cpu(tmp_path, monkeypatch):
    import json
    from experiments.jobs import claim_observer as ob
    job = '20261112T235535Z-13aba806'
    run = tmp_path / 'inbox' / job / 'run'
    run.mkdir(parents=True)
    spec = {'hosts':['gpu6'], 'single_cpu_campaign':True, 'cpu_threads':1,
            'gpu_sharing':{'mode':'exclusive'}, 'inputs':[{'path':'p', 'sha256':'h'}]}
    dc = {'host':'gpu6.example', 'device_uuid':'GPU-a', 'sharing':'exclusive', 'token':'gpu-secret', 'job_dir':str(run)}
    cc = {'host':'gpu6.example', 'threads':1, 'token':'cpu-secret', 'job_dir':str(run)}
    (run/'claim.json').write_text(json.dumps({'device_claim':dc, 'cpu_claim':cc, 'child_pid':71, 'child_proc_start_time':1.5, 'process_group_id':71}))
    (run/'status.json').write_text(json.dumps({'pid':71, 'process_group_id':71, 'proc_start_time':1.5, 'status':'running'}))
    (run/'supervisor.json').write_text(json.dumps({'pid':72, 'proc_start_time':2.5}))
    spec_sha = ob.inbox.digest(spec)
    (run.parent/'spec.json').write_text(json.dumps(spec))
    (run.parent/'ready.json').write_text(json.dumps({'spec_sha256':spec_sha, 'source_sha256':'source', 'published':1.0}))
    gpu = tmp_path/'gpu.json'
    gpu.write_text(json.dumps(dc))
    cpu = tmp_path/'cpu.json'
    cpu.write_text(json.dumps({'allocations':{'cpu-secret':cc}}))
    monkeypatch.setattr(ob, 'get_process_start_time', lambda pid: {71:1.5, 72:2.5}[pid])
    monkeypatch.setattr(ob.os, 'getpgid', lambda pid: 71)
    monkeypatch.setattr(ob.os, 'killpg', lambda pid, signal: None)
    cfg = {'gpu_claim':str(gpu), 'cpu_claim':str(cpu), 'host_fqdn':'gpu6.example', 'gpu_uuid':'GPU-a',
           'inbox':str(tmp_path/'inbox'), 'spec_sha256':spec_sha, 'source_sha256':'source'}
    result = ob.sample_once(cfg, {'job_id':job, 'inputs_sha256':ob.inbox.digest(spec['inputs'])})
    assert result['job_join'] and result['cpu_token_present'] and result['leader_live']
    assert result['gpu_token_sha256'] != dc['token'] and result['cpu_token_sha256'] != cc['token']


def test_reader_timeout_is_unknown_error(monkeypatch):
    import time
    import pytest
    from experiments.jobs import claim_observer as ob

    def slow(cfg, binding):
        time.sleep(2)
        return {}

    monkeypatch.setattr(ob, 'sample_once', slow)
    with pytest.raises(TimeoutError):
        ob.bounded_sample({'read_timeout_seconds':0.05, 'reader_cpu_limit':1}, None)


def test_prepublication_and_cpu_identity_are_required():
    data = valid()
    data[-1]['published_ns'] = 1
    assert observe(CFG, BIND, data)['status'] == 'unknown'
    data = valid()
    data[1]['cpu_token_present'] = False
    assert observe(CFG, BIND, data)['status'] == 'unknown'
    data = valid()
    data[-1]['supervisor_birth'] = 99
    assert observe(CFG, BIND, data)['status'] == 'unknown'


def test_foreign_replacement_is_only_release_after_owned_token_seen():
    data = valid()
    data[-1]['gpu_state'] = 'other'
    data[-1]['gpu_token_sha256'] = 'different'
    assert observe(CFG, BIND, data)['status'] == 'bounded'
    data[1]['gpu_job_dir'] = '/inbox/another/run'
    assert observe(CFG, BIND, data)['status'] == 'unknown'
