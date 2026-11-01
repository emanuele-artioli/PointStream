import numpy as np
import pytest

from experiments.tennis.w8_input_job import judged_bitmap, validate


def test_declared_recipe_has_exact_eighty_ticks():
    ticks = np.unpackbits(np.frombuffer(judged_bitmap("0:240:3"), np.uint8)).nonzero()[0]
    assert ticks.tolist() == list(range(0, 240, 3))
    with pytest.raises(ValueError):
        judged_bitmap("1:240:3")


def test_frozen_window_uses_paid_bitmap_and_exact_bindings(monkeypatch):
    from experiments.tennis import w8_input_job as job, w8_payload

    blob = b'frozen-window'
    monkeypatch.setattr(job, 'WINDOW_SHA', job.digest(blob))
    units = [{'id': name, 'reference': f'ref{i}', 'reference_tick': i, 'box': [1, 2, 3, 4]}
             for i, name in enumerate(job.UNITS)]
    profile = {'window': job.WINDOW, 'judged': 'judged', 'kps': 'kps', 'units': units,
               'match': {'section': 'match'}}
    code = job.Path(job.__file__).resolve().parent
    profile['models'] = {key: job.file_sha(code / filename) for key, filename in
                         [('payload_decoder_code', 'w8_payload.py'), ('conditioning_decoder_code', 'w8_inputs.py'),
                          ('geometry_code', 'k6b.py'), ('packet_code', 'k6b_rate_streams.py')]}
    sections = {'judged': job.judged_bitmap(job.RECIPE), 'kps': b'kps', 'ref0': b'r0', 'ref1': b'r1', 'match': b'm'}
    monkeypatch.setattr(w8_payload, 'unpack_window', lambda packet, expected: (sections, {}))
    def decode(_kps, refs, window, paid_bitmap):
        assert window == job.WINDOW and paid_bitmap is sections['judged'] and set(refs) == set(job.UNITS)
        arrays = {name: {'t': np.arange(240), 'judged': np.arange(240) % 3 == 0,
                         'reference_box': np.array([1, 2, 3, 4])} for name in job.UNITS}
        return arrays, {'reference_frames': {name: i for i, name in enumerate(job.UNITS)}}
    monkeypatch.setattr(job, 'decode_inputs', decode)
    _, _, match = job.decode_frozen_window(blob, profile)
    assert match == b'm'
    sections['judged'] = bytes(30)
    with pytest.raises(ValueError, match='bitmap'):
        job.decode_frozen_window(blob, profile)
    with pytest.raises(ValueError, match='identity'):
        job.decode_frozen_window(b'changed', profile)
    sections['judged'] = job.judged_bitmap(job.RECIPE)
    profile['models']['geometry_code'] = '0' * 64
    with pytest.raises(ValueError, match='decoder code'):
        job.decode_frozen_window(blob, profile)


def test_match_validation_rejects_unpinned_originals_before_loading_models(tmp_path):
    from experiments.tennis import w8_input_job as job

    fits = tmp_path / 'fits.tar'
    guider = tmp_path / 'guider.pth'
    fits.write_bytes(b'wrong')
    guider.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='identity'):
        job.verify_match_payload(b'match', {}, fits, guider)


def test_tensor_fingerprints_bind_name_shape_dtype_bytes_and_negative_zero():
    import copy
    from experiments.tennis import w8_input_job as job

    state = {'reference_unet': {'weight': np.array([0.0], dtype=np.float32)},
             'denoising_unet': {'weight': np.array([1.0], dtype=np.float16)},
             'pose_guider': {'weight': np.array([2.0], dtype=np.float32)}}
    expected = job.tensor_fingerprints(state)
    for replacement in (np.array([-0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
                        np.array([0.0], dtype=np.float16), np.array([[0.0]], dtype=np.float32)):
        changed = copy.deepcopy(state)
        changed['reference_unet']['weight'] = replacement
        assert job.tensor_fingerprints(changed) != expected
    renamed = copy.deepcopy(state)
    renamed['reference_unet']['other'] = renamed['reference_unet'].pop('weight')
    assert job.tensor_fingerprints(renamed) != expected


def test_worker_must_publish_every_expected_match_tensor():
    from experiments.tennis import w8_input_job as job

    fingerprints = {f'tensor-{i}': {'sha256': str(i)} for i in range(480)}
    job.check_worker_match({'match_tensors': fingerprints}, {'tensor_fingerprints': fingerprints})
    with pytest.raises(ValueError, match='fingerprints'):
        job.check_worker_match({}, {'tensor_fingerprints': fingerprints})
    changed = dict(fingerprints)
    changed.pop('tensor-0')
    with pytest.raises(ValueError, match='fingerprints'):
        job.check_worker_match({'match_tensors': changed}, {'tensor_fingerprints': fingerprints})


def test_cpu_bootstrap_receipt_requires_exact_bytes_and_real_denial():
    import errno
    from experiments.tennis import w8_input_job as job

    expected = job.digest(np.array([-1.0, -0.0, 1.0], np.float32).tobytes()
                          + np.array([0.5, -0.0, 1.5], np.float16).tobytes())
    record = {'before_landlock_sha256': expected, 'after_landlock_sha256': expected,
              'cpuinfo_denied_errno': errno.EACCES}
    job.check_bootstrap_receipt(record)
    for name, value in [('before_landlock_sha256', 'changed'), ('after_landlock_sha256', 'changed'),
                        ('cpuinfo_denied_errno', errno.ENOENT)]:
        with pytest.raises(ValueError, match='bootstrap'):
            job.check_bootstrap_receipt({**record, name: value})


def test_landlock_requires_single_thread_and_receipt():
    from experiments.tennis import w8_input_job as job

    job.check_pre_sandbox_threads(['123'], 123)
    job.check_thread_receipt({'pre_sandbox_thread_count': 1})
    for threads in ([], ['123', '456'], ['456']):
        with pytest.raises(RuntimeError, match='exactly'):
            job.check_pre_sandbox_threads(threads, 123)
    for value in ({}, {'pre_sandbox_thread_count': 2}, {'pre_sandbox_thread_count': True}):
        with pytest.raises(ValueError, match='single-thread'):
            job.check_thread_receipt(value)


def test_installed_worker_guider_requires_exact_file_and_profile(tmp_path, monkeypatch):
    from experiments.tennis import w8_input_job as job

    guider = tmp_path / 'guider.pth'
    guider.write_bytes(b'correct')
    monkeypatch.setattr(job, 'GUIDER_SHA', job.file_sha(guider))
    profile = {'models': {'pose_guider': job.GUIDER_SHA}}
    job.check_installed_guider(guider, profile)
    guider.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='guider identity'):
        job.check_installed_guider(guider, profile)
    guider.write_bytes(b'correct')
    profile['models']['pose_guider'] = 'wrong'
    with pytest.raises(ValueError, match='guider identity'):
        job.check_installed_guider(guider, profile)


def test_fixed_inputs_require_pinned_one_core_and_probe_original_aliases(tmp_path, monkeypatch):
    import json
    from experiments.tennis import w8_input_job as job

    stage = tmp_path / 'smoke'
    stage.mkdir()
    inputs = {}
    declared = []
    staged = []
    for name in job.FIXED_INPUTS:
        path = tmp_path / name
        path.write_bytes(name.encode())
        alias = tmp_path / f'{name}-alias'
        alias.symlink_to(path)
        sha = job.file_sha(path)
        inputs[name] = sha
        declared.append({'name': name, 'path': str(alias), 'sha256': sha})
        staged.append({'name': name, 'path': str(path), 'sha256': sha})
    monkeypatch.setattr(job, 'FIXED_INPUTS', inputs)
    spec = {'device': 'cpu', 'cpu_threads': 1, 'single_cpu_campaign': True,
            'defer_failed_publication': True, 'budget_seconds': 300,
            'environment': {'sha256': job.PREP_ENV_SHA}, 'staged_inputs': declared}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [0]}))
    paths, denied = job.fixed_inputs(stage)
    assert paths['source'] in denied and tmp_path / 'source-alias' in denied
    assert paths['guider'] not in denied
    spec['single_cpu_campaign'] = False
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='one-core'):
        job.fixed_inputs(stage)
    spec['single_cpu_campaign'] = True
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    paths['packet'].write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        job.fixed_inputs(stage)


def test_fixed_promotion_requires_seal_and_remaining_budget(tmp_path, monkeypatch):
    import json
    from experiments.tennis import w8_input_job as job

    stage = tmp_path / 'smoke'
    stage.mkdir()
    sealed = {'passed': True, 'worker_sha256': 'worker', 'decoded_sha256': {'a': 'hash'}}
    (stage / 'fixed-validation.json').write_text(json.dumps(sealed))
    (tmp_path / 'spec.json').write_text(json.dumps({'budget_seconds': 300, 'full': {'seconds': 160}}))
    (stage / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 100, 'phase_started': 110, 'budget_end': 400}}))
    (stage / 'staging.json').write_text(json.dumps({'temporary_environment': {'TMPDIR': str(tmp_path)}}))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(job, 'validate_fixed', lambda *_: {**sealed, 'checks': {}})
    monkeypatch.setattr(job, 'recover_fixed_publication', lambda *_: tmp_path)
    monkeypatch.setattr(job.time, 'monotonic', lambda: 140)
    output = stage / 'external.json'
    assert job.fixed_validate_command(stage, output) == 0
    monkeypatch.setattr(job.time, 'monotonic', lambda: 230)
    assert job.fixed_validate_command(stage, output) == 1
    assert not json.loads(output.read_text())['passed']
    monkeypatch.setattr(job.time, 'monotonic', lambda: 140)
    (stage / 'fixed-validation.json').write_text(json.dumps({**sealed, 'worker_sha256': 'changed'}))
    assert job.fixed_validate_command(stage, output) == 1
    assert 'seal mismatch' in json.loads(output.read_text())['error']


def test_fixed_worker_fails_before_output_without_distinct_network_namespace(tmp_path, monkeypatch):
    from experiments.tennis import w8_input_job as job

    monkeypatch.setattr(job.sys, 'platform', 'linux')
    monkeypatch.setattr(job.os, 'readlink', lambda _: 'net:[parent]')
    with pytest.raises(RuntimeError, match='network namespace'):
        job.fixed_worker(tmp_path / 'packet', tmp_path / 'profile', tmp_path / 'guider', tmp_path / 'out', [], 'net:[parent]')
    assert not (tmp_path / 'out').exists()


def test_fixed_worker_rejects_external_network_interface(tmp_path, monkeypatch):
    import socket
    from experiments.tennis import w8_input_job as job

    monkeypatch.setattr(job.sys, 'platform', 'linux')
    monkeypatch.setattr(job.os, 'readlink', lambda _: 'net:[child]')
    monkeypatch.setattr(socket, 'if_nameindex', lambda: [(1, 'lo'), (2, 'eth0')])
    with pytest.raises(RuntimeError, match='only loopback'):
        job.fixed_worker(tmp_path / 'packet', tmp_path / 'profile', tmp_path / 'guider', tmp_path / 'out', [], 'net:[parent]')
    assert not (tmp_path / 'out').exists()


def test_fixed_run_launches_separate_isolated_worker_and_validator(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.tennis import w8_input_job as job

    paths = {name: tmp_path / name for name in job.FIXED_INPUTS}
    forbidden = [paths['prepared'], paths['source']]
    monkeypatch.setattr(job, 'fixed_inputs', lambda _: (paths, forbidden))
    monkeypatch.setattr(job.sys, 'platform', 'linux')
    monkeypatch.setattr(job.os, 'readlink', lambda _: 'net:[parent]')
    monkeypatch.setattr(job.os, 'access', lambda *_: True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(job, 'publish_fixed_outputs', lambda *_: None)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(job.subprocess, 'run', run)
    job.fixed_run(tmp_path, paths)
    assert calls[0][0][:4] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
    assert 'fixed-worker' in calls[0][0] and 'fixed-validate' in calls[1][0]
    assert calls[0][0][calls[0][0].index('--guider') + 1] == str(paths['guider'])
    assert all(call[1]['close_fds'] is True and call[1]['stdin'] == job.subprocess.DEVNULL for call in calls)
    assert 'HOME' not in calls[0][1]['env'] and calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert calls[0][1]['env']['XDG_CACHE_HOME'] == str(tmp_path) == calls[0][1]['env']['TORCH_HOME']


def test_publication_contains_only_checked_outputs_and_detects_tampering(tmp_path):
    import json
    import shutil
    import tarfile
    from experiments.tennis import w8_input_job as job

    stage = tmp_path / 'stage'
    stage.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (scratch / 'fixed-decoded').mkdir()
    for path in [scratch / 'fixed-decoded' / 'worker.json', stage / 'fixed-validation.json',
                 stage / 'fixed-launch.json', stage / 'fixed-resource.json',
                 *(scratch / 'fixed-decoded' / f'{name}.npz' for name in job.UNITS)]:
        path.write_bytes(b'evidence')
    job.publish_fixed_outputs(stage, scratch)
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        tar.add(scratch / 'publish', arcname='publish')
    (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': job.file_sha(stage / 'published.tar')}}))
    shutil.rmtree(scratch)
    recovered = job.recover_fixed_publication(stage, tmp_path)
    assert recovered.is_relative_to(tmp_path) and (recovered / f'{job.UNITS[0]}.npz').read_bytes() == b'evidence'
    (stage / 'fixed-validation.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='receipt mismatch'):
        job.recover_fixed_publication(stage, tmp_path)


def test_publication_rejects_oversize_and_symlinked_worker_output(tmp_path):
    from experiments.tennis import w8_input_job as job

    stage, scratch = tmp_path / 'stage', tmp_path / 'scratch'
    stage.mkdir()
    (scratch / 'fixed-decoded').mkdir(parents=True)
    for path in [scratch / 'fixed-decoded' / 'worker.json', stage / 'fixed-validation.json',
                 stage / 'fixed-launch.json', stage / 'fixed-resource.json',
                 *(scratch / 'fixed-decoded' / f'{name}.npz' for name in job.UNITS)]:
        path.write_bytes(b'x')
    oversized = scratch / 'fixed-decoded' / f'{job.UNITS[0]}.npz'
    with oversized.open('wb') as f:
        f.truncate(20 * 1024 * 1024)
    with pytest.raises(ValueError, match='20 MiB'):
        job.fixed_publication_files(stage, scratch)
    oversized.unlink()
    oversized.symlink_to(stage / 'fixed-validation.json')
    with pytest.raises(ValueError, match='invalid'):
        job.fixed_publication_files(stage, scratch)


def test_validator_fails_closed_without_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path))
    # Avoid imposing a process CPU limit on the pytest runner itself.
    monkeypatch.setattr("experiments.tennis.w8_input_job.cpu_limit", lambda _: None)
    result = validate(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["validator_completed"]
    assert "FileNotFoundError" in result["details"]["error"]


def test_denied_probes_must_exist_and_read(tmp_path):
    from experiments.tennis.w8_input_job import _verified_denials

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"a")
    with pytest.raises(ValueError, match="not a file"):
        _verified_denials([first, second])
    second.write_bytes(b"b")
    _verified_denials([first, second])
    with pytest.raises(ValueError, match="Distinct"):
        _verified_denials([first, first])


def test_allowlist_rejects_forbidden_inside_project(tmp_path):
    from pathlib import Path
    from experiments.tennis.w8_input_job import _allowed_read_paths

    project_file = Path(__file__).resolve()
    with pytest.raises(ValueError, match="forbidden"):
        _allowed_read_paths(tmp_path, tmp_path, [project_file])


def test_stage_validation_survives_scratch_removal(tmp_path, monkeypatch):
    import errno
    import io
    import json
    import shutil
    import tarfile
    from experiments.tennis import w8_input_job as job

    stage = tmp_path / "stage"
    packets = stage / "packets"
    decoded = stage / "decoded"
    packets.mkdir(parents=True)
    decoded.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    temp = tmp_path / "host-tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(job, "cpu_limit", lambda _: None)
    # The real validator starts in a fresh process; pytest may already have
    # consumed more than its 120-second process-wide CPU limit.
    monkeypatch.setattr(job, "cpu_seconds", lambda: 0.0)
    monkeypatch.setattr(job, "validate_golden", lambda *_: None)
    contents = {"keypoints.kps": b"keypoints", **{f"{n}.png.ref": n.encode() for n in job.UNITS}}
    for name, content in contents.items():
        (packets / name).write_bytes(content)
    packet_shas = {name: job.digest(content) for name, content in contents.items()}
    monkeypatch.setattr(job, "PACKET_SHAS", packet_shas)
    prepared = stage / "prepared.tar"
    with tarfile.open(prepared, "w") as tar:
        for name in job.UNITS:
            content = b"golden"
            member = tarfile.TarInfo(f"publish/units/{name}.npz")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(job, "PREPARED_SHA", job.digest(prepared.read_bytes()))
    hashes = {}
    for name in job.UNITS:
        path = decoded / f"{name}.npz"
        np.savez_compressed(path, reference=np.zeros(1), reference_box=np.zeros(4), t=np.arange(240),
                            run=np.zeros(240), box=np.zeros((240, 4)), pose=np.zeros(1),
                            judged=np.arange(240) % 3 == 0)
        hashes[name] = job.digest(path.read_bytes())
    forbidden = ["/golden", "/prepared-copy", "/probe", "/source"]
    (stage / "receipt.json").write_text(json.dumps({"worker_returncode": 0, "packet_dir": "packets", "prepared": "prepared.tar",
                                                    "packets": packet_shas, "recipe": job.RECIPE, "parent_cpu_s": 0, "forbidden": forbidden}))
    import cv2
    (decoded / "worker.json").write_text(json.dumps({"isolation": {"landlock_abi": 1, "denied": {p: errno.EACCES for p in forbidden}},
                                                       "recipe": job.RECIPE, "numpy": np.__version__, "opencv": cv2.__version__,
                                                       "cpu_s": 0, "hashes": hashes}))
    shutil.rmtree(scratch)
    result = job.validate(stage)
    assert result["passed"], result
    assert not scratch.exists()
    monkeypatch.setattr(job, "cpu_seconds", lambda: 121.0)
    assert not job.validate(stage)["checks"]["process_cpu_limits"]
    monkeypatch.setattr(job, "cpu_seconds", lambda: 0.0)
    manifest = job.evidence_manifest(stage)
    (packets / "keypoints.kps").write_bytes(b"tampered")
    assert manifest != job.evidence_manifest(stage)
    assert not job.validate(stage)["passed"]
