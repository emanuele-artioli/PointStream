import hashlib
import json

import numpy as np
import pytest

from experiments.tennis import w8_receiver_generate as witness


def test_generation_recipe_never_requantizes_or_shortens_context(tmp_path):
    for smoke in (False, True):
        args = witness.generator_arguments(tmp_path / 'prepared.tar', tmp_path / 'decoded.pt',
                                           tmp_path / 'base.pt', {'original/x': tmp_path / 'x'},
                                           'Ada', smoke)
        assert '--fit-bits' not in args
        assert '--max-frames' not in args
        assert args[args.index('--steps') + 1] == '30'
        assert args[args.index('--cfg') + 2] == '2'
        assert ('--units' in args) == smoke


def test_crop_validator_rejects_changed_coverage(tmp_path, monkeypatch):
    # Use broadcast-backed zeros to avoid allocating historical image data.
    name = witness.fixed.UNITS[0]
    path = tmp_path / f'{name}@match5.npz'
    path.write_bytes(b'sealed')
    sidecar = {'unit': name, 'profile': 'match5', 'frames_generated': 240, 'kept': 80,
               'steps': 30, 'scope': 'candidate', 'seed': int(hashlib.sha256(name.encode()).hexdigest()[:7], 16),
               'npz_sha256': witness.fixed.file_sha(path)}
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    class Arrays:
        files = ['t', 'crops']
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            if key == 't':
                return np.arange(0, 240, 3)
            return np.broadcast_to(np.uint8(0), (80, 784, 512, 3))
    monkeypatch.setattr(witness.np, 'load', lambda *args, **kwargs: Arrays())
    assert witness.validate_crops(tmp_path, (name,))[name] == sidecar['npz_sha256']
    sidecar['frames_generated'] = 80
    path.with_suffix('.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='sidecar'):
        witness.validate_crops(tmp_path, (name,))


def test_arguments_parse_with_real_generator_interface(tmp_path, monkeypatch):
    from experiments.tennis import k6b_generate

    captured = []
    monkeypatch.setattr(k6b_generate, 'command_generate', lambda args: captured.append(args) or 0)
    for smoke in (True, False):
        argv = witness.generator_arguments(tmp_path / 'prepared.tar', tmp_path / 'decoded.pt',
                                           tmp_path / 'base.pt', {'original/x': tmp_path / 'x'},
                                           'Ada', smoke)
        assert k6b_generate.main(argv) == 0
    for args in captured:
        assert args.fit_bits == []
        assert args.fit == [['match5', str(tmp_path / 'decoded.pt')]]
        assert args.fit_base == [['match5', str(tmp_path / 'base.pt')]]
        assert args.cfg == [['match5', '2']]
        assert args.max_frames == 0 and args.from_t == 0 and not args.keep_all
    assert captured[0].units == witness.fixed.UNITS[0]
    assert captured[1].units == ''


def test_bootstrap_rejects_process_and_network_operations():
    from experiments.tennis import w8_receiver_bootstrap as bootstrap

    for event in ('subprocess.Popen', 'os.fork', 'os.exec', 'socket.connect'):
        with pytest.raises(PermissionError):
            bootstrap._deny_processes(event, ())
    bootstrap._deny_processes('open', ())


def test_bootstrap_denials_require_real_permission_error(tmp_path):
    from experiments.tennis import w8_receiver_bootstrap as bootstrap

    existing = tmp_path / 'readable'
    existing.write_bytes(b'x')
    with pytest.raises(RuntimeError, match='readable'):
        bootstrap._check_denials([existing])
    with pytest.raises(RuntimeError, match='did not establish denial'):
        bootstrap._check_denials([tmp_path / 'missing'])


def test_bootstrap_config_rejects_changed_asset_and_broad_root(tmp_path, monkeypatch):
    from experiments.tennis import w8_receiver_bootstrap as bootstrap

    asset = tmp_path / 'asset'
    asset.write_bytes(b'good')
    output = tmp_path / 'out'
    output.mkdir()
    protected = tmp_path / 'protected'
    protected.write_bytes(b'private')
    alias = tmp_path / 'source_alias'
    alias.symlink_to(protected)
    other = {}
    for kind in ('prepared', 'fit', 'golden', 'masks'):
        other[kind] = tmp_path / kind
        other[kind].write_bytes(kind.encode())
    from experiments.tennis import w8_receiver_preflight as preflight

    protected_files = {'source': protected, **other}
    monkeypatch.setattr(preflight, 'PROTECTED', {k: (str(v), hashlib.sha256(v.read_bytes()).hexdigest())
                                               for k, v in protected_files.items()})
    aliases = [alias]
    for kind, target in other.items():
        alt = tmp_path / f'{kind}_alias'
        alt.symlink_to(target)
        aliases.append(alt)
    verified = {k: {'path': str(v), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest(),
                    'device': v.stat().st_dev, 'inode': v.stat().st_ino} for k, v in protected_files.items()}
    receipt_path = tmp_path / 'receipt.json'
    receipt_path.write_text(json.dumps({'protected_inputs': verified}))
    config = {'parent_user_namespace': 'parent', 'parent_network_namespace': 'parent',
              'allow_paths': [str(asset), str(output)],
              'denied_paths': {'source': [str(protected)], 'aliases': [str(p) for p in aliases],
                               **{k: [str(v)] for k, v in other.items()}},
              'cpu_seconds': 5, 'environment': {},
              'files': {'unit': {'path': str(asset), 'sha256': hashlib.sha256(b'good').hexdigest()}},
              'tensor_fingerprints': {str(i): {} for i in range(480)}, 'unit': 'unit', 'output': str(output),
              'worker_kind': 'generate', 'protected_receipt': {
                  'path': str(receipt_path), 'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()}}
    path = tmp_path / 'config.json'
    def write():
        data = json.dumps(config).encode()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    digest = write()
    assert bootstrap.load_config(path, digest) == config
    config['worker_kind'] = 'cost_probe'
    for steps in (1, 3, 6):
        config['probe_steps'] = steps
        assert bootstrap.load_config(path, write())['probe_steps'] == steps
    config['probe_steps'] = 30
    with pytest.raises(ValueError, match='probe steps'):
        bootstrap.load_config(path, write())
    config['worker_kind'] = 'generate'
    del config['probe_steps']
    digest = write()
    asset.write_bytes(b'bad')
    with pytest.raises(ValueError, match='asset hash'):
        bootstrap.load_config(path, digest)
    asset.write_bytes(b'good')
    config['allow_paths'].append('/')
    with pytest.raises(ValueError, match='broad'):
        bootstrap.load_config(path, write())
    config['allow_paths'].remove('/')
    config['denied_paths']['aliases'].pop()
    with pytest.raises(ValueError, match='each protected category'):
        bootstrap.load_config(path, write())
    config['denied_paths']['aliases'].append(str(aliases[-1]))
    hardlink = tmp_path / 'allowed_hardlink'
    import os
    os.link(protected, hardlink)
    config['allow_paths'].append(str(hardlink))
    with pytest.raises(ValueError, match='protected hardlink'):
        bootstrap.load_config(path, write())


def test_sealed_validation_rejects_receipt_tampering(tmp_path, monkeypatch):
    name = witness.fixed.UNITS[0]
    fingerprints = {str(i): {} for i in range(480)}
    monkeypatch.setattr(witness, 'check_oracle_fingerprints', lambda *_: None)
    files = {'checkpoint': 'checkpoint', 'unit': witness.ORACLE_UNIT_SHA[name], 'decoder_manifest': 'manifest',
             'base': witness.BASE_SHA, 'aa_archive': witness.AA_ARCHIVE_SHA,
             'code:experiments.tennis.k6b_generate': witness.GENERATOR_CODE_SHA,
             'code:experiments.tennis.k6b_fit': witness.FIT_CODE_SHA,
             **{f'weight:{k}': v for k, v in witness.PINNED_WEIGHTS.items()}}
    receipt = {'unit': name, 'packet_sha256': witness.fixed.WINDOW_SHA, 'profile_sha256': witness.fixed.PROFILE_SHA,
               'tensor_fingerprints': fingerprints, 'files': files,
               'recipe': {'steps': 30, 'cfg': 2, 'context': [24, 4], 'size': [512, 784]},
               'isolation': {'landlock_abi': 4, 'pre_sandbox_threads': 1, 'extra_fds': False,
                             'denials': {'/protected': 13}},
               'device': {'name': 'Ada'}, 'kernels': {'attention': 'x'}, 'model_record': {'loaded': True},
               'sidecar': {'npz_sha256': 'crops'}, 'scope': 'candidate',
               'phase_costs': {k: {'wall_seconds': 1.0, 'cpu_seconds': 0.1, 'gpu_event_seconds': 0.5,
                                   'gpu_reserved_wall_seconds': 1.0} for k in ('model_load', 'original_probe', 'context_probe', 'sampling', 'publication')}}
    expected = {key: receipt[key] for key in ('unit', 'packet_sha256', 'profile_sha256',
                                              'tensor_fingerprints', 'files')}
    monkeypatch.setattr(witness, 'validate_crops', lambda *_: {name: 'crops'})
    monkeypatch.setattr(witness, 'check_model_record', lambda *_: None)
    monkeypatch.setattr(witness, 'check_execution_receipt', lambda *_: None)
    receipt['checkpoint_sha256'] = 'checkpoint'
    receipt['decoder_manifest_sha256'] = 'manifest'
    path = tmp_path / 'sealed.json'
    path.write_text(json.dumps(receipt))
    assert witness.validate_sealed(tmp_path, expected)['sha256'] == 'crops'
    receipt['sidecar']['npz_sha256'] = 'tampered'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='output mismatch'):
        witness.validate_sealed(tmp_path, expected)


def test_model_record_rejects_requantization_and_incomplete_loading():
    record = {'profile': 'original', 'dtype': 'float16', 'cfg': 2,
              'reference_unet': {'missing': 0}, 'denoising_unet': {'missing': 0}, 'pose_guider': {'missing': 0},
              'fit': {'checkpoint_sha256': 'checkpoint', 'rank': 4, 'base': {'sha256': witness.BASE_SHA},
                      'reference_unet': {'lora_tensors': 208, 'lora_layers': 104},
                      'denoising_unet': {'lora_tensors': 256, 'lora_layers': 128},
                      'pose_guider': {'keys': 16, 'missing': 0}}}
    witness.check_model_record(record, 'checkpoint')
    record['fit']['billing'] = {'bits': 4}
    with pytest.raises(ValueError, match='loading record'):
        witness.check_model_record(record, 'checkpoint')


def test_denial_groups_flatten_real_paths():
    from experiments.tennis.w8_receiver_bootstrap import flatten_denials

    assert [str(p) for p in flatten_denials({'source': ['/a', '/b'], 'golden': ['/c']})] == ['/a', '/b', '/c']


def test_independent_oracle_fingerprint_binding():
    with pytest.raises(ValueError, match='independent CPU oracle'):
        witness.check_oracle_fingerprints({str(i): {} for i in range(480)})


def test_prepare_packet_accepts_existing_empty_output(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.tennis import w8_payload

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        load=lambda *args, **kwargs: {}, save=lambda _state, path: path.write_bytes(b'checkpoint')))
    monkeypatch.setattr(w8_payload, 'unpack_match', lambda *args: {})
    monkeypatch.setattr(witness.fixed, 'pinned_profile', lambda _: {'match': {'sha256': 'match'}})
    arrays = {name: {'t': np.arange(240)} for name in witness.fixed.UNITS}
    monkeypatch.setattr(witness.fixed, 'decode_frozen_window', lambda *_: (arrays, {}, b'match'))
    monkeypatch.setattr(witness.fixed, 'check_installed_guider', lambda *_: None)
    monkeypatch.setattr(witness.fixed, 'tensor_fingerprints', lambda _: {str(i): {} for i in range(480)})
    monkeypatch.setattr(witness, 'check_oracle_fingerprints', lambda *_: None)
    expected_hashes = {}
    for name in witness.fixed.UNITS:
        file = tmp_path / f'{name}.npz'
        np.savez_compressed(file, **arrays[name])
        expected_hashes[name] = witness.fixed.file_sha(file)
    monkeypatch.setattr(witness, 'ORACLE_UNIT_SHA', expected_hashes)
    bundle = tmp_path / 'packet'
    bundle.write_bytes(b'packet')
    output = tmp_path / 'output'
    output.mkdir()
    result = witness.prepare_packet(bundle, tmp_path / 'profile', tmp_path / 'guider', output,
                                    {str(i): {} for i in range(480)})
    assert result['unit_sha256'] == expected_hashes
    assert (output / 'decoded.pt').read_bytes() == b'checkpoint'
    with pytest.raises(ValueError, match='empty'):
        witness.prepare_packet(bundle, tmp_path / 'profile', tmp_path / 'guider', output,
                               {str(i): {} for i in range(480)})


def test_gpu_qualification_rejects_wrong_kernels():
    receipt = {'device': {'name': 'NVIDIA RTX 6000 Ada Generation', 'major': 8, 'minor': 9},
               'torch_version': '2.7.0+cu128', 'cuda_version': '12.8',
               'kernels': {'kernel_launches': 2, 'attention_family': ['flash', 'mem_efficient'],
                           'tensor_core_like': ['cutlass']},
               'on_cuda': {name: True for name in ('vae', 'image_encoder', 'reference_unet',
                                                     'denoising_unet', 'pose_guider')},
               'executed_modules': {'animate_anyone': '/p'},
               'context_probe_cost': {'frames': 24, 'steps': 1, 'wall_seconds': 1.0, 'cpu_seconds': 0.5,
                                      'gpu_seconds': 0.8, 'scope': 'kernel qualification, not quality evidence'}}
    witness.check_execution_receipt(receipt)
    receipt['kernels']['attention_family'] = ['flash']
    with pytest.raises(ValueError, match='kernel'):
        witness.check_execution_receipt(receipt)


def test_frozen_code_rejects_self_consistent_substitution():
    files = {'code:experiments.tennis.k6b_generate': {'sha256': witness.GENERATOR_CODE_SHA},
             'code:experiments.tennis.k6b_fit': {'sha256': witness.FIT_CODE_SHA},
             'aa_archive': {'sha256': witness.AA_ARCHIVE_SHA}}
    witness.check_frozen_code(files)
    for key in files:
        original = files[key]['sha256']
        files[key]['sha256'] = '0' * 64
        with pytest.raises(ValueError, match='source identity'):
            witness.check_frozen_code(files)
        files[key]['sha256'] = original


def test_timeout_terminates_then_kills_and_reaps_only_owned_worker():
    import subprocess

    class Worker:
        def __init__(self):
            self.waits = 0
            self.actions = []
        def wait(self, timeout):
            self.waits += 1
            self.actions.append(('wait', timeout))
            if self.waits < 3:
                raise subprocess.TimeoutExpired('worker', timeout)
            return -9
        def terminate(self):
            self.actions.append(('terminate',))
        def kill(self):
            self.actions.append(('kill',))
    worker = Worker()
    assert witness.wait_owned_worker(worker, 12) == (-9, True)
    assert worker.actions == [('wait', 12), ('terminate',), ('wait', 5), ('kill',), ('wait', 5)]


def test_cost_probe_cannot_pass_candidate_validation(tmp_path, monkeypatch):
    output = tmp_path / 'sealed.json'
    output.write_text(json.dumps({'scope': 'qualification_only', 'recipe': {'steps': 3}}))
    with pytest.raises(ValueError, match='scope or step'):
        witness.validate_sealed(tmp_path, {})
    calls = []
    monkeypatch.setattr(witness, '_validate_receipt', lambda *args: calls.append(args) or {'scope': args[2]})
    assert witness.validate_cost_probe(tmp_path, {}, 3)['scope'] == 'qualification_only'
    assert calls[0][3] == 3
    for steps in (0, 2, 30, True):
        with pytest.raises(ValueError, match='exactly 1, 3 or 6'):
            witness.validate_cost_probe(tmp_path, {}, steps)


def test_cost_probe_renders_entire_native_runs_with_original_seeds():
    class Renderer:
        def __init__(self):
            self.calls = []
        def render(self, reference, poses, steps, seed):
            self.calls.append((reference, poses.copy(), steps, seed))
            return poses
    unit = {'reference': np.array([1]), 'pose': np.arange(240), 't': np.arange(240),
            'run': np.repeat([2, 5], 120), 'judged': np.arange(240) % 3 == 0}
    for steps in (1, 3, 6, 30):
        renderer = Renderer()
        ticks, crops = witness.render_full_unit(renderer, unit, 41, steps)
        assert ticks == list(range(0, 240, 3))
        assert list(crops) == ticks
        assert [len(call[1]) for call in renderer.calls] == [120, 120]
        assert [call[2:] for call in renderer.calls] == [(steps, 41 * 1009 + 2), (steps, 41 * 1009 + 5)]
