from copy import deepcopy

import pytest

from experiments.generative import gvcrt_peer_job_w8leaves as job


def manifest_for(qp, stream):
    shifts = [0, 2, 0, 1, 0, 1, 0, 1]
    return {
        'schema': 'gvcrt-native-v1', 'source_revision': 'd0e32bfa3e8e282f9a77437c223c605858eea637',
        'frames': 16, 'fps': 30, 'original_width': 1280, 'original_height': 720,
        'canvas_width': 1920, 'canvas_height': 1088, 'base_qp': qp, 'reset_interval': 8,
        'padding_policy': 'released', 'ec_part': 1, 'model_sha256': {'I': job.I_SHA, 'P': job.P_SHA},
        'stream': {'filename': 'stream.bin', 'bytes': 16, 'sha256': job.sha(stream)},
        'placements': [
            {'index': i, 'I': i == 0, 'reset': i > 0 and i % 8 == 1,
             'qp': qp + (shifts[i % 8] if i else 0), 'offset': i, 'bytes': 1}
            for i in range(16)
        ],
    }


@pytest.mark.parametrize('qp', [0, 3, 9])
def test_registered_qp_manifest_and_mutations(tmp_path, qp):
    stream = tmp_path / 'stream.bin'
    stream.write_bytes(b'x' * 16)
    manifest = manifest_for(qp, stream)
    job.check_manifest(manifest, 16, stream, qp)
    for field, value in [('base_qp', (qp + 1) % 10), ('canvas_width', 1280)]:
        altered = deepcopy(manifest)
        altered[field] = value
        with pytest.raises(ValueError):
            job.check_manifest(altered, 16, stream, qp)
    for key, value in [('qp', qp), ('I', True), ('reset', False), ('offset', 0)]:
        altered = deepcopy(manifest)
        altered['placements'][1][key] = value
        with pytest.raises(ValueError):
            job.check_manifest(altered, 16, stream, qp)
    with pytest.raises(ValueError):
        job.check_manifest(manifest, 16, stream, (qp + 1) % 10)


def test_default_unchanged_and_endpoints_bind_run_and_validator():
    deadline = '2026-11-06T12:00:00Z'
    default = job.make_spec(deadline)
    assert default == job.make_spec(deadline, 3)
    for qp in (0, 9):
        spec = job.make_spec(deadline, qp)
        assert spec['arguments'] == [*default['arguments'], '--qp', str(qp)]
        assert spec['validator'] == [*default['validator'], '--qp', str(qp)]
    for qp in (-1, 1, 10, True):
        with pytest.raises(ValueError):
            job.make_spec(deadline, qp)


def test_cpu_table_gate_checks_all_tables_and_shifts():
    from types import SimpleNamespace

    from experiments.generative.gvcrt_qp_coverage import check_tables

    class Tensor:
        def __init__(self, rows):
            self.shape = (rows, 1)

    class Finite:
        def all(self):
            return True

    torch = SimpleNamespace(isfinite=lambda _: Finite())
    i = SimpleNamespace(get_qp_num=lambda: 10, q_scale_enc=Tensor(10), q_scale_dec=Tensor(10),
                        q_scale_recon=Tensor(10))
    p = SimpleNamespace(get_qp_num=lambda: 10, shift_qp=lambda q, index: q + [0, 2, 1][index],
                        q_scale_enc=Tensor(12), q_scale_dec=Tensor(12),
                        q_scale_feature=Tensor(12), q_scale_recon=Tensor(12))
    assert check_tables(i, p, torch)['p_shift_indices']['9'] == [9, 11, 10]
    p.q_scale_feature = Tensor(11)
    with pytest.raises(ValueError):
        check_tables(i, p, torch)
    p.q_scale_feature = Tensor(12)
    p.shift_qp = lambda q, index: q + index
    with pytest.raises(ValueError):
        check_tables(i, p, torch)
