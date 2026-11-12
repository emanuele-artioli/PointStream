import copy
import json

import pytest

from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as encoder
from experiments.tennis.k5d_timing import Timer


def test_cost_forecast_requires_stage_and_campaign_fit(tmp_path):
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 200.}))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 100., 'phase_started': 110., 'budget_end': 2000.}}))
    assert cost.projection(tmp_path, 400.)['full_estimate_with_stage_reserve_seconds'] == 810.
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 225.}))
    with pytest.raises(ValueError, match='forecast'):
        cost.projection(tmp_path, 400.)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 200.}))
    with pytest.raises(ValueError, match='forecast'):
        cost.projection(tmp_path, 1150.)


def test_timer_records_nested_events_and_relocates(tmp_path):
    timer = Timer(tmp_path / 'before.json')
    assert timer.measure('outer', lambda: timer.measure('inner', lambda: 7)) == 7
    timer.relocate(tmp_path / 'after.json')
    events = json.loads((tmp_path / 'after.json').read_text())['events']
    assert [e['name'] for e in events] == ['inner', 'outer']
    assert all(e['completed'] and e['wall_seconds'] >= 0 and e['self_cpu_seconds'] >= 0 for e in events)
    with pytest.raises(ZeroDivisionError):
        timer.measure('error', lambda: 1 / 0)
    assert json.loads((tmp_path / 'after.json').read_text())['events'][-1]['completed'] is False


def test_48_fidelity_receipt_is_not_corpus_authority(tmp_path):
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only',
               'purpose': '48-frame-cost-pilot', 'clip': cost.CLIP, 'max_frames': 48,
               'archive_sha256': 'archive', 'calibration_job_id': cost.ACCEPTED_JOB, 'accepted_smoke_sha256': cost.ACCEPTED_SHA,
               'evidence_sha256': 'a' * 64,
               'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'full_cost']}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(receipt))
    digest = encoder.oracle.sha(path)
    assert encoder.fidelity_gate(path, digest, 'archive', purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=48)
    with pytest.raises(ValueError, match='purpose'):
        encoder.fidelity_gate(path, digest, 'archive')
    with pytest.raises(ValueError, match='frame limit'):
        encoder.fidelity_gate(path, digest, 'archive', purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=188)
    with pytest.raises(ValueError, match='frame limit'):
        encoder.fidelity_gate(path, digest, 'archive', purpose='48-frame-cost-pilot', clip='tracknet/game9/Clip5', frames=48)


def test_exact_contract_rejects_changed_pixel_or_metric():
    identities = {k: {'sha256': k} for k in ('manifest', 'archive', 'balltrack_weight', 'wasb_config', 'exposure', 'balltrack_code', 'wasb_code', 'wasb_weights')}
    rows = [{'frame': i, 'member': str(i), 'source_sha256': str(i), 'label_kind': 1, 'label_x': 1, 'label_y': 2} for i in range(16)]
    source = {'clip': cost.CLIP, 'device': {'capability': [8, 6]}, 'identities': identities, 'rows': rows, 'scores': {}}
    metric = {'frame': 0, 'frame_psnr': 30., 'full_lpips': .1, 'full_dists': .2, 'crop_lpips': .3, 'crop_dists': .4}
    arm = {'zero_ball_wire_sha256': 'z', 'judges': {'wasb': 'a'}, 'metrics': [metric], 'guard_off': False}
    point = {'codec': 'dcvc', 'point': 0, 'native_sha256': 'n', 'wire_sha256': 'w',
             'encoder_decoded_sha256': 'd', 'encoder_rendered_sha256': 'r', 'baseline': arm, 'streak': arm}
    prior = {'clip': cost.CLIP, 'frames': 16, 'fps': '30', 'size': [1280, 720],
             'identities': {n: {'sha256': n} for n in ('image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone')},
             'tools': {'ffmpeg': {'path': '/bin/ffmpeg', 'sha256': 'f'}},
             'metric_models': {'lpips': {'heads_sha256': 'h', 'backbone_sha256': 'a'},
                               'dists': {'weights_sha256': 'w', 'backbone_sha256': 'v'}},
             'conversion': {'yuv_sha256': 'y'}, 'points': [point]}
    cost.compare_contract(prior, source, copy.deepcopy(prior), copy.deepcopy(source), exact=True)
    ada_source = copy.deepcopy(source)
    ada_source['device']['capability'] = [8, 9]
    with pytest.raises(ValueError, match='GPU class'):
        cost.compare_contract(prior, source, prior, ada_source, exact=True)
    cost.compare_contract(prior, source, prior, ada_source, exact=True, capability=(8, 9))
    changed = copy.deepcopy(prior)
    changed['points'][0]['encoder_rendered_sha256'] = 'different'
    with pytest.raises(ValueError, match='pixel parity'):
        cost.compare_contract(prior, source, changed, source, exact=True)
    with pytest.raises(ValueError, match='pixel parity'):
        cost.compare_contract(prior, source, changed, ada_source, exact=True, capability=(8, 9))
    changed = copy.deepcopy(prior)
    changed['points'][0]['baseline']['metrics'][0]['crop_dists'] = .401
    with pytest.raises(ValueError, match='metric output'):
        cost.compare_contract(prior, source, changed, source, exact=True)


def test_generated_spec_freezes_cost_scope_and_required_inputs():
    from experiments.jobs.inbox import validate_spec

    path = cost.fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json'
    base = json.loads(path.read_text())
    gate = '/home/itec/emanuele/Datasets/pointstream-data/tennis/example/approved-48-only.json'
    spec = cost.make_spec(base, gate, 'a' * 64, '2026-11-05T02:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 1700
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 48}
    assert spec['gpu_models'] == ['RTX A6000']
    assert spec['smoke']['seconds'] == 600 and spec['full']['seconds'] == 900
    assert {x['name'] for x in spec['staged_inputs']} >= {'fidelity_gate', 'accepted16'}
    with pytest.raises(ValueError, match='data-root'):
        cost.make_spec(base, '/outside/receipt.json', 'a' * 64, spec['deadline'])
    with pytest.raises(ValueError, match='SHA256'):
        cost.make_spec(base, gate, 'not-a-hash', spec['deadline'])
