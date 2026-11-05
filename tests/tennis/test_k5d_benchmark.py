import json

import pytest

from experiments.tennis import k5d_benchmark_analysis as analysis, k5d_encoder as encoder


def test_independent_gate_must_be_valid_and_hash_bound(tmp_path):
    path = tmp_path / 'gate.json'
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': 'full-retrospective-benchmark', 'archive_sha256': 'archive',
               'calibration_job_id': 'job', 'evidence_sha256': 'a' * 64,
               'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'full_cost']}
    path.write_text(json.dumps(receipt))
    digest = encoder.oracle.sha(path)
    assert encoder.fidelity_gate(path, digest, 'archive') == receipt
    receipt['checks'].remove('full_cost')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='hash'):
        encoder.fidelity_gate(path, digest, 'archive')
    with pytest.raises(ValueError, match='checks'):
        encoder.fidelity_gate(path, encoder.oracle.sha(path), 'archive')


def test_exact_pooled_psnr_rejects_inexact_frame_only_records():
    assert analysis.metric_value([{'sse': 0, 'pixels': 3}], 'frame_psnr') == 100
    assert analysis.metric_value([{'sse': 3 * 255**2, 'pixels': 3}], 'frame_psnr') == 0
    assert analysis.metric_value([{'frame_psnr': 30}], 'frame_psnr') == pytest.approx(30)
    with pytest.raises(ValueError, match='frame PSNR'):
        analysis.metric_value([{'frame_psnr': None}], 'frame_psnr')


def test_false_interpolation_requires_support_and_handles_ties_conservatively():
    native = [{'rate': 10., 'false': 4}, {'rate': 10., 'false': 2}, {'rate': 40., 'false': 6}]
    assert analysis.interpolate_native_false(native, 20.) == pytest.approx(4.)
    assert analysis.interpolate_native_false(native, 9.) is None
    assert analysis.interpolate_native_false(native, 41.) is None


def test_population_gate_counts_all_30_and_unmatched_negative_never_passes(tmp_path, monkeypatch):
    selection = json.loads(analysis.SOURCE.read_text())
    clips = selection['tracknet']['clips']
    paths = []
    for i, clip in enumerate(clips):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / 'pilot.json').write_text(json.dumps({'clip': clip['clip_id']}))
        paths.append(folder)

    def simulated(_path, clip):
        i = clips.index(clip)
        visibility = clip['visibility_counts']
        absent = visibility['0']
        return {'clip_id': clip['clip_id'], 'game': clip['clip_id'].split('/')[1], 'frames': clip['label_rows'],
                'visible': visibility['1'] + visibility['2'], 'absent': absent, 'unknown': visibility['3'],
                'source_qualified': i >= 3, 'current_source_qualified': i >= 3, 'task_pass': 3 <= i < 26,
                'source_scores': {'false_on_absent': 10 if clip['clip_id'] == 'tracknet/game9/Clip5' else 0},
                'low_points': [{'codec': codec, 'setting': point, 'candidate_false': 0,
                                'native_false_at_rate': None if absent and i == first_absent else 0.}
                               for codec, point in analysis.LOW]}

    first_absent = next(i for i, clip in enumerate(clips) if clip['visibility_counts']['0'])
    monkeypatch.setattr(analysis, 'summarize_clip', simulated)
    result = analysis.summarize(selection, paths)
    assert result['coverage'] == {'frames': 5675, 'visible': 5472, 'absent': 197, 'unknown': 6}
    assert result['passing_clips'] == 23
    assert result['qualified_clips'] == 27
    assert result['pooled_mean_excess_false_per_absent'] is None
    assert not result['task_endpoint_pass']
    assert not result['strong_absence_claim_allowed']


def test_fidelity_rates_use_native_and_complete_candidate_bytes():
    report = {'frames': 1, 'fps': '30', 'points': [
        {'codec': 'dcvc', 'point': 0, 'native_bytes': 100, 'wire_bytes': 140,
         'baseline': {'metrics': [{'sse': 1, 'pixels': 3}]},
         'streak': {'metrics': [{'sse': 2, 'pixels': 3}]}}]}
    native, candidate = analysis.points(report, 'frame_psnr')
    assert native.byte_count == 100
    assert candidate.byte_count == 140
    assert candidate.rate == pytest.approx(native.rate * 1.4)


def test_candidate_and_anchor_family_filters_are_independent():
    quality = [{'frame_psnr': 30., 'full_lpips': .2, 'full_dists': .3}]
    report = {'frames': 1, 'fps': '30', 'points': [
        {'codec': codec, 'point': point, 'native_bytes': 100 + point, 'wire_bytes': 150 + point,
         'baseline': {'metrics': quality}, 'streak': {'metrics': quality}}
        for codec, point in [('dcvc', 0), ('svt', 48)] ]}
    rows = analysis.points(report, 'full_lpips', candidate_family='dcvc', anchor_family='svt')
    assert {(r.codec, r.variant) for r in rows} == {('dcvc', 'streak'), ('svt', 'native')}
    assert len(analysis.points(report, 'full_dists', 'dcvc', None)) == 3


def test_historical_psnr_guard_uses_dcvc_against_dcvc_only():
    fidelity = {'frame_psnr': {'dcvc': {'dcvc': {'bd_rate': .02}, 'pooled': {'bd_rate': -.3}},
                               'pooled': {'pooled': {'bd_rate': -.5}}}}
    assert analysis.historical_psnr_guard(fidelity, exact_sse=True, has_cap=False) is False
    fidelity['frame_psnr']['dcvc']['dcvc']['bd_rate'] = .005
    assert analysis.historical_psnr_guard(fidelity, exact_sse=True, has_cap=False) is True
    assert analysis.historical_psnr_guard(fidelity, exact_sse=False, has_cap=True) is None


def test_historical_source_failure_cannot_be_readmitted_by_rerun():
    assert analysis.task_source_eligibility('tracknet/game10/Clip11', True) == (True, False, False)
    assert analysis.task_source_eligibility('tracknet/game10/Clip8', True) == (True, False, False)
    assert analysis.task_source_eligibility('tracknet/game9/Clip7', True) == (True, False, False)
    assert analysis.task_source_eligibility('tracknet/game9/Clip5', True) == (True, True, True)
    assert analysis.task_source_eligibility('tracknet/game9/Clip5', False) == (False, True, False)
