"""CPU tests of W3's pure logic and of the worker's new plan keys; mocks are never codec evidence."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.tennis import w3_analysis as an
from experiments.tennis.w3_neural_baseline import parse_scales, project_full_seconds
from src.codecs import dcvc_uf_worker as worker


def nals(intra: int, per_group: int, groups: int = 3, padding: int = 0) -> list[dict]:
    rows = [{'type': 'I', 'display_frames': 1, 'encoder_padding_frames': 0, 'nal_bytes': intra, 'sps_bytes': 4}]
    rows += [{'type': 'P', 'display_frames': 8, 'encoder_padding_frames': padding, 'nal_bytes': per_group, 'sps_bytes': 0}
             for _ in range(groups)]
    return rows


def test_rate_estimate_is_one_i_frame_plus_mean_p_cost():
    # 1000 B I (+4 sps), 80 B per 8-frame group = 10 B/frame, 240 frames at 25 fps
    expected = (9 + 1004 + 10 * 239) * 8 * 25 / 240 / 1000
    assert an.estimate_rate_kbps(nals(1000, 80), target_frames=240, fps=25) == pytest.approx(expected)
    with pytest.raises(ValueError):
        an.estimate_rate_kbps(nals(1000, 80)[:1], target_frames=240, fps=25)


def candidates(rates: dict[tuple[int, int], float]) -> list[dict]:
    return [{'scale': s, 'qp': q, 'est_kbps': r} for (s, q), r in rates.items()]


def test_selection_is_by_log_rate_distinct_and_ordered():
    grid = candidates({(8, 0): 6, (8, 63): 14, (4, 0): 18, (4, 36): 31, (4, 63): 58, (2, 63): 61})
    three = an.select_points(grid, 3)
    assert [(p['scale'], p['qp']) for p in three] == [(8, 63), (4, 36), (2, 63)]
    assert [p['target_kbps'] for p in three] == [15.0, 30.0, 60.0]
    assert [(p['scale'], p['qp']) for p in an.select_points(grid, 1)] == [(4, 36)]
    assert an.bracket_ok(three)


def test_selection_never_reuses_a_candidate_and_ties_prefer_resolution():
    grid = candidates({(4, 5): 30.0, (2, 5): 30.0, (8, 5): 30.0})
    picks = an.select_points(grid, 3)
    assert len({(p['scale'], p['qp']) for p in picks}) == 3
    assert [p['scale'] for p in picks if p['target_kbps'] == 30.0] == [2]
    with pytest.raises(ValueError):
        an.select_points(grid[:2], 3)
    with pytest.raises(ValueError):
        an.select_points(grid, 2)


def test_bracket_requires_both_ends():
    assert not an.bracket_ok([{'est_kbps': 30.0}, {'est_kbps': 60.0}])
    assert not an.bracket_ok([{'est_kbps': 12.0}, {'est_kbps': 30.0}])


def test_frame_diffs_and_exactness(tmp_path: Path):
    a, b = tmp_path / 'a.yuv', tmp_path / 'b.yuv'
    left = np.arange(30, dtype=np.uint8)
    right = left.copy()
    right[12] += 1
    right[25] += 3
    a.write_bytes(left.tobytes())
    b.write_bytes(right.tobytes())
    diffs = an.frame_max_diffs(a, b, 10, 3)
    assert diffs == [0, 1, 3]
    tolerant = an.exactness(diffs)
    assert (tolerant['passed'], tolerant['first_frame_differing'], tolerant['first_frame_beyond_tolerance']) == (False, 1, 2)
    assert an.exactness(diffs[:2])['passed']
    strict = an.exactness(diffs[:2], levels=0)
    assert not strict['passed'] and strict['first_frame_beyond_tolerance'] == 1
    with pytest.raises(ValueError):
        an.frame_max_diffs(a, b, 7, 3)


def test_luma_psnr_per_frame(tmp_path: Path):
    a, b = tmp_path / 'a.yuv', tmp_path / 'b.yuv'
    frame = np.zeros(6, np.uint8)  # 2x2 luma + 2 chroma bytes
    noisy = frame.copy()
    noisy[0] = 16
    a.write_bytes(np.concatenate([frame, frame]).tobytes())
    b.write_bytes(np.concatenate([frame, noisy]).tobytes())
    values = an.luma_psnr_per_frame(a, b, 2, 2, 2)
    assert values[0] == 99.0
    assert values[1] == pytest.approx(10 * math.log10(255 ** 2 / (256 / 4)))


def curve(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points


def test_envelope_bd_rate_and_dominance():
    svt = {m: curve([(10, 0.5), (17, 0.4), (34, 0.3)]) for m in an.METRICS}
    cheaper = {m: curve([(8, 0.5), (14, 0.4), (28, 0.3)]) for m in an.METRICS}
    result = an.envelope_comparison(cheaper, svt)
    for metric in an.METRICS:
        assert result[metric]['bd_rate_percent'] < 0
        assert result[metric]['rate_dominance']['verdict'] == 'better at every common rate'
    # all distances far below the anchor's: no quality overlap, rate dominance still defined
    far = {m: curve([(12, 0.2), (20, 0.15), (40, 0.1)]) for m in an.METRICS}
    result = an.envelope_comparison(far, svt)
    assert result['full_lpips']['bd_rate_percent'] is None
    assert result['full_lpips']['rate_dominance']['verdict'] == 'better at every common rate'
    worse = {m: curve([(12, 0.9), (20, 0.8), (40, 0.7)]) for m in an.METRICS}
    assert an.envelope_comparison(worse, svt)['full_dists']['rate_dominance']['verdict'] == 'worse at every common rate'
    assert an.rate_dominance([(100, -0.2), (200, -0.1)], [(10, -0.5), (34, -0.3)])['verdict'] is None


def test_scale_list_and_projection():
    assert parse_scales('8,4,2') == [8, 4, 2]
    with pytest.raises(ValueError):
        parse_scales('1')
    timings = dict(setup=10, calibration=20, native_fixed=5, native_variable=1, svt_scale8=2, masks=3, scoring=1)
    expected = 1.35 * (10 + 20 + 15 + 3 * 9.6 * 1 + 42 + 80 / 9 * 3 + 3 * 80 / 9 * 1 + 4 * 6)
    assert project_full_seconds(timings, 6) == pytest.approx(expected)
    with pytest.raises(ValueError):
        project_full_seconds({**timings, 'scoring': -1}, 6)


def test_run_action_sets_cublas_workspace_and_reports_warnings(monkeypatch):
    import warnings

    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)

    def fake(plan):
        warnings.warn('nondeterministic op used', UserWarning)
        return {'operation': 'calibrate'}

    monkeypatch.setattr(worker, 'calibrate', fake)
    payload = worker.run_action('calibrate', {'deterministic': True})
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert payload['warnings'] == ['nondeterministic op used']
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    worker.run_action('calibrate', {})
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    json.dumps(payload)


def fake_worker(flip_second_decoder: bool):
    """A stand-in for run_worker that writes plausible files; exercises code_point's bookkeeping only."""
    import hashlib

    frame = 240 * 136 * 3 // 2
    pixels = np.random.default_rng(0).integers(0, 256, size=2 * frame, dtype=np.uint8)

    def run_worker(action, plan, work, tag, timeout):
        if action == 'encode':
            Path(plan['container']).write_bytes(b'PSDC' + bytes(40))
            Path(plan['i_recon_file']).write_bytes(pixels[:frame].tobytes())
            return {'container_sha256': hashlib.sha256(Path(plan['container']).read_bytes()).hexdigest(),
                    'container_bytes': 44, 'encode_seconds': 1.0, 'dpb_reachable': False, 'nals': [],
                    'environment': {'extension': {'variant': 'sm89'}}, 'warnings': [],
                    'checkpoints': {'stream_policy': 'legacy', 'determinism': {'final': {
                        'deterministic_algorithms': True, 'cudnn_deterministic': True, 'cudnn_benchmark': False}}}}, 2.0
        out = pixels.copy()
        if flip_second_decoder and tag == 'decode-b':
            out[frame + 5] ^= 1
        Path(plan['out_file']).write_bytes(out.tobytes())
        hashes = ['x' if (flip_second_decoder and tag == 'decode-b' and i == 1) else f'h{i}' for i in range(2)]
        return {'deterministic': True, 'container_sha256': 'c', 'environment': {}, 'warnings': [],
                'passes': [{'decode_seconds': 1.0, 'model_decode_seconds': 0.5, 'frame_sha256': hashes}] * 2}, 3.0

    return run_worker


@pytest.mark.parametrize('flip', [False, True])
def test_code_point_accepts_identical_decoders_and_rejects_a_difference(tmp_path, monkeypatch, flip):
    from argparse import Namespace

    from experiments.tennis import w3_neural_baseline as job

    source = tmp_path / 'source.yuv'
    source.write_bytes(np.random.default_rng(1).integers(0, 256, size=2 * 1920 * 1620, dtype=np.uint8).tobytes())
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(job, 'run_worker', fake_worker(flip))
    args = Namespace(image_ckpt=tmp_path / 'i', video_ckpt=tmp_path / 'v')
    point = {'scale': 8, 'qp': 36, 'target_kbps': 30.0, 'est_kbps': 31.0}
    record = job.code_point(args, source, work, 25, 2, point)
    assert record['accepted'] is (not flip)
    assert record['check_a_i_frame']['passed']
    assert record['check_b_two_processes']['max_abs_diff'] == (1 if flip else 0)
    assert record['kbps'] == pytest.approx(44 * 8 * 25 / 2 / 1000)
    json.dumps(record)


def test_score_stream_uses_whole_frame_and_the_foreground_crop(tmp_path, monkeypatch):
    from experiments.tennis import w3_neural_baseline as job
    from experiments.visor import h3b

    width, height, n = 240, 136, 4
    decoded = tmp_path / 'decoded.yuv'
    decoded.write_bytes(np.random.default_rng(2).integers(0, 256, size=n * width * height * 3 // 2, dtype=np.uint8).tobytes())
    truth = np.zeros((n, 1080, 1920, 3), np.uint8)
    foreground = np.zeros((1080, 1920), bool)
    foreground[500:560, 900:1000] = True
    seen = {}
    monkeypatch.setattr(job, 'decoded_rgb', lambda frame, w, h: np.zeros((h, w, 3), np.uint8))
    monkeypatch.setattr(h3b, 'lpips_reduce', lambda net, frames, true, masks, device: [[0.3, 0.4]])

    def dists(net, crops, true, mask, device):
        seen.setdefault('shapes', []).append((crops[0].shape, true.shape, mask.shape))
        return [0.2 if mask.all() else 0.25]

    monkeypatch.setattr(h3b, 'dists_crops', dists)
    scores = job.score_stream({'lpips': None, 'dists': None}, decoded, [width, height], n, [0, 3], truth, {0: foreground, 3: foreground})
    assert scores == {'full_lpips': [0.3, 0.3], 'foreground_lpips': [0.4, 0.4], 'full_dists': [0.2, 0.2], 'foreground_dists': [0.25, 0.25]}
    whole, crop = seen['shapes'][0], seen['shapes'][1]
    assert whole[0] == (1080, 1920, 3)
    assert crop[0][:2] == crop[1][:2] == crop[2] == (60 + 128, 100 + 128)


def test_audited_controls_match_and_mismatch(tmp_path):
    from experiments.tennis import w3_neural_baseline as job

    prior = {m: np.tile(np.array([[0.1], [0.0], [0.2], [0.0], [0.3], [0.0]]), (1, 80)) + k for k, m in enumerate(an.METRICS)}
    ctx = {'prior_scores': prior}

    def result(rate, scores):
        return {'svt': [{'scale': 8, 'kbps': rate, 'scores': {m: [prior[m][0][0] + shift for _ in range(9)] for m, shift in scores.items()}}]}

    ok = job.audited_controls('al', result(10.1975, {m: 0.0 for m in an.METRICS}), ctx, list(range(0, 25, 3)))
    assert ok['scales']['8']['bytes_match'] and ok['scales']['8']['scores_match']
    with pytest.raises(ValueError):
        job.audited_controls('al', result(10.1975, {m: 0.01 for m in an.METRICS}), ctx, list(range(0, 25, 3)))
    other = job.audited_controls('al', result(11.0, {m: 0.01 for m in an.METRICS}), ctx, list(range(0, 25, 3)))
    assert not other['scales']['8']['bytes_match'] and not other['scales']['8']['scores_match']
    assert job.audited_controls('med', {}, ctx, [])['applies'] is False


def test_foreground_masks_control_flow_with_stand_ins(tmp_path, monkeypatch):
    import cv2

    from experiments.tennis import k6, k6b
    from experiments.tennis import w3_neural_baseline as job
    from experiments.tennis.k6b_rate_transport import array_sha, sha
    from src.segmentation.masks import compress

    wid = 'alcaraz_hurkacz/031@129'
    ticks = [0, 3]
    class Truth:  # a 240-frame RGB array without holding 1.5 GB
        def __getitem__(self, t):
            return np.full((1080, 1920, 3), t % 256, np.uint8)

    truth = Truth()
    matte = np.zeros((240, 1, 54, 96), bool)
    matte[:, 0, 20:30, 40:50] = True
    drawn = np.zeros((1080, 1920), bool)
    drawn[100:120, 100:120] = True
    scoring = {}
    for t in ticks:
        r = cv2.resize(matte[t, 0].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        scoring[str(t)] = {'r': sha(np.packbits(r).tobytes()), 's': sha(np.packbits(r | drawn).tobytes())}
    row = {'judged': list(range(0, 240, 3)), 'players': ['p1'], 'fps': 25, 'scored': list(range(240)), 'scoring_masks': scoring}
    anchor = tmp_path / 'anchor'
    anchor.mkdir()
    (anchor / 'keypoints.json.zst').write_bytes(compress(json.dumps({'decoded': {'code': {'p1@0': [[1.0, 2.0]]}}}).encode()))
    wire = tmp_path / 'wire'
    (wire / 'packets' / wid.replace('/', '__')).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(k6b, 'k6_window', lambda directory: {'row': row, 'poses': {}, 'dir': directory})
    monkeypatch.setattr(k6b, 'k6_oracle', lambda t, *rest: calls.append(t) or ({}, {'variant': drawn}))
    monkeypatch.setattr('experiments.tennis.k6b_rate_streams.decode_keypoints', lambda blob: ({'compositing_order': ['p1']}, []))
    monkeypatch.setattr('experiments.tennis.k6b_rate_streams.decode_masks', lambda blob: ({}, matte))
    for name in ('keypoints.kps', 'exact-rle.matte'):
        (wire / 'packets' / wid.replace('/', '__') / name).write_bytes(b'x')
    prior_rows = {wid: {'source_rgb_frame_sha256': {str(t): array_sha(truth[t]) for t in ticks}}}
    masks = job.foreground_masks(wid, {wid: anchor}, wire, truth, ticks, prior_rows)
    assert calls == ticks and set(masks) == set(ticks) and masks[0][110, 110] and masks[0][1000, 1000] is np.False_
    prior_rows[wid]['source_rgb_frame_sha256']['3'] = 'bad'
    with pytest.raises(ValueError):
        job.foreground_masks(wid, {wid: anchor}, wire, truth, ticks, prior_rows)
    assert k6.DISTS_PAD > 0


def synthetic_summary(tmp_path, monkeypatch):
    """A smoke summary whose DCVC records come from code_point and a stand-in worker."""
    from argparse import Namespace

    from experiments.tennis import w3_neural_baseline as job

    source = tmp_path / 'source.yuv'
    source.write_bytes(np.random.default_rng(1).integers(0, 256, size=2 * 1920 * 1620, dtype=np.uint8).tobytes())
    monkeypatch.setattr(job, 'run_worker', fake_worker(False))
    scores = {m: [0.1 + 0.01 * i for i in range(9)] for m in an.METRICS}
    windows = []
    for key in ('al', 'med'):
        work = tmp_path / ('work-' + key)
        work.mkdir()
        point = {'scale': 8, 'qp': 36, 'target_kbps': 30.0, 'est_kbps': 31.0}
        record = job.code_point(Namespace(image_ckpt=tmp_path / 'i', video_ckpt=tmp_path / 'v'), source, work, 25, 2, point)
        record['encode_report']['container_bytes'] = record['bytes']
        record['scores'], record['mean'] = scores, {m: 0.1 for m in an.METRICS}
        record.pop('decoded')
        scale8 = {'scale': 8, 'scores': scores, 'kbps': 10.1975}
        candidates = [{'scale': s, 'qp': q, 'est_kbps': 20.0} for s in job.SCALES for q in job.CAL_QPS]
        windows.append({'window': key, 'accepted': True, 'candidates': candidates, 'chosen': [point], 'bracket_ok': True,
                        'dcvc': [record], 'svt': [scale8], 'timings': {'convert': 5, 'calibration': 20, 'native_fixed': 8,
                        'native_variable': 2, 'svt_scale8': 3, 'masks': 4, 'scoring': 2},
                        'audited_controls': {'applies': key == 'al', 'scales': {'8': {'score_max_error': {'full_lpips': 0.0}}}}})
    return {'frames': 25, 'points': 1, 'gpu_class': 'RTX 6000 Ada', 'gpu_uuid': 'GPU-abc', 'accepted': True, 'windows': windows,
            'setup_seconds': 60.0}


def run_validator(tmp_path, monkeypatch, summary):
    import tarfile

    from experiments.tennis import w3_neural_baseline as job

    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    summary = json.loads(json.dumps(summary))
    (stage / 'w3-summary.json').write_text(json.dumps(summary))
    (stage / 'staging.json').write_text(json.dumps({'publication_seconds': 5.0}))
    (stage.parent / 'spec.json').write_text(Path('docs/experiments/w3-specs/w3.json').read_text())
    inner = tmp_path / 'summary.json'
    inner.write_text(json.dumps(summary))
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        tar.add(inner, arcname='publish/summary.json')
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    job.validate()
    return json.loads((tmp_path / 'validation.json').read_text())


def test_validator_passes_a_good_smoke(tmp_path, monkeypatch):
    result = run_validator(tmp_path, monkeypatch, synthetic_summary(tmp_path, monkeypatch))
    assert result['passed'] and result['projected_full_seconds'] < 2400


def test_validator_catches_a_decoder_difference(tmp_path, monkeypatch):
    bad = synthetic_summary(tmp_path, monkeypatch)
    bad['windows'][1]['dcvc'][0]['identical_frame_hashes'] = False
    with pytest.raises(SystemExit):
        run_validator(tmp_path, monkeypatch, bad)
    checks = json.loads((tmp_path / 'validation.json').read_text())['checks']
    assert [k for k, v in checks.items() if not v] == ['decoders_bit_identical']
