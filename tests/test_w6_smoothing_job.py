import numpy as np

from experiments.tennis.w6_smoothing_job import profile_step


def test_profiles_use_one_flow_previous_raw_and_reset_gaps(monkeypatch):
    from experiments.visor import h3

    observed = []
    def flow(target, source):
        observed.append((int(target[0, 0]), int(source[0, 0])))
        return np.zeros((*target.shape, 2), np.float32)
    monkeypatch.setattr(h3, 'dis_flow', flow)
    first = np.full((64, 64, 3), 20, np.uint8)
    second = np.full_like(first, 100)
    third = np.full_like(first, 180)
    outputs, previous = profile_step(first, 0, None)
    assert all(np.array_equal(x, first) for x in outputs)
    outputs, previous = profile_step(second, 1, previous)
    assert [int(x[0, 0, 0]) for x in outputs] == [100, 80, 60]
    outputs, previous = profile_step(third, 2, previous)
    assert observed == [(100, 20), (180, 100)]
    assert [int(x[0, 0, 0]) for x in outputs] == [180, 160, 140]
    outputs, previous = profile_step(None, 3, previous)
    assert outputs == [] and previous is None
    outputs, _ = profile_step(first, 4, previous)
    assert all(np.array_equal(x, first) for x in outputs)
    assert len(observed) == 2


def test_frozen_timeline_rejects_missing_crop_mask_or_box():
    import pytest
    from experiments.tennis.w6_smoothing_job import require_complete_timeline

    crops = {'p': {'t': np.arange(240), 'crops': [None] * 240}}
    boxes = [{i: (0, 0, 1, 1) for i in range(240)}]
    masks = {0: {i: True for i in range(240)}}
    require_complete_timeline(['p'], crops, boxes, masks)
    for kind in ('crop', 'box', 'mask'):
        changed_crops = {'p': {'t': crops['p']['t'].copy(), 'crops': crops['p']['crops'][:]}}
        changed_boxes = [boxes[0].copy()]
        changed_masks = {0: masks[0].copy()}
        if kind == 'crop':
            changed_crops['p']['t'] = np.delete(changed_crops['p']['t'], 9)
        elif kind == 'box':
            del changed_boxes[0][9]
        else:
            del changed_masks[0][9]
        with pytest.raises(ValueError, match='complete'):
            require_complete_timeline(['p'], changed_crops, changed_boxes, changed_masks)


def test_full_saved_array_audit_recomputes_decisions_and_rejects_tampering(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import tarfile

    import pytest
    from experiments.tennis import w6, w6_smoothing_job as job

    def archive(path, members):
        with tarfile.open(path, 'w') as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    stage = tmp_path / 'full'
    stage.mkdir()
    tools = tmp_path / 'bin'
    tools.mkdir()
    for name in ('ffmpeg', 'dav1d'):
        (tools / name).write_bytes(name.encode())
    monkeypatch.setattr(job, 'FFMPEG_SHA', sha(b'ffmpeg'))
    monkeypatch.setattr(job, 'DAV1D_SHA', sha(b'dav1d'))
    mask = b'fixed mask'
    monkeypatch.setattr(job, 'MASK_NPZ_SHA', sha(mask))
    masks = tmp_path / 'mask.tar'
    archive(masks, {'publish/masks.npz': mask})
    post = tmp_path / 'post.tar'
    meta = {'target_kbps': 15, 'anchor_rates_kbps': {'svt_low': 10, 'svt_high': 20}}
    archive(post, {'publish/w6-judge.json': json.dumps(meta).encode()})
    generated = tmp_path / 'generated.tar'
    archive(generated, {'publish/p0@match5.npz': b'p0', 'publish/p1@match5.npz': b'p1'})
    paths = {'masks': masks, 'post_archive': post, 'generated': generated}
    monkeypatch.setattr(job, 'INPUTS', {})
    code = {'source_sha256': 'code', 'git_head': 'head', 'snapshot_sha256': 'snapshot', 'spec_sha256': 'spec'}
    (tmp_path / 'ready.json').write_text(json.dumps(code))
    monkeypatch.setattr(job, 'inputs', lambda _: (paths, {}, {'code': code}))
    baseline = np.r_[np.nan, np.ones(239)]
    anchor = np.r_[np.nan, np.full(239, 2.0)]
    saved = {'variants': np.array(['hybrid', 'svt_low', 'svt_high']),
             'flicker': np.stack([baseline, anchor, anchor]), 'energy': np.stack([baseline] * 3),
             'energy_src': baseline}
    monkeypatch.setattr(job.ctrl, 'saved_judges', lambda _: (b'', saved))
    flicker = np.stack([baseline, baseline * 0.8, baseline * 1.1])
    energy = np.stack([baseline, baseline, baseline])
    lpips = np.array([[1.0] * 80, [0.9] * 80, [1.0] * 80])
    dists = np.array([[1.0] * 80, [0.9] * 80, [1.0] * 80])
    out = io.BytesIO()
    np.savez_compressed(out, flicker=flicker, energy=energy, source_energy=baseline, lpips=lpips, dists=dists,
                        ticks=np.arange(0, 240, 3), weights=np.array(job.WEIGHTS_FIXED), anchor_flicker=anchor)
    data = out.getvalue()
    means = {'flicker': [1.0, 0.8, 1.1], 'energy': [1.0] * 3,
             'lpips': [1.0, 0.9, 1.0], 'dists': [1.0, 0.9, 1.0]}
    # Reproduce NumPy's exact sums as the worker does.
    means['flicker'] = np.mean(flicker[:, 1:], axis=1).tolist()
    means['lpips'] = np.mean(lpips, axis=1).tolist()
    means['dists'] = np.mean(dists, axis=1).tolist()
    report = {'frames': 240, 'tick_count': 80, 'inputs': {},
              'baseline': job.ctrl.comparison({'flicker': flicker[0], 'energy': energy[0], 'source_energy': baseline}, saved, 240),
              'source_yuv_sha256': w6.SOURCE_YUV_SHA, 'judges_sha256': job.JUDGES_SHA, 'mask_npz_sha256': sha(mask),
              'ffmpeg_sha256': sha(b'ffmpeg'), 'dav1d_sha256': sha(b'dav1d'), 'code_identity': 'code',
              'weights': list(job.WEIGHTS_FIXED), 'metric_controls': [{'tick': t, 'w1': [0.1, 0.2], 'new': [0.1, 0.2]} for t in (0, 3)],
              'metric_control_max_abs_differences': [[0.0, 0.0], [0.0, 0.0]],
              'crop_tracks': {name: {'frames': 240, 'first': 0, 'last': 239, 'npz_sha256': sha(name.encode())}
                              for name in ('p0', 'p1')},
              'means': means, 'anchor_flicker_mean': 2.0, 'source_energy_mean': 1.0,
              'original_time_rule': [w6.time_check(means['flicker'][k], 2.0, 1.0, 1.0) for k in range(3)],
              'promising': [False, True, False], 'series_sha256': sha(data)}
    def publish():
        archive(stage / 'published.tar', {'publish/smoothing.json': json.dumps(report).encode(), 'publish/series.npz': data})
        (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': job.w6.digest(stage / 'published.tar')},
                                                        'environment': {'bin': str(tools)}}))
    publish()
    result = job.audit_stage(stage, full_external=True)
    assert result['pairs'] == 239 and result['ticks'] == 80 and result['promising'] == [False, True, False]
    report['promising'] = [False, False, False]
    publish()
    with pytest.raises(ValueError, match='decision'):
        job.audit_stage(stage, full_external=True)
