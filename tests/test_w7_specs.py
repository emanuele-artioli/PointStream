"""W7 spec builder: shape of the emitted specs and the recorded per-window GPU estimate."""
import json
from pathlib import Path

from experiments.tennis import w7_specs as sp
from experiments.tennis import window_manifest as wm

SELECTION = Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/selection.json'


def test_native_base_specs():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    manifest = (sp.W7_DIR + '/windows.json', 'a' * 64)
    smoke, full = sp.native_base(windows, manifest, 24), sp.native_base(windows, manifest, 240)
    assert [i['name'] for i in smoke['staged_inputs']] == ['windows', 'al028']
    assert [i['name'] for i in full['staged_inputs']] == ['windows'] + [w.name for w in windows]
    for spec in (smoke, full):
        assert spec['verify_staged_inputs'] is True and spec['hosts'] == ['gpu3', 'gpu5', 'gpu6']
        assert spec['smoke']['seconds'] <= 600 and spec['full']['seconds'] <= 2600
        assert all(i['path'].startswith(sp.DATA + '/') for i in spec['staged_inputs'])
        assert spec['arguments'].count('--window') == spec['arguments'].count('--source')


def test_w1_stage_is_single_window_and_gpu_estimate_recorded():
    w = wm.from_selection(json.loads(SELECTION.read_text()))[0]
    spec = sp.w1_window_stage('experiments.tennis.w1_score', w, ('/x/windows.json', 'b' * 64), ('/x/state.json', 'c' * 64), ['--judged', '{judged}'],
                              {}, gpu=True, full_seconds=2400)
    assert spec['arguments'][:6] == ['--windows-file', '{staged:windows}', '--window-state', '{staged:state}', '--window', 'al028']
    assert spec['gpu_models'] == ['RTX A6000', 'RTX 6000 Ada'] and spec['full']['seconds'] == 2400
    assert sp.per_window_gpu_seconds() == 4654


def test_first_window_specs_for_the_generalised_gpu_stages():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    manifest = (sp.W7_DIR + '/windows.json', 'a' * 64)
    d = sp.rgb_diagnostic(windows, manifest, (sp.W7_DIR + '/b.tar', 'b' * 64), 24)
    assert d['gpu_models'] == ['RTX A6000'] and d['arguments'].count('--window') == 1 and d['smoke']['seconds'] <= 600
    t = sp.rate_transport(windows, manifest, [(sp.W7_DIR + '/a.tar', 'c' * 64)], [(sp.W7_DIR + '/p.tar', 'd' * 64)], 1)
    assert t['device'] == 'cpu' and t['arguments'].count('--window') == 1 and t['verify_staged_inputs'] is True
    files = {k: (sp.W7_DIR + f'/{k}', 'e' * 64) for k in ('windows', 'anchor', 'transport', 'masks', 'gpu_base', 'base', 'prior', 'lpips', 'dists')}
    j = sp.real_base_judge(windows, files, [(sp.W7_DIR + '/g0.tar', 'f' * 64)], (sp.W7_DIR + '/rates.json', '1' * 64), 3)
    assert j['arguments'].count('--gen') == 1 and '--no-early-stop' in j['arguments'] and j['scale']['judged']['smoke'] == 3
    full = sp.real_base_judge(windows, files, [(sp.W7_DIR + f'/g{i}.tar', 'f' * 64) for i in range(6)], (sp.W7_DIR + '/rates.json', '1' * 64), 0)
    assert full['arguments'].count('--gen') == 6 and full['full']['seconds'] <= 2600


def test_budget_check_against_the_raised_tier1_budget():
    b = sp.budget_check()
    assert b['per_window_s'] == 4654 and b['first_ok'] and b['total_ok']  # 4,654 s <= 4,680 s; 27,924 s <= 28,800 s
    assert b['margin_total_s'] == 28800 - 6 * 4654
    assert not sp.budget_check(per_window=5000)['total_ok']


def test_first_window_stage_specs_all_stages():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    first = windows[0]
    man, src, pend = (sp.W7_DIR + '/manifest.json', 'a' * 64), (sp.W7_DIR + '/sources/alcaraz_hurkacz__028.mp4', first.source_sha256), (sp.W7_DIR + '/x.tar', sp.PENDING)
    units = ['alcaraz_hurkacz__028_at120__person_000000000000', 'alcaraz_hurkacz__028_at120__person_000000000001']
    specs = {'features': sp.features(first, man, src), 'anchors': sp.anchors(first, man, pend, src), 'prepare': sp.prepare(first, man, pend, pend, src),
             'generate': sp.generate(first, man, pend, units),
             'receiver': sp.receiver(windows, (sp.W7_DIR + '/windows.json', 'b' * 64), pend, pend, [pend], 2),
             'dcvc': sp.dcvc(windows, (sp.W7_DIR + '/windows.json', 'b' * 64), pend, pend, pend, 25)}
    for name, spec in specs.items():
        assert spec['verify_staged_inputs'] is True and spec['smoke']['seconds'] <= 600 and spec['full']['seconds'] <= 2600, name
        assert set(spec['hosts']) >= {'gpu3', 'gpu5', 'gpu6'} or name in ('rgb_diagnostic', 'dcvc')
        placeholders = {a[len('{staged:'):-1] for a in spec['arguments'] if a.startswith('{staged:')}
        assert placeholders <= {i['name'] for i in spec['staged_inputs']}, (name, placeholders)
        assert len({i['name'] for i in spec['staged_inputs']}) == len(spec['staged_inputs'])
    assert specs['generate']['environment'] == sp.JCV_ENVIRONMENT and '--fit' in specs['generate']['arguments']
    assert specs['generate']['arguments'][specs['generate']['arguments'].index('--fit') + 1:][:2] == ['match5', 'ah-C@fit-100-fp16.pt']  # two argv items, as the 2026-10-24 renderer job
    assert set(specs['receiver']['gpu_models']) == {'RTX 6000 Ada', 'RTX A6000'}  # shared, as the 2026-10-26 receiver job
    assert specs['dcvc']['scale']['frames'] == {'smoke': 25, 'full': 240}


def test_receiver_denied_probe_count_follows_the_window_set():
    from experiments.tennis.k6b_receiver_sam import denied_probe_count

    assert denied_probe_count({}) == 4  # frozen pair: two sources, control, transport
    assert denied_probe_count({'windows': 1}) == 2  # one manifest window: its source and the transport


def test_w7_rates_cli_writes_rates_and_state_and_validates(tmp_path, monkeypatch):
    import io
    import tarfile

    from experiments.tennis import w7_rates

    inner = tmp_path / 'bundle.tar'
    manifest = json.dumps({'rows': [{'window': 'al028', 'scale': s, 'ivf': {'bytes': 1000 * s}} for s in (8, 4, 2)]}).encode()
    with tarfile.open(inner, 'w') as t:
        info = tarfile.TarInfo('base/manifest.json')
        info.size = len(manifest)
        t.addfile(info, io.BytesIO(manifest))
    outer = tmp_path / 'published.tar'
    with tarfile.open(outer, 'w') as t:
        t.add(inner, arcname='publish/bundle.tar')
    assert [r['scale'] for r in w7_rates.base_rows(outer)] == [8, 4, 2]


def test_timing_sequences_for_manifest_windows_use_the_base_trace_only():
    from experiments.tennis.k6b_real_base_cost import separated_projection, timing_sequences

    row = {'id': 'alcaraz_hurkacz/028@120', 'ticks': [0, 3, 6], 'frame_seconds': {'0': 3.0, '3': 2.0, '6': 2.1}}
    seq = timing_sequences({'rows': [row]}, ['alcaraz_hurkacz/028@120'])
    assert list(seq) == ['alcaraz_hurkacz/028@120/base']
    projection = separated_projection(60.0, seq, 5.0, expected_sequences=1)
    assert projection['projected_full_seconds'] > projection['projected_worker_seconds'] > 0


def test_packet_file_finds_both_transport_layouts(tmp_path):
    import tarfile

    from experiments.tennis import w1_roi

    w = wm.from_selection(json.loads(SELECTION.read_text()))[0]
    window = w1_roi.window_from_spec(w, {'control_ivf_bytes': {'8': 1, '4': 2, '2': 3}, 'start_base': {'4': 60, '2': 67, '1': 70}, 'kps1_bytes': 1, 'kps2_bytes': 1})
    for prefix in ('', 'publish/'):
        path = tmp_path / (prefix.strip('/') or 'plain')
        path.mkdir()
        member = path / 'x'
        member.write_bytes(b'abc')
        archive = tmp_path / f'{prefix.strip("/") or "plain"}.tar'
        with tarfile.open(archive, 'w') as tar:
            tar.add(member, arcname=prefix + window.packet_dir + 'exact-rle.matte')
        with tarfile.open(archive) as tar:
            stream = w1_roi.packet_file(tar, window, 'exact-rle.matte')
            assert stream is not None and stream.read() == b'abc'
            assert w1_roi.packet_file(tar, window, 'keypoints.kps') is None


def test_dcvc_spec_meets_the_w3_validator_caps():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    pend = (sp.W7_DIR + '/x.tar', sp.PENDING)
    spec = sp.dcvc(windows, (sp.W7_DIR + '/windows.json', 'b' * 64), pend, pend, pend, 25)
    assert spec['budget_seconds'] <= 3300 and spec['hosts'] == ['gpu5', 'gpu6'] and spec['gpu_models'] == ['RTX 6000 Ada']


def test_dcvc_spec_modes_calibrated_named_and_bracketing():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    pend = (sp.W7_DIR + '/x.tar', sp.PENDING)
    man = (sp.W7_DIR + '/windows.json', 'b' * 64)
    calibrated = sp.dcvc(windows, man, pend, pend, pend, 25)
    assert '--points' in calibrated['arguments'] and 'points' in calibrated['scale']
    named = sp.dcvc(windows, man, pend, pend, pend, 25, ('al028:2:24', 'al028:2:24,al028:2:30'))
    assert '--point-list' in named['arguments'] and named['scale']['svt_scales']['full'] == '8'
    bracket = sp.dcvc(windows, man, pend, pend, pend, 25, None, (18.46, 19.68))
    i = bracket['arguments'].index('--bracket')
    assert bracket['arguments'][i + 1:i + 3] == ['18.46', '19.68'] and '--points' not in bracket['arguments'] and 'points' not in bracket['scale']


# ----------------------------------------------------------------------------------------------- batch mode (2026-10-31)

import hashlib  # noqa: E402

import pytest  # noqa: E402

from experiments.tennis import w7_batch as wb  # noqa: E402

WINDOWS_JSON = Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/windows.json'
FIXTURE = Path(__file__).resolve().parent / 'fixtures/w7_first_window.sha256.json'


def _a(name, k):
    return f'{name}={sp.W7_DIR}/{name}.tar:{k * 64}'


#: arguments of `first-window` that produced tests/fixtures/w7_first_window.sha256.json before the builders were generalised
FIRST_WINDOW_CASES = {
    'features': [_a('manifest', 'a')], 'build-inputs': [_a('manifest', 'a'), _a('features', 'b')], 'native-base': [],
    'anchors': [_a('manifest', 'a'), _a('inputs', 'c')], 'prepare': [_a('manifest', 'a'), _a('inputs', 'c'), _a('anchor', 'd')],
    'generate': [_a('manifest', 'a'), _a('prepared', 'e'), 'units=u1,u2:x'], 'rate-transport': [_a('anchor', 'd'), _a('prepared', 'e')],
    'rgb-diagnostic': [_a('base', 'f')], 'receiver': [_a('transport', '1'), _a('anchor', 'd'), _a('generated', '2')],
    'rates': [_a('transport_json', '3'), _a('base', 'f')],
    'judge': [_a('anchor', 'd'), _a('transport', '1'), _a('masks', '4'), _a('gpu_base', '5'), _a('base', 'f'), _a('prior', '6'), _a('generated', '2'), _a('rates', '7')],
    'w1-masks': [_a('state', '8'), _a('anchor', 'd'), _a('transport', '1')],
    'w1-encode-half': [_a('state', '8'), _a('transport', '1'), 'target=18.46:x'], 'w1-encode-quarter': [_a('state', '8'), _a('transport', '1'), 'target=18.46:x'],
    'w1-encode-half-out': [_a('state', '8'), _a('transport', '1'), 'target=18.46:x'], 'w1-encode-quarter-out': [_a('state', '8'), _a('transport', '1'), 'target=18.46:x'],
    'w1-score': [_a('state', '8'), _a('masks', '4'), _a('hybrid', '9'), _a('encode_half', '0'), _a('encode_x0', '1'), _a('encode_x1', '2')],
    'dcvc': [_a('anchor', 'd'), _a('transport', '1'), _a('prior_frame', '6')],
    'dcvc-bracket': [_a('anchor', 'd'), _a('transport', '1'), _a('prior_frame', '6'), 'low=18.46:x', 'high=19.68:x'],
    'dcvc-points': [_a('anchor', 'd'), _a('transport', '1'), _a('prior_frame', '6'), 'smoke_points=al028:2:24:x', 'full_points=al028:2:24,al028:2:30:x'],
}


def test_first_window_specs_are_byte_identical_to_before_the_batch_builders(tmp_path):
    expected = json.loads(FIXTURE.read_text())
    assert set(expected) == set(FIRST_WINDOW_CASES)
    for name, ups in FIRST_WINDOW_CASES.items():
        stage = 'dcvc' if name.startswith('dcvc') else name
        out = tmp_path / name
        sp.first_window([stage, str(WINDOWS_JSON), str(out), *ups])
        assert hashlib.sha256((out / f'{stage}.json').read_bytes()).hexdigest() == expected[name], name


def _campaign():
    return wm.load(WINDOWS_JSON)[1:]


def test_plan_groups_respect_every_cap_and_nest():
    ws = _campaign()
    assert [w.name for w in ws] == ['al034', 'al043', 'med000', 'med024', 'med019']
    groups = wb.plan_groups(ws)
    for stage, gs in groups.items():  # every window exactly once per stage
        assert sorted(w.name for g in gs for w in g) == sorted(w.name for w in ws), stage
    assert len(groups['features']) == 1 and len(groups['generate']) == 5 and len(groups['dcvc']) == 5
    for g in groups['anchors']:
        assert wb.wall('anchors', g) <= 0.9 * sp.FULL_SECONDS
    for g in groups['native-base']:
        assert sum(wb.NATIVE_SMOKE_S[w.fps] for w in g) < 1300 / 37.5 + 1e-9
    for g in groups['receiver']:
        assert sum(wb.RECEIVER_PROJECTION_S * wb.factor(w) for w in g) < 1500
    # a judge group (= receiver group) sits inside one anchor job and one native-base job: those stages take a single archive
    for g in groups['judge']:
        for upstream in ('anchors', 'native-base'):
            assert any({w.name for w in g} <= {w.name for w in u} for u in groups[upstream]), (upstream, [w.name for w in g])


def test_plan_totals_and_the_saving_over_one_job_per_window():
    result = wb.plan(_campaign())
    assert result['saved_gpu_s'] > 0 and result['total_gpu_s'] < result['unbatched_gpu_s']
    assert result['total_gpu_s'] == pytest.approx(sum(j['gpu_s'] for j in result['jobs']))
    plain = wb.plan(_campaign(), 1.0)
    assert plain['total_gpu_s'] < result['total_gpu_s']
    first = {j['stage']: j for j in result['jobs'] if j['index'] == 0}
    assert first['features']['smoke_s'] == 163 and first['generate']['smoke_s'] == 92 and first['dcvc']['smoke_s'] == 66
    assert 'GPU-h' in wb.format_plan(result, plain)


def _batch(tmp_path, stage, names, *ups, extra=()):
    target = wb.batch([stage, str(WINDOWS_JSON), str(tmp_path), '--windows', ','.join(names), '--pending', *extra, *ups])
    return json.loads(target.read_text())


def _names(spec):
    return {i['name'] for i in spec['staged_inputs']}


def _check_common(spec, tmp_name):
    assert spec['verify_staged_inputs'] is True
    assert spec['smoke']['seconds'] <= sp.SMOKE_SECONDS and spec['full']['seconds'] <= sp.FULL_SECONDS, tmp_name
    assert spec['budget_seconds'] >= spec['full']['seconds']
    placeholders = {a[len('{staged:'):-1] for a in spec['arguments'] if a.startswith('{staged:')}
    assert placeholders <= _names(spec), (tmp_name, placeholders - _names(spec))
    assert len(_names(spec)) == len(spec['staged_inputs'])
    assert all(i['path'].startswith(sp.DATA + '/') for i in spec['staged_inputs'])


def _flag_values(spec, flag):
    args = spec['arguments']
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_confirm_family_batches_list_every_window_and_smoke_keeps_the_first(tmp_path):
    groups = wb.plan_groups(_campaign())
    inputs = f'inputs={sp.W7_DIR}/in.tar:{"c" * 64}'
    for stage, upstream in (('features', ()), ('anchors', (inputs,)), ('prepare', (inputs,))):
        group = max(groups[stage], key=len)
        ids, names = [w.id for w in group], [w.name for w in group]
        assert len(group) > 1, stage
        spec = _batch(tmp_path, stage, names, f'manifest={sp.W7_DIR}/manifest.json:{"a" * 64}', *upstream)
        _check_common(spec, stage)
        args = spec['arguments']
        i = args.index('--windows')
        assert args[i + 1:i + 1 + len(ids)] == ids, stage  # every window listed, in order
        j = args.index('--sources')
        assert args[j + 1:j + 1 + len(ids)] == ['{staged:c%d}' % k for k in range(len(ids))]
        sources = [x for x in spec['staged_inputs'] if x['name'].startswith('c')]
        assert [x['path'].rsplit('/', 1)[1] for x in sources] == [w.source_file for w in group]  # one staged source per window
        assert [x['sha256'] for x in sources] == [w.source_sha256 for w in group]
        assert '--max-windows' in args and args[args.index('--max-windows') + 1] == '{windows}'
        assert spec['scale']['windows'] == {'smoke': 1, 'full': 0}, stage  # smoke = first listed window, full = all
        assert 'only the first named window' in spec['smoke']['representative_basis']
    assert [len(g) for g in groups['features']] == [5]  # the five remaining windows in one features job


def test_build_inputs_takes_every_feature_archive_and_the_validator_names_them(tmp_path):
    spec = _batch(tmp_path, 'build-inputs', ['al034', 'med000'], f'manifest={sp.W7_DIR}/manifest.json:{"a" * 64}')
    _check_common(spec, 'build-inputs')
    assert _flag_values(spec, '--feature-archive') == ['{staged:f0}'] and spec['scale']['windows'] == {'smoke': 1, 'full': 0}
    two = wb.batch(['build-inputs', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034,med000', f'manifest={sp.W7_DIR}/manifest.json:{"a" * 64}',
                    f'features0={sp.W7_DIR}/f0.tar:{"1" * 64}', f'features1={sp.W7_DIR}/f1.tar:{"2" * 64}'])
    assert two  # one archive covers the plan's single features job, so features0 is the one taken
    assert json.loads(two.read_text())['staged_inputs'][1]['sha256'] == '1' * 64


def test_prepare_passes_each_anchor_archive_as_its_own_k6_stage(tmp_path):
    ws = _campaign()
    spec = _batch(tmp_path, 'prepare', [w.name for w in ws], f'manifest={sp.W7_DIR}/manifest.json:{"a" * 64}', f'inputs={sp.W7_DIR}/i.tar:{"c" * 64}')
    groups = wb.plan_groups(ws)['anchors']
    assert _flag_values(spec, '--k6-stage') == ['{staged:a%d}' % k for k in range(len(groups))] * 2  # runner part and native part
    assert {'a0', 'a1'} <= _names(spec) if len(groups) > 1 else 'a0' in _names(spec)


def test_window_list_stages_cover_the_whole_group_and_name_their_limits(tmp_path):
    ws = _campaign()
    groups = wb.plan_groups(ws)
    names = [w.name for w in groups['native-base'][0]]
    native = _batch(tmp_path, 'native-base', names)
    _check_common(native, 'native-base')
    assert _flag_values(native, '--window') == names and len(_flag_values(native, '--source')) == len(names)
    assert native['scale']['frames'] == {'smoke': 24, 'full': 240} and native['device'] == 'cpu'
    rgb = _batch(tmp_path, 'rgb-diagnostic', names)
    _check_common(rgb, 'rgb')
    assert _flag_values(rgb, '--window') == names and rgb['gpu_models'] == ['RTX A6000'] and rgb['gpu_sharing']['mode'] == 'shared'
    assert rgb['full']['seconds'] == sp.FULL_SECONDS if len(names) > 1 else True
    if len(names) > 1:
        assert 'no first-window limiter' in rgb['smoke']['representative_basis']
    rates = _batch(tmp_path, 'rates', names)
    assert _flag_values(rates, '--window') == names and rates['scale']['windows'] == {'smoke': len(names), 'full': len(names)}
    transport = _batch(tmp_path, 'rate-transport', [w.name for w in ws])
    _check_common(transport, 'rate-transport')
    assert _flag_values(transport, '--window') == [w.name for w in ws]
    assert _flag_values(transport, '--anchor') == ['{staged:a%d}' % k for k in range(len(groups['anchors']))]


def test_receiver_and_judge_batches_stage_one_source_and_one_generation_per_window(tmp_path):
    ws = _campaign()
    groups = wb.plan_groups(ws)
    group = max(groups['receiver'], key=len)
    names = [w.name for w in group]
    receiver = _batch(tmp_path, 'receiver', names)
    _check_common(receiver, 'receiver')
    assert _flag_values(receiver, '--window') == names and _flag_values(receiver, '--source') == ['{staged:%s}' % n for n in names]
    assert _flag_values(receiver, '--generated') == ['{staged:g%d}' % k for k in range(len(names))]
    assert _flag_values(receiver, '--max-judged') == ['{judged}'] and receiver['scale']['judged'] == {'smoke': 2, 'full': 0}
    assert {x['name'] for x in receiver['staged_inputs']} >= {f'g{k}' for k in range(len(names))}
    judge = _batch(tmp_path, 'judge', names)
    _check_common(judge, 'judge')
    assert _flag_values(judge, '--window') == names and _flag_values(judge, '--gen') == ['{staged:g%d}' % k for k in range(len(names))]
    assert judge['scale']['judged'] == {'smoke': 3, 'full': 0} and '--no-early-stop' in judge['arguments']


def test_judge_refuses_windows_that_span_two_anchor_archives(tmp_path):
    groups = wb.plan_groups(_campaign())
    spanning = [groups['anchors'][0][0].name, groups['anchors'][1][0].name]
    with pytest.raises(SystemExit):
        wb.batch(['judge', str(WINDOWS_JSON), str(tmp_path), '--windows', ','.join(spanning), '--pending'])


def test_one_window_stages_refuse_groups_and_dcvc_takes_a_bracket_per_window(tmp_path):
    with pytest.raises(SystemExit):
        wb.batch(['dcvc', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034,al043', '--pending', '--bracket', 'al034=18.4,19.6'])
    with pytest.raises(SystemExit):
        wb.batch(['generate', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034,al043', '--pending'])
    with pytest.raises(SystemExit):
        wb.batch(['dcvc', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034', '--pending'])  # no bracket given
    for window, bracket in (('al034', '18.4,19.6'), ('med000', '31.25,33.5')):
        spec = _batch(tmp_path, 'dcvc', [window], extra=('--bracket', f'{window}={bracket}'))
        _check_common(spec, 'dcvc')
        low, high = bracket.split(',')
        i = spec['arguments'].index('--bracket')
        assert spec['arguments'][i + 1:i + 3] == [low, high] and _flag_values(spec, '--window') == [window]
        assert spec['budget_seconds'] <= 3300 and spec['hosts'] == ['gpu5', 'gpu6']
        assert spec['scale']['svt_scales'] == {'smoke': '8', 'full': '8,4,2'} and 'points' not in spec['scale']


def test_generation_is_one_window_per_job_with_that_windows_adapter(tmp_path):
    al = _batch(tmp_path, 'generate', ['al034'], extra=('--units', 'al034=a__person_1,a__person_2'))
    med = _batch(tmp_path, 'generate', ['med000'], extra=('--units', 'med000=m__person_1,m__person_2'))
    _check_common(al, 'generate-al')
    _check_common(med, 'generate-med')
    assert al['arguments'][al['arguments'].index('--fit') + 2] == 'ah-C@fit-100-fp16.pt'
    assert med['arguments'][med['arguments'].index('--fit') + 2] == 'med-C@fit-100-fp16.pt'
    assert _flag_values(med, '--windows') == [wm.load(WINDOWS_JSON)[3].id]
    assert med['full']['seconds'] > al['full']['seconds'] and med['full']['seconds'] <= sp.FULL_SECONDS  # 1.25 x for the 50 fps window, still under the cap
    assert al['staged_inputs'][-1]['sha256'] != med['staged_inputs'][-1]['sha256']  # adapter archives differ


def test_w1_stages_batch_with_any_window_and_per_window_archives(tmp_path):
    masks = _batch(tmp_path, 'w1-masks', ['med019'])
    assert _flag_values(masks, '--window') == ['med019']
    with pytest.raises(SystemExit):
        wb.batch(['w1-encode-half', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034', '--pending'])  # needs --target
    enc = _batch(tmp_path, 'w1-encode-half', ['al034'], extra=('--target', 'al034=18.5'))
    assert enc['arguments'][enc['arguments'].index('--target-kbps') + 1] == '18.5' and '--controls' in enc['arguments']
    score = _batch(tmp_path, 'w1-score', ['med024'], f'encode_x0.med024={sp.W7_DIR}/x.tar:{"3" * 64}')
    _check_common(score, 'w1-score')
    assert _flag_values(score, '--encode-extra') == ['{staged:encode_x0}'] and score['gpu_models'] == ['RTX A6000']


def test_batch_without_pending_stops_on_a_missing_upstream_archive(tmp_path):
    with pytest.raises(SystemExit):
        wb.batch(['features', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034,al043'])  # no manifest archive named
    assert not list(tmp_path.glob('*.json'))


def test_group_caps_are_enforced_when_a_job_is_too_big(tmp_path):
    ws = _campaign()
    with pytest.raises(SystemExit):  # five windows through the receiver would project past its 1,500 s smoke check
        wb.batch(['receiver', str(WINDOWS_JSON), str(tmp_path), '--windows', ','.join(w.name for w in ws), '--pending'])
    with pytest.raises(SystemExit):  # four Al-speed windows would take 2,388 s of the 2,600 s anchors cap
        wb.batch(['anchors', str(WINDOWS_JSON), str(tmp_path), '--windows', 'al034,al043,med000,med024,med019', '--pending'])


def test_batch_scores_only_the_rules_svt_anchors(tmp_path):
    """W7's rule names plain half, plain quarter and half +20 outside R; the batch drops +40 and quarter outside-R."""
    assert 'w1-encode-quarter-out' not in wb.MODEL
    out = _batch(tmp_path, 'w1-encode-half-out', ['al034'], extra=('--target', 'al034=18.5'))
    args = out['arguments']
    assert args[args.index('--offsets') + 1:args.index('--offsets') + 2] == ['-20'] and '-40' not in args
    assert out['scale']['max_encodes']['full'] == 6
