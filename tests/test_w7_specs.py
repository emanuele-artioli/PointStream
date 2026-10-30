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
