import json
import struct

import pytest

from experiments.tennis import w8_oracle, w8_payload as w


def test_envelope_roundtrip_and_accounting():
    profile = {'model_sha256': 'a' * 64, 'units': ['p0', 'p1'], 'reference_ticks': [0, 0]}
    data = w.pack(profile, [('ivf', 'frame', b'abc'), ('ref0', 'window', b'xyz'), ('fit', 'match', b'fit')])
    entries, counts = w.unpack(data, profile)
    assert entries == {'ivf': b'abc', 'ref0': b'xyz', 'fit': b'fit'}
    assert sum(counts.values()) == len(data)
    assert counts['frame'] == 3 and counts['match'] == 3
    for corrupted in (data[:-1], data + b'!', data[:-1] + b'X'):
        with pytest.raises(ValueError):
            w.unpack(corrupted, profile)
    with pytest.raises(ValueError):
        w.unpack(data, {})
    for name in ('../p', '/p', '.', 'p/a'):
        with pytest.raises(ValueError):
            w.pack(profile, [(name, 'frame', b'a')])
    with pytest.raises(ValueError):
        w.pack(profile, [('a', 'frame', b'a'), ('a', 'frame', b'b')])


def test_malformed_directory_and_decompression_limits():
    data = w.pack({}, [('a', 'frame', b'a')])
    n = struct.unpack('>I', data[5:9])[0]
    head = json.loads(data[9:9+n])
    head['sections'].append(head['sections'][0])
    raw = w.canonical(head)
    with pytest.raises(ValueError):
        w.unpack(w.MAGIC + struct.pack('>I', len(raw)) + raw + b'aa', {})
    import lzma
    body = lzma.compress(b'x' * 1000)
    for blob, size in ((body, 10), (body + body, 1000), (body[:-1], 1000)):
        with pytest.raises(ValueError):
            w.decompress_exact(blob, size)


def test_quantization_matches_frozen_torch_recipe():
    torch = pytest.importorskip('torch')
    from experiments.tennis.k6b_fit import quantize_tensor
    torch.manual_seed(3)
    for tensor in (torch.randn(8, 128), torch.randn(7), torch.tensor(0.125), torch.zeros(4, 256)):
        expected = quantize_tensor(tensor, 4, [])
        meta, symbols, scales = w.quantized_tensor(tensor, 4)
        actual = w.restore_tensor(meta, symbols, scales)
        assert torch.equal(actual, expected)
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        with pytest.raises(ValueError):
            w.restore_tensor(meta, symbols + b'x', scales)


def test_gate_requires_all_families_and_no_extrapolation():
    anchors = {f: [{'kbps': 10., 'lpips': .3, 'dists': .4}, {'kbps': 40., 'lpips': .2, 'dists': .3}] for f in w8_oracle.FAMILIES}
    assert w8_oracle.gate(20., {'lpips': .1, 'dists': .1}, anchors)['decision'] == 'pass'
    assert w8_oracle.gate(20., {'lpips': .5, 'dists': .1}, anchors)['decision'] == 'fail'
    assert w8_oracle.gate(41., {'lpips': .1, 'dists': .1}, anchors)['decision'] == 'unresolved'
    with pytest.raises(ValueError):
        w8_oracle.gate(20., {'lpips': .1, 'dists': .1}, {'dcvc': anchors['dcvc']})


def test_rate_bills_window_framing_per_window():
    counts = {'frame': 100, 'window': 50, 'reference': 200, 'match': 300}
    cache = {'match_sha256': 'a' * 64, 'reference_sha256_by_unit': {'p0': 'b' * 64, 'p1': 'c' * 64}}
    assert 'conditional_kbps' not in w8_oracle.rates(counts)
    result = w8_oracle.rates(counts, cache_contract=cache, expected_cache=cache)
    assert result['cold_kbps'] == pytest.approx(650 * 8 / 4.8 / 1000)
    assert result['conditional_kbps'] == pytest.approx((150 / 4.8 + 500 / 7200) * 8 / 1000)


def test_nested_match_framing_is_once_per_match():
    torch = pytest.importorskip('torch')
    base = {'w': torch.ones(4)}
    state = {'rank': 4, 'reference_unet': {'w': torch.ones(4)}, 'denoising_unet': {'w': torch.ones(4)},
             'pose_guider': {'w': torch.zeros(4)}}
    match = w.pack_match(state, base, 'a' * 64)
    restored = w.unpack_match(match, w.sha(match), 'a' * 64, base)
    assert restored['rank'] == 4
    profile = {'window': 'nad000', 'models': {k: 'a' * 64 for k in w.REQUIRED_MODELS},
               'recipe': {'profile': 'match5', 'steps': 30, 'guidance': 2, 'context': [24, 4], 'seed_rule': 'sha256(unit_name)[:7] then seed*1009+run', 'native_frames': 240, 'fps': [50, 1], 'saved_ticks': list(range(0, 240, 3)), 'fallback': 'base', 'scheduler': {}, 'rank': 4, 'compositing_order': ['p0', 'p1'], 'color_path': 'rgb'},
               'ivf': 'ivf', 'kps': 'kps', 'judged': 'judged', 'match': {'section': 'match', 'sha256': w.sha(match)},
               'units': [{'id': 'p0', 'reference': 'r0', 'reference_tick': 0, 'box': [1, 2, 3, 4]},
                         {'id': 'p1', 'reference': 'r1', 'reference_tick': 0, 'box': [1, 2, 3, 4]}]}
    body = [('ivf', 'frame', b'a'), ('kps', 'frame', b'b'), ('r0', 'reference', b'c'),
            ('r1', 'reference', b'd'), ('judged', 'window', bytes(30)), ('match', 'match', match)]
    payload = w.pack(profile, body)
    _, counts = w.unpack_window(payload, profile)
    assert counts['match'] == len(match) and counts['reference'] == 2 and counts['frame'] == 2
    assert counts['window'] == len(payload) - len(match) - 4
    with pytest.raises(ValueError):
        w.unpack_match(match, w.sha(match), 'b' * 64, base)


def test_strict_win_intervals_find_crossing():
    anchors = {f: [{'kbps': 10., 'lpips': .3, 'dists': .4}, {'kbps': 40., 'lpips': .1, 'dists': .3}] for f in w8_oracle.FAMILIES}
    out = w8_oracle.strict_win_intervals({'lpips': .2, 'dists': .1}, anchors)
    assert out['maximum_win_rate_supremum_kbps'] == pytest.approx(20.)
    assert out['intervals_kbps'][0][0] == 10.


def test_conditional_cache_requires_exact_identity():
    cache = {'match_sha256': 'a' * 64, 'reference_sha256_by_unit': {'p0': 'b' * 64, 'p1': 'c' * 64}}
    wrong = {'match_sha256': 'd' * 64, 'reference_sha256_by_unit': cache['reference_sha256_by_unit']}
    with pytest.raises(ValueError):
        w8_oracle.rates({'frame': 1, 'window': 2, 'match': 3, 'reference': 4}, cache_contract=cache, expected_cache=wrong)


def test_checkpoint_roundtrip_against_frozen_implementation():
    torch = pytest.importorskip('torch')
    from experiments.tennis.k6b_fit import quantize_adapters

    torch.manual_seed(42)
    base = {'large.weight': torch.randn(8, 128), 'small.bias': torch.randn(8)}
    state = {'rank': 4, 'reference_unet': {'lora.weight': torch.randn(8, 128).half()},
             'denoising_unet': {'lora.weight': torch.randn(3, 4).half()},
             'pose_guider': {name: (value + .02 * torch.randn_like(value)).half() for name, value in base.items()}}
    expected, _ = quantize_adapters(state, 4, base)
    directory, sections = w.encode_checkpoint(state, base)
    decoded = w.decode_checkpoint(directory, {name: data for name, _, data in sections}, base, 4)
    assert w.compare_checkpoint(decoded, expected) == {'reference_unet': 1, 'denoising_unet': 1, 'pose_guider': 2}
    with pytest.raises(ValueError):
        w.decode_checkpoint(directory, {name: data for name, _, data in sections}, {}, 4)


def test_compressed_outer_header_bounded_and_charged():
    blob = w.pack({'data': 'x' * 2000}, [('a', 'frame', b'x')], compact_header=True)
    sections, counts = w.unpack(blob, {'data': 'x' * 2000})
    assert sections == {'a': b'x'} and counts['window'] == len(blob) - 1
    assert len(blob) < 500
    broken = bytearray(blob)
    broken[9:13] = (w.MAX_HEADER + 1).to_bytes(4, 'big')
    with pytest.raises(ValueError):
        w.unpack(bytes(broken), {'data': 'x' * 2000})


def test_negative_zero_symbols_preserve_exact_frozen_bytes():
    torch = pytest.importorskip('torch')
    from experiments.tennis.k6b_fit import quantize_tensor

    tensor = torch.ones(1, 1024)
    tensor[0, 0] = -0.001
    tensor[0, 1] = 0.001
    expected = quantize_tensor(tensor, 4, [])
    assert torch.signbit(expected[0, 0]) and not torch.signbit(expected[0, 1])
    meta, symbols, scales = w.quantized_tensor(tensor, 4)
    actual = w.restore_tensor(meta, symbols, scales)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_saved_w8_accounting_and_one_unbracketed_family():
    counts = {'frame': 23876, 'window': 2404, 'reference': 366161, 'match': 1897395}
    cache = {'match_sha256': 'a' * 64, 'reference_sha256_by_unit': {'p0': 'b' * 64, 'p1': 'c' * 64}}
    result = w8_oracle.rates(counts, cache_contract=cache, expected_cache=cache)
    assert result['conditional_kbps'] == pytest.approx(46.315062222222224)
    assert result['warm_kbps'] == pytest.approx(43.8)
    assert result['cold_kbps'] == pytest.approx(3816.3933333333334)
    points = {f: [{'kbps': 35., 'lpips': .3, 'dists': .3}, {'kbps': 50., 'lpips': .2, 'dists': .2}]
              for f in w8_oracle.FAMILIES}
    points['s2_o20'][1]['kbps'] = 45.82333333333334
    gate = w8_oracle.gate(result['conditional_kbps'], {'lpips': .1, 'dists': .1}, points)
    assert gate['decision'] == 'unresolved'
    assert gate['cells']['s2_o20:lpips']['won'] is None
    assert gate['cells']['dcvc:dists']['won'] is True
