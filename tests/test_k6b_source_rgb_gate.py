"""The exact source-RGB gate must reject even one differing frame hash."""
import pytest

from tools.analysis.k6b_source_rgb_gate import EXPECTED, compare


def test_exact_gate_rejects_hash_change_and_invalid_yuv():
    rows = []
    gpu_rows = []
    for tag, (window, yuv) in EXPECTED.items():
        hashes = ['same'] * 80
        rows.extend({'window': tag, 'scale': scale, 'frames': 240,
                     'source_yuv_sha256': yuv, 'source_rgb_frame_sha256': hashes}
                    for scale in (2, 4, 8))
        gpu_rows.append({'id': window, 'converted': {'sha256': yuv}, 'expected_judged': list(range(0, 240, 3)),
                         'source_rgb_frame_sha256': {str(i): 'same' for i in range(0, 240, 3)}})
    manifest, gpu = {'rows': rows}, {'rows': gpu_rows}
    assert compare(manifest, gpu)['passed']
    gpu_rows[0]['source_rgb_frame_sha256']['3'] = 'different'
    result = compare(manifest, gpu)
    assert not result['passed']
    assert result['rows'][0]['differing_ticks'] == [3]
    rows[0]['source_yuv_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='YUV identity'):
        compare(manifest, gpu)
