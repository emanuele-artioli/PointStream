import json
import struct
from pathlib import Path

import pytest

from experiments.tennis import w7_byteproof as bp, w7_native_yuv_fps as candidate


def test_frozen_mixed_cadence_selection():
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    rows = candidate.smoke_rows(inventory)
    assert [candidate.cadence(row) for row in rows] == [25, 50]
    assert len(candidate.rows(inventory)) == 75


def test_ivf_default_preserves_old_gate_and_explicit_fps(tmp_path):
    source = tmp_path / '25.ivf'
    header = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 960, 544, 25, 1, 240, 0)
    source.write_bytes(header + b''.join(struct.pack('<IQ', 1, i) + b'x' for i in range(240)))
    with pytest.raises(ValueError, match='timestamps'):
        bp.ivf(source)
    assert len(bp.ivf(source, fps=25)[1]) == 240
    assert bp.compare_ivf(source, source, fps=25)['exact_ivf_bytes']
    source.write_bytes(source.read_bytes() + b'x')
    with pytest.raises(ValueError, match='timestamps'):
        bp.ivf(source, fps=25)


def test_probe_checks_per_window_cadence_and_timestamps():
    data = {'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544,
                         'pix_fmt': 'yuv420p', 'r_frame_rate': '25/1',
                         'color_range': 'tv', 'time_base': '1/25'}],
            'frames': [{'best_effort_timestamp': i} for i in range(240)]}
    assert len(bp.probe(data, 960, 544, fps=25)['timestamps']) == 240
    with pytest.raises(ValueError, match='profile'):
        bp.probe(data, 960, 544)
    data['frames'][17]['best_effort_timestamp'] = 18
    with pytest.raises(ValueError, match='PTS'):
        bp.probe(data, 960, 544, fps=25)
