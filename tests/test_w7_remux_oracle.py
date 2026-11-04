from fractions import Fraction
from pathlib import Path

import pytest

from experiments.tennis.w7_remux_oracle import check_hash, commands, validate_probes


def probe(time_base='1/50', ticks=(0, 1, 2), color='tv'):
    return {'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544, 'pix_fmt': 'yuv420p',
                         'r_frame_rate': '50/1', 'time_base': time_base, 'color_range': color,
                         'color_space': 'bt709', 'color_transfer': 'bt709', 'color_primaries': 'bt709'}],
            'frames': [{'best_effort_timestamp': n} for n in ticks]}


def test_native_commands_copy_stream_and_no_overwrite():
    cmds = commands('/ffmpeg', '/ffprobe', Path('/source.ivf'), Path('/new/out.mkv'),
                    Path('/new/orig.yuv'), Path('/new/out.yuv'))
    assert cmds['remux'][cmds['remux'].index('-c:v') + 1] == 'copy'
    for key in ('decode_original', 'decode_remux'):
        cmd = cmds[key]
        assert cmd.index('-hwaccel') < cmd.index('-i') and cmd[cmd.index('-hwaccel') + 1] == 'none'
        assert cmd.index('-c:v') < cmd.index('-i') and cmd[cmd.index('-c:v') + 1] == 'libdav1d'
    assert all('-n' in cmds[k] for k in ('remux', 'decode_original', 'decode_remux'))
    assert all(cmds[k][cmds[k].index('-threads') + 1] == '1'
               for k in ('remux', 'decode_original', 'decode_remux'))


def test_exact_equivalent_timebase_and_reject_drift():
    args = {'width': 960, 'height': 544, 'frames': 3, 'fps': Fraction(50)}
    assert validate_probes(probe(), probe('1/1000', (0, 20, 40)), **args)['duration'] == '3/50'
    with pytest.raises(ValueError, match='timestamps'):
        validate_probes(probe(), probe('1/1000', (0, 20, 41)), **args)
    with pytest.raises(ValueError, match='color'):
        validate_probes(probe(), probe(color='pc'), **args)
    with pytest.raises(ValueError, match='frame count'):
        validate_probes(probe(), probe(ticks=(0, 1)), **args)


def test_hash_identity_must_match(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'test')
    with pytest.raises(ValueError, match='identity mismatch'):
        check_hash(path, '0' * 64)
    with pytest.raises(ValueError, match='lowercase hex'):
        check_hash(path, 'invalid')


@pytest.mark.parametrize('value', ['unknown', 'unspecified', 'reserved', 'reserved0', ''])
def test_unavailable_color_is_not_proof_of_preservation(value):
    altered = probe()
    altered['streams'][0]['color_space'] = value
    with pytest.raises(ValueError, match='color metadata'):
        validate_probes(altered, altered, width=960, height=544, frames=3, fps=Fraction(50))
