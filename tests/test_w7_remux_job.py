import hashlib
from fractions import Fraction

import pytest

from experiments.tennis import w7_remux_job as job
from experiments.tennis.w7_remux_oracle import validate_probes


def test_audit_rejects_changed_stream_and_historical_decode():
    frames = [{'best_effort_timestamp': n} for n in range(240)]
    probe = {'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544,
                          'pix_fmt': 'yuv420p', 'r_frame_rate': '50/1', 'time_base': '1/50',
                          'color_range': 'tv', 'color_space': 'bt709',
                          'color_transfer': 'bt709', 'color_primaries': 'bt709'}], 'frames': frames}
    case = job.CASES['med000']
    payload = b'remux'
    report = {'status': 'passed', 'scope': 'transport only; no quality score',
              'identities': {'input': case['stream'], 'ffmpeg': job.FFMPEG_SHA, 'ffprobe': job.FFPROBE_SHA},
              'historical_decoded_sha256': case['decoded'], 'decoded_yuv_sha256': case['decoded'],
              'historical_decoded_match': True,
              'metadata': validate_probes(probe, probe, width=960, height=544, frames=240, fps=Fraction(50)),
              'remux_mkv_bytes': len(payload), 'remux_sha256': hashlib.sha256(payload).hexdigest(),
              'source_ivf_bytes': 10, 'bytes_saved': 5}
    job.audit_report(report, probe, probe, payload, 'med000')
    with pytest.raises(ValueError, match='disagree'):
        job.audit_report(report, probe, probe, b'other', 'med000')
    report['decoded_yuv_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='disagree'):
        job.audit_report(report, probe, probe, payload, 'med000')
