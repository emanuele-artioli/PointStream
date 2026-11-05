import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.tennis import k5d_opentt_join_witness as witness


def probe(pts):
    return {'streams': [{'width': 1920, 'height': 1080, 'nb_frames': str(len(pts)), 'avg_frame_rate': '120/1',
                         'r_frame_rate': '120/1', 'time_base': '1/15360'}],
            'frames': [{'best_effort_timestamp': x} for x in pts]}


def test_pts_cadence_checks_every_frame_and_exact_tick():
    base, pts = witness.cadence(probe([32, 160, 288]), 3)
    assert base == Fraction(1, 15360)
    assert pts == [32, 160, 288]
    with pytest.raises(ValueError, match='timestamps'):
        witness.cadence(probe([32, 161, 288]), 3)
    incomplete = probe([32, 160])
    incomplete['streams'][0]['nb_frames'] = '3'
    with pytest.raises(ValueError, match='actual decoded frame count'):
        witness.cadence(incomplete, 3)


def test_presentation_pts_and_filter_offset_checked(monkeypatch):
    frame_bytes = 1920 * 1080 * 3
    stderr = (b'[showinfo] config in time_base: 1/15360, frame_rate: 120/1\n'
              b'[showinfo] n: 0 pts: 1280\n[showinfo] n: 1 pts: 1408\n[showinfo] n: 2 pts: 1536\n')
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b'0' * frame_bytes * 3, stderr=stderr)

    monkeypatch.setattr(witness.subprocess, 'run', fake_run)
    frames, _, pts = witness.extract(Path('/ffmpeg'), Path('/source'), 10, None, Fraction(1, 15360), [1280, 1408, 1536])
    assert len(frames) == 3 and pts == [1280, 1408, 1536]
    assert 'select=between(n\\,10\\,12),showinfo' in calls[0]
    with pytest.raises(ValueError, match='PTS'):
        witness.extract(Path('/ffmpeg'), Path('/source'), 10, None, Fraction(1, 15360), [1408, 1536, 1664])


def test_frozen_witness_triplet_is_within_source_context():
    clip, selected = witness.selection('test_2')
    assert selected['frame_indices'] == [1085, 1086, 1087]
    assert all(clip['start_zero_based_provisional'] <= i < clip['start_zero_based_provisional'] + 240
               for i in selected['frame_indices'])
