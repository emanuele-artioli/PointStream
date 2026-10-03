import json
import subprocess
import sys

import pytest

from experiments.gate_a_confirmation.native_anchor_v2 import metric_rows, run_with_receipt


def test_metric_rows_preserve_original_window_and_frame_identity():
    rows = metric_rows([50, 60, 70, 72], [8, 4, 2, 1], [.7, .8, .9, .95], 2, [0, 1])
    assert rows[2] == {'frame_index': 2, 'source_window_index': 1,
                       'window_frame_index': 0, 'vmaf': 70, 'y_mse': 2, 'ssim': .9}
    json.dumps(rows, allow_nan=False)
    with pytest.raises(ValueError, match='every registered frame'):
        metric_rows([50], [8], [.7], 2, [0])


def test_receipt_measures_child_and_raw_constraints(tmp_path):
    with (tmp_path / 'log').open('wb') as log:
        receipt = run_with_receipt([sys.executable, '-c', 'x = bytearray(12 * 1024**2)'], log, 5)
    assert receipt['returncode'] == 0
    assert receipt['child_maxrss_raw'] > 0
    assert receipt['child_user_seconds'] >= 0
    assert receipt['rss_unit'] in ('bytes', 'KiB')
    assert 'rlimit_as_bytes' in receipt['inherited_parent_constraints']
    json.dumps(receipt, allow_nan=False)


def test_failure_receipt_is_retained(tmp_path):
    with (tmp_path / 'log').open('wb') as log:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_with_receipt([sys.executable, '-c', 'raise SystemExit(3)'], log, 5)
    assert caught.value.receipt['returncode'] == 3
    assert caught.value.receipt['child_maxrss_raw'] > 0


def test_timeout_reaps_owned_child_and_retains_receipt(tmp_path):
    with (tmp_path / 'log').open('wb') as log:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run_with_receipt([sys.executable, '-c', 'import time; time.sleep(5)'], log, .05)
    assert caught.value.receipt['timed_out']
    assert caught.value.receipt['returncode'] < 0


def test_full96_source000_identity_is_window_relative():
    rows = metric_rows([60.] * 96, [4.] * 96, [.8] * 96, 96, [0])
    assert len(rows) == 96
    assert rows[0]['source_window_index'] == rows[-1]['source_window_index'] == 0
    assert rows[0]['window_frame_index'] == 0
    assert rows[-1]['window_frame_index'] == 95
    assert all('source_frame_index' not in row for row in rows)
