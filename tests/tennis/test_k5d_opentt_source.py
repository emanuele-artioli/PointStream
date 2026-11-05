import hashlib
import json
import zipfile

import pytest

from experiments.tennis import k5d_opentt_source as source
from experiments.tennis.k5d_analysis import score_frames



def report_witness():
    frames = [{'frame_index': i, 'raw_rgb_sha256': 'a' * 64} for i in (1085, 1086, 1087)]
    evidence = json.dumps({'clip': 'test_2', 'fast_seek_raw_parity': True, 'frames': frames})
    return {'decode_witness': {'passed': True, 'frames': frames}, 'join_evidence_text': evidence,
            'join_witness': {'evidence_sha256': hashlib.sha256(evidence.encode()).hexdigest()}}

def test_metadata_keeps_partial_coordinates_unknown_and_checks_identity(tmp_path):
    archive = tmp_path / 'labels.zip'
    ball = {'10': {'x': 100, 'y': 150}, '11': {'x': -1, 'y': -1}, '12': {'x': -1, 'y': 450}}
    raw = json.dumps(ball).encode()
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('ball_markup.json', raw)
    clip = {'id': 'test_1', 'label_archive_sha256': source.identity.sha(archive),
            'ball_member_sha256': hashlib.sha256(raw).hexdigest(), 'start_zero_based_provisional': 10,
            'labels': [{'frame': 10, 'x': 100, 'y': 150, 'kind': 1},
                       {'frame': 11, 'x': -1, 'y': -1, 'kind': 0},
                       {'frame': 12, 'x': -1, 'y': 450, 'kind': -1}],
            'visible': 1, 'absent': 1, 'unknown': 238}
    manifest = tmp_path / 'selection.json'
    manifest.write_text(json.dumps({'items': [clip]}))
    assert source.metadata(manifest, source.identity.sha(manifest), 'test_1', archive) == clip
    clip['labels'][2]['kind'] = 0
    manifest.write_text(json.dumps({'items': [clip]}))
    with pytest.raises(ValueError, match='frozen manifest'):
        source.metadata(manifest, source.identity.sha(manifest), 'test_1', archive)


def test_join_requires_reviewed_independent_zero_offset_witness(tmp_path):
    path = tmp_path / 'join.json'
    clip = {'id': 'test_1', 'video_sha256': 'v', 'ball_member_sha256': 'b'}
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('independent reviewed evidence')
    row = {'schema_version': 1, 'passed': True, 'clip': 'test_1', 'video_sha256': 'v',
           'ball_markup_sha256': 'b', 'annotation_to_zero_based_offset': 0,
           'reviewer': 'reviewer', 'method': 'independent review', 'evidence_sha256': source.identity.sha(evidence)}
    path.write_text(json.dumps(row))
    assert source.join_witness(path, source.identity.sha(path), clip, evidence) == row
    row['annotation_to_zero_based_offset'] = 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='zero-offset'):
        source.join_witness(path, source.identity.sha(path), clip, evidence)


def test_qualification_recomputes_labels_and_unknown_denominator():
    clip = {'id': 'test_1', 'start_zero_based_provisional': 10, 'visible': 1, 'absent': 1, 'unknown': 238,
            'labels': [{'frame': 10, 'x': 100, 'y': 150, 'kind': 1}, {'frame': 11, 'x': -1, 'y': -1, 'kind': 0}]}
    labels = [(1, 100 * 2 / 3, 100.), (0, -2 / 3, -2 / 3)] + [(-1, 0., 0.)] * 238
    preds = [(True, 100 * 2 / 3, 100.)] + [(False, 0., 0.)] * 239
    rows = [{'frame': i + 10, 'label_kind': label[0], 'label_x': label[1], 'label_y': label[2],
             'detected': pred[0], 'x': pred[1], 'y': pred[2], 'rgb_sha256': 'a' * 64}
            for i, (label, pred) in enumerate(zip(labels, preds))]
    report = {'clip': 'test_1', 'start_zero_based': 10, 'frames': 240, 'rows': rows,
              'scores': score_frames(labels, preds), 'qualified': True, **report_witness()}
    assert source.validate_report(report, clip)['qualified']
    rows[0]['label_x'] += 1
    with pytest.raises(ValueError, match='frozen frame join'):
        source.validate_report(report, clip)


def test_false_positive_on_one_of_six_negatives_fails_qualification():
    clip = {'id': 'test_2', 'start_zero_based_provisional': 1066, 'visible': 85, 'absent': 6, 'unknown': 149,
            'labels': ([{'frame': 1066 + i, 'x': 90, 'y': 90, 'kind': 1} for i in range(85)] +
                       [{'frame': 1151 + i, 'x': -1, 'y': -1, 'kind': 0} for i in range(6)])}
    labels = [(1, 60., 60.)] * 85 + [(0, -2 / 3, -2 / 3)] * 6 + [(-1, 0., 0.)] * 149
    predictions = [(True, 60., 60.)] * 85 + [(True, 0., 0.)] + [(False, 0., 0.)] * 154
    rows = [{'frame': 1066 + i, 'label_kind': label[0], 'label_x': label[1], 'label_y': label[2],
             'detected': pred[0], 'x': pred[1], 'y': pred[2], 'rgb_sha256': 'a' * 64}
            for i, (label, pred) in enumerate(zip(labels, predictions))]
    report = {'clip': 'test_2', 'start_zero_based': 1066, 'frames': 240, 'rows': rows,
              'scores': score_frames(labels, predictions), 'qualified': False, **report_witness()}
    assert not source.validate_report(report, clip)['qualified']
    report['qualified'] = True
    with pytest.raises(ValueError, match='incorrect source qualification'):
        source.validate_report(report, clip)


def test_forecast_separates_fixed_cost_and_fails_closed_on_full_budget():
    report = {'timing': {'preflight_fixed_seconds': 10., 'model_init_seconds': 10.,
                         'source_io_seconds': 1., 'judge_seconds': 2.}}
    execution = {'seconds': 24., 'exit_code': 0}
    clock = {'started': 1., 'phase_started': 2., 'budget_end': 200.}
    estimate = source.forecast_smoke(report, execution, clock, 3.)
    assert estimate['fixed_seconds'] == 21.
    assert estimate['full_estimate_seconds'] == 82.5
    clock['budget_end'] = 90.
    with pytest.raises(ValueError, match='forecast exceeds'):
        source.forecast_smoke(report, execution, clock, 3.)


def test_spec_requires_actual_join_receipt(tmp_path):
    from experiments.tennis import k5d_opentt_source_spec as spec
    with pytest.raises(FileNotFoundError):
        spec.build(tmp_path / 'missing', tmp_path / 'missing-evidence',
                   '/home/itec/emanuele/Datasets/r', '/home/itec/emanuele/Datasets/e',
                   '/home/itec/emanuele/Datasets/s', '2026-11-05T01:00:00+00:00')


def test_spec_binds_reviewed_test2_and_scope(tmp_path):
    from experiments.tennis import k5d_opentt_source_spec as spec
    selection = json.loads(spec.SELECTION.read_text())
    clip = next(x for x in selection['items'] if x['id'] == 'test_2')
    evidence = tmp_path / 'evidence'
    evidence.write_text('synthetic reviewed evidence')
    receipt = tmp_path / 'receipt'
    receipt.write_text(json.dumps({'schema_version': 1, 'passed': True, 'clip': 'test_2',
                                   'video_sha256': clip['video_sha256'], 'ball_markup_sha256': clip['ball_member_sha256'],
                                   'annotation_to_zero_based_offset': 0, 'reviewer': 'test', 'method': 'test',
                                   'evidence_sha256': source.identity.sha(evidence)}))
    data = spec.build(receipt, evidence, '/home/itec/emanuele/Datasets/r', '/home/itec/emanuele/Datasets/e',
                      '/home/itec/emanuele/Datasets/s', '2026-11-05T01:00:00+00:00')
    assert data['scale']['frames'] == {'smoke': 16, 'full': 240}
    assert data['budget_seconds'] == 220 and data['single_cpu_campaign']
    assert next(x for x in data['staged_inputs'] if x['name'] == 'join_witness')['sha256'] == source.identity.sha(receipt)
    evidence.write_text('tampered')
    with pytest.raises(ValueError, match='evidence SHA256 mismatch'):
        spec.build(receipt, evidence, '/home/itec/emanuele/Datasets/r', '/home/itec/emanuele/Datasets/e',
                   '/home/itec/emanuele/Datasets/s', '2026-11-05T01:00:00+00:00')


def test_opencv_witness_rejects_changed_frames_and_seek():
    class Frame:
        shape = (1080, 1920, 3)
        def __init__(self, value):
            self.value = value
        def tobytes(self):
            return self.value.encode()

    class Capture:
        def __init__(self):
            self.pos = 0
        def set(self, prop, value):
            self.pos = value
            return True
        def get(self, prop):
            return self.pos
        def read(self):
            frame = Frame(str(self.pos))
            self.pos += 1
            return True, frame

    class CV:
        CAP_PROP_POS_FRAMES = 1
        COLOR_BGR2RGB = 2
        @staticmethod
        def cvtColor(frame, conversion):
            return frame

    rows = [{'frame_index': i, 'raw_rgb_sha256': hashlib.sha256(str(i).encode()).hexdigest()}
            for i in (1085, 1086, 1087)]
    evidence = json.dumps({'clip': 'test_2', 'fast_seek_raw_parity': True, 'frames': rows})
    witness = {'evidence_sha256': hashlib.sha256(evidence.encode()).hexdigest()}
    record = source.check_decode_witness(Capture(), CV, evidence, witness)
    assert record['frames'] == rows
    record['frames'][1]['raw_rgb_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='instrumentation gap'):
        source.validate_decode_witness(record, evidence, witness)
    capture = Capture()
    capture.set = lambda prop, value: False
    with pytest.raises(ValueError, match='seek'):
        source.check_decode_witness(capture, CV, evidence, witness)
