import hashlib
import json
import zipfile

import pytest

from experiments.tennis import k5d_opentt_source as source
from experiments.tennis.k5d_analysis import score_frames


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
              'scores': score_frames(labels, preds), 'qualified': True}
    assert source.validate_report(report, clip)['qualified']
    rows[0]['label_x'] += 1
    with pytest.raises(ValueError, match='frozen frame join'):
        source.validate_report(report, clip)
