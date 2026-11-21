"""Score image-only ball detections against source labels, or replay saved evidence.

A new JSON input binds labels, geometry, model/config identities and predictions.
The historical mode reads the authenticated saved frame CSV, with no inference.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ARCHIVE_SHA = 'e3ae88ccbab68a19aa0a34a9ad12328352d00a69ae44901edfd7cbe01220aa12'
CSV_SHA = 'e50089b4659ad8c84d84f1794347ad08445b8552784d607cb1af3b68ad9cdfba'
WASB_SHA = '9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb'
BALLTRACK_SHA = 'b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec'
HASH = re.compile(r'^[0-9a-f]{64}$')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def score(labels, baseline, candidate):
    if not len(labels) == len(baseline) == len(candidate) or not labels:
        raise ValueError('Frame coverage differs')
    out = {'visible': 0, 'absent': 0, 'uncertain': 0, 'thresholds': {}}
    for radius in (10, 5):
        out['thresholds'][str(radius)] = {'both_hit': 0, 'baseline_only': 0, 'candidate_only': 0, 'neither_hit': 0}
    absent = {'both_false': 0, 'baseline_only_false': 0, 'candidate_only_false': 0, 'neither_false': 0}
    for tick, (label, a, b) in enumerate(zip(labels, baseline, candidate, strict=True)):
        if label['frame'] != tick or a['frame'] != tick or b['frame'] != tick:
            raise ValueError('Frame order differs')
        kind = label['kind']
        if type(kind) is not int or kind not in (-1, 0, 1):
            raise ValueError('Invalid label kind')
        for pred in (a, b):
            if type(pred['detected']) is not bool:
                raise ValueError('Detected flag must be Boolean')
            if pred['detected'] and any(type(pred[k]) not in (int, float) or not math.isfinite(pred[k]) for k in ('x', 'y')):
                raise ValueError('Detected point must be finite')
        if kind == -1:
            out['uncertain'] += 1
            continue
        if kind == 0:
            out['absent'] += 1
            key = ('both_false' if b['detected'] else 'baseline_only_false') if a['detected'] else ('candidate_only_false' if b['detected'] else 'neither_false')
            absent[key] += 1
            continue
        if any(type(label[k]) not in (int, float) or not math.isfinite(label[k]) for k in ('x', 'y')):
            raise ValueError('Visible label must have finite coordinates')
        out['visible'] += 1
        for radius in (10, 5):
            hits = [p['detected'] and math.hypot(p['x']-label['x'], p['y']-label['y']) <= radius for p in (a,b)]
            key = ('both_hit' if hits[1] else 'baseline_only') if hits[0] else ('candidate_only' if hits[1] else 'neither_hit')
            out['thresholds'][str(radius)][key] += 1
    for counts in out['thresholds'].values():
        counts['baseline_hits'] = counts['both_hit'] + counts['baseline_only']
        counts['candidate_hits'] = counts['both_hit'] + counts['candidate_only']
        counts['baseline_recall'] = counts['baseline_hits']/out['visible'] if out['visible'] else None
        counts['candidate_recall'] = counts['candidate_hits']/out['visible'] if out['visible'] else None
    out['absent_pairs'] = absent
    return out


def historical(path):
    blob = path.read_bytes()
    if digest(blob) != ARCHIVE_SHA:
        raise ValueError('Saved archive hash mismatch')
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        member = archive.extractfile('actual-analysis/frame-transitions.csv')
        if member is None:
            raise ValueError('Missing historical frame CSV')
        data = member.read()
    if digest(data) != CSV_SHA:
        raise ValueError('Saved frame CSV hash mismatch')
    all_rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
    rows = [r for r in all_rows if r['point'] == 'svt-62' and r['judge'] == 'wasb' and r['threshold_px'] == '10']
    if len(rows) != 292 or any(r['member'] != f"Dataset/game9/Clip5/{int(r['frame']):04d}.jpg" for r in rows):
        raise ValueError('Unexpected historical clip or frame coverage')
    labels, baseline, candidate = [], [], []
    for r in rows:
        frame, kind = int(r['frame']), int(r['label_kind'])
        labels.append({'frame': frame, 'kind': kind, 'x': float(r['label_x']), 'y': float(r['label_y'])})
        for field, out in (('baseline', baseline), ('streak', candidate)):
            value = r[field+'_detected']
            if value not in ('True', 'False'):
                raise ValueError('Invalid historical Boolean')
            out.append({'frame': frame, 'detected': value == 'True',
                        'x': float(r[field+'_x']), 'y': float(r[field+'_y'])})
    result = score(labels, baseline, candidate)
    r10, absence = result['thresholds']['10'], result['absent_pairs']
    assert (result['visible'],result['absent'],result['uncertain']) == (261,30,1)
    assert (r10['baseline_hits'],r10['candidate_hits'],r10['candidate_only'],r10['baseline_only']) == (5,229,224,0)
    assert (absence['both_false'],absence['candidate_only_false'],absence['baseline_only_false']) == (3,2,0)
    return {'scope':'historical frame-pair arithmetic replay; no new inference',
            'archive_sha256':ARCHIVE_SHA, 'csv_sha256':CSV_SHA, 'judge_checkpoint_sha256':WASB_SHA,
            'sender_checkpoint_sha256':BALLTRACK_SHA, 'result':result}


def current(path):
    data = json.loads(path.read_text())
    required = ('source_sha256','label_file_sha256','judge_checkpoint_sha256','judge_config_sha256',
                'sender_checkpoint_sha256','baseline_decode_sha256','candidate_decode_sha256')
    if any(not HASH.fullmatch(data.get(key,'')) for key in required):
        raise ValueError('Missing cryptographic bindings')
    canonical_labels = json.dumps(data['labels'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if digest(canonical_labels) != data['label_file_sha256']:
        raise ValueError('Canonical label content differs from declared hash')
    if data['baseline_decode_sha256'] == data['candidate_decode_sha256']:
        raise ValueError('Identical decoded inputs cannot establish a paired effect')
    if data['judge_checkpoint_sha256'] == data['sender_checkpoint_sha256']:
        raise ValueError('Judge checkpoint matches sender')
    if data.get('judge_input') != 'decoded_pixels_only':
        raise ValueError('Judge input must be decoded pixels only')
    if not isinstance(data.get('confidence_policy'), str) or not data['confidence_policy']:
        raise ValueError('Declare frozen postprocessor confidence policy')
    if data.get('coordinate_space') != 'source_native_pixels':
        raise ValueError('Coordinates must use native source pixel geometry')
    width, height = data.get('width'), data.get('height')
    if type(width) is not int or type(height) is not int or min(width,height) < 1:
        raise ValueError('Invalid geometry')
    for rows in (data['labels'], data['baseline'], data['candidate']):
        for row in rows:
            if row.get('detected') or row.get('kind') == 1:
                if not (0 <= row['x'] < width and 0 <= row['y'] < height):
                    raise ValueError('Coordinates outside source frame')
    return {'scope':'paired source labels and caller-supplied frozen image-only judge outputs',
            'input_sha256':digest(path.read_bytes()),'result':score(data['labels'],data['baseline'],data['candidate'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (a.historical is None) == (a.input is None):
        p.error('Provide exactly one of --historical or --input')
    result = historical(a.historical) if a.historical else current(a.input)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result['result']))


if __name__ == '__main__':
    main()
