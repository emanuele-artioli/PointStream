"""Compare paid PTP2 tile centers with sparse official native source labels."""
import hashlib
import json
import math
import struct
import tarfile
from pathlib import Path

ROOT = Path('/home/itec/emanuele/Datasets/pointstream-data')
SOURCES = ROOT/'tennis/confirmation-source-20261115T062148Z-297a05e2/tennis-confirmation-sources.tar'
SOURCE_SHA = 'e5edc12c857658a119bf6d6cbe029ee64c09d6a860f1b975fe0708952895293d'
MANIFEST_SHA = 'df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1'
JOBS = {'match209_000': '20261121T220316Z-7cbf21fc', 'match269_000': '20261121T221317Z-fa6e8d74',
        'match235_000': '20261121T221746Z-7f377ae9'}
HEADER = struct.Struct('>4sIII')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()


def main():
    if sha(SOURCES) != SOURCE_SHA:
        raise ValueError('source archive identity')
    with tarfile.open(SOURCES) as archive:
        raw = archive.extractfile('manifest.json').read()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('source manifest identity')
    manifest = json.loads(raw)
    reports = []
    for source, job in JOBS.items():
        entry, = [item for item in manifest['sources'] if item['id'] == source]
        archive_path = ROOT/'jobs/fleet/inbox'/job/'full/published.tar'
        with tarfile.open(archive_path) as archive:
            record = json.load(archive.extractfile('publish/result.json'))
            score, = [r for r in record['results'] if r['mode']=='tracked' and r['crf']==60]
            packet = archive.extractfile('publish/'+score['file']).read()
        if hashlib.sha256(packet).hexdigest() != score['sha256']:
            raise ValueError('paid packet identity')
        magic, mlen, blen, tlen = HEADER.unpack_from(packet)
        if magic != b'PTP2' or HEADER.size+mlen+blen+tlen+32 != len(packet):
            raise ValueError('packet layout')
        if hashlib.sha256(packet[:-32]).digest() != packet[-32:]:
            raise ValueError('packet checksum')
        meta = json.loads(packet[HEADER.size:HEADER.size+mlen])
        positions = meta['positions']
        if meta['canvas'] != [1280,720] or len(positions) != record['frames']:
            raise ValueError('canvas geometry or frame coverage')
        rows = entry['official_window_rows']
        if len({r['native_frame_index'] for r in rows}) != len(rows):
            raise ValueError('duplicate annotation index')
        visible = [r for r in rows if r['visibility'] == 1]
        hidden = [r for r in rows if r['visibility'] == 0]
        errors = []
        for row in visible:
            tick = row['native_frame_index']
            pos = positions[tick]
            error = None if pos is None else math.hypot(1.5*(pos[0]+16)-row['native_X'],
                                                       1.5*(pos[1]+16)-row['native_Y'])
            errors.append({'frame':tick,'distance_native_px':error,'hit_10_native_px':error is not None and error <= 10})
        not_visible_placements = sum(positions[row['native_frame_index']] is not None for row in hidden)
        reports.append({'source':source,'job':job,'packet_sha256':score['sha256'],'frames':len(positions),
                        'fps':record['fps'],'labeled_visible':len(visible),'visible_with_placement':sum(e['distance_native_px'] is not None for e in errors),
                        'hits_10_native_px':sum(e['hit_10_native_px'] for e in errors),
                        'labeled_not_visible':len(hidden),'placements_when_not_visible':not_visible_placements,
                        'unlabeled':len(positions)-len(rows),'placements_all_frames':sum(p is not None for p in positions),
                        'clipped_edge_centers':sum(p is not None and (p[0] in (0,1248) or p[1] in (0,688)) for p in positions),
                        'visible_rows':errors})
    print(json.dumps({'source_manifest_sha256':MANIFEST_SHA,'threshold_native_px':10,'canvas_to_native':1.5,
                      'scope':'Official sparse labels, paid tile origins; not-visible is not proven absent.',
                      'sources':reports},indent=2))


if __name__ == '__main__':
    main()
