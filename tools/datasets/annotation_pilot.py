"""Prepare a real, source-only boundary-review pilot from frozen K4b artifacts.

No model execution, codec output or quality score is read. Existing human player
instance choices seed proposals only; boundary/racket validation is still needed.
The generated review is development material, never untouched confirmation.

python -m tools.datasets.annotation_pilot --archive K4B.tar --marks MARKS.json \
    --selection SELECTION.json --out NEW_DIR
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis.k4 import upsample
from experiments.tennis.k4b import numbered
from experiments.tennis.k4b_rules import validate_marks
from src.segmentation.masks import ClipMasks, decompress
from tools.datasets.annotation_review import prepare

ARCHIVE_SHA = '2dc9acb6541224f9f72af8ceb9c781308c3e62d19fcb819515736e4b7da261ee'
MARKS_SHA = '05f5130526254316c078003a3a66398c97aa33027ac56519e0f267429eba50ba'
SELECTION_SHA = '7d1a22f62ac03c514b5f47dbc8c4c3864d9e6b92d461ab90175d2b520c6c23ef'
INSTANCES_SHA = '89b78156e85b62741326a5ff2dc8a020ac58934f5dba0ad33a158c887b40f731'
SEED = 'pointstream-submission-boundary-pilot-v1'


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def rank(value: str) -> str:
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def select(frames: list[dict[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    """Round-robin hash selection across source groups, independent of outcomes."""
    if count < 3 or len({r['id'] for r in frames}) != len(frames):
        raise ValueError('need at least three distinct selected frames')
    groups: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        groups.setdefault(frame['match'], []).append(frame)
    if len(groups) < 3 or len(frames) < count:
        raise ValueError('pilot needs at least three source groups and enough frames')
    order = sorted(groups, key=rank)
    queues = {g: sorted(groups[g], key=lambda r: rank(r['id'])) for g in order}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        for group in order:
            if queues[group] and len(chosen) < count:
                chosen.append(queues[group].pop(0))
    return chosen


def member(archive: tarfile.TarFile, name: str) -> bytes:
    info = archive.getmember(name)
    if not info.isfile():
        raise ValueError('required archive member is not a regular file')
    handle = archive.extractfile(info)
    if handle is None:
        raise ValueError('missing archive member')
    return handle.read()


def export(archive_path: Path, marks_path: Path, selection_path: Path, output: Path) -> dict[str, Any]:
    for path, expected in ((archive_path, ARCHIVE_SHA), (marks_path, MARKS_SHA), (selection_path, SELECTION_SHA)):
        if digest(path) != expected:
            raise ValueError(f'frozen input SHA256 mismatch: {path.name}')
    if output.exists():
        raise FileExistsError(output)
    marks = json.loads(marks_path.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    with tarfile.open(archive_path) as archive:
        instances_bytes = member(archive, 'publish/instances.json')
        if hashlib.sha256(instances_bytes).hexdigest() != INSTANCES_SHA:
            raise ValueError('review instances identity mismatch')
        review = json.loads(instances_bytes)
        # The historical validator binds all48 returned source-only marks.
        # Its path identity check is reproduced here without extracting the archive.
        if marks.get('result_sha256') != INSTANCES_SHA or marks.get('selection_sha256') != SELECTION_SHA:
            raise ValueError('marks identify a different source review')
        output.mkdir(parents=True, exist_ok=False)
        inputs = output / 'inputs'
        inputs.mkdir()
        original_review = inputs / 'historical-instances.json'
        original_review.write_bytes(instances_bytes)
        labels = validate_marks(marks, review, original_review)
        selected = select(review['frames'])
        index_by_id = {r['id']: i for i, r in enumerate(review['frames'])}
        group_order = sorted({r['match'] for r in selected}, key=rank)
        blind_ids = {next(r['id'] for r in selected if r['match'] == group) for group in group_order[:3]}
        sources = {(r['match'], r['clip']): r for r in selection['sources']}
        manifest_frames, provenance = [], []
        for i, row in enumerate(selected):
            old_index = index_by_id[row['id']]
            full = json.loads(member(archive, f'publish/frame-{old_index:03d}.json'))
            if any(full[k] != row[k] for k in ('id', 'match', 'clip', 'frame', 'masks_sha256')):
                raise ValueError('embedded source frame differs from indexed review')
            prefix = 'data:image/jpeg;base64,'
            if not full['original'].startswith(prefix):
                raise ValueError('original is not the frozen source JPEG')
            image = base64.b64decode(full['original'][len(prefix):], validate=True)
            raw_masks = member(archive, f'publish/{row["masks"]}')
            if hashlib.sha256(raw_masks).hexdigest() != row['masks_sha256']:
                raise ValueError('raw SAM masks differ from review')
            clip = ClipMasks.from_doc(json.loads(decompress(raw_masks)))
            ids = set(labels[row['id']]['selected'])
            mask = np.zeros((1080, 1920), dtype=bool)
            for proposal in numbered(clip):
                if proposal['id'] in ids:
                    mask |= upsample(proposal['mask'], (1920, 1080))
            source_name, mask_name = f'source-{i:02d}.jpg', f'proposal-{i:02d}.png'
            source_path, mask_path = inputs / source_name, inputs / mask_name
            source_path.write_bytes(image)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            with Image.open(source_path) as source_image:
                if source_image.size != (1920, 1080):
                    raise ValueError('source JPEG geometry changed')
            manifest_frames.append(dict(id=row['id'], source_group=row['match'], frame_index=row['frame'],
                                        blind=row['id'] in blind_ids,
                                        source=dict(path=source_name, sha256=digest(source_path)),
                                        proposal=dict(path=mask_name, sha256=digest(mask_path))))
            source = sources[(row['match'], row['clip'])]
            provenance.append(dict(id=row['id'], source_group=row['match'], clip=row['clip'],
                                   native_frame_index=row['frame'], fps=row['fps'],
                                   source_video_sha256=source['sha256'], converted_source_sha256=source['converted_sha256'],
                                   historical_mask_sha256=row['masks_sha256'],
                                   proposed_player_ids=sorted(ids), historical_missing_player=labels[row['id']]['missing_instance'],
                                   source_jpeg_sha256=digest(source_path), proposal_png_sha256=digest(mask_path),
                                   blind=row['id'] in blind_ids))
    manifest = dict(version=1, review_id='submission-boundary-pilot-20261114-v1', frames=manifest_frames)
    manifest_path = inputs / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    prepared = prepare(manifest_path, output / 'review')
    result = dict(schema=1, purpose='source-only development boundary/racket validation pilot; no codec outcomes',
                  status='awaiting_human_review_no_new_reference_labels', seed=SEED, selection='source-group round robin over SHA256-ranked IDs',
                  count=len(selected), source_groups=group_order, blind_count=len(blind_ids),
                  archive_sha256=ARCHIVE_SHA, old_marks_sha256=MARKS_SHA, old_selection_sha256=SELECTION_SHA,
                  review_sha256=prepared['review_sha256'], manifest_sha256=digest(manifest_path), frames=provenance,
                  annotation_target='union of the actual tennis players and their visible rackets; exclude officials/crowd; ball scored separately',
                  proposal_limit='person/player SAM3.1 prompts; rackets may be absent; selected IDs were human identified historically, boundaries not validated',
                  source_limit='historical JPEG92 source review render from frozen YUV; exposed development matches, not untouched confirmation',
                  correction='approve complete union only; flag missing objects; polygon corrections replace whole union; blind items require full redraw or explicit empty',
                  marking_time='unknown until human timing is returned; browser records elapsed_seconds',
                  scientific_compute=dict(gpu_seconds=0, new_model_inference=0, native_encodes=0))
    (output / 'pilot-provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'README.txt').write_text('Review source frames only. Target = actual players + visible rackets; exclude officials/crowd. Ball is separate.\n'
                                      'Proposals are historically player-ID-selected SAM masks, not complete boundary truth.\n'
                                      'Missing rackets/parts require corrections. Polygon edits replace the whole union.\n'
                                      'Serve review/ with tools/datasets/k8_review_server.py, then download decisions.\n'
                                      'Import decisions through tools.datasets.annotation_review to a NEW output directory.\n'
                                      'Do not promote unreviewed frames or rejected/uncertain frames to ground truth.\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--marks', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.archive, args.marks, args.selection, args.out)
    print(json.dumps(dict(frames=result['count'], groups=len(result['source_groups']), blind=result['blind_count'], output=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
