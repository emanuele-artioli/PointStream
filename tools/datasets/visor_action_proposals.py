"""CPU-only revision proposals for the SAME18 VISOR source frames.

The user's returned decisions remain rejected/uncertain. This module filters
existing official instances by a pilot-specific action ontology; it does not
infer motion, manufacture missing hands or create reference annotations.
No review UI is published. ROOT must inspect rule/assets before another review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.segmentation import visor

POLICY = 'visor-source-action-targets-v2'
KEEP_OBJECTS = {'P35_105': {'saucepan'}, 'P12_101': {'tap'}, 'P11_103': {'pan'}}
DROP_OBJECTS = {'P35_105': {'cloth'}, 'P12_101': {'sink'}, 'P11_103': {'oven', 'hob/cooktop/stovetop'}}


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def decide(annotation: dict[str, Any], video: str) -> tuple[bool, str]:
    if video not in KEEP_OBJECTS:
        raise ValueError('No source-specific reviewed policy for this video')
    if visor.native_class(annotation) in visor.HANDS:
        return True, 'visible hand/forearm candidate retained when official polygon exists'
    name = annotation['name']
    if name in KEEP_OBJECTS[video]:
        return True, 'pilot action-object candidate; current interaction still requires human validation'
    if name in DROP_OBJECTS[video]:
        return False, 'excluded by user action-target feedback for this pilot; not a universal class rule'
    return False, 'UNRESOLVED class: retained in inventory, no automatic foreground/background truth'


def filtered(frame: visor.Frame | None, video: str, *, dense: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    shape = visor.DENSE_SIZE if dense else visor.FRAME_SIZE
    union = np.zeros(shape, bool)
    inventory: list[dict[str, Any]] = []
    if frame is not None:
        for ordinal, annotation in enumerate(frame.annotations):
            keep, reason = decide(annotation, video)
            pixels = visor.polygon_mask(annotation['segments'], source_size=shape, shape=shape)
            inventory.append({'ordinal': ordinal, 'name': annotation['name'], 'class': visor.native_class(annotation),
                              'annotation_fields': sorted(annotation),
                              'attributes_without_segments': {k: v for k, v in annotation.items() if k != 'segments'},
                              'keep_candidate': keep, 'reason': reason, 'native_annotation_canvas_pixels': int(pixels.sum())})
            if keep:
                union |= pixels
    if dense:
        union = cv2.resize(union.astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0
    return union, inventory


def build(intake_path: Path, prior_provenance: Path, output: Path) -> dict[str, Any]:
    started, cpu = time.monotonic(), time.process_time()
    intake = json.loads(intake_path.read_bytes())
    old = json.loads(prior_provenance.read_bytes())
    if old['review_sha256'] != '5d67a28c53714e410dd4ec0d77a765b1a7181575d17abe69a48e9d753fd755b8' or len(old['frames']) != 18:
        raise ValueError('Expected exact prior18 source-only review')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with zipfile.ZipFile(intake['dataset']['outer_zip']) as archive:
        for candidate in intake['selected_provisional']:
            video = candidate['video']
            if video not in KEEP_OBJECTS:
                raise ValueError('Pilot source changed')
            sparse_blob, dense_blob = archive.read(candidate['sparse_member']), archive.read(candidate['dense_member'])
            check = candidate['annotation_check']
            if (hashlib.sha256(sparse_blob).hexdigest() != check['sparse_json_sha256']
                    or hashlib.sha256(dense_blob).hexdigest() != check['dense_zip_sha256']):
                raise ValueError('Official annotation member differs from frozen intake')
            sparse = visor.frames(visor.load_annotations(sparse_blob))
            dense = visor.frames(visor.load_annotations(dense_blob))
            for old_row in [r for r in old['frames'] if r['video'] == video]:
                index, number = old_row['native_index_0based'], old_row['visor_frame_1based']
                if number != index + 1:
                    raise ValueError('Native/released frame mapping changed')
                manual = number in sparse
                present = manual or number in dense
                frame = sparse[number] if manual else dense.get(number)
                if present != old_row['proposal_annotation_known'] or manual != old_row['official_manual_anchor']:
                    raise ValueError('Original proposal presence/tier changed')
                mask, inventory = filtered(frame, video, dense=present and not manual)
                name = f'{video}-action-proposal-{index}.png'
                path = output / name
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                records.append({'id': old_row['id'], 'video': video, 'native_index_0based': index,
                                'visor_frame_1based': number, 'source_png_sha256': old_row['source_png_sha256'],
                                'proposal': name, 'proposal_sha256': sha(path), 'proposal_known': present,
                                'official_tier': 'sparse_human' if manual else 'dense_interpolated' if present else 'unknown_no_annotation',
                                'blind_in_v1': old_row['blind'], 'blind_in_v2': not present,
                                'blind_control_hidden_in_v1': old_row['mechanical_blind_control'],
                                'kept_names': [r['name'] for r in inventory if r['keep_candidate']],
                                'dropped_names': [r['name'] for r in inventory if not r['keep_candidate']],
                                'unresolved_names': [r['name'] for r in inventory if r['reason'].startswith('UNRESOLVED')],
                                'instance_inventory': inventory,
                                'ground_truth': False,
                                'limitation': 'unknown is placeholder only; missing official hands/persons are not invented; every candidate needs human validation'})
    result = {'schema': 'pointstream.visor.action-proposal.v2', 'policy': POLICY,
              'state': 'CPU_proposals_only_ROOT_review_required_before_UI_publication',
              'old_review_sha256': old['review_sha256'], 'intake_sha256': sha(intake_path),
              'prior_provenance_sha256': sha(prior_provenance), 'count': len(records), 'frames': records,
              'rules': {'hands': 'retain all available official left/right hand or glove polygons, including visible forearms',
                        'objects': {k: sorted(v) for k, v in KEEP_OBJECTS.items()},
                        'excluded_for_this_pilot': {k: sorted(v) for k, v in DROP_OBJECTS.items()},
                        'people': 'official selected-frame labels have no full-person class; hand polygons cannot supply full person truth',
                        'tap': 'retain as interaction candidate; actual per-frame interaction remains to be checked',
                        'cloth': 'drop extra towel proposal for this pilot; actively held cloth may still require human correction',
                        'unknowns': 'no official labels at P35@225/238; cannot create missing hands or objects by filtering'},
              'new_reference_annotations': 0, 'GPU_seconds': 0,
              'wall_seconds': time.monotonic()-started, 'process_CPU_seconds': time.process_time()-cpu}
    (output/'filtered-proposals.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, required=True)
    parser.add_argument('--prior-provenance', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    r = build(args.intake, args.prior_provenance, args.out)
    print(json.dumps({k: r[k] for k in ('state', 'count', 'wall_seconds', 'process_CPU_seconds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
