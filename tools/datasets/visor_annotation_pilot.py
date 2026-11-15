"""Source-only VISOR annotation pilot for the reserved confirmation windows.

No inference, codec outcome, metric or new annotation is produced. Export six
native source frames per fixed window: all human anchors, first/last, then
farthest-gap filling with the earliest-index tie rule. Unknown annotations use
a zero placeholder solely to satisfy the review asset interface, are always
blind, and never become empty ground truth. One additional hash-selected blind
control per video hides an available official proposal.

Run on a budgeted CPU host with --intake ROOT_RECEIPT --out NEW_HOST_LOCAL_DIR
--ffmpeg PACKED_FFMPEG. The output is source material for human review only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.segmentation import visor
from tools.datasets.annotation_review import prepare

SEED = 'pointstream-visor-confirmation-review18-v1'
FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
VIDEOS = {'P35_105': (175, 238), 'P12_101': (217, 280), 'P11_103': (229, 292)}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def byte_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_indices(first: int, last: int, anchors: list[int], count: int = 6) -> list[int]:
    if first < 0 or last < first or count < 2 or last - first + 1 < count:
        raise ValueError('invalid bounded source window')
    selected = {first, last, *(i for i in anchors if first <= i <= last)}
    if len(selected) > count:
        raise ValueError('all manual anchors do not fit; do not silently omit any')
    while len(selected) < count:
        remaining = (i for i in range(first, last + 1) if i not in selected)
        selected.add(max(remaining, key=lambda i: (min(abs(i - j) for j in selected), -i)))
    return sorted(selected)


def hash_rank(value: str) -> str:
    return byte_sha(f'{SEED}:{value}'.encode())


def union_proposal(frame: visor.Frame | None, *, dense: bool) -> np.ndarray:
    """Unknown is a placeholder, whose separate status MUST force blind redraw."""
    shape = visor.DENSE_SIZE if dense else visor.FRAME_SIZE
    mask = np.zeros(shape, dtype=bool)
    if frame is not None:
        for annotation in frame.annotations:
            mask |= visor.polygon_mask(annotation['segments'], source_size=shape, shape=shape)
    if dense:
        return cv2.resize(mask.astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def source_info(video: Path, ffprobe: Path) -> dict[str, Any]:
    command = [str(ffprobe), '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=codec_name,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries,r_frame_rate,avg_frame_rate,nb_frames',
               '-of', 'json', str(video)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout)['streams'][0]


def add_context(review_root: Path, rows: list[dict[str, Any]]) -> None:
    """Show source-only ontology/provenance/unknown flags without changing core UI."""
    path = review_root / 'index.html'
    html = path.read_text()
    notes = {r['id']: {'targets': r['foreground_target_names'], 'tier': r['proposal_provenance'],
                       'native': r['native_index_0based'], 'released': r['visor_frame_1based'],
                       'reason': r['blind_reason'], 'source': r['video']} for r in rows}
    encoded = json.dumps(notes).replace('<', '\\u003c')
    html = html.replace('<div id="info">', '<div id="pilot-context" style="padding:10px;background:#fff8dd"></div><div id="info">', 1)
    context_script = '''<script>
const pilotContext=__CONTEXT__, originalPilotLoad=load;
load=function(){originalPilotLoad();const c=pilotContext[review.frames[index].id];
document.getElementById('pilot-context').textContent=`Target: visible hands/forearms and released active objects (${c.targets.join(', ')}). Exclude unrelated scene objects. Video ${c.source}: native frame ${c.native} (0-based) = VISOR frame ${c.released} (1-based). Proposal: ${c.tier}. ${c.reason||'Review all boundaries and missing visible target objects.'} Polygon corrections REPLACE the whole union. Unknown proposal is NOT empty ground truth. Mark uncertain if visibility is unclear.`;};load();
</script>'''.replace('__CONTEXT__', encoded)
    path.write_text(html + context_script)


def export(intake_path: Path, output: Path, ffmpeg: Path) -> dict[str, Any]:
    started, cpu = time.monotonic(), time.process_time()
    if output.exists():
        raise FileExistsError(output)
    intake = json.loads(intake_path.read_bytes())
    if intake['status'] != 'provisional_source_reservation_only__annotation_ineligible__confirmation_not_opened':
        raise ValueError('confirmation has changed status; inspect before preparing review')
    candidates = intake['selected_provisional']
    if {r['video'] for r in candidates} != set(VIDEOS):
        raise ValueError('fixed reserved source groups differ')
    if sha(ffmpeg) != FFMPEG_SHA:
        raise ValueError('packed ffmpeg identity mismatch')
    ffprobe = ffmpeg.with_name('ffprobe')
    dataset = intake['dataset']
    manifest_path = Path(dataset['manifest'])
    if sha(manifest_path) != dataset['manifest_sha256']:
        raise ValueError('installed dataset manifest differs from intake')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    frames_manifest, records, commands = [], [], []
    with zipfile.ZipFile(dataset['outer_zip']) as archive:
        mapping = archive.read(candidates[0]['annotation_check']['mapping_member'])
        if byte_sha(mapping) != candidates[0]['annotation_check']['mapping_file_sha256']:
            raise ValueError('official sparse-to-native mapping file differs')
        mapping_document = json.loads(mapping)
        for candidate in candidates:
            video_name = candidate['video']
            first, last = VIDEOS[video_name]
            if candidate['requested_native_indices_0based'] != list(range(first, last + 1)):
                raise ValueError('reserved window changed')
            video_path = Path(candidate['source_actual_hash_readback']['path'])
            if sha(video_path) != candidate['source_actual_hash_readback']['sha256']:
                raise ValueError('actual source video differs from accepted intake hash')
            info = source_info(video_path, ffprobe)
            if info != candidate['native_info']:
                raise ValueError('native video metadata differs from accepted intake')
            sparse_blob = archive.read(candidate['sparse_member'])
            dense_blob = archive.read(candidate['dense_member'])
            check = candidate['annotation_check']
            if byte_sha(sparse_blob) != check['sparse_json_sha256'] or byte_sha(dense_blob) != check['dense_zip_sha256']:
                raise ValueError('official annotation member differs from intake')
            sparse_doc, dense_doc = visor.load_annotations(sparse_blob), visor.load_annotations(dense_blob)
            sparse, dense = visor.frames(sparse_doc), visor.frames(dense_doc)
            video_mapping = mapping_document[video_name]
            if byte_sha(canonical(video_mapping)) != check['mapping_video_canonical_sha256']:
                raise ValueError('video mapping canonical identity differs')
            anchors = visor.keyframe_anchors(sparse_doc, video_mapping, 50.0)
            actual_anchors = {n - 1 for n in sparse if first <= n - 1 <= last}
            if any(anchors.get(n + 1) != n for n in actual_anchors):
                raise ValueError('human frame0/frame1 mapping is not exact')
            dense_alignment = visor.frame_alignment(dense_doc, anchors, 1.0)
            selected = select_indices(first, last, sorted(actual_anchors))
            available = [n for n in selected if n + 1 in sparse or n + 1 in dense]
            if not available:
                raise ValueError('no available proposal for mechanical blind control')
            blind_control = min(available, key=lambda n: hash_rank(f'{video_name}:{n}'))
            targets = sorted({a['name'] for doc in (sparse, dense) for n, frame in doc.items()
                              if first <= n - 1 <= last for a in frame.annotations})
            pattern = inputs / (video_name + '-source-%02d.png')
            select_filter = '+'.join(f'eq(n\\,{n})' for n in selected)
            command = [str(ffmpeg), '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1',
                       '-i', str(video_path), '-map', '0:v:0', '-an', '-vf',
                       'select=' + select_filter + ',scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=full',
                       '-fps_mode', 'passthrough', '-frames:v', str(len(selected)), '-pix_fmt', 'rgb24',
                       '-threads:v', '1', str(pattern)]
            before = time.monotonic()
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if done.returncode:
                raise RuntimeError(f'ffmpeg source-only export failed: {done.stderr[-2000:]}')
            commands.append({'video': video_name, 'command': command, 'wall_seconds': time.monotonic() - before})
            for ordinal, index in enumerate(selected, 1):
                number = index + 1
                manual = number in sparse
                present = manual or number in dense
                frame = sparse[number] if manual else dense.get(number)
                if present and not manual:
                    alignment = dense_alignment.get(number)
                    if alignment is None or alignment['index'] != index or not alignment['exact'] or alignment['drift'] != 0:
                        raise ValueError('dense proposal has unverified source alignment')
                if not present:
                    proposal_tier = 'unknown_no_proposal_NOT_empty_ground_truth'
                    reason = 'Unknown annotation: draw complete visible target union or explicitly confirm empty; no zero GT is supplied.'
                else:
                    proposal_tier = 'official_sparse_human_proposal' if manual else 'official_dense_interpolated_proposal'
                    reason = 'Mechanical blind control: proposal hidden; redraw complete union or explicitly confirm empty.' if index == blind_control else ''
                blind = not present or index == blind_control
                source = inputs / (video_name + f'-source-{ordinal:02d}.png')
                proposal = inputs / (video_name + f'-proposal-{index}.png')
                if not source.is_file():
                    raise ValueError('selected source PNG unavailable')
                with Image.open(source) as image:
                    if image.size != (1920, 1080) or image.mode != 'RGB' or image.format != 'PNG':
                        raise ValueError('native source geometry/lossless PNG format differs')
                mask = union_proposal(frame, dense=present and not manual)
                Image.fromarray(mask.astype(np.uint8) * 255).save(proposal)
                frame_id = f'{video_name}/native-{index}/visor-{number}'
                frames_manifest.append(dict(id=frame_id, source_group=video_name, frame_index=index, blind=blind,
                                            source=dict(path=source.name, sha256=sha(source)),
                                            proposal=dict(path=proposal.name, sha256=sha(proposal))))
                records.append(dict(id=frame_id, video=video_name, participant=candidate['participant'], native_index_0based=index,
                                    visor_frame_1based=number, original_video_sha256=candidate['source_actual_hash_readback']['sha256'],
                                    source_png_sha256=sha(source), proposal_png_sha256=sha(proposal),
                                    proposal_provenance=proposal_tier, proposal_annotation_known=present,
                                    official_manual_anchor=manual, blind=blind, blind_reason=reason,
                                    mechanical_blind_control=index == blind_control,
                                    proposal_object_names=[] if frame is None else sorted({a['name'] for a in frame.annotations}),
                                    foreground_target_names=targets,
                                    official_sparse_sha256=check['sparse_json_sha256'], official_dense_zip_sha256=check['dense_zip_sha256'],
                                    source_frame_mapping='native zero-based index = released VISOR one-based frame minus1; exact50fps mapping',
                                    proposal_raster='manual native1920x1080 polygons; dense854x480 raster nearest resized to1920x1080; unknown placeholder invisible',
                                    native_info=info))
    manifest = dict(version=1, review_id='visor-confirmation-source-review18-v1', frames=frames_manifest)
    manifest_path = inputs / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    review = prepare(manifest_path, output / 'review')
    add_context(output / 'review', records)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(schema=1, status='source_only_pilot_awaiting_human_review_NOT_confirmation_eligible',
                  intake_sha256=sha(intake_path), seed=SEED, frames=records, count=len(records),
                  selection_rule='all manual anchors and both endpoints; farthest-gap spread with earlier-index ties; six/video',
                  annotation_reference_status='none newly human reviewed; full192-frame reference creation remains pending',
                  proposal_unknown_count=sum(not r['proposal_annotation_known'] for r in records),
                  blind_count=sum(r['blind'] for r in records), mechanical_blind_count=sum(r['mechanical_blind_control'] for r in records),
                  review_sha256=review['review_sha256'], manifest_sha256=sha(manifest_path),
                  source_render='native geometry/cadence preserved; selected decoded BT709 full-range source frames rendered RGB8, stored losslessly as PNG; no codec/model output',
                  ffmpeg_sha256=FFMPEG_SHA, ffprobe_sha256=sha(ffprobe), commands=commands,
                  python=sys.version, script_sha256=sha(Path(__file__)), visor_module_sha256=sha(Path(visor.__file__)),
                  wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
                  child_cpu_seconds=child.ru_utime+child.ru_stime, gpu_seconds=0,
                  limitations=['pilot18 does not repair all192 annotation records', 'P35unknown frames require explicit redraw, never inferred empty',
                               'P11same-frame contradictions are included; manual proposals are distinguished', 'P12intermediate tap visibility remains uncertain, not proven missing',
                               'sourceparticipant exposure and HInt unresolved provenance remain as intake', 'marking time is unmeasured until actual human decisions'])
    (output / 'pilot-provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'README.txt').write_text('Source-only VISOR pilot: visible hands/forearms and released active-object targets shown for each video.\n'
                                      'No codec outputs or model generation.18 frames only; not all192 reference creation.\n'
                                      'Official manual masks are proposals at sparse anchors; other masks are interpolated proposals.\n'
                                      'Unknown records have NO annotation; invisible zero placeholders are NOT empty ground truth.\n'
                                      'Blind frames require full redraw or explicit empty confirmation. Corrections replace the whole union.\n'
                                      'Use uncertain/rejected flags where visibility is unclear. Do not discard such cases.\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.intake, args.out, args.ffmpeg)
    print(json.dumps({k: result[k] for k in ('status', 'count', 'blind_count', 'proposal_unknown_count', 'review_sha256', 'wall_seconds', 'process_cpu_seconds', 'child_cpu_seconds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
