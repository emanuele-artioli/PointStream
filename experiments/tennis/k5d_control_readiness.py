"""CPU-only readiness for a full-context K5d stage; never scores or encodes."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5b_trackers as trk, k5d_checkpoint as cp
from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle


def jpeg_size(header: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding pixels or loading a model."""
    f = io.BytesIO(header)
    if f.read(2) != b'\xff\xd8':
        raise ValueError('invalid JPEG start marker')
    while True:
        b = f.read(1)
        if not b:
            raise ValueError('JPEG header lacks dimensions')
        if b != b'\xff':
            raise ValueError('invalid JPEG marker sequence')
        marker = f.read(1)
        while marker == b'\xff':
            marker = f.read(1)
        if not marker or marker in (b'\xd9', b'\xda', b'\x00'):
            raise ValueError('JPEG dimensions absent before image data')
        if marker in (b'\x01', *[bytes([i]) for i in range(0xd0, 0xd8)]):
            continue
        length = f.read(2)
        if len(length) != 2:
            raise ValueError('truncated JPEG segment')
        size = int.from_bytes(length, 'big')
        if size < 2:
            raise ValueError('invalid JPEG segment length')
        if marker[0] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            data = f.read(5)
            if len(data) != 5:
                raise ValueError('truncated JPEG dimensions')
            return int.from_bytes(data[3:5], 'big'), int.from_bytes(data[1:3], 'big')
        f.seek(size - 2, io.SEEK_CUR)


def source(archive: Path, clip_id: str, frames: int) -> dict[str, Any]:
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(archive) != fleet.EXPECTED['archive']:
        raise ValueError('source selection or archive differs')
    expected = oracle.select(json.loads(fleet.SOURCE.read_text()), clip_id, 0, frames)
    if frames != expected['label_rows']:
        raise ValueError('readiness requires entire registered clip')
    game = int(clip_id.split('/')[1][4:])
    clips = [c for c in trk.tracknet_clips(archive, [game]) if c['id'] == clip_id]
    if len(clips) != 1:
        raise ValueError('source clip missing or ambiguous')
    clip = clips[0]
    oracle.validate_rows(clip['rows'], expected)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or hashlib.sha256(z.read(expected['label'])).hexdigest() != expected['label_sha256']:
            raise ValueError('source archive labels or ZIP entries differ')
        for row in clip['rows']:
            with z.open(f"{clip['dir']}/{row[0]}") as image:
                if jpeg_size(image.read(65536)) != (1280, 720):
                    raise ValueError('source frame JPEG dimensions differ')
    return {'clip': clip_id, 'frames': frames, 'labels': len(clip['rows']),
            'visibility_counts': expected['visibility_counts'], 'archive_sha256': fleet.EXPECTED['archive'],
            'scope': 'full timeline labels and JPEG headers; no source image decode'}


def asset_inputs(args: Any, stage: Path) -> dict[str, Any]:
    """Check data/model files without consulting GPU or execution metadata."""
    staged = {r['name']: r for r in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('archive', 'balltrack_weight', 'wasb_config', 'image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone'):
        if name == 'archive':
            record = staged[name]
            if record['sha256'] != fleet.EXPECTED[name] or Path(record['path']) != Path(args.archive):
                raise ValueError('staged source identity differs')
            # source() checks the actual archive immediately before scanning it.
            continue
        if oracle.sha(Path(getattr(args, name))) != fleet.EXPECTED[name]:
            raise ValueError('readiness asset identity differs: ' + name)
    if oracle.sha(Path(args.wasb_weights) / 'wasb_tennis_best.pth.tar') != fleet.EXPECTED['wasb_weights']:
        raise ValueError('readiness WASB weight differs')
    trees = {}
    for name in ('balltrack_code', 'wasb_code'):
        path = Path(getattr(args, name))
        archive = path.parent / Path(staged[name]['source']).name
        if oracle.sha(archive) != staged[name]['sha256']:
            raise ValueError('model source archive differs')
        trees[name] = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(path) != trees[name]:
            raise ValueError('model source tree differs')
    oracle.bind_config(Path(args.wasb_code), fleet.EXPECTED['wasb_config'])
    return {'model_source_tree_sha256': trees, 'source_hash_deferred_to_scan': True}


def inputs(args: Any, stage: Path, identity: dict[str, str]) -> dict[str, Any]:
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (dispatch['staged_environment']['sha256'] != identity['environment_sha256']
            or dispatch['code']['source_sha256'] != identity['source_sha256']
            or dispatch['code']['snapshot_sha256'] != identity['snapshot_sha256']
            or 'RTX A6000' not in dispatch['environment']['gpu']['name']
            or 'Ada' in dispatch['environment']['gpu']['name']):
        raise ValueError('source, runtime, or A6000 GPU class differs')
    checked = asset_inputs(args, stage)
    source_result = source(Path(args.archive), args.clip, args.frames)
    enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                      purpose=getattr(args, 'fidelity_purpose', 'full-retrospective-benchmark'),
                      clip=args.clip, frames=args.frames)
    checkpoint = None
    if args.action == 'score':
        checkpoint = cp.verify(Path(args.checkpoint) / 'publish', args.checkpoint_sha256, identity)
        if checkpoint['clip'] != args.clip or checkpoint['frames'] != args.frames:
            raise ValueError('checkpoint clip or native context differs')
    return {'source': source_result, 'model_source_tree_sha256': checked['model_source_tree_sha256'], 'identity': identity,
            'checkpoint_sha256': args.checkpoint_sha256 if checkpoint else None,
            'scope': 'CPU asset, source header and checkpoint readiness; no GPU scoring'}
