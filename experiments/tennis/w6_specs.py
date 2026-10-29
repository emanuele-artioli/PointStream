"""W6 fleet job specifications.

    python -m experiments.tennis.w6_specs OUT_DIR generate            # one job per Al player
    python -m experiments.tennis.w6_specs OUT_DIR masks  GEN_TAR GEN_TAR
    python -m experiments.tennis.w6_specs OUT_DIR judge  GEN_TAR GEN_TAR MASKS_TAR

``GEN_TAR`` and ``MASKS_TAR`` are PATH@SHA256 of the ``published.tar`` of the earlier stage's full run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from experiments.tennis.w1_roi import ANCHOR_SHA, SOURCE_SHA, TRANSPORT_SHA

SAM_SHA = 'FILL'  # sam3.1_multiplex.pt
DATA = '/home/itec/emanuele/Datasets/pointstream-data'
MODELS = '/home/itec/emanuele/Models'
TEMPLATE = Path(__file__).with_name('w6_generate_template.json')
UNITS = ('alcaraz_hurkacz__031_at129__person_000000000001', 'alcaraz_hurkacz__031_at129__person_000000000006')
ENVIRONMENT = {'path': DATA + '/environments/pointstream-20261009T214002Z.tar.gz',
               'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'}
FILES = {
    'al': (DATA + '/tennis/k6b-confirm-20261024-execution/sources/alcaraz_hurkacz__031.mp4', SOURCE_SHA),
    'transport': (DATA + '/jobs/fleet/inbox/20261025T082944Z-29f021a5/full/published.tar', TRANSPORT_SHA),
    'anchor': (DATA + '/jobs/fleet/inbox/20261024T201059Z-29586984/full/published.tar', ANCHOR_SHA),
    'old_generated': (DATA + '/tennis/k6b-reference-quality-20261026T093850Z/raw/generated-full-al-0.tar',
                      '6a3f620b394871154290340d7700831aea62f098e2569bf0d130e6735a4a1ef1'),
    'old_masks': (DATA + '/tennis/k6b-receiver-frame-20261026T192107Z/evidence/raw/k6b-frame-full-published.tar',
                  '610a09038d13a8ad0b5aab84489fdb6128c46361b2853060cb0151b232401768'),
    'sam': (DATA + '/tennis/k6b-sam-inputs-20261026T182100Z/sam3.1_multiplex.pt', SAM_SHA),
    'base': (DATA + '/jobs/fleet/inbox/20261027T072713Z-2888cefe/full/published.tar',
             '00b13e89077a27a3e6ff23bf476e9795c2120c5b1f5252ddfdd6f533a8119a39'),
    'gpu_base': (DATA + '/jobs/fleet/inbox/20261027T074823Z-135421ee/full/published.tar',
                 '5c4a0cfced5501e085f67d394f3a50ca7f92d2cb592f73fbd3ad661461cb691b'),
    'encode_half': (DATA + '/jobs/fleet/inbox/20261029T092919Z-a9e696a8/full/published.tar',
                    'daf906285f206c7821c29856eddf12a6e39c390319ac5c501e958876415faa87'),
    'pose': (MODELS + '/YOLO/yolo11x-pose.pt', '013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8'),
}
BOTH = ['RTX 6000 Ada', 'RTX A6000']


def pair(argument: str) -> tuple[str, str]:
    """``PATH@SHA256`` (the server's files are not mounted where the specification is written)."""
    path, _, digest = argument.rpartition('@')
    if len(digest) != 64:
        raise SystemExit(f'give {argument} as PATH@SHA256')
    return path, digest


def deadline(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def staged(items: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    return [{'name': n, 'path': p, 'sha256': d, 'extract': False} for n, (p, d) in items.items()]


def shared(basis: str) -> dict[str, Any]:
    return {'gpu_sharing': {'mode': 'shared', 'basis': basis, 'margin_mib': 8192},
            'contention': {'policy': 'continue', 'basis': basis, 'resume_attempts': 0}}


def generate(unit: str, gpu_class: str = 'RTX A6000') -> dict[str, Any]:
    template = json.loads(TEMPLATE.read_text())
    arguments = list(template['arguments'])
    start = arguments.index('--') + 1
    native = arguments[start:]
    native[native.index('--units') + 1] = unit
    native[native.index('--gpu-class') + 1] = gpu_class
    native.append('--keep-all')
    ada = gpu_class == 'RTX 6000 Ada'
    basis = 'Generation outputs do not depend on wall time (fixed seeds); peak 12.2 GiB in the model-GPU table.'
    items = template['staged_inputs']
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu5', 'gpu6'] if ada else ['gpu3'], 'gpu_models': [gpu_class], 'gpu_memory_mib': 24000 if ada else 16000, 'cpu_threads': 8,
            'local_storage_gib': 40, 'environment': template['environment'], 'entrypoint': ['-m', 'experiments.tennis.k6b_generate'],
            'arguments': native, 'scale': {'frames': {'smoke': 8, 'full': 0}}, 'staged_inputs': items,
            'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in items], 'required_commands': [],
            'smoke': {'seconds': 400, 'representative_basis': 'The same player unit, first 8 frames, frozen 30 DDIM steps / 512x784 / 4 bit / C base plus matching adapter / guidance 2.0, the released long-video sampler and --keep-all; the full stage renders all 240 frames of the unit.'},
            'full': {'seconds': 2400}, 'validator': ['{python}', '-m', 'experiments.tennis.w6', 'validate-generate'], 'validator_seconds': 60,
            'budget_seconds': 3000, 'stall_seconds': 1500.0, 'deadline': deadline(10), **shared(basis)}


def masks(generated: list[str]) -> dict[str, Any]:
    items = {**{k: FILES[k] for k in ('transport', 'al', 'old_generated', 'old_masks')}, 'weights': FILES['sam'],
             **{f'g{i}': pair(g) for i, g in enumerate(generated)}}
    arguments = ['masks', '--transport', '{staged:transport}', '--source', '{staged:al}', '--old-generated', '{staged:old_generated}',
                 '--old-masks', '{staged:old_masks}', '--weights', '{staged:weights}', '--judged', '{judged}']
    for i in range(len(generated)):
        arguments += ['--generated', f'{{staged:g{i}}}']
    basis = 'SAM predictions on saved crops are deterministic and do not depend on wall time; 12 GB plus the sharing margin.'
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu2', 'gpu3', 'gpu5', 'gpu6'], 'gpu_models': BOTH, 'gpu_memory_mib': 12000, 'cpu_threads': 4,
            'local_storage_gib': 35, 'environment': ENVIRONMENT, 'required_commands': ['ffmpeg'],
            'entrypoint': ['-m', 'experiments.tennis.w6'], 'arguments': arguments, 'scale': {'judged': {'smoke': 2, 'full': 0}},
            'staged_inputs': staged(items), 'inputs': [{'path': p, 'sha256': d} for p, d in items.values()],
            'smoke': {'seconds': 560, 'representative_basis': 'Both Al players, first 2 regenerated frames each, the isolated Landlock worker with every-frame fresh prompts (k6b_receiver_sam --independent-frames), the comparison with the saved crops and masks; the full stage runs all 240 frames.'},
            'full': {'seconds': 2400}, 'validator': ['{python}', '-m', 'experiments.tennis.w6', 'validate-masks'], 'validator_seconds': 60,
            'budget_seconds': 3200, 'stall_seconds': 900.0, 'deadline': deadline(10), **shared(basis)}


def judge(generated: list[str], masks_tar: str) -> dict[str, Any]:
    items = {k: FILES[k] for k in ('al', 'transport', 'anchor', 'base', 'gpu_base', 'encode_half', 'pose')}
    items['masks'] = pair(masks_tar)
    for i, g in enumerate(generated):
        items[f'g{i}'] = pair(g)
    arguments = ['judge', '--al', '{staged:al}', '--transport', '{staged:transport}', '--anchor', '{staged:anchor}', '--base', '{staged:base}',
                 '--gpu-base', '{staged:gpu_base}', '--encode-half', '{staged:encode_half}', '--pose', '{staged:pose}', '--masks', '{staged:masks}',
                 '--frames', '{frames}', '--clips']
    for i in range(len(generated)):
        arguments += ['--generated', f'{{staged:g{i}}}']
    basis = 'Scores on saved pixels and a deterministic pose detector; nothing depends on wall time.'
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu3', 'gpu5', 'gpu6'], 'gpu_models': BOTH, 'gpu_memory_mib': 12000, 'cpu_threads': 8,
            'local_storage_gib': 40, 'environment': ENVIRONMENT, 'required_commands': ['ffmpeg'],
            'entrypoint': ['-m', 'experiments.tennis.w6'], 'arguments': arguments, 'scale': {'frames': {'smoke': 16, 'full': 240}},
            'staged_inputs': staged(items), 'inputs': [{'path': p, 'sha256': d} for p, d in items.values()],
            'smoke': {'seconds': 560, 'representative_basis': 'The first 16 frames through the full path: source conversion, paid scale-4 base decode with its saved ticks compared, both W1 half-resolution streams decoded with the encoder job hashes, composite with the receiver masks, YOLO11x-pose on source/hybrid/anchors, flow residuals, racket measure and both clips.'},
            'full': {'seconds': 2400}, 'validator': ['{python}', '-m', 'experiments.tennis.w6', 'validate-judge'], 'validator_seconds': 120,
            'budget_seconds': 3200, 'stall_seconds': 900.0, 'deadline': deadline(10), **shared(basis)}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    kind, rest = sys.argv[2], sys.argv[3:]
    specs: dict[str, dict[str, Any]]
    if kind == 'generate':
        specs = {f'generate-{i}': generate(u) for i, u in enumerate(UNITS)}
    elif kind == 'masks':
        specs = {'masks': masks(rest)}
    elif kind == 'judge':
        specs = {'judge': judge(rest[:-1], rest[-1])}
    else:
        raise SystemExit('kind: generate | masks | judge')
    for name, spec in specs.items():
        (out / f'{name}.json').write_text(json.dumps(spec, indent=1) + '\n')
        print(out / f'{name}.json')


if __name__ == '__main__':
    main()
