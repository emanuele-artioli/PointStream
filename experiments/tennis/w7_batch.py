"""W7 Tier 1: the five remaining windows batched by stage (python -m experiments.tennis.w7_specs plan|batch ...).

One fleet job per stage and group of windows, so the smoke (about 650 GPU-s over the chain for one window) is paid per job, not per window.
Nothing is submitted here. What a module allows decides the groups (docs/experiments/w7-untouched.md, 2026-10-31 entry):

- features, build-inputs, anchors, prepare (`k6b_confirm*`): `--windows`/`--sources` are parallel lists and `--max-windows N` keeps the first N, so the
  smoke is the first named window and the full stage is all of them (`{windows}` 1 / 0).
- rgb diagnostic, receiver, judge, native base, rate transport, rates: these take `--window`/`--source` lists but have no first-window limiter, so the
  smoke runs EVERY named window on the smoke grid. Their smoke validators then bound the group: native base projects 37.5 x the smoke CPU
  seconds against 1,300 s, the receiver projects the full stage from the smoke against 1,500 s (601 s per Al window), the judge, receiver (`--masks`,
  `--prior`), rgb diagnostic (`--base`) and rates (`--base`) take ONE upstream archive, the judge ONE anchor archive.
- generation (per adapter, 1,324 s per Al window), DCVC (`--bracket LOW HIGH` is one pair for every window of the job, and each window's pair is
  its own hybrid rates), the w1_* stages (one `--window` by design): one job per window.

The groups are searched, not typed in: receiver/judge groups are the smallest set under their caps, anchors and native base are merged from them
under theirs, so each judge group sits inside one anchor, native-base and receiver archive.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from experiments.tennis import w7_specs as sp
from experiments.tennis import window_manifest as wm

MED_FACTOR = 1.25  # 50 fps clips: untimed, assumed up to this much slower than the 25 fps first window (lead, 2026-10-31)
CAP_FRACTION = 0.9  # a group's estimated full stage stays under this share of the 2,600 s stage cap
SMOKE_EXTRA = 0.25  # added smoke GPU-s per extra window, as a share of the one-window smoke, where the smoke runs every window (estimate)
RESERVE_MARGIN = 1.3  # `full.seconds` = estimate x this, at least the single-window reservation
NATIVE_SMOKE_S = {25: 8.0, 50: 10.0}  # native-base smoke wall seconds of one window (8 s measured, job f6699fe5); validator limit 1300 / 37.5
NATIVE_SMOKE_LIMIT = 1300 / 37.5
RECEIVER_PROJECTION_S = 601.0  # the first window's smoke projected 601 s for its full stage (job 6191ce05); the validator requires < 1,500 s
RECEIVER_PROJECTION_LIMIT = 1450.0


@dataclass(frozen=True)
class StageModel:
    gpu: bool
    smoke_s: float  # successful smoke GPU seconds of the first window (execution.json); 0 for CPU
    full_s: float  # successful full GPU seconds of the first window (25 fps)
    wall_s: float  # full-stage wall seconds per window, for the 2,600 s cap
    smoke_all: bool  # no first-window limiter: the smoke runs every named window
    kind: str  # grouping: pack | anchors | native | receiver | window


#: Measured on the first window (docs/experiments/w7-untouched.md, "Projection for the other five Tier 1 windows"). CPU stage walls: native base 58 s,
#: rate transport 18 s, prepare 27 s measured; build-inputs and rates were not timed (assumed 300 s and 60 s). DCVC full = 243 s measured +
#: about 30 s for the new bracket probe (one encode-only 240-frame candidate, not yet run); its smoke is the 66 s of the first run.
MODEL: dict[str, StageModel] = {
    'features': StageModel(True, 163, 156, 156, False, 'pack'),
    'build-inputs': StageModel(False, 0, 0, 300, False, 'pack'),
    'anchors': StageModel(True, 136, 597, 597, False, 'anchors'),
    'prepare': StageModel(False, 0, 0, 27, False, 'pack'),
    'generate': StageModel(True, 92, 1324, 1324, False, 'window'),
    'rate-transport': StageModel(False, 0, 0, 18, True, 'pack'),
    'native-base': StageModel(False, 0, 0, 58, True, 'native'),
    'rates': StageModel(False, 0, 0, 60, True, 'native'),
    'rgb-diagnostic': StageModel(True, 18.5, 107.5, 107.5, True, 'native'),
    'receiver': StageModel(True, 104, 331, 331, True, 'receiver'),
    'judge': StageModel(True, 34, 283, 283, True, 'receiver'),
    'w1-masks': StageModel(False, 0, 0, 150, False, 'window'),
    'w1-encode-half': StageModel(False, 0, 0, 600, False, 'window'),
    'w1-encode-quarter': StageModel(False, 0, 0, 600, False, 'window'),
    'w1-encode-half-out': StageModel(False, 0, 0, 900, False, 'window'),
    'w1-encode-quarter-out': StageModel(False, 0, 0, 900, False, 'window'),
    'w1-score': StageModel(True, 106, 870, 870, False, 'window'),
    'dcvc': StageModel(True, 66, 273, 273, False, 'window'),
}
STAGE_ORDER = list(MODEL)
LIMITER = {'features', 'build-inputs', 'anchors', 'prepare'}  # `--max-windows {windows}` keeps the first window in the smoke


def factor(w: wm.WindowSpec, med_factor: float = MED_FACTOR) -> float:
    return med_factor if w.fps >= 50 else 1.0


def wall(stage: str, ws: list[wm.WindowSpec], med_factor: float = MED_FACTOR) -> float:
    return sum(MODEL[stage].wall_s * factor(w, med_factor) for w in ws)


def gpu_seconds(stage: str, ws: list[wm.WindowSpec], med_factor: float = MED_FACTOR) -> tuple[float, float]:
    """(smoke, full) GPU seconds of one job: one smoke (extended by the windows of a limiter-less smoke) plus each window's full stage."""
    m = MODEL[stage]
    if not m.gpu:
        return 0.0, 0.0
    smoke = m.smoke_s * (1 + SMOKE_EXTRA * (len(ws) - 1) if m.smoke_all else 1)
    return smoke, sum(m.full_s * factor(w, med_factor) for w in ws)


def group_ok(kind: str, stage: str, ws: list[wm.WindowSpec]) -> bool:
    if kind == 'window':
        return len(ws) == 1 and wall(stage, ws) <= sp.FULL_SECONDS * CAP_FRACTION
    if kind == 'native':
        return sum(NATIVE_SMOKE_S[25 if w.fps < 50 else 50] for w in ws) <= NATIVE_SMOKE_LIMIT
    if kind == 'receiver':
        return sum(RECEIVER_PROJECTION_S * factor(w) for w in ws) <= RECEIVER_PROJECTION_LIMIT
    return wall(stage, ws) <= sp.FULL_SECONDS * CAP_FRACTION


def set_partitions(items: list[Any]) -> Iterator[list[list[Any]]]:
    if not items:
        yield []
        return
    head, rest = items[0], items[1:]
    for part in set_partitions(rest):
        yield [[head], *part]
        for i in range(len(part)):
            yield [*part[:i], [head, *part[i]], *part[i + 1:]]


def ordered(group: list[wm.WindowSpec], order: list[wm.WindowSpec]) -> list[wm.WindowSpec]:
    return sorted(group, key=order.index)


def plan_groups(windows: list[wm.WindowSpec]) -> dict[str, list[list[wm.WindowSpec]]]:
    """The groups of every stage for `windows`: receiver/judge groups first (fewest jobs), then anchors and native base/rgb/rates merged from
    them under their caps (fewest jobs, then the lowest peak load). Deterministic in the manifest order."""
    if len(windows) > 8:
        raise ValueError('plan_groups searches set partitions: at most 8 windows')
    best: tuple[Any, ...] | None = None
    for recv in set_partitions(list(windows)):
        if not all(group_ok('receiver', 'receiver', g) for g in recv):
            continue
        blocks = list(range(len(recv)))
        merged: dict[str, list[list[int]]] = {}
        for kind, stage in (('anchors', 'anchors'), ('native', 'native-base')):
            options = [p for p in set_partitions(blocks) if all(group_ok(kind, stage, sum((recv[i] for i in b), [])) for b in p)]
            if not options:
                break
            merged[kind] = min(options, key=lambda p: (len(p), max(wall(stage, sum((recv[i] for i in b), [])) for b in p)))
        else:
            key = (len(recv), len(merged['anchors']), len(merged['native']), max(wall('anchors', sum((recv[i] for i in b), [])) for b in merged['anchors']))
            if best is None or key < best[0]:
                best = (key, recv, merged)
    if best is None:
        raise ValueError('no grouping satisfies the receiver, anchor and native-base caps')
    _, recv, merged = best

    def join(blocks_: list[list[int]]) -> list[list[wm.WindowSpec]]:
        groups = [ordered(sum((recv[i] for i in b), []), windows) for b in blocks_]
        return sorted(groups, key=lambda g: windows.index(g[0]))

    out: dict[str, list[list[wm.WindowSpec]]] = {}
    receiver_groups = sorted((ordered(g, windows) for g in recv), key=lambda g: windows.index(g[0]))
    for stage, m in MODEL.items():
        if m.kind == 'anchors':
            out[stage] = join(merged['anchors'])
        elif m.kind == 'native':
            out[stage] = join(merged['native'])
        elif m.kind == 'receiver':
            out[stage] = receiver_groups
        elif m.kind == 'window':
            out[stage] = [[w] for w in windows]
        else:
            out[stage] = pack(stage, windows)
    return out


def pack(stage: str, windows: list[wm.WindowSpec]) -> list[list[wm.WindowSpec]]:
    """Fewest groups under the stage's cap (then the lowest peak), manifest order inside each."""
    options = [[ordered(g, windows) for g in p] for p in set_partitions(list(windows)) if all(group_ok('pack', stage, g) for g in p)]
    if not options:
        raise ValueError(f'{stage}: a single window exceeds the cap')
    best = min(options, key=lambda p: (len(p), max(wall(stage, g) for g in p)))
    return sorted(best, key=lambda g: windows.index(g[0]))


def plan(windows: list[wm.WindowSpec], med_factor: float = MED_FACTOR) -> dict[str, Any]:
    groups = plan_groups(windows)
    jobs: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        for k, g in enumerate(groups[stage]):
            smoke, full = gpu_seconds(stage, g, med_factor)
            jobs.append({'stage': stage, 'index': k, 'windows': [w.name for w in g], 'smoke_s': round(smoke, 1), 'full_s': round(full, 1),
                         'gpu_s': round(smoke + full, 1), 'wall_s': round(wall(stage, g, med_factor), 1), 'gpu': MODEL[stage].gpu})
    unbatched = sum(sum(gpu_seconds(stage, [w], med_factor)) for stage in STAGE_ORDER for w in windows)
    total = sum(j['gpu_s'] for j in jobs)
    return {'windows': [w.name for w in windows], 'med_factor': med_factor, 'jobs': jobs, 'total_gpu_s': round(total, 1),
            'total_gpu_h': round(total / 3600, 2), 'unbatched_gpu_s': round(unbatched, 1), 'saved_gpu_s': round(unbatched - total, 1)}


def format_plan(result: dict[str, Any], plain: dict[str, Any] | None = None) -> str:
    lines = [f"windows: {', '.join(result['windows'])}; Med windows assumed {result['med_factor']}x the 25 fps full times",
             f"{'stage':<22}{'job':>4}  {'windows':<34}{'smoke':>7}{'full':>8}{'GPU-s':>8}"]
    cpu_single: dict[str, list[str]] = {}
    for j in result['jobs']:
        if not j['gpu'] and len(j['windows']) == 1 and MODEL[j['stage']].kind == 'window':
            cpu_single.setdefault(j['stage'], []).append(j['windows'][0])
            continue
        lines.append(f"{j['stage']:<22}{j['index']:>4}  {','.join(j['windows']):<34}{j['smoke_s']:>7.0f}{j['full_s']:>8.0f}{j['gpu_s']:>8.0f}" + ('' if j['gpu'] else '   (CPU)'))
    lines += [f"{stage:<22}{'1-' + str(len(ws)):>4}  one job per window: {','.join(ws)}   (CPU, 0 GPU-s)" for stage, ws in cpu_single.items()]
    lines.append(f"total {result['total_gpu_s']:.0f} GPU-s = {result['total_gpu_h']:.2f} GPU-h; one job per window and stage would be {result['unbatched_gpu_s']:.0f} GPU-s "
                 f"({result['unbatched_gpu_s'] / 3600:.2f} GPU-h); batching saves {result['saved_gpu_s']:.0f} GPU-s")
    if plain is not None:
        lines.append(f"with the Med windows as fast as the 25 fps ones (factor {plain['med_factor']}): {plain['total_gpu_s']:.0f} GPU-s = {plain['total_gpu_h']:.2f} GPU-h "
                     f"(unbatched {plain['unbatched_gpu_s']:.0f})")
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------------- specs


def parse_ups(items: list[str]) -> dict[str, tuple[str, str]]:
    out = {}
    for a in items:
        name, value = a.split('=', 1)
        path, digest = value.rsplit(':', 1)
        out[name] = (path, digest)
    return out


def per_window_values(items: list[str] | None) -> dict[str, str]:
    return {k: v for k, v in (x.split('=', 1) for x in items or [])}


class Upstream:
    """Upstream archives by role: `role` (one archive covers every window), `roleK` (the archive of the stage's K-th job, from `plan`) or
    `role.WINDOW` (one archive per window, e.g. `generated.al034`). `pending` fills a missing one with the all-zero hash (a spec to complete)."""

    def __init__(self, ups: dict[str, tuple[str, str]], pending: bool) -> None:
        self.ups, self.pending = ups, pending

    def need(self, key: str) -> tuple[str, str]:
        if key in self.ups:
            return self.ups[key]
        if self.pending:
            return (f'{sp.W7_DIR}/pending/{key}', sp.PENDING)
        raise SystemExit(f'missing upstream archive {key}=PATH:SHA256 (or --pending)')

    def pick(self, role: str, names: list[str], groups: list[list[str]]) -> tuple[str, str]:
        if role in self.ups:
            return self.ups[role]
        for k, g in enumerate(groups):
            if set(names) <= set(g):
                return self.need(f'{role}{k}')
        raise SystemExit(f'{role}: the windows {names} span several upstream jobs {groups}; run them as separate jobs')

    def pick_all(self, role: str, names: list[str], groups: list[list[str]]) -> list[tuple[str, str]]:
        if role in self.ups:
            return [self.ups[role]]
        found = [self.need(f'{role}{k}') for k, g in enumerate(groups) if set(names) & set(g)]
        if not found:
            raise SystemExit(f'{role}: no upstream job holds {names}')
        return found

    def window(self, role: str, name: str) -> tuple[str, str]:
        return self.need(f'{role}.{name}')


def finish(stage: str, spec: dict[str, Any], ws: list[wm.WindowSpec]) -> dict[str, Any]:
    """Batch timing: the reservation follows the group's estimate (never below the single-window spec, never above 2,600 s), the basis says what the smoke covers."""
    m = MODEL[stage]
    spec['smoke']['representative_basis'] += (' In this batch job the smoke keeps only the first named window (`--max-windows 1`).' if stage in LIMITER
                                                 else ' This module has no first-window limiter, so in a batch job the smoke runs every named window on this grid.'
                                                 if m.smoke_all and len(ws) > 1 else '')
    if stage == 'dcvc':
        return spec  # the W3 validator caps the reservation (2,600 s, budget 3,300 s)
    wanted = min(sp.FULL_SECONDS, math.ceil(RESERVE_MARGIN * wall(stage, ws) / 60) * 60)
    if stage == 'rgb-diagnostic' and len(ws) > 1:
        wanted = sp.FULL_SECONDS  # the smoke validator projects 10 x the smoke against this reservation; reservation only (about 110 s per window expected)
    if wanted > spec['full']['seconds']:
        spec['full']['seconds'] = wanted
        spec['budget_seconds'] = max(spec['budget_seconds'], wanted + 800)
    assert spec['smoke']['seconds'] <= sp.SMOKE_SECONDS and spec['full']['seconds'] <= sp.FULL_SECONDS
    return spec


def check_group(stage: str, ws: list[wm.WindowSpec]) -> None:
    m = MODEL[stage]
    if not group_ok(m.kind if m.kind in ('window', 'native', 'receiver') else 'pack', stage, ws):
        raise SystemExit(f'{stage}: {[w.name for w in ws]} breaks the stage cap (one window per job / native smoke projection / receiver projection / 2,340 s)')


def build_spec(stage: str, ws: list[wm.WindowSpec], campaign: list[wm.WindowSpec], windows_ref: tuple[str, str], up: Upstream,
               opts: argparse.Namespace) -> dict[str, Any]:
    groups = {s: [[w.name for w in g] for g in gs] for s, gs in plan_groups(campaign).items()}
    names = [w.name for w in ws]
    sources = source_list(ws)
    anchor_groups, native_groups, receiver_groups = groups['anchors'], groups['native-base'], groups['receiver']
    confirm = up.need('manifest') if stage in ('features', 'anchors', 'prepare', 'generate', 'build-inputs') else windows_ref
    if stage == 'features':
        return sp.features_for(ws, confirm, sources)
    if stage == 'build-inputs':
        return sp.build_inputs_for(confirm, up.pick_all('features', names, groups['features']))
    if stage == 'anchors':
        return sp.anchors_for(ws, confirm, up.need('inputs'), sources)
    if stage == 'prepare':
        return sp.prepare_for(ws, confirm, up.need('inputs'), up.pick_all('anchor', names, anchor_groups), sources)
    if stage == 'generate':
        units = per_window_values(opts.units).get(ws[0].name)
        if units is None:
            if not opts.pending:
                raise SystemExit('generate needs --units WINDOW=unit1,unit2 (the prepare output\'s player-unit names)')
            units = f'{ws[0].safe_id}__person_0,{ws[0].safe_id}__person_1'
        return sp.generate(ws[0], confirm, up.need('prepared'), units.split(','))
    if stage == 'rate-transport':
        return sp.rate_transport(ws, windows_ref, up.pick_all('anchor', names, anchor_groups), [up.need('prepared')], 0)
    if stage == 'native-base':
        return sp.native_base(ws, windows_ref, 240)
    if stage == 'rates':
        return sp.rates_job(ws, windows_ref, up.need('transport_json'), up.pick('base', names, native_groups))
    if stage == 'rgb-diagnostic':
        return sp.rgb_diagnostic(ws, windows_ref, up.pick('base', names, native_groups), 240, shared=not opts.exclusive)
    if stage == 'receiver':
        return sp.receiver(ws, windows_ref, up.need('transport'), up.pick_all('anchor', names, anchor_groups),
                           [up.window('generated', n) for n in names], 2, all_windows=True)
    if stage == 'judge':
        files = {'windows': windows_ref, 'anchor': up.pick('anchor', names, anchor_groups), 'transport': up.need('transport'),
                 'masks': up.pick('masks', names, receiver_groups), 'gpu_base': up.pick('gpu_base', names, native_groups),
                 'base': up.pick('base', names, native_groups), 'prior': up.pick('prior', names, receiver_groups), **sp.assets('lpips', 'dists')}
        return sp.real_base_judge(ws, files, [up.window('generated', n) for n in names], up.pick('rates', names, native_groups), 0)
    if len(ws) != 1:
        raise SystemExit(f'{stage}: one window per job')
    w = ws[0]
    if stage == 'w1-masks':
        return sp.w1_masks_spec(w, windows_ref, up.pick('state', names, native_groups), up.pick('anchor', names, anchor_groups), up.need('transport'))
    if stage.startswith('w1-encode-'):
        kind = stage[len('w1-encode-'):]
        target = per_window_values(opts.target).get(w.name)
        if target is None:
            raise SystemExit(f'{stage} needs --target {w.name}=KBPS (the judge\'s scale-4 total_kbps, KPS2)')
        return sp.w1_encode_spec(w, windows_ref, up.pick('state', names, native_groups), up.need('transport'), 2 if kind.startswith('half') else 4,
                                 float(target), controls=kind == 'half', outside=kind.endswith('-out'))
    if stage == 'w1-score':
        extra = [up.window(f'encode_x{i}', w.name) for i in range(8) if f'encode_x{i}.{w.name}' in up.ups]
        return sp.w1_score_spec(w, windows_ref, up.pick('state', names, native_groups), up.window('w1masks', w.name), up.pick('hybrid', names, receiver_groups),
                                [up.window('encode_half', w.name), *extra])
    if stage == 'dcvc':
        bracket = per_window_values(opts.bracket).get(w.name)
        if bracket is None:
            raise SystemExit(f'dcvc needs --bracket {w.name}=LOW,HIGH: the window\'s hybrid KPS2 and KPS1 total kbps (the job takes one pair per window)')
        low, high = (float(x) for x in bracket.split(','))
        prior_role = 'prior' if any(k.startswith('prior') for k in up.ups) or up.pending else 'prior_frame'
        return sp.dcvc([w], windows_ref, up.pick('anchor', names, anchor_groups), up.need('transport'), up.pick(prior_role, names, receiver_groups), 25,
                       None, (low, high))
    raise SystemExit(f'unknown stage {stage}')


def source_list(ws: list[wm.WindowSpec]) -> list[tuple[str, str]]:
    return [sp.source_files([w])[w.name] for w in ws]


def batch(argv: list[str]) -> Path:
    p = argparse.ArgumentParser(prog='w7_specs batch', description='the spec of one stage for the named windows (one job); see `plan` for the jobs')
    p.add_argument('stage')
    p.add_argument('windows_json', type=Path)
    p.add_argument('out_dir', type=Path)
    p.add_argument('--windows', required=True, help='comma-separated window names of this job (the smoke keeps the first where the module has a limiter)')
    p.add_argument('--campaign', help='the windows the plan is made for (default: every manifest window but the first, which ran as the first window)')
    p.add_argument('--pending', action='store_true', help='fill missing upstream archives with the all-zero hash instead of stopping')
    p.add_argument('--exclusive', action='store_true', help='rgb diagnostic: exclusive GPU (default shared, as the job that ran)')
    p.add_argument('--units', action='append', help='generate: WINDOW=unit1,unit2 (prepare output names)')
    p.add_argument('--bracket', action='append', help='dcvc: WINDOW=LOW,HIGH kbps (KPS2 and KPS1 hybrid totals of the window)')
    p.add_argument('--target', action='append', help='w1-encode-*: WINDOW=KBPS (hybrid total at scale 4, KPS2)')
    p.add_argument('upstream', nargs='*', help='NAME=PATH:SHA256, roles NAME, NAMEk (k-th job of the upstream stage) or NAME.WINDOW')
    a = p.parse_intermixed_args(argv)
    if a.stage not in MODEL:
        raise SystemExit(f'unknown stage {a.stage}; stages: {", ".join(MODEL)}')
    every = wm.load(a.windows_json)
    by_name = {w.name: w for w in every}
    chosen_names = a.windows.split(',')
    campaign = [by_name[n] for n in a.campaign.split(',')] if a.campaign else every[1:]
    if len(set(chosen_names)) != len(chosen_names) or any(n not in by_name for n in chosen_names):
        raise SystemExit('--windows: unknown or repeated window name')
    ws = [by_name[n] for n in chosen_names]
    if not set(chosen_names) <= {w.name for w in campaign}:
        raise SystemExit('--windows must belong to the campaign')
    check_group(a.stage, ws)
    windows_ref = (sp.W7_DIR + '/windows.json', wm.file_sha256(a.windows_json))
    spec = finish(a.stage, build_spec(a.stage, ws, campaign, windows_ref, Upstream(parse_ups(a.upstream), a.pending), a), ws)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    target = a.out_dir / f"{a.stage}-{'+'.join(chosen_names)}.json"
    target.write_text(json.dumps(spec, indent=1) + '\n')
    smoke, full = gpu_seconds(a.stage, ws)
    print(f'{target} smoke {smoke:.0f} + full {full:.0f} GPU-s (estimate), full reservation {spec["full"]["seconds"]} s')
    return target


def plan_command(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog='w7_specs plan')
    p.add_argument('windows_json', type=Path)
    p.add_argument('--windows', help='comma-separated names (default: every manifest window but the first)')
    p.add_argument('--med-factor', type=float, default=MED_FACTOR)
    p.add_argument('--json', type=Path, help='also write the plan here')
    a = p.parse_args(argv)
    every = wm.load(a.windows_json)
    by_name = {w.name: w for w in every}
    ws = [by_name[n] for n in a.windows.split(',')] if a.windows else every[1:]
    result = plan(ws, a.med_factor)
    print(format_plan(result, plan(ws, 1.0) if a.med_factor != 1.0 else None))
    if a.json:
        a.json.write_text(json.dumps(result, indent=1) + '\n')

