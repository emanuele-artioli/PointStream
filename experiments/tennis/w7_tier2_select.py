"""Reproducible scene-shot draw and K8b registration for W7 Tier 2.

Register drawn shots in order, checkpointing each native 200-frame chunk. No
quality values are read here. All coordinates are decoded broadcast frames.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k8, k8b, w7_select

FRAMES = 115_505
SPLIT = 57_753
SEED = 20261030


def draw_shots(scene_text: str, half: str) -> list[tuple[int, int]]:
    scenes = k8.parse_scenes(scene_text, 50)
    cuts = sorted({int(r["frame"]) for r in scenes
                   if r["score"] >= k8b.DECISION["scene_threshold"] and 0 < r["frame"] < FRAMES})
    shots = k8.shot_spans(cuts, FRAMES)
    if half == "fit":
        eligible = [(a, b) for a, b in shots if b <= SPLIT and b - a >= 240]
    elif half == "eval":
        eligible = [(a, b) for a, b in shots if a >= SPLIT and b - a >= 240]
    else:
        raise ValueError("half must be fit or eval")
    order = np.random.default_rng(SEED).permutation(len(eligible))
    return [eligible[int(i)] for i in order]


def _register(video: str, start: int, end: int, ffmpeg: str) -> list[dict[str, Any]]:
    return k8b.register_chunk(video, start, end, ffmpeg)


def _register_cached(video: str, offset: int, start: int, end: int) -> list[dict[str, Any]]:
    from experiments.tennis.w7_tier2_cache import register_raw

    return register_raw(Path(video), offset, start, end)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    order = draw_shots(Path(args.scenes).read_text(), args.half)
    cache: dict[str, Any] | None = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache = json.loads((cache_dir / 'index.json').read_text())
        if cache is None:
            raise ValueError('cached index is null')
        raw = cache_dir / 'frames.yuv'
        from experiments.tennis.w7_tier2_cache import FRAME_BYTES, chunks_for

        starts = chunks_for(Path(args.scenes).read_text(), int(cache['count']))
        offsets: dict[str, int] = {}
        position = 0
        for start in starts:
            offsets[str(start)] = position
            position += min(k8b.CHUNK, FRAMES - start) * FRAME_BYTES
        if cache['offsets'] != offsets or cache['raw_bytes'] != position:
            raise ValueError('cached frame offsets differ from the frozen draw')
        if (cache['scene_sha256'] != w7_select.file_sha256(Path(args.scenes))
                or cache['source_sha256'] != w7_select.file_sha256(Path(args.video))
                or cache['raw_sha256'] != w7_select.file_sha256(raw)
                or raw.stat().st_size != cache['raw_bytes'] or cache['numpy'] != np.__version__):
            raise ValueError('cached input identity differs')
    selections: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for a, b in order:
            # The original broadcast is one clip: K8b's 200-frame chunk grid
            # is anchored at frame zero, not at the shot boundary.
            starts = list(range(a // k8b.CHUNK * k8b.CHUNK, b, k8b.CHUNK))
            paths = [out / f"chunk-{s:06d}.json" for s in starts]
            missing = [(s, p) for s, p in zip(starts, paths) if not p.exists()]
            if cache is not None:
                if any(str(s) not in cache['offsets'] for s, _ in missing):
                    raise ValueError(f'cache does not cover shot {a}:{b}')
                futures = [(s, p, pool.submit(_register_cached, str(raw), cache['offsets'][str(s)], s, min(s + k8b.CHUNK, FRAMES)))
                           for s, p in missing]
            else:
                futures = [(s, p, pool.submit(_register, args.video, s, min(s + k8b.CHUNK, FRAMES), args.ffmpeg))
                           for s, p in missing]
            for s, p, future in futures:
                rows = future.result()
                if [row["frame"] for row in rows] != list(range(s, min(s + k8b.CHUNK, FRAMES))):
                    raise ValueError(f"incomplete registration chunk {s}")
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(rows))
                tmp.replace(p)
            rows = [row for p in paths for row in json.loads(p.read_text()) if a <= row["frame"] < b]
            if [row["frame"] for row in rows] != list(range(a, b)):
                raise ValueError(f"incomplete shot {a}:{b}")
            result = {"shot_start": a, "shot_end": b, **w7_select.first_window(rows)}
            if result["start"] is None:
                rejected.append(result)
            else:
                result["broadcast_start"] = a + result["start"]
                selections.append(result)
            report = {"seed": SEED, "numpy": np.__version__, "half": args.half,
                      "draw_order": order, "selected": selections, "rejected": rejected}
            (out / f"{args.half}-selection.json").write_text(json.dumps(report, indent=2))
            print(args.half, a, b, "selected" if result["start"] is not None else "rejected", flush=True)
            if len(selections) == 3:
                return
    raise RuntimeError(f"only {len(selections)} shots admitted")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("half", choices=("fit", "eval"))
    p.add_argument("--scenes", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", help="verified local native-frame cache")
    run(p.parse_args())


if __name__ == "__main__":
    main()
