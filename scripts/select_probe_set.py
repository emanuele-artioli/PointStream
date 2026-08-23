"""Deprecated wrapper. Do not use this as the probe-set regenerator.

The v1 writer in this file produced a silently unusable set: the manifest
and the view named different clips, and conditioning frames were resolved
by reconstructing ``frame_{source_id}.png`` under ``_skeleton``. The v2
path is ``python -m experiments.probe_set regenerate``.

This script exists so an old command line still hits that path instead of
rewriting v1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.probe_set.materialize import regenerate
from experiments.probe_set.schema import (
    DEFAULT_CLIP_LEN_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_NUM_CLIPS,
    DEFAULT_SEED,
    TRAINING_SPLIT_VIDEOS,
    ProbeSetError,
)
from experiments.probe_set.verify import verify


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("assets/dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/probe_set"),
        help="Probe-set tree to write (manifest, clips view, training view).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Deprecated. If set, the output tree is this file's parent directory.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=list(TRAINING_SPLIT_VIDEOS),
        help="Training-split videos to sample from. Held-out videos are rejected.",
    )
    parser.add_argument("--num-clips", type=int, default=DEFAULT_NUM_CLIPS)
    parser.add_argument("--clip-len-frames", type=int, default=DEFAULT_CLIP_LEN_FRAMES)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--materialize-training-view",
        type=Path,
        default=None,
        help="Ignored. The v2 regenerator always writes training_view/ inside --output.",
    )
    parser.add_argument(
        "--refresh-training-view",
        action="store_true",
        help="Ignored. The v2 regenerator always rebuilds the training view.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output = args.manifest.parent if args.manifest is not None else args.output
    print(
        "select_probe_set.py is not the regenerator; calling "
        f"experiments.probe_set regenerate -> {output}",
        file=sys.stderr,
    )
    try:
        manifest = regenerate(
            dataset_root=args.dataset_root,
            output_dir=output,
            seed=args.seed,
            num_clips=args.num_clips,
            clip_len_frames=args.clip_len_frames,
            min_frames=args.min_frames,
            training_videos=tuple(args.videos),
        )
    except (ProbeSetError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"Wrote {manifest['schema']} ({manifest['num_probe_clips']} clips, "
        f"coordinate_system={manifest['coordinate_system']}) to {output}"
    )
    verify(
        output,
        dataset_root=args.dataset_root,
        check_locked_split=tuple(args.videos) == TRAINING_SPLIT_VIDEOS,
    )
    print("Verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
