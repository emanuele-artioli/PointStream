"""CLI: ``python -m experiments.probe_set regenerate|verify ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.probe_set.materialize import regenerate
from experiments.probe_set.repair_links import repair
from experiments.probe_set.schema import (
    DEFAULT_CLIP_LEN_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_NUM_CLIPS,
    DEFAULT_SEED,
    TRAINING_SPLIT_VIDEOS,
    ProbeSetError,
)
from experiments.probe_set.verify import collect_violations, verify
from src.contracts import paths as ps_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    regen = sub.add_parser("regenerate", help="Rebuild the probe-set view and derive the manifest from it")
    regen.add_argument("--dataset-root", type=Path, default=ps_paths.assets() / "dataset")
    regen.add_argument("--output", type=Path, default=ps_paths.assets() / "probe_set")
    regen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    regen.add_argument("--num-clips", type=int, default=DEFAULT_NUM_CLIPS)
    regen.add_argument("--clip-len-frames", type=int, default=DEFAULT_CLIP_LEN_FRAMES)
    regen.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    regen.add_argument(
        "--videos",
        nargs="+",
        default=list(TRAINING_SPLIT_VIDEOS),
        help="Training-split videos to sample from. Held-out videos are rejected.",
    )

    fix = sub.add_parser(
        "repair-links",
        help="Retarget a pre-existing view's symlinks at the data root, without rebuilding",
    )
    fix.add_argument("--root", type=Path, default=None)
    fix.add_argument("--data-root", type=Path, default=None)
    fix.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change and touch nothing.",
    )

    check = sub.add_parser("verify", help="Fail loudly if the probe set is not usable")
    check.add_argument("--root", type=Path, default=ps_paths.assets() / "probe_set")
    check.add_argument("--dataset-root", type=Path, default=None)
    check.add_argument(
        "--locked-split",
        action="store_true",
        help="Also assert the locked 5-train / 2-held-out video names.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "repair-links":
        report = repair(root=args.root, data_root=args.data_root, apply=not args.dry_run)
        print(report.summary())
        for line in report.unresolved[:20]:
            print(f"  unresolved: {line}")
        if len(report.unresolved) > 20:
            print(f"  ... and {len(report.unresolved) - 20} more")
        return 0 if report.ok else 1
    if args.command == "regenerate":
        try:
            manifest = regenerate(
                dataset_root=args.dataset_root,
                output_dir=args.output,
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
            f"coordinate_system={manifest['coordinate_system']}) to {args.output}"
        )
        verify(
            args.output,
            dataset_root=args.dataset_root,
            check_locked_split=tuple(args.videos) == TRAINING_SPLIT_VIDEOS,
        )
        print("Verifier: OK")
        return 0

    violations = collect_violations(
        args.root,
        dataset_root=args.dataset_root,
        check_locked_split=args.locked_split,
    )
    if violations:
        print(ProbeSetError(violations), file=sys.stderr)
        return 1
    print(f"Verifier: OK ({args.root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
