"""Tiny-frame smoke for the E1 harness. Not a 4K sweep.

Encodes two 64x64 frames once as the conventional fallback and once as the
independent reference, checks identity/checkpoint helpers, and refuses a
mismatched curve. Does not load long scenes or call PointStream ``run()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.tier.low_rate_checkpoint import load_checkpoint, save_checkpoint
from experiments.tier.low_rate_fallback import run_fallback_control
from experiments.tier.low_rate_identity import (
    assert_same_input,
    input_identity,
    references_path,
)
from experiments.tier.low_rate_validate import DECLARED_FPS, OUT_DIR
from src.contracts.config import FallbackConfig


def _tiny_clip() -> np.ndarray:
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    frames[:, 16:48, 16:48] = 180
    frames[1, 20:44, 20:44] = 40
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--qp", type=int, default=63)
    parser.add_argument("--preset", default="0")
    args = parser.parse_args(argv)

    dest_dir = Path(args.out_dir) if args.out_dir else OUT_DIR / "smoke"
    dest_dir.mkdir(parents=True, exist_ok=True)

    identity = input_identity(
        video="smoke",
        scenes=("tiny_a", "tiny_b"),
        frames_per_scene=2,
        codec=args.codec,
        fps=DECLARED_FPS,
    )
    path = references_path(identity, root=dest_dir)
    if "n2" not in path.name or "tiny_a+tiny_b" not in path.name:
        raise SystemExit(f"smoke identity missing from path: {path.name}")

    other = input_identity(
        video="smoke",
        scenes=("tiny_a", "tiny_b"),
        frames_per_scene=96,
        codec=args.codec,
        fps=DECLARED_FPS,
    )
    try:
        assert_same_input(identity, other)
    except SystemExit:
        pass
    else:
        raise SystemExit("identity check accepted a duration mismatch")

    ckpt = dest_dir / "points"
    save_checkpoint(ckpt, "probe", {"bytes": 1, "name": "probe"})
    resumed = load_checkpoint(ckpt, "probe")
    if resumed is None or resumed.get("bytes") != 1:
        raise SystemExit("checkpoint did not round-trip")
    if load_checkpoint(ckpt, "missing") is not None:
        raise SystemExit("missing checkpoint should be None")

    result = run_fallback_control(
        _tiny_clip(),
        FallbackConfig(),
        codec=args.codec,
        qp=int(args.qp),
        preset=str(args.preset),
        fps=DECLARED_FPS,
    )
    comparison = result["comparison"]
    print(
        f"fallback {result['fallback']['bytes']} B vs "
        f"reference {result['reference']['bytes']} B  "
        f"rate_rel={comparison['rate_rel']}  held={comparison['held']}",
        flush=True,
    )
    save_checkpoint(ckpt, "fallback-equivalence", result)
    if not comparison["held"]:
        print("ALARM conventional-fallback did not reproduce the reference", flush=True)
        return 1
    print(f"smoke ok  {dest_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
