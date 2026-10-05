"""Bounded prompt pilot for one domain: which text does each backend need?

SAM 3.1 is the reference, so its own prompts cannot be scored against it; each
SAM variant gets coverage numbers and a contact sheet for visual review. YOLOE
variants are benchmarked against every SAM variant and get sheets too.

    python -m experiments.segmentation.prompt_pilot --domain egocentric --frames 48 --clips 2 --out DIR

Exploratory and not citable: the variants below are the hypotheses under test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.segmentation import REFERENCE_BACKEND, ClipMasks, load_domain
from src.segmentation import __main__ as cli

# name -> (backend, --prompt overrides, --option overrides)
VARIANTS: dict[str, dict[str, tuple[str, list[str], list[str]]]] = {
    "egocentric": {
        "sam_arm_hand": ("sam31", [], []),
        "sam_forearm_hand": ("sam31", ["arm=forearm"], []),
        "sam_person_hand": ("sam31", ["arm=person"], []),
        "yoloe-26x_arm_hand": ("yoloe-26x", [], []),
        "yoloe-26x_arm_hand_open": (
            "yoloe-26x",
            [],
            ["yoloe:new_track_conf=0.03", "yoloe:min_hits=1", "yoloe:hold_frames=0"],
        ),
        "yoloe-26x_person_hand": ("yoloe-26x", ["yoloe:arm=person"], []),
        "yoloe-26x_person_glove": ("yoloe-26x", ["yoloe:arm=person", "yoloe:hand=glove"], []),
        "yoloe-26n_person_hand": ("yoloe-26n", ["yoloe:arm=person"], []),
    },
    "tennis": {
        "sam_player_racket": ("sam31", [], []),
        "sam_person_racket": ("sam31", ["player=person"], []),
        "yoloe-26x_player_racket": ("yoloe-26x", [], []),
        "yoloe-26x_person_racket": ("yoloe-26x", ["yoloe:player=person"], []),
        "yoloe-26n_person_racket": ("yoloe-26n", ["yoloe:player=person"], []),
    },
}


def coverage(run: Path) -> dict[str, object]:
    masks = ClipMasks.load(run)
    area = masks.height * masks.width
    per_class = {
        name: sum(1 for frame in masks.frames if any(i.class_name == name for i in frame))
        for name in masks.classes
    }
    fg = [masks.foreground(i).mean() for i in range(len(masks))]
    return {
        "frames": len(masks),
        "frames_with_class": per_class,
        "mean_fg_fraction": round(float(sum(fg) / len(fg)), 4) if fg else None,
        "pixels_per_frame": area,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--clips", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or Path(os.environ["PS_STAGE_DIR"])
    clips = cli.resolve_sources(args.domain, None)[: args.clips]
    summary: dict[str, object] = {"domain": args.domain, "frames": args.frames, "variants": {}}
    for name, (backend, prompts, options) in VARIANTS[args.domain].items():
        flags = [f for p in prompts for f in ("--prompt", p)]
        flags += [f for o in options for f in ("--option", o)]
        code = cli.main(
            [
                "suite",
                "--domain",
                args.domain,
                "--backends",
                backend,
                "--max-frames",
                str(args.frames),
                "--limit",
                str(args.clips),
                "--out",
                str(out / name),
                *flags,
            ]
        )
        if code:
            raise SystemExit(f"variant {name} failed")
        runs = {clip: out / name / args.domain / backend / clip for clip, _ in clips}
        summary["variants"][name] = {  # type: ignore[index]
            "backend": backend,
            "prompts": load_domain(args.domain)
            .with_overrides(prompts, options)
            .prompts_for("sam" if backend == REFERENCE_BACKEND else "yoloe"),
            "options": options,
            "coverage": {clip: coverage(run) for clip, run in runs.items()},
        }
    sams = [n for n, v in VARIANTS[args.domain].items() if v[0] == REFERENCE_BACKEND]
    others = [n for n, v in VARIANTS[args.domain].items() if v[0] != REFERENCE_BACKEND]
    agreement = {}
    for ref in sams:
        report = cli.bench(
            out / ref / args.domain / REFERENCE_BACKEND,
            [out / n / args.domain / VARIANTS[args.domain][n][0] for n in others],
        )
        # bench rows run clip-major, candidate-minor.
        agreement[ref] = [
            {
                "variant": others[i % len(others)],
                "clip": row["clip"],
                **{k: row["scopes"]["foreground"][k] for k in ("J", "F", "recall", "precision")},
            }
            for i, row in enumerate(report["rows"])
        ]
    summary["agreement"] = agreement
    for clip, source in clips:
        step = max(1, args.frames // 4)
        frames = ",".join(str(i) for i in range(0, args.frames, step))
        names = list(VARIANTS[args.domain])
        cli.main(
            [
                "sheet",
                "--source",
                str(source),
                "--frames",
                frames,
                "--width",
                "360",
                "--out",
                str(out / "sheets" / f"{clip}.jpg"),
                "--labels",
                ",".join(names),
                "--runs",
                *[str(out / n / args.domain / VARIANTS[args.domain][n][0] / clip) for n in names],
            ]
        )
    (out / "pilot.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def validate(argv: list[str] | None = None) -> int:
    """Fleet smoke gate: every variant covered every clip and every sheet exists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--domain", required=True, choices=sorted(VARIANTS))
    args = parser.parse_args(argv)
    out = Path(os.environ["PS_STAGE_DIR"])
    checks = []
    pilot = json.loads((out / "pilot.json").read_text()) if (out / "pilot.json").is_file() else {}
    checks.append({"name": "pilot summary written", "passed": bool(pilot)})
    for name in VARIANTS[args.domain]:
        cov = (pilot.get("variants") or {}).get(name, {}).get("coverage") or {}
        ok = bool(cov) and all(c["frames"] == pilot["frames"] for c in cov.values())
        checks.append({"name": f"{name}: every clip segmented", "passed": ok, "detail": cov})
    sheets = sorted((out / "sheets").glob("*.jpg"))
    checks.append(
        {"name": "contact sheets", "passed": bool(sheets), "detail": [p.name for p in sheets]}
    )
    result = {"passed": all(c["passed"] for c in checks), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(validate() if "--validate" in sys.argv else main())
