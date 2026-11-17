"""Audit complete reference-pilot outputs and build a durable blind review."""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis.player_requalification import sha256, validate_output
from experiments.tennis.player_requalification_preview import encode, resolve_units, source_crops

UNIT_PREFIX = "nadal_kyrgios__000_at120__person_"
UNITS = [UNIT_PREFIX + "000000000001", UNIT_PREFIX + "000000000000"]
LABELS = {UNITS[0]: "Player 2", UNITS[1]: "Player 1"}


def resolved(config_path: Path, expected: str, conditioning: Path, base: Path) -> dict[str, Any]:
    if sha256(config_path) != expected:
        raise ValueError("config hash mismatch")
    config = resolve_units(json.loads(config_path.read_text()), [["conditioning", str(conditioning)]])
    config["tennis_checkpoint"]["path"] = str(base)
    return config


def audit(config_path: Path, expected: str, conditioning: Path, base: Path,
          smoke: Path, full: Path) -> dict[str, Any]:
    """Validate both real stages before independently comparing the repeated player."""
    config = resolved(config_path, expected, conditioning, base)
    pilot = config.get("reference_pilot", {})
    if (pilot.get("quality") not in (70, 40)
            or [u["unit"] for u in config["units"]] != UNITS):
        raise ValueError("wrong pilot quality or player order")
    sr = validate_output(smoke, config, expected, "tennis_only", 1)
    fr = validate_output(full, config, expected, "tennis_only", 2)
    if sr["plan"]["units"][0] != fr["plan"]["units"][0]:
        raise ValueError("smoke/full input identity mismatch")
    filename = f"{UNITS[0]}@tennis_only.npz"
    with np.load(smoke / filename, allow_pickle=False) as a, np.load(full / filename, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("smoke/full ticks, boxes or crops differ")
    return {"quality": pilot["quality"], "smoke_receipt_sha256": sha256(smoke / "receipt.json"),
            "full_receipt_sha256": sha256(full / "receipt.json"), "repeated_unit": UNITS[0],
            "input_identities_and_output_arrays_bit_exact": True}


def build(control_config: Path, control_hash: str, control_conditioning: Path, base: Path,
          control: Path, variants: list[tuple[Path, str, Path, Path, Path]],
          source: Path, output: Path, seed: int) -> None:
    """Produce local HTML/media and a separately opened reveal key; no server."""
    if not 1 <= len(variants) <= 2 or output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("one or two completed variants and an output outside the checkout required")
    cc = resolved(control_config, control_hash, control_conditioning, base)
    cr = validate_output(control, cc, control_hash, "tennis_only", 2)
    from experiments.tennis.player_requalification import make_plan

    plan, arrays = make_plan(cc, control_hash, 2)
    if set(u["unit"] for u in plan["units"]) != set(UNITS):
        raise ValueError("control unit identity mismatch")
    source_arrays, deltas = source_crops(source, plan, arrays, control_config,
                                         [["conditioning", str(control_conditioning)]])
    by_unit = {u["unit"]: (a, s) for u, a, s in zip(plan["units"], arrays, source_arrays)}
    checks = []
    candidate_paths: dict[str, Path] = {"lossless": control}
    costs: dict[str, Any] = {}
    for config_path, digest, conditioning, smoke, full in variants:
        checked = audit(config_path, digest, conditioning, base, smoke, full)
        q = checked["quality"]
        name = f"quality-{q}"
        if name in candidate_paths:
            raise ValueError("duplicate quality")
        vc = resolved(config_path, digest, conditioning, base)
        vr = json.loads((full / "receipt.json").read_text())
        control_input = {u["unit"]: u["path"] for u in cc["units"]}
        for unit in vc["units"]:
            with np.load(control_input[unit["unit"]], allow_pickle=False) as original, np.load(unit["path"], allow_pickle=False) as variant:
                if (set(original.files) != set(variant.files)
                        or any(not np.array_equal(original[k], variant[k]) for k in original.files if k != "reference")):
                    raise ValueError("variant changed non-reference conditioning")
        control_units = {u["unit"]: u for u in cr["plan"]["units"]}
        for u in vr["plan"]["units"]:
            c = control_units[u["unit"]]
            if any(u[k] != c[k] for k in ("unit", "run", "seed", "t", "pose_sha256", "box_sha256")):
                raise ValueError("variant differs from matched control conditioning")
        if any(vr["plan"][k] != cr["plan"][k] for k in ("source", "settings")):
            raise ValueError("variant source or settings mismatch")
        candidate_paths[name] = full
        costs[name] = sum(x["encoded_ref1_bytes"] for x in vc["reference_pilot"]["units"])
        checks.append(checked)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    key: dict[str, Any] = {"seed": seed, "mappings": {}, "reference_bytes": {"lossless": 366161, **costs},
                           "source_sha256": sha256(source), "lossless_receipt_sha256": sha256(control / "receipt.json"),
                           "source_reference_max_rgb_deltas": deltas, "audits": checks,
                           "completed_qualities": [check["quality"] for check in checks],
                           "complete_candidate_family": len(variants) == 2}
    sections = []
    for unit in UNITS:
        order = list(candidate_paths)
        rng.shuffle(order)
        labels = "ABC"[:len(order)]
        key["mappings"][unit] = dict(zip(labels, order))
        _, src = by_unit[unit]
        panels = []
        for label, name in zip(labels, order):
            with np.load(candidate_paths[name] / f"{unit}@tennis_only.npz", allow_pickle=False) as f:
                crops = f["crops"]
                for fps in (50, 10):
                    encode(crops, output / f"{unit}-{label}-{fps}.mp4", fps)
                Image.fromarray(crops[12]).save(output / f"{unit}-{label}.png")
            panels.append(f'<div><b>{label}</b><video controls loop src="{unit}-{label}-50.mp4"></video>'
                          f'<p>Slowed to 10 fps</p><video controls loop src="{unit}-{label}-10.mp4"></video>'
                          f'<img src="{unit}-{label}.png"></div>')
        for fps in (50, 10):
            encode(src, output / f"{unit}-source-{fps}.mp4", fps)
        sections.append(f'<h2>{html.escape(LABELS[unit])}</h2><div class="grid"><div><b>Source</b>'
                        f'<video controls loop src="{unit}-source-50.mp4"></video><p>Slowed to 10 fps</p>'
                        f'<video controls loop src="{unit}-source-10.mp4"></video></div>{"".join(panels)}</div>')
    completion_note = ("" if len(variants) == 2 else
                       '<p>Partial candidate family: review covers completed outputs only. '
                       'The absent setting remains untested.</p>')
    (output / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>Player appearance review</title>'
        f'<style>.grid{{display:grid;grid-template-columns:repeat({len(candidate_paths) + 1},1fr);gap:1em}}video,img{{width:100%}}</style>'
        '<h1>Player appearance review</h1><p>24 native ticks: 0.48 s at 50 fps. The 10 fps clips are slowed. '
        'Assess body/clothing and pose against source, and record judgments before opening reveal.json. '
        'Clips need not play synchronously. This is not longer-motion or codec evidence.</p>'
        + completion_note + "".join(sections))
    (output / "reveal.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--control-hash", required=True)
    parser.add_argument("--control-conditioning", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--variant", nargs=5, action="append", required=True, metavar=("CONFIG", "SHA", "CONDITIONING", "SMOKE", "FULL"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    a = parser.parse_args()
    variants = [(Path(c), h, Path(d), Path(s), Path(f)) for c, h, d, s, f in a.variant]
    build(a.control_config, a.control_hash, a.control_conditioning, a.base, a.control, variants, a.source, a.output, a.seed)


if __name__ == "__main__":
    main()
