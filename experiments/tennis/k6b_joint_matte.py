"""Frozen saved FULL-reference renders with existing paid dynamic matte packets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any


from experiments.tennis import k6b_matte_judge as judge
from experiments.tennis.k6b_foreground_video import PILOTS, SOURCE_SHA
from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA, comparison
from experiments.tennis.k6b_rate_transport import sha, write

REFERENCE_REPORT_SHA = "e9bbb4aca54cd5453f6c7e3472097881c37a200f6b685ec31221b5aff6c10797"
CONTROL_SHA = "d5a9670d5a223d6c231476cbf7211442aa2ab77dfc01ba1cffbfb7265dee7457"


def assess(
    diagnostic: dict[str, Any],
    reference: dict[str, Any],
    transport: dict[str, Any],
    measured: list[dict[str, Any]],
) -> dict[str, Any]:
    if [r["id"] for r in measured] != list(PILOTS) or any(
        r["judged"] != r["original_judged"] for r in measured
    ):
        raise ValueError("Only both complete original pilot windows decide quality")
    original = {r["window"]: r for r in diagnostic["rows"]}
    refs = {r["window"]: r for r in reference["settings"]["full"]["rows"]}
    wire = {r["window"]: r for r in transport["rows"]}
    out: dict[str, Any] = {}
    for scale in judge.VARIANTS:
        cells = []
        for row in measured:
            wid = row["id"]
            base = original[wid]
            quality = row["means"][str(scale)]
            seconds = base["registered_seconds"]
            packet = row["selected_packets"][str(scale)] if scale else wire[wid]["mattes"]["exact"]
            matte_rate = packet["bytes"] * 8 / seconds / 1000
            rate = refs[wid]["kbps"] + (matte_rate if scale else 0)
            actual = {**base, "method": quality}
            cell = {
                "window": wid,
                "scale": scale,
                "kbps": rate,
                "matte_kbps": matte_rate,
                "matte_packet": packet,
                "reference_rate_components": refs[wid]["rate_components"],
                "quality": quality,
                "primary": {m: comparison(actual, m, rate) for m in ("lpips", "dists")},
                "exact_matte_paid": {
                    m: comparison(actual, m, rate + matte_rate) for m in ("lpips", "dists")
                }
                if not scale
                else None,
                "base_charged": {
                    m: comparison(actual, m, rate + base["base_checkpoint_bytes"] * 8 / 7200 / 1000)
                    for m in ("lpips", "dists")
                },
                "renderer_sensitivity": {
                    m: {
                        str(d): comparison(
                            {**actual, "method": {**quality, m: quality[m] + d}}, m, rate
                        )
                        for d in (-1e-3, 1e-3)
                    }
                    for m in ("lpips", "dists")
                },
            }
            cells.append(cell)
        wins = sum(c["primary"][m]["won"] is True for c in cells for m in ("lpips", "dists"))
        out[str(scale)] = {
            "rows": cells,
            "strict_wins": wins,
            "unresolved": sum(
                c["primary"][m]["won"] is None for c in cells for m in ("lpips", "dists")
            ),
            "passes_3_of_4": wins >= 3,
            "scope": "granted-exact-matte-control"
            if not scale
            else "paid-compressed-dynamic-matte",
        }
    return {
        "settings": out,
        "full12": "not performed",
        "background": "target-derived Telea granted",
        "base": "common shipped base primary; separately charged over7200s",
        "source_independence": "unresolved",
        "115px": "exploratory",
        "experience_track": "untested",
    }


def validate(archive: Path, stage: str) -> dict[str, Any]:
    checked = judge.validate(archive, stage)
    with tarfile.open(archive) as t:
        f = t.extractfile("publish/judge.json")
        if f is None:
            raise ValueError("Missing joint receipt")
        r = json.load(f)
    checks = checked["checks"]
    checks["frozen_windows"] = r["requested"] == list(PILOTS)
    checks["frozen_inputs"] = (
        r["reference_report_sha256"] == REFERENCE_REPORT_SHA
        and r["control_archive_sha256"] == CONTROL_SHA
    )
    for row in r["rows"]:
        checks[row["id"] + ":saved_rgb"] = (
            row["control_variant"] == "full" and row["control_composite_hashes_match"]
        )
        checks[row["id"] + ":source"] = row["source_sha256"] == SOURCE_SHA[PILOTS.index(row["id"])]
        expected_ticks = {str(t) for t in row["judged"]}
        checks[row["id"] + ":frame_hashes"] = all(
            set(row[key]) == expected_ticks
            and all(len(digest) == 64 for digest in row[key].values())
            for key in ("source_rgb_frame_sha256", "source_player_masks_sha256")
        )
        checks[row["id"] + ":all240_matte_hashes"] = set(row["decoded_matte_frame_sha256"]) == {
            str(s) for s in judge.SCALES
        } and all(
            len(hashes) == 240 and all(len(h) == 64 for h in hashes)
            for hashes in row["decoded_matte_frame_sha256"].values()
        )
        checks[row["id"] + ":crop_accounting"] = set(row["crop_coverage"]) == {
            str(t) for t in row["judged"]
        } and all(
            len(c) == 4
            and all(x["requested_union"] - x["written_union"] == x["missing_union"] >= 0 for x in c)
            for c in row["crop_coverage"].values()
        )
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "reference-report", "transport", "matte"):
        p.add_argument("--" + name, type=Path)
    for name in ("anchor", "control", "generated", "source", "backbone"):
        p.add_argument("--" + name, type=Path, action="append", default=[])
    p.add_argument("--max-judged", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        checked = validate(stage / "published.tar", os.environ.get("PS_STAGE", "full"))
        write(Path(os.environ["PS_VALIDATION_PATH"]), checked)
        return 0 if checked["passed"] else 1
    if (
        sha(a.diagnostic.read_bytes()) != DIAGNOSTIC_SHA
        or sha(a.reference_report.read_bytes()) != REFERENCE_REPORT_SHA
        or len(a.control) != 1
        or sha(a.control[0].read_bytes()) != CONTROL_SHA
    ):
        raise ValueError("Frozen reference/diagnostic/control changed")
    import cv2
    import torch
    from experiments.audit.env_smoke import device_record
    from experiments.tennis.k6b import load_generated
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record

    torch.set_num_threads(8)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    device = device_record("RTX 6000 Ada")
    limit = judge.gpu_memory_limit()
    if not device["class_matches"]:
        raise ValueError("Qualified Ada GPU required")
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    )
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    anchors = judge.extract(a.anchor, scratch / "anchors")
    controls = judge.extract(a.control, scratch / "controls")
    for name in ("transport", "matte"):
        with tarfile.open(getattr(a, name)) as t:
            t.extractall(scratch / name, filter="data")
    packet_root = scratch / "transport"
    if (packet_root / "publish").is_dir():
        packet_root /= "publish"
    mask_root = scratch / "matte/publish"
    masks = json.loads((mask_root / "matte.json").read_text())
    mask_rows = {r["window"]: r for r in masks["rows"]}
    generated = load_generated([str(q) for q in a.generated], scratch / "generated")
    nets = {"lpips": load_lpips(a.backbone[0], "cuda"), "dists": load_dists(a.backbone[1], "cuda")}
    sources = {sha(q.read_bytes()): q for q in a.source}
    if set(sources) != set(SOURCE_SHA):
        raise ValueError("Both frozen source clips required")
    result: dict[str, Any] = {
        "requested": list(PILOTS),
        "rows": [],
        "device": device,
        "gpu_memory_limit_mib": limit,
        "on_cuda": {n: next(net.parameters()).is_cuda for n, net in nets.items()},
        "torch": torch.__version__,
        "code_sha256": sha(Path(__file__).read_bytes()),
        "lpips": lpips_record(a.backbone[0]),
        "dists_backbone_sha256": sha(a.backbone[1].read_bytes()),
        "reference_report_sha256": REFERENCE_REPORT_SHA,
        "control_archive_sha256": CONTROL_SHA,
        "inputs": {
            str(q): sha(q.read_bytes())
            for q in [
                a.diagnostic,
                a.reference_report,
                a.transport,
                a.matte,
                *a.anchor,
                *a.control,
                *a.generated,
                *a.source,
            ]
        },
        "missing_pixels": "native Telea fill outside available generated crop; no target foreground reconstruction",
    }
    began = time.monotonic()
    for wid, source_sha in zip(PILOTS, SOURCE_SHA):
        result["rows"].append(
            judge.judge_window(
                wid,
                anchors[wid],
                controls[wid],
                packet_root / "packets",
                mask_root,
                mask_rows[wid],
                generated,
                sources[source_sha],
                source_sha,
                nets,
                scratch,
                publish,
                a.max_judged,
                control_variant="full",
            )
        )
        result["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["seconds"] = time.monotonic() - began
        if a.max_judged == 0 and len(result["rows"]) == 2:
            result["assessment"] = assess(
                json.loads(a.diagnostic.read_text()),
                json.loads(a.reference_report.read_text()),
                json.loads((packet_root / "transport.json").read_text()),
                result["rows"],
            )
        write(publish / "judge.json", result)
        write(stage / "judge.json", result)
        with tarfile.open(stage / "published.tar", "w") as t:
            t.add(publish, arcname="publish")
    print(
        json.dumps(
            {
                "windows": len(result["rows"]),
                "seconds": result["seconds"],
                "peak_gib": result["gpu_peak_gib"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
