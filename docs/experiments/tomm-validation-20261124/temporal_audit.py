"""Audit temporal motion receipts and 24-frame context-boundary periodicity."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "temporal-audit.json"


def inspect_delta_receipt(path: Path, name: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    block = data.get("summary") or data.get("arms")
    rows: dict[str, Any] = {}
    for arm, vals in block.items():
        s_delta = vals["source_delta"]
        d_delta = vals["decoded_delta"]
        rows[arm] = {
            "n": vals["n"],
            "error": vals["error"],
            "source_delta": s_delta,
            "decoded_delta": d_delta,
            "decoded_to_source_delta_ratio": d_delta / s_delta if s_delta else None,
        }
    return {"match": name, "receipt": str(path.relative_to(ROOT)), "arms": rows}


def inspect_boundary_periodicity(path: Path, arm_name: str, region_filter: set[str]) -> dict[str, Any]:
    """Check whether error or energy spikes at 24-frame context boundaries (stride 20, overlap 4)."""
    data = json.loads(path.read_text())
    rows = [r for r in data.get("rows", []) if r.get("arm") == arm_name and r.get("region") in region_filter]
    if not rows:
        return {"available": False, "receipt": str(path.relative_to(ROOT))}

    # Pose2VideoPipeline uses context_frames=24, context_overlap=4 -> stride 20.
    # Transitions crossing window starts/overlaps occur around tick % 20 in {0, 1, 2, 3, 4} (after tick 0).
    seam_ticks: list[dict[str, Any]] = []
    interior_ticks: list[dict[str, Any]] = []
    for r in rows:
        tick = int(r["tick"])
        mod = tick % 20
        target = seam_ticks if (tick >= 20 and mod in (0, 1, 2, 3, 4)) else interior_ticks
        target.append(r)

    return {
        "available": True,
        "receipt": str(path.relative_to(ROOT)),
        "arm": arm_name,
        "n_rows": len(rows),
        "seam_count": len(seam_ticks),
        "interior_count": len(interior_ticks),
        "seam_mean_error": mean(r["error"] for r in seam_ticks) if seam_ticks else None,
        "interior_mean_error": mean(r["error"] for r in interior_ticks) if interior_ticks else None,
        "seam_mean_energy": mean(r["energy"] for r in seam_ticks) if seam_ticks else None,
        "interior_mean_energy": mean(r["energy"] for r in interior_ticks) if interior_ticks else None,
    }


def main() -> None:
    shanghai = inspect_delta_receipt(
        ROOT / "docs/experiments/test-crop-joint-20261122/match102-motion-four-arm.json",
        "Shanghai (match102)",
    )
    sydney = inspect_delta_receipt(
        ROOT / "docs/experiments/test-joint-extension-20261122/native/full-confirmation-20261123/temporal-matched.json",
        "Sydney (match127)",
    )

    nadal = json.loads((ROOT / "docs/experiments/joint-trajectory-20261122/temporal-with-neural.json").read_text())
    alcaraz_joint = json.loads((ROOT / "docs/experiments/al028-temporal-20261122/joint-result.json").read_text())
    alcaraz_ctrl = json.loads((ROOT / "docs/experiments/al028-temporal-20261122/controls.json").read_text())

    seam_checks = {
        "nadal_starved": inspect_boundary_periodicity(
            ROOT / "docs/experiments/starved-temporal-20261122/result.json",
            "pointstream",
            {"p1", "p2"},
        ),
        "alcaraz_joint": inspect_boundary_periodicity(
            ROOT / "docs/experiments/al028-temporal-20261122/joint-result.json",
            "joint",
            {"archived-track-0", "archived-track-1"},
        ),
    }

    payload = {
        "external_delta_receipts": [shanghai, sydney],
        "nadal_means": nadal["means"],
        "alcaraz_joint_means": alcaraz_joint["means"],
        "alcaraz_control_means": alcaraz_ctrl["means"],
        "seam_checks": seam_checks,
        "note": (
            "Shanghai and Sydney JSONs store aggregate means without per-tick rows, "
            "and current receipts measure source-flow-warped error only (not self-flow flicker or tLPIPS)."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
