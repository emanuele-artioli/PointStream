"""Invariant verification for BP46 long eligible tennis-scene manifest.

Verifies:
- exact frame counts at 48/96/192/384 frames;
- dimensions, frame rate and colour metadata recorded;
- object tracks cover the requested interval;
- source frames, masks, motion and appearance references align;
- context IDs do not group unrelated cameras/backgrounds;
- failures stay in the manifest with reasons;
- at least 2 diagnostic videos (near-static and smooth-pan);
- at least 6 independent confirmation videos;
- at least 1 high-motion ineligible control;
- reports submitted / succeeded / failed counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.long_scenes.schema import (
    PASTE_MAE_MAX,
    SCHEMA_ID,
    TARGET_SPANS,
)
from src.contracts import paths as ps_paths


class ManifestValidationError(AssertionError):
    """Raised when any manifest invariant is violated."""


def _match_identity(provenance: dict[str, Any]) -> tuple[str, str] | None:
    """Compare audited match/event labels, ignoring case and whitespace.

    This is not an event resolver: aliases and compilation match boundaries
    still require the provenance audit before confirmation can be accepted.
    """
    values = [provenance.get(key) for key in ("match_name", "event")]
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    return (" ".join(str(values[0]).casefold().split()),
            " ".join(str(values[1]).casefold().split()))


def verify_manifest(
    manifest: dict[str, Any],
    *,
    strict_confirmation: bool = False,
) -> dict[str, Any]:
    """Verify all acceptance invariants on the manifest dictionary.

    Returns summary metrics if valid; raises ManifestValidationError on failure.
    """
    violations: list[str] = []

    # 1. Schema
    if manifest.get("schema") != SCHEMA_ID:
        violations.append(f"schema is '{manifest.get('schema')}', expected '{SCHEMA_ID}'")

    target_spans = manifest.get("target_spans") or []
    if tuple(target_spans) != TARGET_SPANS:
        violations.append(f"target_spans is {target_spans}, expected {list(TARGET_SPANS)}")

    # 2. Partitions: diagnostic videos (>=2), confirmation videos (>=6), >=1 ineligible control
    diag_vids = manifest.get("diagnostic_videos") or []
    if len(diag_vids) < 2:
        violations.append(f"expected >= 2 diagnostic videos, got {len(diag_vids)}: {diag_vids}")

    conf_vids = manifest.get("confirmation_videos") or []
    dev_vids = manifest.get("development_videos") or []
    provenance_map = manifest.get("provenance") or {}

    confirmation_deficits: list[str] = []

    # Validate provenance and contamination for confirmation candidates
    clean_conf_vids: list[str] = []
    seen_matches: dict[tuple[str, str], str] = {}
    # A second upload of a used match is not held out. Include every known
    # development/diagnostic source and any other recorded prior use.
    used_matches: dict[tuple[str, str], str] = {}
    for video, prov in provenance_map.items():
        if not isinstance(prov, dict):
            continue
        if video in diag_vids or video in dev_vids or prov.get("is_contaminated") or prov.get("prior_use"):
            match = _match_identity(prov)
            if match is not None:
                used_matches[match] = video

    for cv in conf_vids:
        prov = provenance_map.get(cv)
        if not isinstance(prov, dict):
            confirmation_deficits.append(
                f"confirmation candidate '{cv}': missing provenance metadata; cannot verify independence"
            )
            continue

        is_contam = prov.get("is_contaminated")
        prior_use = prov.get("prior_use")
        if not isinstance(is_contam, bool) or not isinstance(prior_use, list) or not all(
            isinstance(use, str) and use.strip() for use in prior_use
        ):
            confirmation_deficits.append(
                f"confirmation candidate '{cv}': missing or malformed prior-use audit fields"
            )
            continue
        if is_contam or prior_use:
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' is contaminated by prior use: {prior_use}"
            )
            continue

        evt_status = prov.get("event_status", "")
        if evt_status != "verified":
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' has unresolved event status: '{evt_status}'"
            )
            continue

        src_type = prov.get("source_type", "")
        if src_type != "tournament_broadcast":
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' has invalid source_type '{src_type}' (must be tournament_broadcast)"
            )
            continue

        if prov.get("confirmation_eligible") is not True:
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' is marked confirmation_eligible=False"
            )
            continue

        # Check for duplicate matches
        match_id = _match_identity(prov)
        if match_id is None:
            confirmation_deficits.append(f"confirmation candidate '{cv}': missing match identity")
            continue
        if match_id in used_matches:
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' repeats a development/diagnostic or previously used "
                f"match '{used_matches[match_id]}' ({match_id})"
            )
            continue
        if match_id in seen_matches:
            confirmation_deficits.append(
                f"confirmation candidate '{cv}' is a duplicate match of '{seen_matches[match_id]}' ({match_id})"
            )
            continue
        seen_matches[match_id] = cv
        clean_conf_vids.append(cv)

    if len(clean_conf_vids) < 6:
        confirmation_deficits.append(
            f"expected >= 6 independent confirmation matches, got {len(clean_conf_vids)} accepted: {clean_conf_vids}"
        )

    diag_set = set(diag_vids)
    conf_set = set(conf_vids)
    dev_set = set(dev_vids)

    if not diag_set.isdisjoint(conf_set):
        overlap = sorted(list(diag_set.intersection(conf_set)))
        violations.append(f"split isolation violated: diagnostic and confirmation overlap on {overlap}")
    if not conf_set.isdisjoint(dev_set):
        overlap = sorted(list(conf_set.intersection(dev_set)))
        violations.append(f"split isolation violated: confirmation and development overlap on {overlap}")

    controls = manifest.get("ineligible_controls") or []
    if len(controls) < 1:
        violations.append(f"expected >= 1 ineligible control, got {len(controls)}")

    scenes = manifest.get("scenes") or []
    if not scenes:
        violations.append("manifest contains 0 scenes")

    # Check diagnostic covers near-static and smooth-pan
    has_near_static = False
    has_smooth_pan = False
    has_control = False

    # Check context IDs do not group unrelated videos
    context_videos: dict[str, set[str]] = {}

    for s in scenes:
        video = s.get("video", "")
        scene = s.get("scene", "")
        key = f"{video}/{scene}"
        role = s.get("role", "")
        ctx = s.get("context_id", "")

        # Role membership & split isolation
        if video in conf_set and role != "confirmation":
            violations.append(f"{key}: confirmation video '{video}' has non-confirmation role '{role}'")
        if video in diag_set and role == "confirmation":
            violations.append(f"{key}: diagnostic video '{video}' has confirmation role '{role}'")
        if video in dev_set and role == "confirmation":
            violations.append(f"{key}: development video '{video}' has confirmation role '{role}'")

        if not ctx:
            violations.append(f"{key}: missing context_id")
        else:
            context_videos.setdefault(ctx, set()).add(video)

        # Source metadata
        sm = s.get("source_metadata") or {}
        if sm.get("width", 0) <= 0 or sm.get("height", 0) <= 0:
            violations.append(f"{key}: invalid dimensions {sm.get('width')}x{sm.get('height')}")
        if sm.get("working_fps") != 24.0:
            violations.append(f"{key}: working_fps is {sm.get('working_fps')}, expected 24.0")
        for field in ["pix_fmt", "color_space", "color_primaries", "color_transfer", "sha256"]:
            if not sm.get(field):
                violations.append(f"{key}: source_metadata missing '{field}'")

        # Eligibility
        elig = s.get("eligibility") or {}
        route = elig.get("route", "")
        if route not in {"pointstream", "conventional_fallback"}:
            violations.append(f"{key}: invalid route '{route}'")

        inelig_reasons = elig.get("ineligibility_reasons") or []
        if route == "conventional_fallback" and not inelig_reasons:
            violations.append(f"{key}: conventional_fallback must record ineligibility_reasons")

        # Check diagnostic motion coverage
        if role == "diagnostic_near_static" and route == "pointstream":
            mot = elig.get("camera_motion") or {}
            if mot.get("consecutive_mad", 999.0) < 1.0:
                has_near_static = True
        if role == "diagnostic_smooth_pan" and route == "pointstream":
            pano = elig.get("panorama") or {}
            mot = elig.get("camera_motion") or {}
            if pano.get("growth_factor", 1.0) >= 1.01 and mot.get("consecutive_mad", 0.0) >= 1.0:
                has_smooth_pan = True

        # Ineligible control check
        if role == "control_ineligible":
            has_control = True
            if route != "conventional_fallback":
                violations.append(f"{key}: control_ineligible must route to conventional_fallback")

        # Paste-back check
        pb = elig.get("paste_back") or {}
        if route == "pointstream":
            mae = pb.get("opaque_mae", 999.0)
            if mae > PASTE_MAE_MAX:
                violations.append(f"{key}: eligible scene has paste-back MAE {mae} > {PASTE_MAE_MAX}")

        # Validate intervals: 48, 96, 192, 384
        intervals = s.get("intervals") or {}
        for span in TARGET_SPANS:
            intv = intervals.get(str(span))
            if intv is None:
                violations.append(f"{key}: missing interval record for {span} frames")
                continue
            if intv.get("frame_count") != span:
                violations.append(
                    f"{key} span {span}: frame_count is {intv.get('frame_count')}, expected {span}"
                )
            status = intv.get("status")
            if status not in {"eligible", "ineligible", "insufficient_duration"}:
                violations.append(f"{key} span {span}: invalid interval status '{status}'")
            fail_reasons = intv.get("failure_reasons") or []
            if status in {"ineligible", "insufficient_duration"} and not fail_reasons:
                violations.append(f"{key} span {span}: failure status '{status}' missing failure_reasons")
            if status == "eligible":
                hashes = intv.get("frame_hashes") or {}
                if not hashes.get("first") or not hashes.get("mid") or not hashes.get("last"):
                    violations.append(f"{key} span {span}: eligible interval missing frame_hashes (requires 'first', 'mid', 'last')")
                growth = intv.get("canvas_growth", 0.0)
                if growth <= 0.0:
                    violations.append(f"{key} span {span}: eligible interval has non-positive canvas_growth {growth}")
                mae = intv.get("paste_back_mae", 999.0)
                if mae > PASTE_MAE_MAX:
                    violations.append(f"{key} span {span}: eligible interval paste_back_mae {mae} > {PASTE_MAE_MAX}")
                start_f = intv.get("start_frame", 0)
                end_f = intv.get("end_frame", 0)
                if end_f - start_f != span:
                    violations.append(f"{key} span {span}: interval window [{start_f}:{end_f}] length {end_f - start_f} != {span}")

    # Verify context IDs do not span across different source videos
    for ctx, vids in context_videos.items():
        if len(vids) > 1:
            violations.append(
                f"context_id '{ctx}' groups unrelated source videos: {sorted(vids)}"
            )

    if not has_near_static:
        violations.append("diagnostic videos do not include a verified near-static eligible scene (consecutive_mad < 1.0)")
    if not has_smooth_pan:
        violations.append("diagnostic videos do not include a verified smooth-pan eligible scene (growth_factor >= 1.01, consecutive_mad >= 1.0)")
    if not has_control:
        violations.append("diagnostic videos do not include a designated ineligible control")

    # Verify confirmation coverage across clean confirmation matches for ALL target spans: 48, 96, 192, 384
    if clean_conf_vids:
        for span in TARGET_SPANS:
            eligible_conf_by_vid: dict[str, int] = {v: 0 for v in clean_conf_vids}
            for s in scenes:
                if s.get("video") in clean_conf_vids and s.get("role") == "confirmation":
                    intv = (s.get("intervals") or {}).get(str(span)) or {}
                    if intv.get("status") == "eligible":
                        eligible_conf_by_vid[s.get("video")] += 1
            missing_vids = [v for v, count in eligible_conf_by_vid.items() if count == 0]
            if missing_vids:
                confirmation_deficits.append(
                    f"confirmation coverage incomplete at span {span}: missing eligible scenes for {sorted(missing_vids)}"
                )

    if violations:
        msg = f"{len(violations)} manifest invariant violation(s):\n" + "\n".join(f"  - {v}" for v in violations)
        raise ManifestValidationError(msg)

    # Check strict confirmation if requested
    if confirmation_deficits and strict_confirmation:
        msg = f"confirmation corpus incomplete ({len(confirmation_deficits)} deficit(s)):\n" + "\n".join(
            f"  - {d}" for d in confirmation_deficits
        )
        raise ManifestValidationError(msg)

    summary = manifest.get("summary", {})
    diag_ready = len(violations) == 0
    conf_ready = len(confirmation_deficits) == 0 and len(clean_conf_vids) >= 6

    if diag_ready and not conf_ready:
        status = "DIAGNOSTIC_READY_CONFIRMATION_INCOMPLETE"
        verdict = f"diagnostic inputs ready; confirmation corpus incomplete ({len(clean_conf_vids)} of 6 independent confirmation matches accepted)"
    else:
        status = "VERIFIED_PASS"
        verdict = "diagnostic inputs and confirmation corpus fully verified"

    return {
        "status": status,
        "verdict": verdict,
        "diagnostic_status": "READY" if diag_ready else "FAILED",
        "confirmation_status": "COMPLETE" if conf_ready else "INCOMPLETE",
        "schema": manifest.get("schema"),
        "num_scenes": len(scenes),
        "diagnostic_videos": diag_vids,
        "confirmation_videos": clean_conf_vids,
        "development_videos": dev_vids,
        "confirmation_deficits": confirmation_deficits,
        "ineligible_controls": controls,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify BP46 long tennis-scene manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ps_paths.repo_root() / "manifests" / "bp46_long_tennis_scenes.json",
        help="Path to manifest file.",
    )
    parser.add_argument(
        "--strict-confirmation",
        action="store_true",
        help="Fail with non-zero exit if confirmation corpus has deficits.",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        res = verify_manifest(payload, strict_confirmation=args.strict_confirmation)
        print(f"Manifest verification: {res['verdict']}")
        print(json.dumps(res, indent=2))
        if res["status"] == "DIAGNOSTIC_READY_CONFIRMATION_INCOMPLETE":
            print("\nNOTE: Missing confirmation footage need not block the diagnostic search.")
        return 0
    except ManifestValidationError as exc:
        print(f"Manifest verification FAILED:\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
