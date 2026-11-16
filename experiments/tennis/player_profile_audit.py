"""Strict join of independently extracted player-generation receipt facts.

This validates the supplied facts, not their extraction or pixel-level mask coverage.
Each record must carry a source identifier so a successful join is traceable.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label}: missing or invalid SHA256")
    return value


def _ticks(values: Any, label: str) -> set[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}: missing ticks")
    if any(type(t) is not int or t < 0 for t in values) or len(set(values)) != len(values):
        raise ValueError(f"{label}: invalid or duplicate ticks")
    return set(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_join(data: dict[str, Any]) -> dict[str, Any]:
    """Validate exact judge-to-preparation-to-output and fit identities.

    `generated` records describe independently hash-checked NPZ members and their
    actual t arrays. A mask claim requires explicit per-tick checks, not counts.
    Missing evidence fails; no profile substitution is permitted.
    """
    for field in ("prepared_sha256", "judge_prepared_sha256"):
        _sha(data.get(field), field)
    if data["prepared_sha256"] != data["judge_prepared_sha256"]:
        raise ValueError("judge/preparation SHA256 mismatch")
    expected_archives = [_sha(x, "judge generated archive") for x in data["judge_generated_sha256"]]
    if not expected_archives or len(set(expected_archives)) != len(expected_archives):
        raise ValueError("missing or duplicate judge generated archive")
    generated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in data["generated"]:
        key = (row["unit"], row["profile"])
        if key in generated:
            raise ValueError(f"duplicate generated unit/profile: {key}")
        archive = _sha(row.get("archive_sha256"), "generated archive")
        if archive not in expected_archives:
            raise ValueError(f"unreferenced generated archive: {key}")
        if _sha(row.get("npz_sha256"), "NPZ hash") != _sha(row.get("receipt_npz_sha256"), "receipt NPZ hash"):
            raise ValueError(f"NPZ hash mismatch: {key}")
        if not row.get("source"):
            raise ValueError(f"missing member provenance: {key}")
        _ticks(row.get("ticks"), f"{key} generated")
        generated[key] = row
    if {r["archive_sha256"] for r in generated.values()} != set(expected_archives):
        raise ValueError("judge archive lacks generated records")
    fits: dict[tuple[str, str], dict[str, Any]] = {}
    for fit in data["fits"]:
        key = (fit["unit"], fit["profile"])
        if key in fits:
            raise ValueError(f"duplicate fit attribution: {key}")
        fits[key] = fit
    count = 0
    required_fits: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    window_ids: set[str] = set()
    for window in data["windows"]:
        if window["id"] in window_ids or not window["units"] or not window["profiles"]:
            raise ValueError("duplicate window or empty units/profiles")
        window_ids.add(window["id"])
        if not set(window["adapted_profiles"]).issubset(window["profiles"]):
            raise ValueError("adapted profile absent from judge profiles")
        ticks = _ticks(window.get("judged"), f"{window['id']} judged")
        for unit in window["units"]:
            for profile in window["profiles"]:
                key = (unit, profile)
                if key in seen:
                    raise ValueError(f"duplicate expected unit/profile: {key}")
                seen.add(key)
                if profile in window["adapted_profiles"]:
                    required_fits.add(key)
                    if key not in fits or fits[key].get("match") != window["match"]:
                        raise ValueError(f"missing or wrong match fit: {key}")
                if key not in generated:
                    raise ValueError(f"missing generated unit/profile: {key}")
                missing = ticks - set(generated[key]["ticks"])
                if missing:
                    raise ValueError(f"missing generated ticks: {key}: {sorted(missing)}")
                checks = data.get("mask_checks", {}).get(unit, {})
                if set(checks) != {str(t) for t in ticks} or any(value is not True for value in checks.values()):
                    raise ValueError(f"missing or failed mask coverage evidence: {key}")
                count += len(ticks)
    if not seen:
        raise ValueError("no expected unit/profile pairs")
    if set(fits) != required_fits:
        raise ValueError("unexpected or missing fit attribution")
    for fit in fits.values():
        if not fit.get("source") or not fit.get("match"):
            raise ValueError("fit match or provenance mismatch")
        if _sha(fit.get("generation_checkpoint_sha256"), "generation checkpoint") != _sha(fit.get("fit_checkpoint_sha256"), "fit checkpoint"):
            raise ValueError("fit checkpoint mismatch")
        base = fit.get("generation_base_sha256")
        fit_base = fit.get("fit_base_sha256")
        if base is not None or fit_base is not None:
            if _sha(base, "generation base") != _sha(fit_base, "fit base"):
                raise ValueError("base checkpoint mismatch")
    return {"status": "pass", "unit_profile_pairs": len(seen), "judged_unit_profile_ticks": count}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facts", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_join(json.loads(args.facts.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
