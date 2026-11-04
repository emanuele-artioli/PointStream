"""Frozen K5d calibration entrypoint and fleet smoke promotion validator.

This wrapper binds source-snapshot receipts, staged archive identities and the
packed runtime. Submission/admission and process-tree budget remain fleet duties.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.tennis import k5d_encoder as encoder, k5d_source_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/experiments/k5d-inputs/study-selection-20261104.json"
EXPOSURE = ROOT / "docs/experiments/k5c-inputs/wasb-reported-split-provenance.json"
SOURCE_SHA = "d572e991d89f52264f6017e0d034e93dcfbb9766c7693093a4502e8f144da343"
EXPOSURE_SHA = "854183164f8120cf869775d8f77acba38ae1e4e89129a05bde09a17e327ede62"
EXPECTED = {
    "archive": "bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18",
    "wasb_weights": "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb",
    "wasb_config": "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d",
    "balltrack_weight": "b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec",
    "image_ckpt": "b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302",
    "video_ckpt": "934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3",
    "lpips_backbone": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    "dists_backbone": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
}


def archive_tree_sha(archive_path: Path) -> str:
    """Derive tree_sha from the verified tar without trusting the cached tree."""
    found: dict[str, str] = {}
    with tarfile.open(archive_path) as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe code archive member")
            if path.suffix not in (".py", ".yaml", ".yml"):
                continue
            if not member.isfile():
                raise ValueError("code archive has non-regular source member")
            name = path.as_posix()
            if name in found:
                raise ValueError("duplicate source in code archive")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError("unreadable code archive member")
            h = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
            found[name] = h.hexdigest()
    if not found:
        raise ValueError("empty source archive")
    digest = hashlib.sha256()
    for name, value in sorted(found.items()):
        digest.update(name.encode() + b"\0" + bytes.fromhex(value))
    return digest.hexdigest()


def snapshot_head(stage: Path) -> str:
    """Read the fleet-verified snapshot's HEAD from the stage dispatch receipt."""
    code = json.loads((stage / "dispatch.json").read_text())["code"]
    head = code["git_head"]
    if not isinstance(head, str) or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError("invalid fleet snapshot Git HEAD")
    for name in ("source_sha256", "snapshot_sha256", "spec_sha256"):
        value = code.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid fleet snapshot identity")
    return head


def projection(stage: Path, now: float) -> dict[str, float]:
    """Gate 188 frames using measured smoke and one-core process-tree bounds."""
    execution = json.loads((stage / "execution.json").read_text())
    clock = json.loads((stage / "dispatch.json").read_text())["campaign_clock"]
    elapsed = execution["seconds"]
    start, phase, end = clock["started"], clock["phase_started"], clock["budget_end"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (elapsed, start, phase, end, now)):
        raise ValueError("invalid measured campaign clock")
    if execution.get("exit_code") != 0 or elapsed <= 0 or not (0 < start <= phase <= now < end):
        raise ValueError("inconsistent measured campaign clock")
    projected = elapsed * (188 / 16) * 1.25 + 60
    remaining = end - now
    if projected > 2400 or projected + 60 > remaining:
        raise ValueError("188-frame forecast exceeds stage or campaign budget")
    return {"smoke_seconds": elapsed, "multiplier": 188 / 16, "safety_factor": 1.25,
            "projected_full_seconds_including_60s": projected, "remaining_seconds": remaining,
            "campaign_reserve_seconds": 60.0}


def validate_smoke() -> None:
    target = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        stage = Path(os.environ["PS_STAGE_DIR"])
        with tempfile.TemporaryDirectory(prefix="k5d-validator-", dir=os.environ["TMPDIR"]) as tmp:
            with tarfile.open(stage / "published.tar", "r:") as archive:
                members = archive.getmembers()
                if len(members) > 200 or sum(m.size for m in members) > 2_000_000_000:
                    raise ValueError("unexpected smoke archive size")
                if any(not (m.isfile() or m.isdir()) for m in members):
                    raise ValueError("unexpected link or special archive member")
                archive.extractall(tmp, filter="data")
            result = encoder.validate(Path(tmp) / "publish")
        if result["frames"] != 16 or result["points"] != 6 or not result["full_metrics"]:
            raise ValueError("full-metrics 16-frame smoke incomplete")
        cost = projection(stage, time.monotonic())
        result = {"passed": True, "checks": {"source_qualification": True, "six_native_psbl_points": True,
                  "frame_metrics_and_judges": True, "receiver_parity": True, "full_stage_forecast": True},
                  "cost_projection": cost, "scope": result["scope"]}
    except Exception as exc:
        target.write_text(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    target.write_text(json.dumps(result, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-smoke", action="store_true")
    for name in ("archive", "balltrack-code", "balltrack-weight", "wasb-code", "wasb-weights", "wasb-config",
                 "image-ckpt", "video-ckpt", "lpips-backbone", "dists-backbone"):
        parser.add_argument("--" + name)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    if args.validate_smoke:
        validate_smoke()
        return
    names = ("archive", "balltrack_code", "balltrack_weight", "wasb_code", "wasb_weights", "wasb_config",
             "image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone")
    if any(not getattr(args, name) for name in names) or args.frames not in (16, 188):
        parser.error("all staged paths and 16 or 188 frames required")
    expected_stage = {"smoke": 16, "full": 188}
    if expected_stage.get(os.environ.get("PS_STAGE", "")) != args.frames:
        raise ValueError("stage and frame count mismatch")
    # Reconstruct code-tree hashes from the verified tar itself: fleet's cached
    # extracted trees are read-only but are not re-hashed on every cache hit.
    if oracle.sha(SOURCE) != SOURCE_SHA or oracle.sha(EXPOSURE) != EXPOSURE_SHA:
        raise ValueError("source snapshot manifest or exposure receipt changed")
    dispatch_code = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "dispatch.json").read_text())["code"]
    data: dict[str, Any] = vars(args).copy()
    data["snapshot_identity"] = {name: dispatch_code.get(name) for name in
                                 ("git_head", "source_sha256", "snapshot_sha256", "tracked_worktree_patch_sha256", "included_untracked_sha256")}
    data.update(code_revision=snapshot_head(Path(os.environ["PS_STAGE_DIR"])), manifest=str(SOURCE), manifest_sha256=oracle.sha(SOURCE), exposure=str(EXPOSURE),
                exposure_sha256=oracle.sha(EXPOSURE), clip="tracknet/game10/Clip3", start_frame=0,
                output=Path(os.environ["PS_SCRATCH_DIR"]) / "publish", feasibility_no_metrics=False)
    for name, digest in EXPECTED.items():
        data[name + "_sha256"] = digest
    staging = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "staging.json").read_text())
    entries = {item["name"]: item for item in staging["inputs"]}
    for name in ("wasb_code", "balltrack_code"):
        entry = entries[name]
        source_archive = Path(data[name]).parent / Path(entry["source"]).name
        if oracle.sha(source_archive) != entry["sha256"]:
            raise ValueError("code archive differs from fleet staging identity")
        expected_tree = archive_tree_sha(source_archive)
        if oracle.tree_sha(Path(data[name])) != expected_tree:
            raise ValueError("cached code tree differs from verified archive")
        data[name + "_sha256"] = expected_tree
    for name, binary in (("ffmpeg", "ffmpeg"), ("encoder", "SvtAv1EncApp"), ("decoder", "dav1d")):
        data[name] = str(Path(sys.prefix) / "bin" / binary)
    encoder.run(argparse.Namespace(**data))


if __name__ == "__main__":
    main()
