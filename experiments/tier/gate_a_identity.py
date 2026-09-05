"""Gate 0 identity: source proof, manifest verification, and implementation digest.

Gate A requires:
- Frozen source: alcaraz_highlights, ordered scenes scene_000 then scene_028
- Context: alcaraz_highlights_main_court, native 3840x2160, 24 fps
- Known 48-frame prefix RGB SHA-256:
    scene_000: 388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6
    scene_028: e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb
- Selected-record manifest SHA-256:
    840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6
- Any mismatch stops the run.
- Missing 384-frame contiguous prefix stops that duration; it is never filled from another shot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


import numpy as np

from experiments.tier.bp52_background_search import _manifest_snapshot
from experiments.tier.low_rate_checkpoint import fingerprint, implementation_digest, source_identity
from experiments.tier.low_rate_clips import load_e1_sequence

MEASUREMENT_FILES = (
    "experiments/tier/gate_a_long_context.py",
    "experiments/tier/gate_a_identity.py",
    "experiments/tier/gate_a_tools.py",
    "experiments/tier/gate_a_controls.py",
    "experiments/tier/low_rate_measure.py",
    "src/components/codec/encode.py",
    "src/components/codec/measure.py",
)


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ).splitlines()
    dirty_paths = sorted(
        {line[3:] for line in dirty if len(line) > 3 and (root / line[3:]).is_file()}
    )
    return {
        "commit": commit,
        "dirty": bool(dirty),
        "dirty_files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in dirty_paths
        },
        "measurement_files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in MEASUREMENT_FILES
        },
    }


VIDEO = "alcaraz_highlights"
SCENES = ("scene_000", "scene_028")
CONTEXT_ID = "alcaraz_highlights_main_court"
FPS = 24
NATIVE_SHAPE = (2160, 3840, 3)  # H, W, C

KNOWN_48_PREFIX_HASHES: dict[str, str] = {
    "scene_000": "388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6",
    "scene_028": "e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb",
}

KNOWN_MANIFEST_SHA256 = "840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6"

FROZEN_PREFIX_HASHES: dict[int, dict[str, str]] = {
    48: KNOWN_48_PREFIX_HASHES,
    96: {
        "scene_000": "09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573",
        "scene_028": "2009a1bb6934c4109f16f73951acb2af5a22d91a3cb15f7f79fc3dc883bb6c6b",
    },
    192: {
        "scene_000": "943bb165838cf5a3189798a4e6dd2de20531ee1619766741ff303b34377c769b",
        "scene_028": "f8e19178e6335c32c5dbce9a20b96840e8059c80cae795fdcf4bb2f0e0dc2664",
    },
}


def verify_manifest(video: str = VIDEO, scenes: tuple[str, ...] = SCENES) -> dict[str, Any]:
    """Verify the selected-record manifest SHA-256 matches the pre-registered identity."""
    manifest = _manifest_snapshot(video, list(scenes))
    actual_sha256 = manifest.get("selected_scene_records_sha256")
    if actual_sha256 != KNOWN_MANIFEST_SHA256:
        raise SystemExit(
            f"Gate A manifest SHA-256 mismatch: expected {KNOWN_MANIFEST_SHA256}, "
            f"got {actual_sha256}. Source identity verification failed."
        )
    return manifest


def verify_source_clips(
    video: str = VIDEO,
    scenes: tuple[str, ...] = SCENES,
    *,
    n_frames: int = 48,
) -> list[Any]:
    """Load and verify input clips against frozen prefix hashes and metadata."""
    if video != VIDEO or tuple(scenes) != SCENES:
        raise SystemExit(
            f"Gate A source is fixed to {VIDEO} {SCENES}, got video={video} scenes={scenes}"
        )

    verify_manifest(video, scenes)

    try:
        clips = load_e1_sequence(video, list(scenes), n_frames=n_frames)
    except Exception as exc:
        raise SystemExit(
            f"Gate A cannot load {n_frames} contiguous frames for {video} {scenes}: {exc}. "
            "Never fill missing frames from another shot."
        ) from exc

    for clip in clips:
        if clip.context_id != CONTEXT_ID:
            raise SystemExit(
                f"Gate A context mismatch for {clip.scene}: expected {CONTEXT_ID}, got {clip.context_id}"
            )
        _, h, w, c = clip.frames.shape
        if (h, w, c) != NATIVE_SHAPE:
            raise SystemExit(
                f"Gate A frame shape mismatch for {clip.scene}: expected {NATIVE_SHAPE}, got {(h, w, c)}"
            )

    # 1. Verify 48-frame prefix on all clips
    for clip in clips:
        prefix_48 = np.ascontiguousarray(clip.frames[:48])
        prefix_hash = hashlib.sha256(prefix_48.data).hexdigest()
        expected = KNOWN_48_PREFIX_HASHES.get(clip.scene)
        if prefix_hash != expected:
            raise SystemExit(
                f"Gate A 48-frame prefix RGB hash mismatch for {clip.scene}: "
                f"expected {expected}, got {prefix_hash}"
            )

    # 2. If duration is in FROZEN_PREFIX_HASHES, verify full duration hash
    expected_duration = FROZEN_PREFIX_HASHES.get(n_frames)
    if expected_duration is not None:
        for clip in clips:
            full_hash = hashlib.sha256(np.ascontiguousarray(clip.frames).data).hexdigest()
            expected = expected_duration.get(clip.scene)
            if full_hash != expected:
                raise SystemExit(
                    f"Gate A {n_frames}-frame prefix RGB hash mismatch for {clip.scene}: "
                    f"expected {expected}, got {full_hash}"
                )

    return clips


def build_gate_a_identity(
    output_dir: Path,
    *,
    n_frames: int = 48,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete immutable identity record for a Gate A duration run."""
    clips = verify_source_clips(n_frames=n_frames)
    manifest = verify_manifest()
    root = Path(__file__).resolve().parents[2]
    impl_digest = implementation_digest(root)

    sources = source_identity(clips)
    identity: dict[str, Any] = {
        "gate": "A",
        "git": _git_identity(root),
        "native_shape": list(NATIVE_SHAPE),
        "dtype": "uint8",
        "colour_policy": "RGB source -> yuv420p anchors -> RGB decode",
        "eligibility": ["near-static", "smooth-pan", "same compatible court context"],
        "video": VIDEO,
        "scenes": list(SCENES),
        "frames_per_scene": n_frames,
        "fps": FPS,
        "context_id": CONTEXT_ID,
        "sources": sources,
        "source_hash_policy": "sha256(contiguous uint8 RGB bytes); shape and dtype stored beside digest",
        "manifest_sha256": manifest["selected_scene_records_sha256"],
        "implementation_digest": impl_digest,
        "timing_boundaries": [
            "encoder_seconds",
            "client_seconds",
            "evaluation_seconds",
        ],
    }
    if extra:
        identity.update(extra)

    return identity


def write_gate_a_identity(output_dir: Path, identity: dict[str, Any]) -> None:
    """Safely write identity.json with fingerprint checking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity.json"
    fp = fingerprint(identity)
    payload = {"fingerprint": fp, "identity": identity}

    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing.get("fingerprint") != fp:
            raise SystemExit(
                f"{output_dir} already contains a different identity. "
                "Gate A requires a fresh output directory per duration / configuration change."
            )
        return

    identity_path.write_text(json.dumps(payload, indent=2) + "\n")
