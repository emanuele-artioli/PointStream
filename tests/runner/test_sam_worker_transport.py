from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_dataset_pipeline import _load_sam_worker


def _metadata(tmp_path: Path) -> Path:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 1
    arrays_path = tmp_path / "sam31_masks.npz"
    np.savez_compressed(arrays_path, mask_000000=mask)
    payload = {
        "schema": "pointstream.sam31-worker-result.v1",
        "arrays_path": str(arrays_path),
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "provenance": {
            "name": "sam3.1_multiplex",
            "model_revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "policy": "offline_bidirectional",
        },
        "outputs": [
            {
                "role": "player",
                "frame_index": 0,
                "object_id": "source:player:1",
                "tracker_id": 3,
                "score": 0.9,
                "status": "observed",
                "reason": None,
                "mask_key": "mask_000000",
            }
        ],
        "prompts": [],
        "retries": [],
        "sam_runtime": {},
        "gpu_memory_bytes": {},
        "model_load_seconds": 1.0,
        "inference_seconds": 2.0,
    }
    path = tmp_path / "sam31_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_worker_mask_archive_roundtrips_with_provenance(tmp_path: Path) -> None:
    result = _load_sam_worker(_metadata(tmp_path))
    observed = result["outputs"]["player"][(0, "source:player:1")]
    assert observed.tracker_id == 3
    assert observed.mask.shape == (5, 7)
    assert np.count_nonzero(observed.mask) == 12
    assert result["provenance"].policy == "offline_bidirectional"


def test_worker_mask_archive_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = _metadata(tmp_path)
    archive = tmp_path / "sam31_masks.npz"
    np.savez_compressed(archive, mask_000000=np.ones((5, 7), dtype=np.uint8))
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_sam_worker(path)
