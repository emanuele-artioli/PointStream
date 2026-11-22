"""Independently compare all saved scores to the full W7 reference control."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T215432Z-c35b82f6/full")
HYBRID = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261030T185245Z-e51a4dec/full/published.tar")
EXPECTED_HYBRID = "cbc26962935b3542b0b7cf77f724010de19323fedd6fe02cb822ec6e43e75837"
NAMES = ["control_s4_crf63", "plain_crf36", "plain_crf48", "plain_crf60"]
SERIES = ["full_lpips", "foreground_lpips", "full_dists", "foreground_dists"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_archive(path, member):
    with tarfile.open(path) as tar:
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError("Missing scoring archive member")
        return stream.read()


def main():
    summary = json.loads((ROOT / "score-summary.json").read_text())
    assert digest(HYBRID) == EXPECTED_HYBRID
    with np.load(io.BytesIO(array_archive(HYBRID, "publish/alcaraz_hurkacz__028@120/scores.npz")),
                 allow_pickle=False) as old:
        reference_ticks = old["ticks"].tolist()
        reference = np.stack([old[name][2] for name in SERIES])
    with np.load(io.BytesIO(array_archive(ROOT / "published.tar", "publish/scores.npz")),
                 allow_pickle=False) as scores:
        assert scores["ticks"].tolist() == reference_ticks == list(range(0, 240, 3))
        assert summary["ticks"] == reference_ticks
        assert sorted(scores.files) == sorted(["ticks", *NAMES])
        assert [row["name"] for row in summary["points"]] == NAMES
        errors = np.max(np.abs(scores[NAMES[0]] - reference), axis=1)
        assert np.isfinite(errors).all() and np.max(errors) <= 1e-3
        for index, name in enumerate(SERIES):
            assert abs(float(errors[index]) - summary["control_errors"][name]) < 1e-12
        results = {}
        for row in summary["points"]:
            values = scores[row["name"]]
            assert values.shape == (4, 80) and np.isfinite(values).all()
            assert np.allclose(values.mean(axis=1), summary["means"][row["name"]], rtol=0, atol=1e-12)
            results[row["name"]] = {"rate_kbps": row["rate_kbps"], "means": values.mean(axis=1).tolist()}
    print(json.dumps({"passed": True, "ticks": len(reference_ticks), "series": SERIES,
                      "maximum_control_errors": dict(zip(SERIES, errors.tolist())), "results": results,
                      "published_tar_sha256": digest(ROOT / "published.tar"),
                      "summary_sha256": digest(ROOT / "score-summary.json"), "hybrid_sha256": EXPECTED_HYBRID}))


if __name__ == "__main__":
    main()
