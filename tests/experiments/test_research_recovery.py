"""Historical evidence cannot be rewritten through the integration overlay."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def test_historical_scientific_record_cannot_be_whitelisted_as_review_patch(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    dossier = tmp_path / "docs" / "research-recovery"
    shutil.copytree(repository / "docs" / "research-recovery", dossier)
    source_map = tmp_path / "docs" / "workflow" / "reconciliation-source-map.json"
    source_map.parent.mkdir()
    shutil.copyfile(repository / "docs" / "workflow" / source_map.name, source_map)
    for relative in (
        "workflow/pointstream-handoff.md",
        "workflow/long-jobs.md",
        "workflow/repository-maintenance.md",
        "research/README.md",
    ):
        destination = tmp_path / "docs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / "docs" / relative, destination)

    def verify():
        result = subprocess.run(
            [sys.executable, str(dossier / "verify.py")], capture_output=True, text=True, timeout=15
        )
        return result, json.loads(result.stdout)

    result, report = verify()
    assert result.returncode == 0 and report["passed"]
    assert report["source_checks"].startswith("not requested")
    record = dossier / "records" / "code" / "all-pr-dispositions.json"
    record.write_bytes(record.read_bytes() + b"\n")
    mapping = json.loads(source_map.read_text())
    for row in mapping["files"]:
        if row["path"] == "docs/research-recovery/records/code/all-pr-dispositions.json":
            row["review_changed"] = True
            row["integrated_sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    source_map.write_text(json.dumps(mapping))
    result, report = verify()
    assert result.returncode == 1 and not report["passed"]
    assert "Dossier file identity records/code/all-pr-dispositions.json" in report["errors"]
