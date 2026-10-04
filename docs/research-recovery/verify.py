#!/usr/bin/env python3
"""Validate dossier links/indexes; optionally verify recovered sources outside Git."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

p = argparse.ArgumentParser()
p.add_argument("--source-root", type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parent
errors = []
checked = {"json": 0, "jsonl_rows": 0, "relative_links": 0, "local_anchors": 0, "source_hashes": 0}


def check(ok, msg):
    if not ok:
        errors.append(msg)


def read(path):
    return json.loads(path.read_text())


for f in sorted(root.rglob("*")):
    if not f.is_file():
        continue
    if f.suffix == ".json":
        try:
            read(f)
            checked["json"] += 1
        except Exception as e:
            errors.append(f"{f.relative_to(root)}: {e}")
    elif f.suffix == ".jsonl":
        try:
            for line in f.read_text().splitlines():
                if line.strip():
                    json.loads(line)
                    checked["jsonl_rows"] += 1
        except Exception as e:
            errors.append(f"{f.relative_to(root)}: {e}")
    elif f.suffix == ".md":
        for match in re.finditer(r"!?\[[^\]\n]*\]\(([^\n]+?)\)", f.read_text()):
            dest = match.group(1).strip().split(' "')[0].strip("<>")
            if re.match(r"[A-Za-z][\w+.-]*:", dest) or dest.startswith(("/", "#")):
                continue
            target, _, fragment = unquote(dest).partition("#")
            checked["relative_links"] += 1
            linked = f.parent / target if target else f
            check(linked.exists(), f"{f.relative_to(root)} broken link: {dest}")
            if fragment and linked.exists() and linked.suffix == ".md":
                content = linked.read_text()
                anchors = set(re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)', content))
                for heading in re.findall(r"^#+\s+(.+)", content, re.M):
                    heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading).lower()
                    anchors.add(
                        "".join(c for c in heading if c.isalnum() or c in " -_").replace(" ", "-")
                    )
                checked["local_anchors"] += 1
                check(fragment in anchors, f"{f.relative_to(root)} broken anchor: {dest}")
pr = read(root / "records/code/all-pr-dispositions.json")["records"]
check({x["pr"] for x in pr} == set(range(1, 150)), "PR set differs from 1–149")
check(len(pr) == 149, "Duplicate PR records")
commits = read(root / "records/code/all-commit-dispositions.json")["commits"]
check(
    len(commits) == 1165 and len({x["sha"] for x in commits}) == 1165,
    "Commit inventory not 1165 unique objects",
)
check(all(x.get("disposition") for x in commits), "Missing commit disposition")
paper = read(root / "records/manuscript/all-commit-dispositions.json")
paper_rows = paper.get("commits", paper.get("records", [])) if isinstance(paper, dict) else paper
check(
    len(paper_rows) == 63 and len({x["sha"] for x in paper_rows}) == 63,
    "Paper commit inventory not 63 unique objects",
)
check(all(x.get("disposition") for x in paper_rows), "Missing paper commit disposition")
families = [
    json.loads(x)
    for x in (root / "records/server/outputs_actionable_source_map.jsonl").read_text().splitlines()
    if x.strip()
]
check(
    len(families) == 179 and len({x["family"] for x in families}) == 179,
    "Output map not 179 unique families",
)
check(
    sum(x["recursive_status"] == "complete" for x in families) == 169,
    "Complete output traversal count changed",
)
check(
    sum(x["recursive_status"] == "complete_with_pruned_media_dirs" for x in families) == 10,
    "Pruned output traversal count changed",
)
for claim in read(root / "records/headline-claims.json")["claims"]:
    check(
        bool(claim.get("classification")) and bool(claim.get("contradiction_or_limit")),
        f"Incomplete headline qualification {claim.get('id')}",
    )
    for path in claim["evidence"]:
        check((root / path).is_file(), f"Headline evidence missing {claim['id']}: {path}")
for x in read(root / "records/sessions/index.json")["records"]:
    if x["path"].startswith("dossier/"):
        file = root / x["path"].removeprefix("dossier/")
        check(
            file.exists() and hashlib.sha256(file.read_bytes()).hexdigest() == x["sha256"],
            f"Session record identity: {x['path']}",
        )
for x in read(root / "records/demo/report-index.json"):
    file = root / "records/demo" / x["recovered_small_file"]
    check(
        file.exists() and hashlib.sha256(file.read_bytes()).hexdigest() == x["sha256"],
        f"Demo copied report identity: {file}",
    )
exp = [
    json.loads(x)
    for x in (root / "records/experiments/family-dispositions.jsonl").read_text().splitlines()
    if x.strip()
]
check(all(x.get("disposition") for x in exp), "Missing experiment disposition")
check(
    {x["family"] for x in exp} == {x["family"] for x in families},
    "Experiment/server family dispositions do not reconcile",
)
if a.source_root:
    s = a.source_root

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    code = s / "source/code.git"
    actual = set(
        subprocess.check_output(
            ["git", f"--git-dir={code}", "rev-list", "--all"], text=True
        ).split()
    )
    check(actual == {x["sha"] for x in commits}, "Recovered Git union differs from commit index")
    actualpaper = set(
        subprocess.check_output(
            ["git", f"--git-dir={s / 'source/paper.git'}", "rev-list", "--all"], text=True
        ).split()
    )
    check(
        actualpaper == {x["sha"] for x in paper_rows},
        "Recovered paper Git union differs from index",
    )

    def retained_path(raw):
        old = Path("/private/tmp/pointstream-recovery-20260930")
        path = Path(raw)
        return s / path.relative_to(old) if path.is_relative_to(old) else path

    for x in pr:
        actualref = subprocess.check_output(
            ["git", f"--git-dir={code}", "rev-parse", f"refs/heads/pr/{x['pr']}"], text=True
        ).strip()
        check(actualref == x["pr_head_sha"], f"PR{x['pr']} head mismatch")
    for x in read(root / "records/server/dirty-source-snapshots.json"):
        path = retained_path(x["artifact"])
        check(path.exists(), f"Missing snapshot {path}")
        if path.exists():
            check(sha(path) == x["sha256"], f"Snapshot checksum {path}")
            checked["source_hashes"] += 1
    for x in read(root / "records/initial-state.json")["live_demo_files"]:
        path = retained_path(x["snapshot"])
        check(path.exists(), f"Missing demo snapshot {path}")
        if path.exists():
            check(sha(path) == x["sha256"], f"Demo snapshot checksum {path}")
            checked["source_hashes"] += 1
    for x in read(root / "records/demo/closeout-addendum-source-index.json")["files"]:
        if "snapshot" not in x:
            continue
        path = retained_path(x["snapshot"])
        check(path.exists() and sha(path) == x["sha256"], f"Demo addendum checksum {path}")
        checked["source_hashes"] += 1
    for x in csv.DictReader(
        open(root / "records/server/outputs_selected_report_hashes.tsv"), delimiter="\t"
    ):
        paths = [
            s / "workers/server/outputs" / sub / x["path"]
            for sub in ["selected_reports", "additional_selected_reports"]
        ]
        present = [f for f in paths if f.is_file()]
        check(
            bool(present)
            and sha(present[0]) == x["sha256"]
            and present[0].stat().st_size == int(x["bytes"]),
            f"Selected report checksum {x['path']}",
        )
        checked["source_hashes"] += 1
    for group in read(root / "records/experiments/artifact-index.json")["by_sha256"]:
        for x in group["captures"]:
            file = (
                (root / x["retained"])
                if x["retained"].startswith("records/")
                else s / x["retained"]
            )
            check(file.exists() and sha(file) == x["sha256"], f"Experiment source identity {file}")
            checked["source_hashes"] += 1
    final = read(root / "records/final-source-state.json")
    for x in final["late_small_sources"] + [final["late_changed_snapshot"]]:
        file = retained_path(x["snapshot"])
        check(file.exists() and sha(file) == x["sha256"], f"Late source identity {file}")
        checked["source_hashes"] += 1
    manifest = s / "source/retained-source-manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            x = json.loads(line)
            file = s / x["path"]
            check(
                file.exists() and sha(file) == x["sha256"] and file.stat().st_size == x["bytes"],
                f"Retained manifest identity {x['path']}",
            )
            checked["source_hashes"] += 1
    for name, expected in [
        ("bp21-report.json", "b6f8b8463c73f7d3a54a918b677bdb5b1e18ae7cf789490c900e023fa1d21240"),
        ("gate-a-report.json", "71d82a51c6bf831c6a336d12a2cb05b1496912caa17a702b9e490f28380d939a"),
        ("je10-result.json", "1cac287bc1bd84b3d6db650234068748c576a4aef0e9e27eb47b766f60895249"),
    ]:
        path = s / "workers/coordinator" / name
        check(path.exists() and sha(path) == expected, f"Headline source checksum {name}")
        checked["source_hashes"] += 1
manifest = root / "records/dossier-files.jsonl"
integration_map = root.parent / "workflow/reconciliation-source-map.json"
integrated = (
    {row["path"]: row for row in read(integration_map)["files"]} if integration_map.exists() else {}
)
if manifest.exists():
    for line in manifest.read_text().splitlines():
        x = json.loads(line)
        file = root / x["path"]
        current = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        original_matches = (
            file.is_file() and file.stat().st_size == x["bytes"] and current == x["sha256"]
        )
        # Only the reader index and this verifier may be reviewed overlays.
        # Scientific records remain byte-exact against the historical manifest.
        overlay = integrated.get("docs/research-recovery/" + x["path"], {})
        reviewed_overlay = (
            x["path"] in {"README.md", "verify.py"}
            and overlay.get("review_changed") is True
            and overlay.get("source_sha256") == x["sha256"]
            and overlay.get("integrated_sha256") == current
        )
        check(original_matches or reviewed_overlay, f"Dossier file identity {x['path']}")

out = {
    "checks": checked,
    "source_checks": "performed"
    if a.source_root
    else "not requested; recover external archive for source checks",
    "errors": errors,
    "passed": not errors,
}
print(json.dumps(out, indent=2))
raise SystemExit(1 if errors else 0)
