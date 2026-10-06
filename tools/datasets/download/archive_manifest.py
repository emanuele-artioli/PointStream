"""Write the manifest for Datasets/archive/pre-reset-2026-10-05 from the find listing and sha256 pass."""

import collections
import datetime as dt
import json
import os
import sys

ARCHIVE = "/home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05"
OUT = f"{ARCHIVE}/MANIFEST.json"
listing, sha_file, revision = sys.argv[1], sys.argv[2], sys.argv[3]

sha = {}
for line in open(sha_file):
    digest, path = line.rstrip("\n").split("  ", 1)
    sha[path] = digest

entries, totals = [], collections.defaultdict(lambda: collections.Counter())
for line in open(listing):
    kind, size, mtime, path, target = line.rstrip("\n").split("\t")
    top = path.split("/", 1)[0]
    e = {"path": path, "type": {"f": "file", "d": "dir", "l": "symlink"}.get(kind, kind),
         "bytes": int(size), "mtime": dt.datetime.fromtimestamp(float(mtime), dt.timezone.utc).isoformat(timespec="seconds")}
    if kind == "f":
        e["sha256"] = sha[path]
        totals[top]["files"] += 1
        totals[top]["bytes"] += int(size)
    elif kind == "l":
        e["target"] = target
        totals[top]["symlinks"] += 1
    else:
        totals[top]["dirs"] += 1
    entries.append(e)

missing = [e["path"] for e in entries if e["type"] == "file" and not e.get("sha256")]
assert not missing, missing[:5]
manifest = {
    "archive": ARCHIVE,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "reason": "Reset plan 2026-10-05 (Part A step 5): unlabelled datasets superseded by labelled datasets; kept, not deleted, until the labelled datasets show where we stand. The demo depends on them.",
    "operation": "mv (rename within the same NFS filesystem) on 2026-10-05 ~23:05 CEST; nothing copied, modified or deleted",
    "moves": {
        "/home/itec/emanuele/Datasets/tennis_games": f"{ARCHIVE}/tennis_games",
        "/home/itec/emanuele/Datasets/Egocentric-10K": f"{ARCHIVE}/Egocentric-10K",
        "/home/itec/emanuele/Datasets/pointstream-demo": f"{ARCHIVE}/pointstream-demo",
    },
    "checks_before_move": "no process on gpu1-3,5,6 referenced the paths (gpu4 locked for measurements, not checked); no symlinks under Datasets (depth 3) pointed into them; no active fleet jobs",
    "not_moved": "old job directories and manifests under Datasets/pointstream-data stay in place; paths they record now resolve under this archive",
    "permissions": "made read-only (chmod -R a-w) after this manifest was written",
    "tool": {"script": "tools/datasets/download/archive_manifest.py", "git_revision": revision, "host": os.uname().nodename},
    "totals": {k: dict(v) for k, v in totals.items()},
    "entries": entries,
}
tmp = OUT + ".tmp"
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=0)
os.chmod(tmp, 0o444)
os.rename(tmp, OUT)
print(json.dumps(manifest["totals"]))
