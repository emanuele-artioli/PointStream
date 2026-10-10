#!/usr/bin/env bash
# Vendor NVRC (G5's oracle, docs/experiments.md) into PREFIX/opt/NVRC, patched, and
# record it in PREFIX/opt/PROVENANCE.json.
#
#   env/nvrc.sh PREFIX WORK
#
# Called by env/build.sh after the other vendored trees. NVRC has no packaging and
# runs as a script (`python main_nvrc.py` from its directory), so it needs no .pth.
# The patch makes deepspeed optional (only a FLOPs profiler uses it) and lets a
# fourth PNG channel mask the training loss (experiments/background/g5.py).
set -euo pipefail
PREFIX=$(realpath "${1:?usage: env/nvrc.sh PREFIX WORK}")
WORK=$(realpath -m "${2:?usage: env/nvrc.sh PREFIX WORK}")
ENV_DIR=$(cd "$(dirname "$0")" && pwd)
NVRC_URL=https://github.com/hmkx/NVRC
NVRC_REV=ccc432d21bdb6a04a453de588c12396d16af0d81      # NeurIPS 2024, MIT
PATCH=$ENV_DIR/patches/nvrc-pointstream.patch

rm -rf "$WORK/NVRC" "$PREFIX/opt/NVRC"
git clone -q "$NVRC_URL" "$WORK/NVRC"
git -C "$WORK/NVRC" checkout -q "$NVRC_REV"
git -C "$WORK/NVRC" apply "$PATCH"
mkdir -p "$PREFIX/opt/NVRC"
git -C "$WORK/NVRC" archive "$NVRC_REV" | tar -x -C "$PREFIX/opt/NVRC"
git -C "$WORK/NVRC" diff | (cd "$PREFIX/opt/NVRC" && patch -s -p1)
"$PREFIX/bin/python" - "$PREFIX/opt/PROVENANCE.json" "$NVRC_URL" "$NVRC_REV" "$PATCH" <<'PY'
import hashlib, json, pathlib, sys
path, url, rev, patch = sys.argv[1:]
record = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else {}
record["NVRC"] = {"url": url, "revision": rev, "path": "opt/NVRC",
                  "patch": "env/patches/nvrc-pointstream.patch",
                  "patch_sha256": hashlib.sha256(pathlib.Path(patch).read_bytes()).hexdigest()}
pathlib.Path(path).write_text(json.dumps(record, indent=2) + "\n")
PY
