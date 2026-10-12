#!/usr/bin/env bash
# Build the pointstream-jcv prefix (H3e's hand generator) and write its locks.
#
#   env/jcv/build.sh ROOT
#
# ROOT must be host-local disk. The prefix is ROOT/pointstream-jcv; JointControlVideo's
# tree (a DiffSynth-Studio fork, package `diffsynth`) is vendored into
# PREFIX/opt/JointControlVideo at a pinned revision and installed without dependencies.
# Pack with `python3 -m experiments.jobs.environment pack` (docs/fleet.md).
set -euo pipefail
ROOT=$(realpath -m "${1:?usage: env/jcv/build.sh ROOT}")
ENV_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$(dirname "$ENV_DIR")")
PREFIX=$ROOT/pointstream-jcv
WORK=$ROOT/work
LOCKS=$REPO/env/locks
CONDA=${CONDA_EXE:-conda}
JCV_URL=https://github.com/ZhangCYG/JointControlVideo
JCV_REV=f96b2c377f26a165bf63cd9fcb495bb9fd3d3054      # ECCV 2026, Apache-2.0
# The fork omits diffsynth/models/wan_video_camera_controller.py, which wan_video_dit.py imports (SimpleAdapter,
# used only by Wan-Fun camera models). Taken from upstream DiffSynth-Studio (Apache-2.0) at a pinned commit.
CAMERA_URL=https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/c795e35142cc44a8d924e682853066816ebe8468/diffsynth/models/wan_video_camera_controller.py
CAMERA_SHA256=2bad958eb9893105cc7215a4bb91b2c039bde5eb6af8ddf0f02c09e6f7883d61

case "$ROOT" in /home/*) echo "ROOT must be host-local, not the NFS home: $ROOT" >&2; exit 2;; esac
[ ! -e "$PREFIX" ] || { echo "refusing to overwrite $PREFIX; build into a new ROOT" >&2; exit 2; }
mkdir -p "$WORK" "$LOCKS"
export CONDA_CHANNEL_PRIORITY=strict CONDA_PKGS_DIRS=$ROOT/conda-pkgs PIP_CACHE_DIR=$ROOT/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$REPO"
"$CONDA" env create -p "$PREFIX" -f env/jcv/environment.yaml
PY=$PREFIX/bin/python

rm -rf "$WORK/JointControlVideo"
git clone -q "$JCV_URL" "$WORK/JointControlVideo"
git -C "$WORK/JointControlVideo" checkout -q "$JCV_REV"
mkdir -p "$PREFIX/opt/JointControlVideo"
git -C "$WORK/JointControlVideo" archive "$JCV_REV" | tar -x -C "$PREFIX/opt/JointControlVideo"
curl -fsSL "$CAMERA_URL" -o "$PREFIX/opt/JointControlVideo/diffsynth/models/wan_video_camera_controller.py"
echo "$CAMERA_SHA256  $PREFIX/opt/JointControlVideo/diffsynth/models/wan_video_camera_controller.py" | sha256sum -c -
"$PY" -m pip install --no-deps --no-build-isolation "$PREFIX/opt/JointControlVideo"
"$PY" - "$PREFIX/opt/PROVENANCE.json" "$JCV_URL" "$JCV_REV" "$CAMERA_URL" "$CAMERA_SHA256" <<'PY'
import json, pathlib, sys
path, url, rev, camera_url, camera_sha = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({"JointControlVideo": {
    "url": url, "revision": rev, "path": "opt/JointControlVideo",
    "added": {"diffsynth/models/wan_video_camera_controller.py": {"url": camera_url, "sha256": camera_sha}}}}, indent=2) + "\n")
PY
# Import check of the inference path (CPU only).
"$PY" -c "from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig; from diffsynth.models.wan_video_joint_control import HandConditioningModule; print('import ok')"

"$CONDA" list -p "$PREFIX" --explicit --md5 > "$LOCKS/pointstream-jcv.conda.txt"
"$PY" -m pip freeze --all > "$LOCKS/pointstream-jcv.pip.txt"
"$PY" -m pip check > "$LOCKS/pointstream-jcv.pip-check.txt" 2>&1 || true
cp "$PREFIX/opt/PROVENANCE.json" "$LOCKS/pointstream-jcv.opt.json"
echo "built $PREFIX"
