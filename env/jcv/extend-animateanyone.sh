#!/usr/bin/env bash
# Extend the pointstream-jcv pack with Moore-AnimateAnyone (K6b players, 2026-10-22) and write its locks.
#
#   env/jcv/extend-animateanyone.sh PACK DIFFUSERS_WHEEL AA_ARCHIVE ROOT
#
# PACK is pointstream-jcv-20261012T085233Z.tar.gz; DIFFUSERS_WHEEL is diffusers-0.30.3-py3-none-any.whl (PyPI);
# AA_ARCHIVE is `git archive --format=tar` of the fork below at AA_REV. All three are sha256-checked, and nothing is
# fetched from the network (the hosts reach pypi.org but not files.pythonhosted.org). ROOT must be host-local disk.
# The result is ROOT/pointstream-jcv: a copy of the packed prefix plus diffusers 0.30.3 and the AnimateAnyone fork,
# both installed without dependencies (diffusers' declared ones are already in the prefix; checked below).
#
# Why here and not in `pointstream`: AnimateAnyone needs transformers (its CLIP image encoder), and transformers 4.x
# requires huggingface_hub < 1.0 (`pointstream` has 2.x; diffusers 0.30.3 also imports HfFolder, removed in
# huggingface_hub 1.0). That is conflict 10, which pointstream-jcv already resolves, so a second generator goes into
# it instead of a third environment. Called only through experiments/tennis/k6b_generate.py.
# Pack with `python3 -m experiments.jobs.environment pack` (docs/fleet.md).
set -euo pipefail
usage="usage: env/jcv/extend-animateanyone.sh PACK DIFFUSERS_WHEEL AA_ARCHIVE ROOT"
PACK=$(realpath "${1:?$usage}")
WHEEL=$(realpath "${2:?$usage}")
AA_ARCHIVE=$(realpath "${3:?$usage}")
ROOT=$(realpath -m "${4:?$usage}")
ENV_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$(dirname "$ENV_DIR")")
PREFIX=$ROOT/pointstream-jcv
WORK=$ROOT/work
LOCKS=${LOCKS:-$REPO/env/locks}
PACK_SHA256=2bec91d460df5487acd10d5dfe062040db2117238fd0fbb72ecf947cfb7bec74
WHEEL_SHA256=1b70209e4d2c61223b96a7e13bc4d70869c8b0b68f54a35ce3a67fcf813edeee  # PyPI's digest for diffusers 0.30.3
AA_URL=https://github.com/emanuele-artioli/Moore-AnimateAnyone
# The fork of MooreThreads/Moore-AnimateAnyone (Apache-2.0) at a914ef3 that packages the code as `animate_anyone`
# (upstream's top-level `src` collides with PointStream's) and ports it to diffusers 0.30; the tennis fine-tune in
# Models/AnimateAnyone/profiles/finetuned_tennis was trained with its finetune_stage{1,2}.py.
AA_REV=67c42297f8024ddb0fd04b69e42ac71fe6bbf3d8
AA_ARCHIVE_SHA256=bfca6cf1129a0154bf2c3431c0e01b34f5e815eda5ba504f54e001c8ceeb99ee  # git archive --format=tar AA_REV

case "$ROOT" in /home/*) echo "ROOT must be host-local, not the NFS home: $ROOT" >&2; exit 2;; esac
[ ! -e "$PREFIX" ] || { echo "refusing to overwrite $PREFIX; build into a new ROOT" >&2; exit 2; }
printf '%s  %s\n%s  %s\n%s  %s\n' "$PACK_SHA256" "$PACK" "$WHEEL_SHA256" "$WHEEL" "$AA_ARCHIVE_SHA256" "$AA_ARCHIVE" | sha256sum -c -
mkdir -p "$PREFIX" "$WORK" "$LOCKS"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1
tar -xzf "$PACK" -C "$PREFIX"
PY=$PREFIX/bin/python
"$PY" -m pip freeze --all > "$WORK/before.txt"

"$PY" -m pip install --no-deps "$WHEEL"
# diffusers 0.30.3's declared dependencies, all already in the prefix at the versions the locks list (its
# importlib_metadata is the backport, imported only on Python < 3.8; 3.12 uses importlib.metadata).
for module in importlib.metadata filelock huggingface_hub numpy regex requests safetensors PIL; do
  "$PY" -c "import $module" || { echo "missing dependency: $module" >&2; exit 3; }
done

mkdir -p "$PREFIX/opt/Moore-AnimateAnyone"
tar -xf "$AA_ARCHIVE" -C "$PREFIX/opt/Moore-AnimateAnyone"
"$PY" -m pip install --no-deps --no-build-isolation "$PREFIX/opt/Moore-AnimateAnyone"
"$PY" - "$PREFIX/opt/PROVENANCE.json" "$AA_URL" "$AA_REV" "$AA_ARCHIVE_SHA256" "$PACK_SHA256" "$WHEEL_SHA256" <<'PY'
import json, pathlib, sys
path, url, rev, archive, pack, wheel = sys.argv[1:]
record = json.loads(pathlib.Path(path).read_text())
record["Moore-AnimateAnyone"] = {"url": url, "revision": rev, "path": "opt/Moore-AnimateAnyone", "archive_sha256": archive,
                                 "upstream": "https://github.com/MooreThreads/Moore-AnimateAnyone@a914ef3"}
record["diffusers"] = {"version": "0.30.3", "wheel_sha256": wheel}
record["extended_from_pack_sha256"] = pack
pathlib.Path(path).write_text(json.dumps(record, indent=2) + "\n")
PY
# Import check of the inference path (CPU only), JointControlVideo's included.
"$PY" -c "from diffusers import AutoencoderKL, DDIMScheduler; from transformers import CLIPVisionModelWithProjection; \
from animate_anyone.models.pose_guider import PoseGuider; from animate_anyone.models.unet_2d_condition import UNet2DConditionModel; \
from animate_anyone.models.unet_3d import UNet3DConditionModel; from animate_anyone.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline; \
from diffsynth.pipelines.wan_video import WanVideoPipeline; print('import ok')"

"$PY" -m pip freeze --all > "$WORK/after.txt"
diff "$WORK/before.txt" "$WORK/after.txt" > "$LOCKS/pointstream-jcv.animateanyone-delta.txt" || true
cp "$WORK/after.txt" "$LOCKS/pointstream-jcv.pip.txt"
"$PY" -m pip check > "$LOCKS/pointstream-jcv.pip-check.txt" 2>&1 || true
cp "$PREFIX/opt/PROVENANCE.json" "$LOCKS/pointstream-jcv.opt.json"
echo "built $PREFIX"
