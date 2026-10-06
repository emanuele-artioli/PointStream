#!/usr/bin/env bash
# Build the PointStream runtime prefix and write its lock files.
#
#   env/build.sh ROOT [FIRST_STEP]
#
# Steps: runtime, build, dcvc (sources, extensions, vendored trees), locks.
# FIRST_STEP resumes a failed build in the same ROOT without redoing earlier steps.
#
# ROOT must be host-local disk (never the NFS home): the prefix is ROOT/pointstream,
# the extension build prefix ROOT/pointstream-build, sources and wheels ROOT/work.
# Pack the result with `python3 -m experiments.jobs.environment pack` (docs/fleet.md).
set -euo pipefail

ROOT=$(realpath -m "${1:?usage: env/build.sh ROOT [FIRST_STEP]}")
FIRST=${2:-runtime}
ENV_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$ENV_DIR")
PREFIX=$ROOT/pointstream
BUILD=$ROOT/pointstream-build
WORK=$ROOT/work
LOCKS=$ENV_DIR/locks
CONDA=${CONDA_EXE:-conda}

DCVC_URL=https://github.com/microsoft/DCVC
DCVC_REV=cbdae87a5445114cdc7f48816da63ea80bdeac40      # DCVC-UF, CVPR 2026
CUTLASS_URL=https://github.com/NVIDIA/cutlass
CUTLASS_TAG=v4.4.1                                      # DCVC README
WILOR_URL=https://github.com/rolpotamias/WiLoR
WILOR_REV=fcb911312a38fa8badd30d9656a167485d61b8f9

# Inference-extension variants. DCVC selects tuned CUTLASS hint tables at compile
# time (CURRENT_DEVICE_SM) and the kernel path at run time from the device, so one
# variant serves sm_70 to sm_86 (sm_80 hints) and one serves Ada (sm_89 hints).
declare -A GENCODE=(
  [sm80]="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86"
  [sm89]="-gencode=arch=compute_89,code=sm_89"
)

case "$ROOT" in /home/*) echo "ROOT must be host-local, not the NFS home: $ROOT" >&2; exit 2;; esac
STEPS=(runtime build dcvc locks)
case " ${STEPS[*]} " in *" $FIRST "*) ;; *) echo "unknown step $FIRST" >&2; exit 2;; esac
run_step() {  # run_step NAME: true when NAME is FIRST or comes after it
  local name
  for name in "${STEPS[@]}"; do
    [ "$name" = "$FIRST" ] && return 0
    [ "$name" = "$1" ] && return 1
  done
}
if [ "$FIRST" = runtime ]; then
  for existing in "$PREFIX" "$BUILD"; do
    [ ! -e "$existing" ] || { echo "refusing to overwrite $existing; build into a new ROOT" >&2; exit 2; }
  done
fi
mkdir -p "$ROOT" "$WORK" "$LOCKS"
export CONDA_CHANNEL_PRIORITY=strict CONDA_PKGS_DIRS=$ROOT/conda-pkgs PIP_CACHE_DIR=$ROOT/pip-cache PIP_DISABLE_PIP_VERSION_CHECK=1
log() { printf '\n== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cd "$REPO"   # environment.yaml refers to env/requirements.txt relative to the repository
PY=$PREFIX/bin/python
if run_step runtime; then
  log "runtime prefix $PREFIX"
  "$CONDA" env create -p "$PREFIX" -f environment.yaml
  "$PY" -m pip install --no-deps -r env/no-deps.txt
fi

if run_step build; then
  log "build prefix $BUILD"
  "$CONDA" env create -p "$BUILD" -f env/build-environment.yaml
fi

if run_step dcvc; then
log "DCVC-UF sources"
rm -rf "$WORK/DCVC"
git clone -q "$DCVC_URL" "$WORK/DCVC"
git -C "$WORK/DCVC" checkout -q "$DCVC_REV"
git clone -q --depth 1 --branch "$CUTLASS_TAG" "$CUTLASS_URL" "$WORK/DCVC/third_party/cutlass"
CUTLASS_REV=$(git -C "$WORK/DCVC/third_party/cutlass" rev-parse HEAD)
git -C "$WORK/DCVC" apply "$ENV_DIR/patches/dcvc-build-targets.patch"

log "DCVC-UF extensions"
rm -rf "$WORK/wheels"
mkdir -p "$WORK/wheels/cpu"
(
  # conda-forge keeps the CUDA headers and libraries under targets/, which
  # torch's extension builder does not search.
  export PATH=$BUILD/bin:$PATH CUDA_HOME=$BUILD
  export CPATH=$BUILD/targets/x86_64-linux/include LIBRARY_PATH=$BUILD/targets/x86_64-linux/lib
  cd "$WORK/DCVC/src/cpp"
  "$BUILD/bin/python" -m pip wheel --no-build-isolation --no-deps . -w "$WORK/wheels/cpu"
  cd "$WORK/DCVC/src/layers/extensions/inference"
  for variant in sm80 sm89; do
    rm -rf build ./*.egg-info
    DCVC_SM=${variant#sm} DCVC_GENCODE=${GENCODE[$variant]} \
      "$BUILD/bin/python" -m pip wheel --no-build-isolation --no-deps . -w "$WORK/wheels/$variant"
  done
)
"$PY" -m pip install --no-deps "$WORK"/wheels/cpu/*.whl

log "vendored sources into $PREFIX/opt"
rm -rf "$PREFIX/opt"
mkdir -p "$PREFIX/opt/DCVC" "$PREFIX/opt/WiLoR"
# The clean pinned tree: the build patch touches only setup.py, which never runs here.
git -C "$WORK/DCVC" archive "$DCVC_REV" | tar -x -C "$PREFIX/opt/DCVC"
for variant in sm80 sm89; do
  mkdir -p "$PREFIX/opt/dcvc-extensions/$variant"
  "$PY" -m zipfile -e "$WORK"/wheels/$variant/*.whl "$PREFIX/opt/dcvc-extensions/$variant"
done
rm -rf "$WORK/WiLoR"
git clone -q "$WILOR_URL" "$WORK/WiLoR"
git -C "$WORK/WiLoR" checkout -q "$WILOR_REV"
git -C "$WORK/WiLoR" archive "$WILOR_REV" | tar -x -C "$PREFIX/opt/WiLoR"
# WiLoR has no packaging; a relocatable .pth puts its tree on sys.path.
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "import os, sys; sys.path.append(os.path.join(sys.prefix, 'opt', 'WiLoR'))" > "$SITE/pointstream_wilor.pth"

"$PY" - "$PREFIX/opt/PROVENANCE.json" <<EOF
import hashlib, json, pathlib, sys
work = pathlib.Path("$WORK")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
wheels = {str(p.relative_to(work)): sha(p) for p in sorted(work.glob("wheels/*/*.whl"))}
record = {
    "DCVC": {"url": "$DCVC_URL", "revision": "$DCVC_REV", "path": "opt/DCVC"},
    "cutlass": {"url": "$CUTLASS_URL", "tag": "$CUTLASS_TAG", "revision": "$CUTLASS_REV"},
    "dcvc_build_patch_sha256": sha(pathlib.Path("$ENV_DIR/patches/dcvc-build-targets.patch")),
    "dcvc_extension_gencode": {"sm80": "${GENCODE[sm80]}", "sm89": "${GENCODE[sm89]}"},
    "wheels_sha256": wheels,
    "WiLoR": {"url": "$WILOR_URL", "revision": "$WILOR_REV", "path": "opt/WiLoR"},
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
EOF

fi

log "locks"
"$CONDA" list --explicit --md5 -p "$PREFIX" > "$LOCKS/pointstream.conda.txt"
"$PY" -m pip freeze --all > "$LOCKS/pointstream.pip.txt"
cp "$PREFIX/opt/PROVENANCE.json" "$LOCKS/pointstream.opt.json"
"$CONDA" list --explicit --md5 -p "$BUILD" > "$LOCKS/pointstream-build.conda.txt"
"$BUILD/bin/python" -m pip freeze --all > "$LOCKS/pointstream-build.pip.txt"
# Known, deliberate gaps (env/no-deps.txt) are the only acceptable pip check output.
"$PY" -m pip check > "$LOCKS/pointstream.pip-check.txt" 2>&1 || true
log "done"
