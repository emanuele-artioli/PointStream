#!/usr/bin/env bash
# Build libaom's `inspect` with per-symbol bit accounting, for H1's bits-per-region measure
# (docs/experiments.md, 2026-10-08 H1). Run on a host, on host-local disk:
#   tools/aom/build-inspect.sh /dev/shm/$USER-aom
# env/patches/aom-inspect-ps1.patch makes `inspect` print each frame's order hint (so bits map to
# display frames) and every symbol, including the first of each block (upstream prints the
# block's context in its place, dropping its bits).
set -euo pipefail
work=${1:?work directory}
here=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$work" && cd "$work"
[ -d aom ] || git clone -q --depth 1 --branch v3.12.1 https://aomedia.googlesource.com/aom aom
test "$(git -C aom rev-parse HEAD)" = 10aece4157eb79315da205f39e19bf6ab3ee30d0
git -C aom apply "$here/env/patches/aom-inspect-ps1.patch"
mkdir -p build && cd build
cmake ../aom -DCMAKE_BUILD_TYPE=Release -DCONFIG_ACCOUNTING=1 -DCONFIG_INSPECTION=1 -DENABLE_EXAMPLES=1 \
  -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DENABLE_TOOLS=0 -DCONFIG_AV1_ENCODER=0 > cmake.log
make -j16 inspect > make.log
sha256sum examples/inspect
