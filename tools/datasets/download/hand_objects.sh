#!/bin/bash
# EPIC-KITCHENS-100 hand-object detections (doi:10.5523/bris.3l8eci2oqgst92n14w2yqi5ytu) for
# the videos named as arguments; run on gpu1 on 2026-10-07 for B1b (the 34 videos of VISOR
# evaluation set v2). data.bris publishes no checksum: the sha256 is recorded while streaming
# and the size checked against the server's Content-Length. P01_109 and P27_103 are
# mis-extracted on data.bris (the release README links replacements); neither is in set v2,
# so they are refused here rather than fetched wrong.
set -euo pipefail
D=/home/itec/emanuele/Datasets/EPIC-KITCHENS-hand-objects
B=https://data.bris.ac.uk/datasets/3l8eci2oqgst92n14w2yqi5ytu/hand-objects
mkdir -p "$D"
for v in "$@"; do
  case "$v" in P01_109|P27_103) echo "REFUSE $v (mis-extracted on data.bris)"; exit 1;; esac
  p=${v%%_*}
  o="$D/hand-objects/$p/$v.pkl"
  if [ -e "$o" ]; then echo "HAVE $v"; continue; fi
  mkdir -p "$D/hand-objects/$p"
  h=$(mktemp)
  digest=$(curl -sS -fL --retry 3 -D "$h" "$B/$p/$v.pkl" | tee "$o.part" | sha256sum | cut -d" " -f1)
  want=$(tr -d '\r' < "$h" | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  got=$(stat -c %s "$o.part")
  if [ "$want" != "$got" ]; then echo "FAIL $v size $got != Content-Length $want"; exit 1; fi
  mv "$o.part" "$o"
  chmod 444 "$o"
  printf 'hand-objects/%s/%s.pkl\t%s\tcomputed_while_streaming_download_size_equals_content_length\n' \
    "$p" "$v" "$digest" >> "$D/download.sha256.tsv"
  rm -f "$h"
  echo "OK $v $got $digest"
done
