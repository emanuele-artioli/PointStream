#!/bin/bash
# HInt (HaMeR hand keypoints), run on gpu1 on 2026-10-06. The host's TLS certificate
# expired 2025-05-08 and no checksum is published, so verification is off (-k) at the
# user's request; the sha256 is recorded while streaming and the size checked against
# the server's Content-Length (download-headers.txt).
set -euo pipefail
D=/home/itec/emanuele/Datasets/HInt
mkdir -p "$D" && cd "$D"
curl -k -sS -fL --retry 3 -D download-headers.txt \
  https://fouheylab.eecs.umich.edu/~dandans/projects/hamer/HInt_annotation_partial.zip \
  | tee HInt_annotation_partial.zip.part | sha256sum > HInt_annotation_partial.zip.sha256.part
mv HInt_annotation_partial.zip.part HInt_annotation_partial.zip
mv HInt_annotation_partial.zip.sha256.part HInt_annotation_partial.zip.sha256
chmod 444 ./*
