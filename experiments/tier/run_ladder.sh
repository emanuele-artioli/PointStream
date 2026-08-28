#!/usr/bin/env bash
# Drive the BP24 paired ladder over its axes, one invocation per axis.
#
# Sequential on purpose. The runs are CPU-heavy and this host's home directory —
# which holds the conda environment as well as the repo — is an NFS mount that
# has been measured at roughly 70 KB/s under load. Running these in parallel
# multiplies the import tax rather than the throughput.
#
# Each invocation writes its own JSON, so a run that dies part-way leaves the
# axes that finished behind rather than one truncated file.
#
#   bash experiments/tier/run_ladder.sh [n_frames]
set -u

FRAMES="${1:-8}"
OUT="outputs/bp24-ladder"
RUN="conda run -n pointstream --no-capture-output python -u -m experiments.tier.ladder"

mkdir -p "$OUT"

echo "=== [1/4] av1, payload sweep, low motion (P0 item 2) ==="
$RUN --codecs av1 --frames "$FRAMES" --tier balanced --sweep payload \
     --video alcaraz_highlights --scene scene_000 \
     --out "$OUT/av1-payload-lowmotion.json"

echo "=== [2/4] av1, residual-coarseness sweep, low motion (P0 item 3) ==="
$RUN --codecs av1 --frames "$FRAMES" --tier balanced --sweep coarseness \
     --video alcaraz_highlights --scene scene_000 \
     --out "$OUT/av1-coarseness-lowmotion.json"

# The motion axis. `outputs/bp24-ladder/motion-survey.json` measured this scene
# at 7.70 grey levels between consecutive frames against 0.33 for the one above
# — a factor of 23. Findings §7 predicts PointStream does much worse here,
# because the plate it transmits is the first source frame.
echo "=== [3/4] av1, payload sweep, HIGH motion (findings §7 re-measure) ==="
$RUN --codecs av1 --frames "$FRAMES" --tier balanced --sweep payload \
     --video federer_djokovic --scene scene_003 \
     --out "$OUT/av1-payload-highmotion.json"

# One gain per codec, each against itself. Never ranked against each other:
# the presets are not equal effort (findings §1).
echo "=== [4/4] the rest of the roster, low motion ==="
$RUN --codecs hevc avc vvc --frames "$FRAMES" --tier balanced --sweep payload \
     --video alcaraz_highlights --scene scene_000 \
     --out "$OUT/roster-payload-lowmotion.json"

echo "=== ladder script finished ==="
