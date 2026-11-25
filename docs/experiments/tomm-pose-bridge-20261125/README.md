# Masked interior pose gaps on held-out broadcasts

## Decision recorded before scoring

Hypothesis: linear interpolation across a bracketed five- or ten-frame gap
preserves pose geometry better than holding the last transmitted pose.
Competing explanation: players accelerate or change direction enough that
the improvement exists only in the average, with substantial errors on some
gaps, especially on the far player.

Use the complete 240-tick paid PSCP tracks from Shanghai (match102) and
Sydney (match127) if available locally. The selected paid carriers must match
the existing frozen pose-machine inputs. Scan all interior starts for lengths
5 and 10, separately by role and broadcast. Compare the interpolated pose and
the held pose to the original paid pose on each omitted tick. Use the original
decoded crop height for PCK@0.1 and its area for COCO OKS normalization. A joint
counts toward the target when its paid confidence is at least 0.3. Interpolation
uses endpoint coordinates and requires the joint to exist at both endpoints.

Promote only to a rendered validation if mean PCK@0.1 is at least 0.95,
the 10th percentile of per-gap PCK@0.1 is at least 0.80, and mean OKS beats
the last-pose hold, for each role and gap length. This is agreement with
the transmitted trajectory under artificial omissions, not human anatomical
accuracy, natural-occlusion behavior, or rendered quality. DISTS and temporal
seams require an actual native rendered comparison on the same delivery.
Stop after this CPU diagnostic. Budget: 120 seconds CPU, 0 GPU-hours.

## Outcome

Sydney passes the coordinate gate at five omitted frames for both roles
and at ten for the far player only. The near player's 10-frame mean
PCK@0.1 was 0.930, below the 0.95 gate, although mean OKS improved from
0.751 to 0.913 compared with holding the last pose. Narrow the next
rendered test to five frames for both roles or ten for the far role.
A separate Shanghai run requires its hash-matched paid carrier. The
receipt records the device, commit, scorer and exact Sydney carrier.
