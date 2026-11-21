# Machine localization on three frozen match windows

[Vector figure](ball-machine-three.pdf) · [numerical receipts, jobs and methods](../../../experiments/ball-machine-three-20261122/README.md)

The left three panels count detections within 10 pixels at native 1920×1080 source geometry for the exact 9/12/11 official visible labels. The right panel counts predicted positions on the five source rows labeled not visible; it is not an absent-ball false-positive rate. The direct points are decoded fitted tile centers, with clipping. The other arms run the same frozen WASB model on images only, at source cadence. No confidence intervals or population claim are justified by these short windows, and the detector's training overlap is unknown. Source detection itself is 1/9 on match209.

Rate matching is approximate: plain/ROI/tracked rates are 61.35/60.76/61.77, 60.48/60.56/60.83, and 58.85/52.69/60.58 kbps for matches 209, 269, and 235. The AV1 and ROI arms shown are the same frozen near-target base streams as the visual ball study, not necessarily the strongest native-resolution anchors in its separate plot. Receipt hashes are fixed and checked by [plot.py](plot.py).
