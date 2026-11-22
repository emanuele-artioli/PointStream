"""Compare adjacent-frame luma residuals on the authenticated low-rate displays."""

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
import motion


def main(args):
    labels_path = motion.crops.LABELS
    assert (
        motion.crops.file_sha(labels_path)
        == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    )
    labels = {f["tick"]: f for f in json.loads(labels_path.read_text())["frames"]}
    yy, xx = np.mgrid[:544, :960]
    checked = {}
    for t, frame in labels.items():
        regions = {}
        for i, player in enumerate(frame["players"], 1):
            data = np.frombuffer(base64.b64decode(player["mask"]["png_base64"]), np.uint8)
            mask = cv2.imdecode(data, 0)
            regions[f"p{i}"] = cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST) > 0
        x, y = frame["ball"]["center_xy"]
        regions["ball"] = (abs(xx - x * 0.5) < 16) & (abs(yy - y * 544 / 1080) < 32 * 544 / 1080)
        checked[t] = regions
    rows = []
    previous = {}
    start = time.process_time()

    def observe(t, source, plain, joint):
        current = {"source": source}
        for name, frame in (("plain", plain), ("joint", joint)):
            current[name] = cv2.resize(frame, (960, 544), interpolation=cv2.INTER_AREA)
        if t:
            flow = dis_flow(
                cv2.cvtColor(source, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(previous["source"], cv2.COLOR_RGB2GRAY),
            )
            x, y = xx + flow[:, :, 0], yy + flow[:, :, 1]
            valid = (x >= 0) & (y >= 0) & (x < 959) & (y < 543)
            regions = {"whole": valid, **{k: valid & v for k, v in checked.get(t, {}).items()}}
            delta = {k: luma(v) - luma(warp(previous[k], flow)) for k, v in current.items()}
            for region, mask in regions.items():
                for name, residual in delta.items():
                    error = float(np.mean(np.abs(residual[mask] - delta["source"][mask])))
                    if name == "source":
                        assert error == 0
                    rows.append(
                        dict(
                            tick=t,
                            region=region,
                            arm=name,
                            error=error,
                            energy=float(np.mean(np.abs(residual[mask]))),
                        )
                    )
        previous.update(current)
        cost = time.process_time() - start
        if t == 12:
            assert cost * 240 / 12 * 1.5 < 600
        assert cost < 600

    motion.main(args, observe)
    summary = {}
    for arm in previous:
        summary[arm] = {}
        for region in ("whole", "ball", "p1", "p2"):
            subset = [r for r in rows if r["arm"] == arm and r["region"] == region]
            assert len(subset) == (239 if region == "whole" else 12)
            summary[arm][region] = {
                key: float(np.mean([r[key] for r in subset])) for key in ("energy", "error")
            }
    old = json.loads((motion.HERE.parent / "starved-temporal-20261122/result.json").read_text())
    assert summary["source"] == old["means"]["source"]
    receipt = json.loads((args.output / "motion-receipt.json").read_text())
    result = dict(
        means=summary,
        rows=rows,
        joint_rgb_sha256=receipt["original_joint_240_rgb_sha256"],
        ordinary_native_sha256=receipt["ordinary_yuv_sha256"],
        cpu_seconds=time.process_time() - start,
        script_sha256=motion.crops.file_sha(Path(__file__)),
        equal_previous_source=True,
        opencv=cv2.__version__,
        kbps={"joint_cached": 21.545, "plain": 21.623333333333335},
    )
    (args.output / "temporal.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("packet", "controls", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
