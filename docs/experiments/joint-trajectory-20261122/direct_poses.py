"""Verify that direct native-coordinate pose readout supplies the sealed renderer's conditioning."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import cv2
import numpy as np

from experiments.tennis import common_player_packet as player
from experiments.tennis import w8_payload as transport
from experiments.tennis import k6b

HERE = Path(__file__).resolve().parent


def read_pose(wire):
    profile = player._embedded_profile(wire)
    sections, _roles = transport.unpack(wire, profile)
    _, tracks = player._bounded_keypoints(sections["poses"], profile["player"]["order"])
    return profile["player"]["canvas_size"], sections["poses"], tracks


def array_hash(array):
    h = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    h.update(np.ascontiguousarray(array))
    return h.hexdigest()


def measure(folder):
    cv2.setNumThreads(1)
    old = [
        json.loads(
            (
                HERE.parent
                / "receiver-replay-20261118/independent-idle-20261118"
                / f"player{i}-full.json"
            ).read_text()
        )
        for i in (1, 2)
    ]
    checked = {r["unit"]: r for r in old}
    ledger = json.loads((HERE / "result.json").read_text())["streams"]
    rows = {}
    start = time.process_time()
    for cap, costs in ledger.items():
        wire = (folder / f"joint-{cap}.psw").read_bytes()
        assert (
            hashlib.sha256(wire).hexdigest() == costs["wire_sha256"] and len(wire) == costs["bytes"]
        )
        canvas, signals, tracks = read_pose(wire)
        assert canvas == [1920, 1080] and len(signals) == costs["pose_bytes"] == 1623
        ns = []
        for _ in range(11):
            t = time.perf_counter_ns()
            canvas, signals, tracks = read_pose(wire)
            ns.append(time.perf_counter_ns() - t)
        units = {}
        for track in tracks:
            n = track["name"]
            kp = track["kp"]
            assert np.array_equal(track["t"], np.arange(240)) and kp.shape == (240, 17, 3)
            assert set(np.unique(kp[:, :, 2])).issubset({0.0, 1.0})
            if cap == "20":
                boxes = np.asarray(k6b.run_boxes(list(kp)))
                run = {"kp": kp, "boxes": boxes, "t": track["t"]}
                expected = checked[n]
                rh = hashlib.sha256(f"{np.dtype(np.uint8).str}:{(240, 784, 512, 3)}:".encode())
                assert array_hash(boxes) == expected["box_sha256"]
                prefix_start = time.process_time()
                for tick in range(240):
                    raster = player.pose_for_tick(run, tick)
                    assert raster.shape == (784, 512, 3) and raster.dtype == np.uint8
                    rh.update(np.ascontiguousarray(raster))
                    if tick == 11:
                        assert (time.process_time() - prefix_start) * 40 < 60
                assert rh.hexdigest() == expected["pose_sha256"]
            units[n] = {
                "coordinates_and_visibility_sha256": array_hash(kp),
                "visible_joint_ticks": int(kp[:, :, 2].sum()),
                "renderer_pose_sha256": checked[n]["pose_sha256"],
            }
        rows[cap] = {
            "wire_sha256": costs["wire_sha256"],
            "pose_section_sha256": hashlib.sha256(signals).hexdigest(),
            "pose_bytes": len(signals),
            "running_kbps": costs["kbps"],
            "cold_kbps": costs["cold_kbps"],
            "units": units,
            "parse_ns": ns,
            "median_ns": statistics.median(ns),
        }
    assert (
        rows["20"]["units"] == rows["33"]["units"]
        and rows["20"]["pose_section_sha256"] == rows["33"]["pose_section_sha256"]
    )
    assert time.process_time() - start < 60
    return {
        "streams": rows,
        "image_size": canvas,
        "shape_per_person": [240, 17, 3],
        "ticks": 240,
        "model_inference": False,
        "full_raster_hashes_verified": True,
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = measure(args.folder)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v["median_ns"] for k, v in result["streams"].items()})
