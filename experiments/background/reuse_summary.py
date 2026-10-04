"""Check frame metrics and accounting in a completed registered control report."""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np


def check_quality(quality, expected_frames):
    mse = np.asarray(quality["mse_per_frame"], dtype=np.float64)
    if len(mse) != expected_frames or not np.isfinite(mse).all() or (mse < 0).any():
        raise ValueError("invalid all-frame quality denominator")
    if not math.isclose(float(mse.mean()), quality["pooled_mse"], abs_tol=1e-10):
        raise ValueError("pooled MSE arithmetic failed")
    perfect = int((mse == 0).sum())
    if perfect != quality["perfect_frames"]:
        raise ValueError("perfect-frame denominator failed")
    if not perfect:
        mean = float((10 * np.log10(255**2 / mse)).mean())
        if not math.isclose(mean, quality["mean_frame_psnr_db"], abs_tol=1e-10):
            raise ValueError("mean frame PSNR arithmetic failed")
    elif quality["mean_frame_psnr_db"] is not None:
        raise ValueError("infinite mean-frame PSNR must be null")
    if mse.mean():
        pooled = 10 * math.log10(255**2 / float(mse.mean()))
        if not math.isclose(pooled, quality["pooled_psnr_db"], abs_tol=1e-10):
            raise ValueError("pooled PSNR arithmetic failed")


def summarize(report_path):
    report = json.loads(Path(report_path).read_text())
    if report.get("status") != "complete":
        raise ValueError("completed report required")
    rows = []
    horizons = []
    identities = []
    for source in report["sources"]:
        n = source["frame_count"]
        duration = source["observed_seconds"]
        if len(source["frame_sha256"]) != n or len(source["selected_pts_seconds"]) != n:
            raise ValueError("frame identity denominator failed")
        identities.append(
            {
                k: source[k]
                for k in [
                    "source_id",
                    "source",
                    "scene_metadata",
                    "raw",
                    "frame_count",
                    "fps",
                    "observed_seconds",
                    "requested_seek_seconds",
                    "selected_source_pts_first_seconds",
                    "selected_source_pts_last_seconds",
                    "registered_cut_frames",
                    "joined_cut_metadata",
                ]
            }
        )
        anchors = {row["crf"]: row for row in source["arms"] if row["name"] == "continuous"}
        for arm in source["arms"]:
            check_quality(arm["quality"], n)
            native = arm["bytes"] - arm["manifest"]["bytes"]
            if native <= 0 or ("stream_bytes" in arm and native != arm["stream_bytes"]):
                raise ValueError("physical manifest/native ledger failed")
            anchor = anchors[arm["crf"]]
            rows.append(
                {
                    "source": source["source_id"],
                    "crf": arm["crf"],
                    "arm": arm["name"],
                    "frames": n,
                    "observed_seconds": duration,
                    "bytes": arm["bytes"],
                    "native_stream_bytes": native,
                    "manifest_bytes": arm["manifest"]["bytes"],
                    "packet_count": arm["packet_count"],
                    "bits_per_second": arm["bytes"] * 8 / duration,
                    "mean_frame_y_psnr_db": arm["quality"]["mean_frame_psnr_db"],
                    "pooled_y_psnr_db": arm["quality"]["pooled_psnr_db"],
                    "pooled_mse": arm["quality"]["pooled_mse"],
                    "rate_ratio_to_full_continuous_same_crf": arm["bytes"] / anchor["bytes"],
                    "receiver_receipt": arm["receiver_receipt"],
                    "manifest": arm["manifest"],
                }
            )
            arm_horizons = arm.get("horizons", [])
            registered = report["registration"]["horizons_seconds"]
            if arm_horizons and len(arm_horizons) != len(registered):
                raise ValueError("registered horizon coverage failed")
            for requested, horizon in zip(registered, arm_horizons):
                fps = Fraction(source["fps"])
                expected = min(n, round(requested * fps))
                check_quality(horizon["quality"], expected)
                if not math.isclose(horizon["seconds"], float(expected / fps), abs_tol=1e-10):
                    raise ValueError("registered horizon duration failed")
                if not math.isclose(
                    horizon["bits_per_second"],
                    horizon["bytes"] * 8 / horizon["seconds"],
                    abs_tol=1e-10,
                ):
                    raise ValueError("observed horizon rate failed")
                horizons.append(
                    {
                        "source": source["source_id"],
                        "crf": arm["crf"],
                        "arm": arm["name"],
                        "observed_seconds": horizon["seconds"],
                        "bytes": horizon["bytes"],
                        "bits_per_second": horizon["bits_per_second"],
                        "mean_frame_y_psnr_db": horizon["quality"]["mean_frame_psnr_db"],
                        "pooled_y_psnr_db": horizon["quality"]["pooled_psnr_db"],
                        "manifest": horizon["manifest"],
                    }
                )
    return {
        "report_sha256": hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
        "code_revision": report["code_revision"],
        "worker_sha256": report["worker"]["sha256"],
        "smoke": report["smoke"],
        "registration": report["registration"],
        "identities": identities,
        "rows": rows,
        "hold_only_horizon_ledgers": horizons,
        "scope": "Fixed-image-region native control; horizon ledgers have no matched shorter native anchors. All rate ratios are same-CRF full-interval descriptions, not matched-quality BD-rates or complete semantic codec gains.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
