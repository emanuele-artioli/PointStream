"""Replay three persisted guided-development semantic transport packages.

No model inference or new encoding. Native video comparison is withheld.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
from experiments.tier.receiver_replay import digest, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    os.nice(19)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    if os.getloadavg()[0] > 32:
        raise RuntimeError("host load above policy")
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.data_root / "outputs/sam31-unification/pilot-v1-20260927-run-05"
    audit_path = base / "audit.json"
    audit = json.loads(audit_path.read_text())
    scenes = audit["scenes"]
    expected = {
        "alcaraz_highlights_scene_000",
        "alcaraz_highlights_scene_010",
        "alcaraz_perricard_scene_007",
    }
    if len(scenes) != 3 or {s["source_id"] for s in scenes} != expected:
        raise ValueError("three named pilot windows required")
    report = {
        "code_revision": os.environ["PS_CODE_REVISION"],
        "worker_sha256": digest(__file__),
        "audit": str(audit_path),
        "audit_sha256": digest(audit_path),
        "smoke": args.smoke,
        "rows": [],
    }
    for s in scenes[:1] if args.smoke else scenes:
        if os.getloadavg()[0] > 32:
            raise RuntimeError("host load above policy")
        out = args.out / s["source_id"]
        out.mkdir()
        original = Path(s["client_roundtrip"]["payload_path"])
        package = out / "package.npz"
        shutil.copyfile(original, package)
        if digest(package) != s["client_roundtrip"]["payload_sha256"]:
            raise ValueError("saved package identity mismatch")
        if (
            package.stat().st_size
            != s["client_roundtrip"]["pointstream_runner_size_ledger"]["transport_total"]
        ):
            raise ValueError("physical ledger mismatch")
        decoded_path = out / "decoded.npy"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            "-m",
            "experiments.tier.receiver_replay",
            "--data-root",
            str(args.data_root),
            "--out",
            str(args.out),
            "--child-package",
            str(package),
            "--child-output",
            str(decoded_path),
        ]
        with tempfile.TemporaryDirectory(prefix="ps-pilot-hidden-") as cwd:
            child = subprocess.run(
                command, cwd=cwd, env=env, capture_output=True, text=True, timeout=300
            )
        if child.returncode:
            raise RuntimeError(child.stderr)
        receiver = json.loads(child.stdout.strip().splitlines()[-1])
        decoded = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
        scene = s["scene"]
        paths = scene["frame_paths"]
        if len(paths) != 16 or decoded.shape != (16, 2160, 3840, 3):
            raise ValueError("incomplete 16-frame 4K output/source")
        source_hashes = []
        mse = []
        for i, path in enumerate(paths):
            if digest(path) != scene["frame_file_sha256"][i]:
                raise ValueError("source PNG identity differs")
            frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ph = hashlib.sha256(frame.tobytes()).hexdigest()
            source_hashes.append(ph)
            if ph != s["frame_sha256"][i]:
                raise ValueError("source RGB identity differs")
            mse.append(psnr(frame[None], np.asarray(decoded[i : i + 1]))["frame_mse_y"][0])
        row = {
            "source_id": s["source_id"],
            "frame_ids": s["frame_ids"],
            "source_paths": paths,
            "source_pixel_hashes": source_hashes,
            "package": str(original),
            "package_sha256": digest(package),
            "complete_file_bytes": package.stat().st_size,
            "saved_ledger": s["client_roundtrip"]["pointstream_runner_size_ledger"],
            "receiver": receiver,
            "command": command,
            "parity_saved_decoded": receiver["pixels_sha256"]
            == s["client_roundtrip"]["decoded_frames_sha256"],
            "quality": {
                "frames": 16,
                "frame_mse_y": mse,
                "mean_frame_y_psnr_db": float(np.mean(10 * np.log10(255**2 / np.array(mse)))),
                "pooled_y_psnr_db": float(10 * np.log10(255**2 / np.mean(mse))),
            },
            "classification": "guided development semantic transport; absent background; no matched native anchor",
        }
        report["rows"].append(row)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
