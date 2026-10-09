"""Clips and labels for PLAN step G2 (background evaluation protocol).

    python tools/datasets/g2_inputs.py --datasets ROOT --g1-clips CLIPS.JSON \\
        --g1-job ID=DIR ... --out TAR

Packs one tar for every G2 job:

* ``clips.json``: G1's 8 holding clips with the protocol's warm-up W and spans
  (docs/experiments.md, background evaluation protocol), G1's clip records
  (source, frame range, analysis rate) and where each clip's masks come from.
* ``masks/<clip>/masks.rle``: G1's SAM 3.1 masks, taken from the G1 job's
  ``published.tar`` and checked against the sha256 the job recorded.
* ``labels/<clip>/ball.json``: labelled ball positions by source frame index
  (OpenTTGames ``ball_markup.json``; TrackNet ``Label.csv``, visibility >= 1).
* ``ott_masks/<clip>/<index>.png``: OpenTTGames' own segmentation masks
  (320x128) at G1's analysis frames, to check SAM's players against.

Run with the PointStream environment's interpreter, in local scratch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any

#: The protocol's warm-up W (G1's 99% warm-up), seconds.
WARMUP_S = {
    "ott/test_1": 4.5, "ott/test_2": 9.0, "ott/test_4": 5.0, "ott/test_5": 1.0, "ott/test_6": 2.0, "ott/test_7": 2.0,
    "tracknet/game8/Clip3": 1.0, "tracknet/game10/Clip1": 0.5,
}
SPAN_S = 10.0
LEAD_S = 2.0
PILOT = ["ott/test_5", "tracknet/game10/Clip1"]
SEED = "pointstream-g2"


def safe(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o444
    tar.addfile(info, io.BytesIO(data))


def analysis_indices(clip: dict[str, Any]) -> list[int]:
    """G1's analysis frames (``experiments.background.g1.plan`` with no limit)."""
    fps, rate = float(clip["fps"]), float(clip["analysis"]["fps"])
    first, last = int(clip["first_index"]), int(clip["last_index"])
    out, k = [], 0
    while (at := first + int(round(k * fps / rate))) <= last:
        out.append(at)
        k += 1
    return out


def spans(clip: dict[str, Any], warmup: float) -> dict[str, Any]:
    fps = float(clip["fps"])
    duration = (int(clip["last_index"]) - int(clip["first_index"]) + 1) / fps
    first = int(clip["first_index"])
    end_e = min(warmup + SPAN_S, duration)
    out: dict[str, Any] = {
        "warmup_s": warmup, "duration_s": round(duration, 4),
        "E": [warmup, end_e], "lead_s": max(0.0, warmup - LEAD_S),
    }
    if warmup + 2 * SPAN_S <= duration:
        out["E2"] = [warmup + SPAN_S, warmup + 2 * SPAN_S]
    # Source frame ranges [first, last] (inclusive) for each span.
    out["frames"] = {name: [first + int(round(out[name][0] * fps)), first + int(round(out[name][1] * fps)) - 1]
                     for name in ("E", "E2") if name in out}
    out["frames"]["lead"] = first + int(round(out["lead_s"] * fps))
    return out


def ball_ott(archive: zipfile.ZipFile) -> dict[str, list[float]]:
    markup = json.loads(archive.read("ball_markup.json"))
    return {str(int(k)): [float(v["x"]), float(v["y"])] for k, v in markup.items()
            if float(v["x"]) >= 0 and float(v["y"]) >= 0}


def ball_tracknet(archive: zipfile.ZipFile, member_dir: str) -> dict[str, list[float]]:
    text = archive.read(f"{member_dir}/Label.csv").decode("utf-8")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        if int(row["visibility"]) >= 1 and row["x-coordinate"] and row["y-coordinate"]:
            index = int(Path(row["file name"]).stem)
            out[str(index)] = [float(row["x-coordinate"]), float(row["y-coordinate"])]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--g1-clips", required=True)
    parser.add_argument("--g1-job", action="append", required=True, help="ID=DIR of a G1 job (its full/ stage)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.datasets)
    g1_clips = {c["id"]: c for c in json.loads(Path(args.g1_clips).read_text())["clips"]}
    recorded: dict[str, tuple[str, str, Path]] = {}
    for pair in args.g1_job:
        job, directory = pair.split("=", 1)
        stage = Path(directory)
        for clip_id, record in json.loads((stage / "g1.json").read_text())["sam"]["clips"].items():
            recorded[clip_id] = (job, record["masks_rle_sha256"], stage / "published.tar")
    clips = []
    with tarfile.open(args.out, "x") as out:
        for clip_id, warmup in WARMUP_S.items():
            clip = g1_clips[clip_id]
            job, expected, published = recorded[clip_id]
            with tarfile.open(published) as tar:
                handle = tar.extractfile(f"publish/clips/{safe(clip_id)}/masks.rle")
                if handle is None:
                    raise SystemExit(f"{clip_id}: no masks in {published}")
                data = handle.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f"{clip_id}: masks sha256 differs from job {job}'s record")
            add_bytes(out, f"masks/{safe(clip_id)}/masks.rle", data)
            indices = analysis_indices(clip)
            ott_masks = 0
            if clip["dataset"] == "openttgames":
                with zipfile.ZipFile(root / "OpenTTGames" / "raw" / f"{clip['video']}.zip") as archive:
                    ball = ball_ott(archive)
                    names = {int(m.group(1)): n for n in archive.namelist()
                             if (m := re.fullmatch(r"segmentation_masks/(\d+)\.png", n))}
                    for index in indices:
                        if index in names:
                            add_bytes(out, f"ott_masks/{safe(clip_id)}/{index}.png", archive.read(names[index]))
                            ott_masks += 1
            else:
                with zipfile.ZipFile(root / "TrackNet" / "Dataset.zip") as archive:
                    ball = ball_tracknet(archive, clip["source"]["member_dir"])
            add_bytes(out, f"labels/{safe(clip_id)}/ball.json", json.dumps(ball, sort_keys=True).encode())
            clips.append({**clip, "g2": spans(clip, warmup), "analysis_frames": len(indices),
                          "masks": {**clip["masks"], "g1_job": job, "masks_rle_sha256": expected},
                          "ball_labels": len(ball), "ott_masks": ott_masks})
        doc = {
            "name": "g2-clips-v1", "seed": SEED, "pilot": PILOT,
            "rule": "G1's clips that hold (docs/experiments.md, G1 outcome); warm-up W = G1's 99% warm-up; "
                    f"E = [W, W + {SPAN_S:g} s]; E2 = [W + {SPAN_S:g}, W + {2 * SPAN_S:g} s] where the clip is long "
                    f"enough; codecs start {LEAD_S:g} s before W (or at the clip start); pilot: the smallest "
                    f"sha256('{SEED}:<clip>') per dataset",
            "clips": clips,
        }
        add_bytes(out, "clips.json", json.dumps(doc, indent=1, sort_keys=True).encode())
    for clip in clips:
        print(clip["id"], clip["g2"]["frames"], clip["ball_labels"], clip["ott_masks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
