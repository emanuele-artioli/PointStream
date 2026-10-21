"""K4b.3: keypoint rate against keypoint error, on K4b.2's x1 crop keypoints (CPU).

Each player's fitted frames form tracks (a gap starts a new one). Every code
quantizes at q px, predicts each sent frame from the decoded ones (previous or
constant velocity, closed loop), sends every k-th frame and interpolates the rest.
Rate: pooled empirical entropy of residual and visibility symbols plus the
absolute starts; error: displacement and OKS against the uncoded keypoints.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4
from experiments.tennis.k4b_pose import SIGMAS
from experiments.visor.b1 import file_sha256, write_json
from src.segmentation.masks import decompress

STEPS_PX = (1, 2, 4, 8)
PREDICTORS = ("previous", "velocity")
EVERY = (1, 2, 4)
OKS_BAR = .95


def tracks(rows: list[dict[str, Any]], setting: str = "crop1") -> list[dict[str, Any]]:
    """Runs of consecutive fitted frames per player: keypoints (n, 17, 3), area and box height per frame."""
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["track"], r["t"])):
        if r["settings"][setting]["fitted"]:
            by_player[r["track"]].append(r)
    runs: list[tuple[str, list[dict[str, Any]]]] = []
    for player, seq in sorted(by_player.items()):
        for r in seq:
            if runs and runs[-1][0] == player and r["t"] == runs[-1][1][-1]["t"] + 1:
                runs[-1][1].append(r)
            else:
                runs.append((player, [r]))
    return [{"player": player, "role": run[0]["role"], "t": [x["t"] for x in run],
             "kps": np.asarray([x["settings"][setting]["kps"] for x in run], np.float64),
             "area": np.asarray([x["area"] for x in run], np.float64),
             "height": np.asarray([x["height"] * 1080 for x in run], np.float64)} for player, run in runs]


def code(kps: np.ndarray, q: float, predictor: str, every: int) -> tuple[np.ndarray, list[int], list[int], float]:
    """Decode one track: (decoded (n, 17, 2), residual symbols, visibility symbols, absolute-start bits)."""
    n = len(kps)
    sent = list(range(0, n, every))
    decoded_sent: list[np.ndarray] = []
    residuals: list[int] = []
    visibility: list[int] = []
    start_bits = 17 * (math.log2(1920 / q) + math.log2(1080 / q))
    for j in sent:
        value = kps[j, :, :2]
        visibility.extend((kps[j, :, 2] >= k4.KP_CONF).astype(int).tolist())
        if not decoded_sent:
            decoded_sent.append(np.round(value / q) * q)
            continue
        if predictor == "velocity" and len(decoded_sent) >= 2:
            prediction = 2 * decoded_sent[-1] - decoded_sent[-2]
        else:
            prediction = decoded_sent[-1]
        r = np.round((value - prediction) / q)
        residuals.extend(r.astype(int).ravel().tolist())
        decoded_sent.append(prediction + r * q)
    decoded = np.empty((n, 17, 2))
    for a, (i, j) in enumerate(zip(sent, sent[1:])):
        for t in range(i, j):
            w = (t - i) / (j - i)
            decoded[t] = (1 - w) * decoded_sent[a] + w * decoded_sent[a + 1]
    decoded[sent[-1]:] = decoded_sent[-1]
    return decoded, residuals, visibility, start_bits


def entropy_bits(counter: Counter[int]) -> float:
    total = sum(counter.values())
    return -sum(c * math.log2(c / total) for c in counter.values()) if total else 0.


def evaluate(windows: list[dict[str, Any]], q: float, predictor: str, every: int) -> dict[str, Any]:
    """One code over every window's tracks: pooled rate and the error by role."""
    residuals: Counter[int] = Counter()
    visibility: Counter[int] = Counter()
    starts = 0.
    player_frames = frames = 0
    seconds = 0.
    zero = 0
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for w in windows:
        frames += w["frames"]
        seconds += w["frames"] / w["fps"]
        for tr in w["tracks"]:
            decoded, res, vis, start = code(tr["kps"], q, predictor, every)
            residuals.update(res)
            zero += sum(r == 0 for r in res)
            visibility.update(vis)
            starts += start
            player_frames += len(tr["kps"])
            confident = tr["kps"][:, :, 2] >= k4.KP_CONF
            d = np.linalg.norm(decoded - tr["kps"][:, :, :2], axis=2)
            for t in range(len(d)):
                if not confident[t].any():
                    continue
                e = np.exp(-d[t] ** 2 / ((2 * SIGMAS) ** 2) / (tr["area"][t] + np.spacing(1)) / 2)[confident[t]].mean()
                for group in (tr["role"], "all"):
                    errors[group]["oks"].append(float(e))
                    errors[group]["px"].extend(d[t][confident[t]].tolist())
                    errors[group]["height_share"].extend((d[t][confident[t]] / tr["height"][t]).tolist())
    bits = entropy_bits(residuals) + entropy_bits(visibility) + starts
    players_per_frame = player_frames / frames
    summary = {group: {"frames": len(v["oks"]), "oks_median": float(np.median(v["oks"])), "oks_p05": float(np.percentile(v["oks"], 5)),
                       "px_median": float(np.median(v["px"])), "px_p95": float(np.percentile(v["px"], 95)),
                       "height_share_median": float(np.median(v["height_share"]))} for group, v in errors.items()}
    return {"q_px": q, "predictor": predictor, "every": every, "bits": bits,
            "bits_per_player_frame": bits / player_frames, "kbps_two_players": bits / seconds / 1000 * 2 / players_per_frame,
            "residual_bits_per_value": entropy_bits(residuals) / max(sum(residuals.values()), 1),
            "zero_residual_share": zero / max(sum(residuals.values()), 1), "start_bits": starts,
            "players_per_frame": players_per_frame, "error": summary}


def load(jobs: list[Path], inputs_index: Path) -> list[dict[str, Any]]:
    """Windows of locally validated K4b.2 full stages, with their frame rates from K4b.2's inputs."""
    index = json.loads(inputs_index.read_text())["windows"]
    out = []
    for root in jobs:
        validation = json.loads((root / "local-validation.json").read_text())
        summary = json.loads((root / "k4b-pose.json").read_text())
        if validation.get("passed") is not True or not all(validation["checks"].values()) or summary["stage"] != "full":
            raise ValueError(f"{root}: only locally validated full K4b.2 stages")
        if summary["inputs_index_sha256"] != file_sha256(inputs_index):
            raise ValueError("K4b.2 ran on a different inputs index")
        for w in summary["windows"]:
            path = root / "publish" / k4.safe(w["id"]) / "pose-crops.json.zst"
            if file_sha256(path) != w["files"]["pose-crops.json.zst"]:
                raise ValueError(f"Changed K4b.2 artifact: {path}")
            rows = json.loads(decompress(path.read_bytes()))["rows"]
            out.append({"id": w["id"], "frames": w["frames"], "fps": float(index[w["id"]]["window"]["fps"]), "tracks": tracks(rows)})
    if len({w["id"] for w in out}) != len(out):
        raise ValueError("Duplicate window")
    return out


def decide(codes: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [c for c in codes if all(c["error"].get(g, {}).get("oks_median", 0) >= OKS_BAR for g in ("far", "near"))]
    chosen = min(passing, key=lambda c: c["kbps_two_players"]) if passing else None
    return {"rule": f"lowest-rate code with median OKS >= {OKS_BAR} against the uncoded keypoints for far and near",
            "chosen": {k: chosen[k] for k in ("q_px", "predictor", "every", "kbps_two_players")} if chosen else None,
            "passing": len(passing), "reference_svtav1_player_kbps": {"crf62": 41.9, "crf48": 155.0}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", nargs="+", type=Path, help="locally validated K4b.2 job directories")
    p.add_argument("--inputs-index", type=Path, required=True, help="K4b.2's inputs.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    windows = load(args.jobs, args.inputs_index)
    if len(windows) != 48:
        raise ValueError(f"{len(windows)} of 48 windows")
    codes = [evaluate(windows, q, pred, every) for q, pred, every in product(STEPS_PX, PREDICTORS, EVERY)]
    result = {"codes": codes, "decision": decide(codes), "windows": len(windows),
              "inputs": {str(j): file_sha256(j / "k4b-pose.json") for j in args.jobs} | {str(args.inputs_index): file_sha256(args.inputs_index)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    for c in sorted(codes, key=lambda c: c["kbps_two_players"]):
        print(f"q={c['q_px']} {c['predictor']:8s} every={c['every']} kbps={c['kbps_two_players']:6.2f} "
              f"zero={c['zero_residual_share']:.2f} oks far/near={c['error']['far']['oks_median']:.3f}/{c['error']['near']['oks_median']:.3f} "
              f"px med/p95={c['error']['all']['px_median']:.2f}/{c['error']['all']['px_p95']:.2f}")
    print(json.dumps(result["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
