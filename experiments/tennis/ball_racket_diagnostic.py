"""Evaluate labeled rackets on independently replayed frozen ball-study streams."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import ball_confirmation as prior
from experiments.tennis import ball_confirmation_score as scorer
from experiments.tennis.ball_starved_scaled import render_scaled
from experiments.tennis.k5d_encoder import crop64
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch


def read_member(tar, member):
    with tarfile.open(tar) as archive:
        f = archive.extractfile(member)
        if f is None:
            raise ValueError(f"missing {member}")
        return f.read()


def run(args):
    import torch

    start = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "racket-replay"
    work.mkdir()
    publication = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publication.mkdir()
    receipt = json.loads(read_member(args.packets, "publish/result.json"))
    source_receipt = json.loads(read_member(args.source, "publish/receipt.json"))
    if receipt["source"] != args.match + "_000" or source_receipt["source_id"] != receipt["source"]:
        raise ValueError("wrong source identity")
    rgb_path = work / "source.rgb"
    rgb_path.write_bytes(read_member(args.source, "publish/source.rgb"))
    if prior.sha(rgb_path) != source_receipt["source"]["rgb_sha256"]:
        raise ValueError("source checksum")
    n, fps = receipt["frames"], receipt["fps"]
    source = np.memmap(rgb_path, np.uint8, "r", shape=(n, 720, 1280, 3))
    labels_manifest = json.loads(args.rackets.read_text())
    annotations = labels_manifest["sources"][args.match]
    if annotations["frames"] != n or not annotations["rows"]:
        raise ValueError("no aligned racket labels")
    rows = annotations["rows"]
    if os.environ["PS_STAGE"] == "smoke":
        rows = [r for r in rows if r["objects"]][:2]
    if not rows:
        raise ValueError("no labeled racket objects")
    net = quality.load_dists(args.weights, "cpu")
    decoder = svtav1.tool("dav1d")
    if decoder["sha256"] != receipt["decoder"]["sha256"]:
        raise ValueError("decoder identity changed")
    ball_manifest, _ = scorer.authenticated_manifest(args.balls)
    ball_labels = scorer.labels_for_source(ball_manifest, receipt["source"])
    ball_tick = next(i for i, label in enumerate(ball_labels) if label[0] == 1)
    output = []
    setup = time.monotonic() - start
    for entry in receipt["results"]:
        data = read_member(args.packets, "publish/" + entry["file"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("packet hash mismatch")
        folder = work / entry["mode"]
        folder.mkdir()
        if entry["mode"] == "tracked":
            decoded = render_scaled(data, folder, decoder["path"])
            if hashlib.sha256(decoded.tobytes()).hexdigest() != entry["rendered_sha256"]:
                raise ValueError("tracked reconstruction checksum mismatch")
        else:
            raw = folder / "decoded.yuv"
            meta = {"width":320,"height":180,"fps":[fps,1],"frames":n}
            decoded_info = tracked_patch.decode_native(data, raw, meta, decoder["path"])
            if decoded_info["decoded_yuv_sha256"] != entry["decoded_sha256"]:
                raise ValueError("baseline reconstruction checksum mismatch")
            yuv = np.memmap(raw, np.uint8, "r", shape=(n,270,320))
            decoded = np.stack([cv2.resize(tracked_patch.yuv_to_rgb(f,320,180,False),(1280,720),interpolation=cv2.INTER_CUBIC) for f in yuv])
        _, bx, by = ball_labels[ball_tick]
        ball = dists_crops(net,[crop64(decoded[ball_tick],bx,by)],crop64(source[ball_tick],bx,by),np.ones((64,64),bool),"cpu")[0]
        previous = next(x["dists"] for x in entry["crop"] if x["tick"] == ball_tick)
        if abs(ball-previous) > 1e-5:
            raise ValueError("ball control failed")
        scores = []
        for row in rows:
            tick = row["tick"]
            for obj in row["objects"]:
                x,y,w,h = obj["bbox_xywh"]
                x0,y0 = max(0,math.floor(x*2/3)),max(0,math.floor(y*2/3))
                x1,y1 = min(1280,math.ceil((x+w)*2/3)),min(720,math.ceil((y+h)*2/3))
                if x1 <= x0 or y1 <= y0:
                    continue
                truth = cv2.resize(source[tick,y0:y1,x0:x1],(64,64),interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(decoded[tick,y0:y1,x0:x1],(64,64),interpolation=cv2.INTER_LINEAR)
                value = dists_crops(net,[image],truth,np.ones((64,64),bool),"cpu")[0]
                if not np.isfinite(value):
                    raise ValueError("nonfinite racket DISTS")
                scores.append({"tick":tick,"box":[x0,y0,x1,y1],"dists":float(value)})
        output.append({"mode":entry["mode"],"kbps":entry["kbps"],"packet_sha256":entry["sha256"],"reproduced_ball_tick":ball_tick,"ball_dists":ball,"original_ball_dists":previous,"rackets":scores,"mean_dists":float(np.mean([s["dists"] for s in scores]))})
    report = {"source":args.match,"phase":os.environ["PS_STAGE"],"frames":n,"fps":fps,"racket_manifest_sha256":prior.sha(args.rackets),"labeled_ticks":len(rows),"objects":sum(len(row["objects"]) for row in rows),"setup_seconds":setup,"wall_seconds":time.monotonic()-start,"results":output}
    (publication/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    (Path(os.environ["PS_STAGE_DIR"])/"result.json").write_text(json.dumps(report,indent=2)+"\n")


def validate(args):
    result = json.loads((Path(os.environ["PS_STAGE_DIR"])/"result.json").read_text())
    checks = {"source":result["source"]==args.match,"streams":len(result["results"])==3,"positive":result["objects"]>0,"same_support":len({len(r["rackets"]) for r in result["results"]})==1,"ball_control":all(abs(r["ball_dists"]-r["original_ball_dists"])<1e-5 for r in result["results"])}
    if result["phase"]=="smoke":
        checks["forecast"] = result["wall_seconds"] * args.forecast_factor * 1.5 < args.full_limit
    report = {"passed":all(checks.values()),"checks":checks,"smoke_seconds":result["wall_seconds"]}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","packets","balls","rackets","weights"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--match",required=True)
    parser.add_argument("--frames",type=int)
    parser.add_argument("--forecast-factor",type=float,required=True)
    parser.add_argument("--full-limit",type=float,required=True)
    parser.add_argument("--validate",action="store_true")
    args=parser.parse_args()
    validate(args) if args.validate else run(args)


if __name__ == "__main__":
    main()
