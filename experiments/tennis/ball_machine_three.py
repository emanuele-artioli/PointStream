"""Frozen WASB judge on decoded three-match ball packets and source."""
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
from experiments.tennis import k5b_trackers as trackers
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis.ball_starved_scaled import render_scaled, unpack_scaled
from src.codecs import svtav1
from src.pointstream import tracked_patch

WEIGHT = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"


def member(path, name):
    with tarfile.open(path) as archive:
        data = archive.extractfile(name)
        if data is None:
            raise ValueError(f"missing {name}")
        return data.read()


def run(args):
    import torch

    start = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("GPU or CPU isolation")
    torch.set_num_threads(1)
    oracle.bind_config(args.wasb_code, CONFIG)
    if oracle.sha(args.wasb_weights / "wasb_tennis_best.pth.tar") != WEIGHT:
        raise ValueError("checkpoint identity")
    decoder = svtav1.tool("dav1d")
    receipt = json.loads(member(args.packets, "publish/result.json"))
    src_receipt = json.loads(member(args.source, "publish/receipt.json"))
    if receipt["source"] != args.match + "_000" or src_receipt["source_id"] != receipt["source"]:
        raise ValueError("source identity")
    if decoder["sha256"] != receipt["decoder"]["sha256"]:
        raise ValueError("decoder identity")
    count, fps = receipt["frames"], receipt["fps"]
    if args.frames > count:
        raise ValueError("too many frames")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    work = root / "machine-test"
    work.mkdir()
    source_path = work / "source.rgb"
    source_path.write_bytes(member(args.source, "publish/source.rgb"))
    if prior.sha(source_path) != src_receipt["source"]["rgb_sha256"]:
        raise ValueError("source pixels")
    source = np.memmap(source_path, np.uint8, "r", shape=(count,720,1280,3))
    manifest, raw_manifest = scorer.authenticated_manifest(args.labels)
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    if manifest_hash != "df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1":
        raise ValueError("label identity")
    item, = [v for v in manifest["sources"] if v["id"] == receipt["source"]]
    labels = [r for r in item["official_window_rows"] if r["native_frame_index"] < args.frames]
    model_start = time.monotonic()
    zoo = trackers.Zoo(args.wasb_code,args.wasb_weights,names=("wasb-step1",))
    settings = zoo.settings["wasb-step1"]
    if next(settings["net"].parameters()).device.type != "cuda":
        raise ValueError("WASB not on CUDA")
    model_seconds = time.monotonic() - model_start
    results = {}
    streams = [("source",None)] + [(x["mode"],x) for x in receipt["results"]]
    for name, entry in streams:
        positions = None
        if entry is None:
            images = source
            frame_hash = prior.sha(source_path)
            packet_hash = None
            rate = None
        else:
            data = member(args.packets,"publish/"+entry["file"])
            packet_hash = hashlib.sha256(data).hexdigest()
            if packet_hash != entry["sha256"]:
                raise ValueError("packet identity")
            folder = work / name
            folder.mkdir()
            if name == "tracked":
                meta, _, _ = unpack_scaled(data)
                positions = meta["positions"]
                images = render_scaled(data,folder,decoder["path"])
                frame_hash = hashlib.sha256(images.tobytes()).hexdigest()
                if frame_hash != entry["rendered_sha256"]:
                    raise ValueError("tracked reconstruction")
            else:
                raw = folder / "decoded.yuv"
                meta = {"width":320,"height":180,"fps":[fps,1],"frames":count}
                decoded = tracked_patch.decode_native(data,raw,meta,decoder["path"])
                if decoded["decoded_yuv_sha256"] != entry["decoded_sha256"]:
                    raise ValueError("baseline reconstruction")
                yuv = np.memmap(raw,np.uint8,"r",shape=(count,270,320))
                images = np.stack([cv2.resize(tracked_patch.yuv_to_rgb(f,320,180,False),(1280,720),interpolation=cv2.INTER_CUBIC) for f in yuv])
                frame_hash = hashlib.sha256(images.tobytes()).hexdigest()
            rate = entry["kbps"]
        infer_start = time.monotonic()
        predictions = zoo.run("wasb-step1",images[:args.frames])
        infer_seconds = time.monotonic()-infer_start
        if len(predictions) != args.frames:
            raise ValueError("WASB output cadence")
        rows = []
        for label in labels:
            tick = label["native_frame_index"]
            visible, x, y = predictions[tick]
            error = math.hypot(1.5*x-label["native_X"],1.5*y-label["native_Y"]) if visible and label["visibility"]==1 else None
            row = {"tick":tick,"visibility":label["visibility"],"detected":bool(visible),"native_error":error,"hit_10px":error is not None and error<=10}
            if positions is not None:
                pos = positions[tick]
                direct_error = math.hypot(1.5*(pos[0]+16)-label["native_X"],1.5*(pos[1]+16)-label["native_Y"]) if pos is not None and label["visibility"]==1 else None
                row.update({"direct_present":pos is not None,"direct_error":direct_error,"direct_hit_10px":direct_error is not None and direct_error<=10})
            rows.append(row)
        results[name] = {"kbps":rate,"packet_sha256":packet_hash,"rgb_sha256":frame_hash,"seconds":infer_seconds,"rows":rows}
    report = {"match":args.match,"frames":args.frames,"source_frames":count,"fps":fps,"phase":os.environ["PS_STAGE"],"source_rgb_sha256":prior.sha(source_path),"manifest_sha256":manifest_hash,"checkpoint_sha256":WEIGHT,"gpu":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0)),"settings":{"frames_in":settings["cfg"]["model"]["frames_in"],"frames_out":settings["cfg"]["model"]["frames_out"],"step":settings["cfg"]["detector"]["step"],"threshold":settings["cfg"]["detector"]["postprocessor"]["score_threshold"]},"model_seconds":model_seconds,"elapsed_seconds":time.monotonic()-start,"results":results}
    publish = root / "publish"
    publish.mkdir()
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    (publish/"result.json").write_text(payload)
    (Path(os.environ["PS_STAGE_DIR"])/"result.json").write_text(payload)


def validate(args):
    data = json.loads((Path(os.environ["PS_STAGE_DIR"])/"result.json").read_text())
    results = data["results"]
    checks = {"streams":set(results)=={"source","plain","roi","tracked"},"checkpoint":data["checkpoint_sha256"]==WEIGHT,"cuda":data["capability"]==[8,9],"settings":data["settings"]=={"frames_in":3,"frames_out":3,"step":1,"threshold":0.5},"labels":all(len(x["rows"])>0 for x in results.values()),"same_support":len({tuple(r["tick"] for r in x["rows"]) for x in results.values()})==1}
    if data["phase"]=="smoke":
        checks["forecast"] = data["model_seconds"] + 1.5*sum(x["seconds"] for x in results.values())*(data["source_frames"]/data["frames"]) + 180 < args.full_limit
    report={"passed":all(checks.values()),"checks":checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report)+"\n")
    if not report["passed"]:
        raise SystemExit(str(checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","packets","labels","wasb-code","wasb-weights"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--match",required=True)
    parser.add_argument("--frames",type=int)
    parser.add_argument("--full-limit",type=int,required=True)
    parser.add_argument("--validate",action="store_true")
    args=parser.parse_args()
    validate(args) if args.validate else run(args)


if __name__=="__main__":
    main()
