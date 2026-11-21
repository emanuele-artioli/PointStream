"""Full-resolution native and ROI anchors on frozen RacketVision sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from fractions import Fraction

import numpy as np

from experiments.tennis.ball_starved_scaled import (
    copy_member as copy_member,
    score_frame as score_frame,
)
from experiments.tennis import ball_confirmation as prior
from experiments.tennis import ball_confirmation_score as scorer
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch

SOURCES = {
    "269": ("match269_000", "a7e1ca9f15e9cba300e2cbeed976f30499211018b441d11aa5b8c6c2cadb5aab", 64, 25),
    "235": ("match235_000", "506485f5c85ef7ec4ecacecf920a0c0039174db5d1717bab41b0aff2a4de5311", 154, 60),
}

MANIFEST = "df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1"
VGG = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
W, H = 1280, 720
FRAME_BYTES = W * H * 3 // 2


def run(args):
    import torch

    started = time.monotonic()
    phase = os.environ["PS_STAGE"]
    source_id, archive_hash, total_frames, fps_value = SOURCES[args.source_id]
    fps = Fraction(fps_value)
    n = 16 if phase == "smoke" else total_frames
    if args.frames != n or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Stage or CPU allocation mismatch")
    if prior.sha(args.bases) != archive_hash or prior.sha(args.weights) != VGG:
        raise ValueError("Input identity mismatch")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "ball-starved-native-controls"
    work.mkdir()
    for member, target in (("publish/source.yuv", "source-full.yuv"),
                           ("publish/source.rgb", "source-full.rgb"),
                           ("publish/bases/sender.json", "sender.json"),
                           ("publish/receipt.json", "receipt.json")):
        copy_member(args.bases, member, work / target)
    receipt = json.loads((work / "receipt.json").read_text())
    if (receipt["source_id"] != source_id or receipt["frames"] != total_frames
            or prior.sha(work / "source-full.yuv") != receipt["result"]["source_yuv_sha256"]
            or prior.sha(work / "source-full.rgb") != receipt["source"]["rgb_sha256"]):
        raise ValueError("Accepted source provenance failed")
    source = np.memmap(work / "source-full.rgb", np.uint8, "r", shape=(total_frames,H,W,3))[:n]
    sender = json.loads((work / "sender.json").read_text())
    prior.validate_sender(sender, fps, total_frames)
    point = next(r for r in sender["points"] if r["codec"] == "svt" and r["point"] == 62)
    track = prior.checked_track(point["centres"][:n], n)
    if not any(point is not None for point in track):
        raise ValueError("No real tracker hints in selected frames")
    source_yuv = work / "source.yuv"
    with (work / "source-full.yuv").open("rb") as reader, source_yuv.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024*1024)
        writer.truncate(n * FRAME_BYTES)
    manifest, manifest_raw = scorer.authenticated_manifest(args.sources)
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST:
        raise ValueError("Source label manifest changed")
    labels = scorer.labels_for_source(manifest, source_id)[:n]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    net = quality.load_dists(args.weights, "cpu")
    self_score = dists_crops(net, [source[0,:64,:64]], source[0,:64,:64],
                             np.ones((64,64), bool), "cpu")[0]
    if not np.isfinite(self_score) or abs(self_score) > 1e-5:
        raise ValueError("DISTS self-comparison failed")
    encoder, decoder = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    if (encoder["sha256"] != "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4"
            or decoder["sha256"] != "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"):
        raise ValueError("Codec identity differs from confirmation")
    setup_seconds = time.monotonic() - started
    roi = work / "roi.txt"
    roi.write_text(prior.roi_map(track, n))
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publish.mkdir()
    results = []
    crfs = [70] if n == 16 else [68,70]
    for crf in crfs:
        for mode in ("plain", "roi"):
            folder = work / f"{mode}-{crf}"
            folder.mkdir()
            record = svtav1.code(source_yuv, folder, width=W, height=H, fps=fps, frames=n,
                                  crf=crf, preset=4, threads=1, full_range=False,
                                  encoder=encoder["path"], decoder=decoder["path"], timeout=600,
                                  extra=["--roi-map-file",str(roi)] if mode=="roi" else None)
            if record["decoded_frames"] != n or record["temporal_units"] != n:
                raise ValueError("Native frame count failed")
            ivf = Path(record["stream"]).read_bytes()
            dest = publish / f"{mode}-{crf}.ivf"
            dest.write_bytes(ivf)
            yuv = np.memmap(record["decoded"],np.uint8,"r",shape=(n,H*3//2,W))
            rgb = np.stack([tracked_patch.yuv_to_rgb(frame,W,H,False) for frame in yuv])
            vals = score_frame(net, source, rgb, labels)
            results.append({"mode":mode,"crf":crf,"bytes":len(ivf),
                            "kbps":len(ivf)*8*fps_value/n/1000,"sha256":prior.sha(dest),
                            "decoded_sha256":prior.sha(Path(record["decoded"])),
                            "crop":vals,"mean_dists":float(np.mean([v["dists"] for v in vals])) if vals else None,
                            "file":dest.name})
    report = {"source":source_id,"frames":n,"fps":fps_value,"phase":phase,"track":"accepted fitted svt62",
              "source_archive":archive_hash,"manifest_sha256":MANIFEST,"visible_labels":sum(x[0]==1 for x in labels),
              "positions":sum(p is not None for p in track),
              "self_dists":float(self_score),"encoder":encoder,"decoder":decoder,
              "metric":quality.dists_record(args.weights),"results":results,
              "setup_seconds":setup_seconds,"wall_seconds":time.monotonic()-started}
    (publish/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    (Path(os.environ["PS_STAGE_DIR"])/"result.json").write_text(json.dumps(report,indent=2)+"\n")


def validate(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage/"result.json").read_text())
    source_id, archive_hash, total_frames, fps_value = SOURCES[args.source_id]
    n = 16 if os.environ["PS_STAGE"]=="smoke" else total_frames
    checks = {"source":report["source"]==source_id and report["source_archive"]==archive_hash,
              "count":report["frames"]==n and report["fps"]==fps_value,
              "self":abs(report["self_dists"])<1e-5,
              "tracker":report["positions"]>0,
              "matrix":len(report["results"])==(2 if n==16 else 4)}
    with tarfile.open(stage/"published.tar") as tar:
        for row in report["results"]:
            data = tar.extractfile("publish/"+row["file"]).read()
            ok = hashlib.sha256(data).hexdigest()==row["sha256"] and len(data)==row["bytes"]
            ok &= abs(row["kbps"]-len(data)*8*fps_value/n/1000)<1e-8
            ok &= all(np.isfinite(item["dists"]) and item["dists"]>=0 for item in row["crop"])
            ok &= len(svtav1.ivf_frames(data))==n
            checks[row["mode"]+str(row["crf"])] = bool(ok)
    projection = (report["setup_seconds"] + (report["wall_seconds"]-report["setup_seconds"]) * (total_frames/16) * 2 * 1.5 + 120) if n==16 else None
    if n==16:
        checks["forecast"] = projection < args.full_limit
    else:
        checks["positive_labels"] = report["visible_labels"]>=1
    result = {"passed":all(checks.values()),"checks":checks,"projected_full_seconds":projection}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result,indent=2)+"\n")
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--source-id", choices=tuple(SOURCES), required=True)
    parser.add_argument("--full-limit", type=float, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate(args)
    elif args.bases and args.sources and args.weights and args.frames:
        run(args)
    else:
        parser.error("Missing pilot inputs")


if __name__=="__main__":
    main()
