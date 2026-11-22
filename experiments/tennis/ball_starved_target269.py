"""RacketVision match209 starved native/ROI/tracked-pixel CPU pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import time
from fractions import Fraction

import numpy as np
import cv2

from experiments.tennis.ball_starved_scaled import (
    copy_member as copy_member,
    pack_scaled as pack_scaled,
    unpack_scaled as unpack_scaled,
    render_scaled as render_scaled,
    scaled_roi as scaled_roi,
    score_frame as score_frame,
)
from experiments.tennis import ball_confirmation as prior
from experiments.tennis import ball_confirmation_score as scorer
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch

SOURCE = "match269_000"
ARCHIVE = "a7e1ca9f15e9cba300e2cbeed976f30499211018b441d11aa5b8c6c2cadb5aab"
MANIFEST = "df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1"
VGG = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
FPS = Fraction(25)
W, H = 1280, 720
FRAME_BYTES = W * H * 3 // 2
LOW_W, LOW_H = 320, 180
HEADER = struct.Struct(">4sIII")


def run(args):
    import torch

    started = time.monotonic()
    phase = os.environ["PS_STAGE"]
    n = 16 if phase == "smoke" else 64
    if args.frames != n or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Stage or CPU allocation mismatch")
    if prior.sha(args.bases) != ARCHIVE or prior.sha(args.weights) != VGG:
        raise ValueError("Input identity mismatch")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "ball-starved-269"
    work.mkdir()
    for member, target in (("publish/source.yuv", "source-full.yuv"),
                           ("publish/source.rgb", "source-full.rgb"),
                           ("publish/bases/sender.json", "sender.json"),
                           ("publish/receipt.json", "receipt.json")):
        copy_member(args.bases, member, work / target)
    receipt = json.loads((work / "receipt.json").read_text())
    if (receipt["source_id"] != SOURCE or receipt["frames"] != 64
            or prior.sha(work / "source-full.yuv") != receipt["result"]["source_yuv_sha256"]
            or prior.sha(work / "source-full.rgb") != receipt["source"]["rgb_sha256"]):
        raise ValueError("Accepted source provenance failed")
    source = np.memmap(work / "source-full.rgb", np.uint8, "r", shape=(64,H,W,3))[:n]
    sender = json.loads((work / "sender.json").read_text())
    prior.validate_sender(sender, FPS, 64)
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
    labels = scorer.labels_for_source(manifest, SOURCE)[:n]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    net = quality.load_dists(args.weights, "cpu")
    self_score = dists_crops(net, [source[0,:64,:64]], source[0,:64,:64],
                             np.ones((64,64), bool), "cpu")[0]
    if not np.isfinite(self_score) or abs(self_score) > 1e-5:
        raise ValueError("DISTS self-comparison failed")
    encoder, decoder = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    ffmpeg = svtav1.tool("ffmpeg")
    if ffmpeg["sha256"] != "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2":
        raise ValueError("FFmpeg identity differs from confirmation")
    if (encoder["sha256"] != "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4"
            or decoder["sha256"] != "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"):
        raise ValueError("Codec identity differs from confirmation")
    patches, placements = prior.patch_frames(source_yuv, track, n)
    patch_source = work / "tile.yuv"
    patch_source.write_bytes(patches.tobytes())
    tile_dir = work / "tile"
    tile_dir.mkdir()
    tile = svtav1.code(patch_source, tile_dir, width=64, height=64, fps=FPS, frames=n,
                       crf=20, preset=4, threads=1, full_range=False, encoder=encoder["path"],
                       decoder=decoder["path"], timeout=600)
    roi = work / "roi.txt"
    roi.write_text(scaled_roi(track, n))
    small = work / "small.yuv"
    subprocess.run([ffmpeg["path"],"-v","error","-f","rawvideo","-pixel_format","yuv420p",
                    "-video_size","1280x720","-framerate","25","-i",str(source_yuv),
                    "-vf","scale=320:180:flags=area,format=yuv420p","-frames:v",str(n),
                    "-f","rawvideo",str(small)], check=True, timeout=300)
    if small.stat().st_size != n * LOW_W * LOW_H * 3 // 2:
        raise ValueError("scaled source frame count")
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publish.mkdir()
    results = []
    for mode, crf in (("plain", args.plain), ("roi", args.roi), ("tracked", args.tracked)):
        plain_path = None
        folder = work / f"{mode}-{crf}"
        folder.mkdir()
        record = svtav1.code(small, folder, width=LOW_W, height=LOW_H, fps=FPS, frames=n,
                              crf=crf, preset=4, threads=1, full_range=False,
                              encoder=encoder["path"], decoder=decoder["path"], timeout=600,
                              extra=["--roi-map-file",str(roi)] if mode=="roi" else None)
        if record["decoded_frames"] != n or record["temporal_units"] != n:
            raise ValueError("Native frame count failed")
        ivf = Path(record["stream"]).read_bytes()
        dest = publish / f"{mode}-{crf}.ivf"
        dest.write_bytes(ivf)
        yuv = np.memmap(record["decoded"],np.uint8,"r",shape=(n,LOW_H*3//2,LOW_W))
        rgb = np.stack([cv2.resize(tracked_patch.yuv_to_rgb(frame,LOW_W,LOW_H,False),(W,H),interpolation=cv2.INTER_CUBIC) for frame in yuv])
        vals = score_frame(net, source, rgb, labels)
        if mode != "tracked":
            results.append({"mode":mode,"crf":crf,"bytes":len(ivf),
                        "kbps":len(ivf)*8*25/n/1000,"sha256":prior.sha(dest),
                        "decoded_sha256":prior.sha(Path(record["decoded"])),
                        "crop":vals,"mean_dists":float(np.mean([v["dists"] for v in vals])) if vals else None,
                        "file":dest.name})
        if mode != "tracked":
            continue
        plain_path = dest
        origins = [[p["left"],p["top"]] if p["present"] else None for p in placements]
        native = {"version":1,"profile":"native-av1","width":LOW_W,"height":LOW_H,
                  "fps":[25,1],"frames":n,"full_range":False,"decoder_sha256":decoder["sha256"]}
        meta = {"native":native,"canvas":[W,H],"positions":origins}
        data = pack_scaled(meta, plain_path.read_bytes(), Path(tile["stream"]).read_bytes())
        target = publish / f"tracked-{crf}.ptp2"
        target.write_bytes(data)
        render_dir = work / f"render-{crf}"
        render_dir.mkdir()
        rgb = render_scaled(data,render_dir,decoder["path"])
        rendered_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        vals = score_frame(net,source,rgb,labels)
        results.append({"mode":"tracked","crf":crf,"bytes":len(data),"kbps":len(data)*8*25/n/1000,
                        "sha256":prior.sha(target),"rendered_sha256":rendered_sha256,"crop":vals,
                        "mean_dists":float(np.mean([v["dists"] for v in vals])) if vals else None,
                        "file":target.name})
    report = {"source":SOURCE,"frames":n,"fps":25,"phase":phase,"track":"accepted fitted svt62",
              "source_archive":ARCHIVE,"manifest_sha256":MANIFEST,"visible_labels":sum(x[0]==1 for x in labels),
              "tile_bytes":tile["file_bytes"],"positions":sum(p is not None for p in track),
              "self_dists":float(self_score),"encoder":encoder,"decoder":decoder,"ffmpeg":ffmpeg,
              "metric":quality.dists_record(args.weights),"results":results,
              "wall_seconds":time.monotonic()-started}
    (publish/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    (Path(os.environ["PS_STAGE_DIR"])/"result.json").write_text(json.dumps(report,indent=2)+"\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage/"result.json").read_text())
    n = 16 if os.environ["PS_STAGE"]=="smoke" else 64
    checks = {"source":report["source"]==SOURCE and report["source_archive"]==ARCHIVE,
              "count":report["frames"]==n and report["fps"]==25,
              "self":abs(report["self_dists"])<1e-5,
              "tracker":report["positions"]>0,
              "matrix":len(report["results"])==3}
    with tarfile.open(stage/"published.tar") as tar:
        for row in report["results"]:
            data = tar.extractfile("publish/"+row["file"]).read()
            ok = hashlib.sha256(data).hexdigest()==row["sha256"] and len(data)==row["bytes"]
            ok &= abs(row["kbps"]-len(data)*8*25/n/1000)<1e-8
            ok &= all(np.isfinite(item["dists"]) and item["dists"]>=0 for item in row["crop"])
            if row["mode"]=="tracked":
                meta, base, patch = unpack_scaled(data)
                ok &= len(meta["positions"])==n and bool(base) and bool(patch)
            else:
                ok &= len(svtav1.ivf_frames(data))==n
            checks[row["mode"]+str(row["crf"])] = bool(ok)
    projection = report["wall_seconds"]*(64/16)*1.5+120 if n==16 else None
    if n==16:
        checks["forecast"] = projection < 700
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
    parser.add_argument("--plain", type=float, default=41.0)
    parser.add_argument("--roi", type=float, default=38.0)
    parser.add_argument("--tracked", type=float, default=58.25)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.bases and args.sources and args.weights and args.frames:
        run(args)
    else:
        parser.error("Missing pilot inputs")


if __name__=="__main__":
    main()
