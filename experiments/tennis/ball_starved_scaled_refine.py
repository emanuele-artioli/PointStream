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

from experiments.tennis import ball_confirmation as prior
from experiments.tennis import ball_confirmation_score as scorer
from experiments.tennis.k5d_encoder import crop64
from experiments.tennis import w1_roi
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch, codec

SOURCE = "match209_000"
ARCHIVE = "e9875c1aa522d86d2f0a96df11cec89429800d437964ab5d58d4f0a9de3173a9"
MANIFEST = "df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1"
VGG = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
FPS = Fraction(25)
W, H = 1280, 720
FRAME_BYTES = W * H * 3 // 2
LOW_W, LOW_H = 320, 180
HEADER = struct.Struct(">4sIII")


def copy_member(archive, name, output):
    with tarfile.open(archive) as tar:
        member = tar.extractfile(name)
        if member is None:
            raise ValueError(f"Missing archive member {name}")
        with output.open("xb") as stream:
            shutil.copyfileobj(member, stream)


def pack_scaled(meta, base, tile):
    if set(meta) != {"native", "canvas", "positions"}:
        raise ValueError("unsupported scaled metadata")
    native = meta["native"]
    codec.check_native(native, base)
    codec.check_native({**native, "width": 64, "height": 64}, tile)
    if native["width"] != LOW_W or native["height"] != LOW_H or meta["canvas"] != [W, H]:
        raise ValueError("invalid scaling geometry")
    positions = meta["positions"]
    if len(positions) != native["frames"]:
        raise ValueError("invalid positions")
    for pos in positions:
        if pos is not None and (len(pos) != 2 or any(type(v) is not int for v in pos)
                                or pos[0] < 0 or pos[1] < 0 or pos[0] % 2 or pos[1] % 2
                                or pos[0] + 32 > W or pos[1] + 32 > H):
            raise ValueError("invalid patch position")
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body = HEADER.pack(b"PTP2", len(meta_bytes), len(base), len(tile)) + meta_bytes + base + tile
    return body + hashlib.sha256(body).digest()


def unpack_scaled(packet):
    if len(packet) < HEADER.size + 32 or hashlib.sha256(packet[:-32]).digest() != packet[-32:]:
        raise ValueError("invalid packet checksum")
    magic, msize, bsize, tsize = HEADER.unpack_from(packet)
    if magic != b"PTP2" or HEADER.size + msize + bsize + tsize + 32 != len(packet):
        raise ValueError("invalid scaled header")
    offset = HEADER.size
    meta = json.loads(packet[offset:offset+msize])
    offset += msize
    base = packet[offset:offset+bsize]
    tile = packet[offset+bsize:-32]
    if pack_scaled(meta, base, tile) != packet:
        raise ValueError("noncanonical packet")
    return meta, base, tile


def render_scaled(packet, work, decoder):
    meta, base, tile = unpack_scaled(packet)
    if svtav1.tool(decoder)["sha256"] != meta["native"]["decoder_sha256"]:
        raise ValueError("native decoder identity mismatch")
    raw_base = work / "base.yuv"
    raw_tile = work / "tile.yuv"
    tracked_patch.decode_native(base, raw_base, meta["native"], decoder)
    tracked_patch.decode_native(tile, raw_tile, {**meta["native"], "width": 64, "height": 64}, decoder)
    count = meta["native"]["frames"]
    frames = np.memmap(raw_base, np.uint8, "r", shape=(count, LOW_H * 3 // 2, LOW_W))
    patches = np.memmap(raw_tile, np.uint8, "r", shape=(count, 96, 64))
    result = []
    for tick, position in enumerate(meta["positions"]):
        image = cv2.resize(tracked_patch.yuv_to_rgb(frames[tick], LOW_W, LOW_H, False),
                           (W, H), interpolation=cv2.INTER_CUBIC)
        if position is not None:
            x, y = position
            image[y:y+32,x:x+32] = tracked_patch.yuv_to_rgb(patches[tick],64,64,False)[16:48,16:48]
        result.append(image)
    return np.stack(result)


def scaled_roi(track, frames):
    masks = np.zeros((frames, LOW_H, LOW_W), dtype=bool)
    for index, point in enumerate(track):
        if point is None:
            continue
        x, y = point[0] // 4, point[1] // 4
        masks[index, max(0,y-8):min(LOW_H,y+8), max(0,x-8):min(LOW_W,x+8)] = True
    rows = w1_roi.roi_text(masks, LOW_W, LOW_H, -40, outside=True).splitlines()
    for index, point in enumerate(track):
        if point is None:
            rows[index] = str(index) + ' ' + ' '.join(['0'] * (((LOW_H+63)//64)*((LOW_W+63)//64)))
    return "\n".join(rows) + "\n"


def score_frame(net, source, decoded, labels):
    values = []
    for index, (visible, x, y) in enumerate(labels):
        if visible != 1:
            continue
        target = crop64(source[index], x, y)
        candidate = crop64(decoded[index], x, y)
        value = dists_crops(net, [candidate], target, np.ones((64, 64), bool), "cpu")[0]
        if not np.isfinite(value) or value < 0:
            raise ValueError("Non-finite ball crop DISTS")
        values.append({"tick": index, "dists": float(value)})
    return values


def run(args):
    import torch

    started = time.monotonic()
    phase = os.environ["PS_STAGE"]
    n = 16 if phase == "smoke" else 64
    if args.frames != n or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Stage or CPU allocation mismatch")
    if prior.sha(args.bases) != ARCHIVE or prior.sha(args.weights) != VGG:
        raise ValueError("Input identity mismatch")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "ball-starved-209"
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
    crfs = [68] if n == 16 else [56,68,70]
    for crf in crfs:
        plain_path = None
        for mode in ("plain", "roi"):
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
            results.append({"mode":mode,"crf":crf,"bytes":len(ivf),
                            "kbps":len(ivf)*8*25/n/1000,"sha256":prior.sha(dest),
                            "decoded_sha256":prior.sha(Path(record["decoded"])),
                            "crop":vals,"mean_dists":float(np.mean([v["dists"] for v in vals])) if vals else None,
                            "file":dest.name})
            if mode == "plain":
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
              "matrix":len(report["results"])==(3 if n==16 else 9)}
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
    projection = report["wall_seconds"]*(64/16)*3*1.5+120 if n==16 else None
    if n==16:
        checks["forecast"] = projection < 1000
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
