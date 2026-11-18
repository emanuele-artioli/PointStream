"""Validate the selected three complete base streams on the declared Linux CPU runtime."""
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

import cv2
import numpy as np

from experiments.tennis import common_receiver_manifest as manifests
from experiments.tennis import w8_payload as wire
from src.pointstream.tracked_patch import yuv_to_rgb

root = Path(sys.argv[1]).resolve()
env = Path(sys.prefix)
ffmpeg = env / "bin/ffmpeg"
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
started = time.monotonic()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cpu():
    return sum(getattr(resource.getrusage(k), f) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
               for f in ("ru_utime", "ru_stime"))


def rgba_canvas(yuv):
    return cv2.resize(yuv_to_rgb(yuv, 960, 544, False), (1920, 1080), interpolation=cv2.INTER_CUBIC)


def prepare(row):
    folder = root / str(row["crf"])
    for name, expected in row["sha256"].items():
        assert sha((folder / name).read_bytes()) == expected
    blob = (folder / "players.pscp").read_bytes()
    header = json.loads(blob[9:9+int.from_bytes(blob[5:9], "big")])
    profile = header["profile"]
    resources = profile["resources"]
    assert resources["native_decoder"] == ledger["linux_native_ffmpeg_sha256"]
    assert profile["coded_size"] == [960, 544] and profile["canvas_size"] == [1920, 1080]
    assert profile["recipe"]["base_to_canvas"] == "RGB8-cv2-INTER_CUBIC"
    parts, _ = wire.unpack(blob, profile)
    base = parts["base"]
    assert sha(base) == row["base_ivf_sha256"]
    manifests.verify(folder / "receiver.manifest", sha(blob))
    assert len(base) == row["base_ivf_bytes"]
    assert struct.unpack_from("<4sHH4sHHIIII",base) == (b"DKIF",0,32,b"AV01",960,544,50,1,240,0)
    ts = []
    offset = 32
    while offset < len(base):
        length,tick = struct.unpack_from("<IQ", base, offset)
        assert length > 0
        ts.append(tick)
        offset += 12+length
    assert offset == len(base) and ts == list(range(240))
    ivf = folder / "paid.ivf"
    with ivf.open("xb") as file:
        file.write(base)
    out = folder / "base.yuv"
    args = [str(ffmpeg), "-nostdin", "-v", "error", "-threads", "1", "-c:v", "libdav1d",
            "-i",str(ivf), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-n",str(out)]
    subprocess.run(args, check=True, timeout=15)
    assert out.stat().st_size == 240*960*544*3//2
    arr=np.memmap(out, dtype=np.uint8, mode="r", shape=(240,816,960))
    assert sha(arr) == expected_native[sha(base)]
    return folder,arr,args,ts


ledger = json.loads((root/"rates.json").read_text())
historical = json.loads((root/"full.json").read_text())
expected_native = {r["stream_sha256"]:r["decoded_sha256"] for r in historical["rows"]}
assert sha(ffmpeg.read_bytes()) == ledger["linux_native_ffmpeg_sha256"]
assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and cv2.getNumThreads()==1
receipt = {"source_sha256":ledger["source_sha256"],"private_remote":str(root),"host":os.uname().nodename,
           "runtime":"32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661",
           "ffmpeg":str(ffmpeg),"ffmpeg_sha256":ledger["linux_native_ffmpeg_sha256"],
           "ffmpeg_version":subprocess.check_output([str(ffmpeg),"-version"],text=True).splitlines()[0],
           "numpy":np.__version__,"opencv":cv2.__version__, "script_sha256":sha(Path(__file__).read_bytes()),
           "sam_env_source":str(Path(wire.__file__).resolve().parents[2]),
           "conversion_code_sha256":sha(Path(sys.modules[yuv_to_rgb.__module__].__file__).read_bytes()),
           "historical_report_sha256":sha((root/"full.json").read_bytes()),"rates_sha256":sha((root/"rates.json").read_bytes()),"rows":[]}
for i,row in enumerate(ledger["rows"]):
    folder,arr,args,pts = prepare(row)
    before=cpu()
    early = [sha(rgba_canvas(arr[j])) for j in (0,11,120,239)]
    if not i:
        elapsed = cpu()-before
        forecast = (elapsed/4)*720
        assert forecast < 250, forecast
        receipt["smoke"]={"ticks":[0,11,120,239],"canvas_cpu_seconds":elapsed,"three_rate_conversion_cpu_forecast":forecast}
    planes = {p:hashlib.sha256() for p in "YUV"}
    canvases = hashlib.sha256()
    frame_rows=[]
    for t in pts:
        first=arr[t,:544]
        u,v=arr[t,544:].reshape(2,272,480)
        rgb=rgba_canvas(arr[t])
        assert rgb.shape==(1080,1920,3) and rgb.dtype==np.uint8
        frame_rows.append({"pts":t,"yuv420p_sha256":sha(arr[t]),"rgb24_sha256":sha(rgb)})
        for h,plane in zip(planes.values(),(first,u,v)):
            h.update(plane)
        canvases.update(rgb)
    assert early==[frame_rows[t]["rgb24_sha256"] for t in (0,11,120,239)]
    detail=folder/"native-frames.json"
    detail.write_text(json.dumps(frame_rows)+"\n")
    receipt["rows"].append({"crf":row["crf"],"players_and_joint_wire_sha256":row["sha256"],
        "player_kbps":row["player_kbps"],"joint_kbps":row["joint_kbps"],"native_ivf_sha256":row["base_ivf_sha256"],
        "command":args,"pts":{"first":0,"last":239,"count":240,"step":1,"timebase":"1/50"},
        "yuv420p_bytes":arr.nbytes,"yuv420p_sha256":sha(arr),"historical_orig_match":True,
        "plane_sha256":{k:v.hexdigest() for k,v in planes.items()},"plane_shapes":[[544,960],[272,480],[272,480]],
        "rgb24_shape":[240,1080,1920,3],"rgb24_sha256":canvases.hexdigest(),
        "native_frame_details_sha256":sha(detail.read_bytes()),"repeated_rgb_ticks": [0,11,120,239]})
    assert cpu()<300 and time.monotonic()-started<135
    print("verified",row["crf"],round(cpu(),2),flush=True)
receipt.update(cpu_seconds=cpu(),wall_seconds=time.monotonic()-started,gpu=None)
(root/"native-report.json").write_text(json.dumps(receipt,indent=2)+"\n")
