"""Registered fixed-region native AV1 hold/refresh/reset control, CPU only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import subprocess
import time
from fractions import Fraction
import numpy as np

REGISTRATION = {
    "sources": [
        {"id": "alcaraz_perricard", "start_seconds": 1, "duration_seconds": 30, "scene_cuts_seconds": []},
        {"id": "federer_djokovic", "start_seconds": 20, "duration_seconds": 30,
         "scene_cuts_seconds": [28.445, 43.460]},
    ],
    "geometry": "source upper quarter, scale bicubic to 640x90, yuv420p; fixed image region, no semantic mask claim",
    "quality": "all-frame original-target Y mean-frame PSNR and pooled MSE/PSNR; no altered target",
    "crfs": [32, 44], "refresh_seconds": [None, 1, 5, 10],
    "cut_policy": "separate five-second hold+scene-metadata-cut reset and separate native cut-reset anchor",
    "horizons_seconds": [1, 5, 10, 30],
    "encoder": "libaom-av1 cpu-used=6 row-mt=1 threads=4 crf, IVF; anchors g=9999; each refresh native one-frame stream",
    "scope": "two consecutive intervals; metadata cut times are retained labels, not independently labeled cut truth; no foreground/correction costs or complete codec",
    "exposure": "no learned model; historical data accessed during development; not held-out training evidence",
    "extraction": "rounded nominal start frame converted to input -ss timestamp; passthrough decoded frames, no fps filter; require every selected showinfo PTS delta nominal +/-50us; metadata cut reset on first selected PTS at or after cut",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        started=time.monotonic(); count=0
        while b := f.read(1024**2):
            h.update(b); count += len(b)
            delay=count/(20*1024**2)-(time.monotonic()-started)
            if delay>0: time.sleep(delay)
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}

def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def guard():
    if os.getloadavg()[0] > 40: raise RuntimeError("host load > registered 40; retained partial output")

def run(cmd, commands, *, output=False, stderr=False):
    guard()
    commands.append(cmd)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors="replace")[-2000:])
    return r.stderr if stderr else r.stdout if output else None

def schedule(n, fps, every=None, cuts=()):
    indexes = {0}
    if every is not None:
        k=1
        while round(k*every*fps)<n:
            indexes.add(round(k*every*fps)); k+=1
    indexes.update(int(x) for x in cuts if 0 < int(x) < n)
    return sorted(indexes)

def score(y, decoded):
    assert y.shape == decoded.shape
    mse = np.square(y.astype(np.float64)-decoded.astype(np.float64)).mean(axis=(1,2))
    psnr = np.where(mse > 0, 10*np.log10(255**2/np.maximum(mse, 1e-300)), np.inf)
    return {"frames": len(y), "mse_per_frame": mse.tolist(),
            "mean_frame_psnr_db": float(psnr.mean()) if np.isfinite(psnr).all() else None,
            "perfect_frames": int((mse == 0).sum()), "pooled_mse": float(mse.mean()),
            "pooled_psnr_db": float(10*math.log10(255**2/mse.mean())) if mse.mean() else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--code-revision", required=True)
    p.add_argument("--smoke", action="store_true"); a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    os.nice(19); cores=sorted(os.sched_getaffinity(0))[:8]; os.sched_setaffinity(0, cores)
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    subprocess.run(["ionice","-c","3","-p",str(os.getpid())], check=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    ff=Path("/opt/local/bin/ffmpeg"); probe=Path("/opt/local/bin/ffprobe")
    commands=[]
    report={"registration":REGISTRATION,"smoke":a.smoke,"code_revision":a.code_revision,
            "worker":sha(__file__),"hostname":platform.node(),"affinity":cores,
            "resource_policy":{"cpu_threads":8,"native_threads":4,"max_load":40,"address_space_gib":16,"nice":19,"io":"idle","gpu":"none"},
            "ffmpeg":sha(ff),"ffprobe":sha(probe),"version":run([str(ff),"-version"],commands,output=True).decode(),
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
            "gpu_inventory":run(["nvidia-smi","--query-gpu=uuid,name","--format=csv,noheader"],commands,output=True).decode(),
            "sources":[],"commands":commands}
    save(a.out/"registration-before-run.json", report)
    for spec in REGISTRATION["sources"]:
        base=a.out/spec["id"];base.mkdir()
        source=a.data_root/"assets/raw_4k"/(spec["id"]+".mp4")
        meta=a.data_root/"assets/dataset"/spec["id"] /"scene_metadata.json"
        info=json.loads(run([str(probe),"-v","error","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate,avg_frame_rate,nb_frames","-of","json",str(source)],commands,output=True))["streams"][0]
        fps=Fraction(info["r_frame_rate"]); duration=11 if a.smoke else 30
        start=round(spec["start_seconds"]*fps); n=round(duration*fps)
        raw=base/"source.yuv"
        # Seek is exact timestamp plus normal decoded frame enumeration; raw frame hash certifies scored output.
        extraction_log=run([str(ff),"-v","info","-threads","4","-ss",str(float(start/fps)),"-i",str(source),"-frames:v",str(n),"-vf","crop=iw:ih/4:0:0,scale=640:90:flags=bicubic,showinfo","-fps_mode","passthrough","-pix_fmt","yuv420p","-f","rawvideo",str(raw)],commands,stderr=True)
        (base/"extraction.log").write_bytes(extraction_log)
        times=[float(x) for x in re.findall(rb"pts_time:([-0-9.]+)",extraction_log)]
        if len(times)!=n or not np.allclose(np.diff(times),float(1/fps),atol=5e-5,rtol=0):
            raise ValueError("selected source frames are not complete uniform nominal-timebase consecutive frames")
        pixels=np.fromfile(raw,dtype=np.uint8).reshape(n,640*90*3//2); y=pixels[:,:640*90].reshape(n,90,640)
        cuts=[int(np.searchsorted(times,c-float(start/fps),side="left")) for c in spec["scene_cuts_seconds"]]
        result={"source_id":spec["id"],"source":sha(source),"scene_metadata":sha(meta),"probe":info,
                "raw":sha(raw),"frame_sha256":[hashlib.sha256(v.tobytes()).hexdigest() for v in pixels],
                "requested_start_frame":start,"requested_seek_seconds":float(start/fps),"frame_count":n,"fps":str(fps),"observed_seconds":float(n/fps),
                "registered_cut_frames":cuts,"selected_pts_seconds":times,"extraction_log":sha(base/"extraction.log"),"arms":[]}
        cache={}
        def coded(index,count,crf,name):
            key=(index,count,crf)
            if key in cache:return cache[key]
            inp=base/(name+".yuv"); pixels[index:index+count].tofile(inp)
            bit=base/(name+".ivf")
            run([str(ff),"-v","error","-f","rawvideo","-pixel_format","yuv420p","-video_size","640x90","-framerate",str(fps),"-i",str(inp),"-an","-c:v","libaom-av1","-cpu-used","6","-row-mt","1","-threads","4","-crf",str(crf),"-b:v","0","-g","9999","-f","ivf",str(bit)],commands)
            decoded=run([str(ff),"-v","error","-threads","4","-i",str(bit),"-pix_fmt","yuv420p","-f","rawvideo","pipe:1"],commands,output=True)
            dec=np.frombuffer(decoded,dtype=np.uint8).reshape(count,640*90*3//2)[:,:640*90].reshape(count,90,640)
            inp.unlink();cache[key]=(bit,dec);return bit,dec
        for crf in [32] if a.smoke else REGISTRATION["crfs"]:
            refresh=[("hold",None,()),("refresh1",1,()),("refresh5",5,()),("refresh10",10,()),("refresh5_cutreset",5,cuts)]
            for name,every,extra in refresh:
                indexes=schedule(n,fps,every,extra); pred=np.empty_like(y); packets=[]
                for j,index in enumerate(indexes):
                    end=indexes[j+1] if j+1<len(indexes) else n
                    bit,dec=coded(index,1,crf,f"still_q{crf}_f{index:05d}");pred[index:end]=dec[0]
                    packets.append({"frame":index,"hold_until":end,"stream":sha(bit)})
                manifest={"format":"PointStream fixed-region hold control v1","geometry":[640,90],"fps":str(fps),"frames":n,"crf":crf,"packets":packets,"deployment":"native AV1 decoder; no models"}
                manifest_path=base/f"{name}_q{crf}.json";save(manifest_path,manifest)
                total=sum(v["stream"]["bytes"] for v in packets)+manifest_path.stat().st_size
                horizons=[]
                for sec in REGISTRATION["horizons_seconds"]:
                    end=min(n,round(sec*fps)); prefix=json.loads(json.dumps(manifest));prefix["frames"]=end
                    prefix["packets"]=[v for v in prefix["packets"] if v["frame"]<end]
                    for v in prefix["packets"]:v["hold_until"]=min(v["hold_until"],end)
                    mp=base/f"{name}_q{crf}_h{sec}.json";save(mp,prefix)
                    b=sum(v["stream"]["bytes"] for v in prefix["packets"])+mp.stat().st_size
                    horizons.append({"seconds":float(end/fps),"bytes":b,"bits_per_second":b*8/float(end/fps),"quality":score(y[:end],pred[:end]),"manifest":sha(mp)})
                result["arms"].append({"name":name,"crf":crf,"bytes":total,"cold_start_stream_bytes":packets[0]["stream"]["bytes"],"stream_bytes":sum(v["stream"]["bytes"] for v in packets),"manifest":sha(manifest_path),"quality":score(y,pred),"horizons":horizons,"packet_count":len(packets)})
            for name,indexes in [("continuous",[0]),("reset10",schedule(n,fps,10)),("cutreset",schedule(n,fps,None,cuts))]:
                pred=np.empty_like(y);packets=[]
                for j,index in enumerate(indexes):
                    end=indexes[j+1] if j+1<len(indexes) else n
                    bit,dec=coded(index,end-index,crf,f"{name}_q{crf}_f{index:05d}");pred[index:end]=dec
                    packets.append({"frame":index,"end":end,"stream":sha(bit)})
                mp=base/f"anchor_{name}_q{crf}.json";save(mp,{"geometry":[640,90],"fps":str(fps),"frames":n,"packets":packets})
                result["arms"].append({"name":name,"crf":crf,"bytes":sum(v["stream"]["bytes"] for v in packets)+mp.stat().st_size,"manifest":sha(mp),"packet_count":len(packets),"quality":score(y,pred)})
        report["sources"].append(result);save(a.out/"report.partial.json",report)
    report["status"]="complete";save(a.out/"report.json",report)
    print(json.dumps({"status":"complete","report":sha(a.out/"report.json")}))

if __name__=="__main__":main()
