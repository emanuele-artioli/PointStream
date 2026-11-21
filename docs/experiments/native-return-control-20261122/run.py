"""Measure consecutive-frame and pre-cut-reference AV1 on one registered passage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import shlex
import struct
import subprocess
import time

import numpy as np

from src.codecs import svtav1

SOURCE = Path("/Users/manu/Documents/PointStream-review/2026-11-18-continuous-return/source-only-review.mp4")
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    return sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])


def main(root):
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    tools = {k:svtav1.tool(k) for k in ["ffmpeg","dav1d"]}
    version=subprocess.check_output(["pkg-config","--modversion","aom"],text=True).strip()
    assert version=="3.14.1"
    flags=shlex.split(subprocess.check_output(["pkg-config","--cflags","--libs","aom"],text=True))
    binary=root/"ref"
    commands=[["cc","-O2","-Wall","-Wextra",str(HERE/"ref.c"),"-o",str(binary),*flags]]
    assert sha(SOURCE)=="0d20d883430ecfea33c474f7e7c7f42df6052437898d390af254148def687b50"
    raw=root/"source.yuv"
    commands.append([tools["ffmpeg"]["path"],"-v","error","-threads","1","-i",str(SOURCE),"-vf","scale=320:180:flags=area","-pix_fmt","yuv420p","-f","rawvideo",str(raw)])
    for command in commands:
        subprocess.run(command,check=True,timeout=45)
    shape=(245,270,320)
    assert raw.stat().st_size==math.prod(shape)
    truth=np.memmap(raw,"uint8","r",shape=shape)[:,:180]
    rows=[]
    for qp in [45,35,55]:
        prefix=[]
        for golden in [0,1]:
            ivf=root/f"q{qp}-golden{golden}.ivf"
            yuv=ivf.with_suffix(".yuv")
            calls=[[str(binary),str(raw),str(ivf),str(qp),str(golden)],svtav1.decode_command(tools["dav1d"]["path"],ivf,yuv,threads=1)]
            for command in calls:
                subprocess.run(command,check=True,timeout=60)
            commands+=calls
            data=ivf.read_bytes()
            sizes=svtav1.ivf_frames(data)
            offsets=[32]
            assert len(sizes)==245
            for t,n in enumerate(sizes):
                assert struct.unpack_from("<IQ",data,offsets[-1])==(n,t)
                offsets.append(offsets[-1]+12+n)
            assert offsets[-1]==len(data) and yuv.stat().st_size==raw.stat().st_size
            prefix.append(hashlib.sha256(data[:offsets[145]]).hexdigest())
            displayed=np.memmap(yuv,"uint8","r",shape=shape)[:,:180]
            difference=np.square(displayed.astype(np.float32)-truth)
            rows.append(dict(q=qp,extra_golden=bool(golden),bytes=len(data),return_bytes=len(data)-offsets[145],
                             total_kbps=len(data)*8/4.9/1000,first_return_bytes=sizes[145]+12,
                             whole_y_psnr=10*math.log10(255**2/float(difference.mean(dtype=np.float64))),
                             return_y_psnr=10*math.log10(255**2/float(difference[145:].mean(dtype=np.float64))),
                             prefix_sha256=prefix[-1],ivf_sha256=sha(ivf),decode_sha256=sha(yuv)))
        assert prefix[0]==prefix[1]
        if qp==45:
            assert cpu()*4.5<300 and (time.monotonic()-started)*4.5<600
    result=dict(rows=rows,libaom=version,library_sha256=sha(Path(subprocess.check_output(["pkg-config","--variable=libdir","aom"],text=True).strip())/"libaom.dylib"),
                compiler=subprocess.check_output(["cc","--version"],text=True).splitlines()[0],tools=tools,host=platform.platform(),device="CPU",
                source_sha256=sha(SOURCE),scaled_yuv_sha256=sha(raw),driver_sha256=sha(Path(__file__)),code_sha256=sha(HERE/"ref.c"),
                binary_sha256=sha(binary),commands=commands,elapsed_seconds=time.monotonic()-started,cpu_seconds=cpu())
    (root/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print([(r["q"],r["extra_golden"],r["bytes"],round(r["return_y_psnr"],3)) for r in rows])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    main(cli.parse_args().output)
