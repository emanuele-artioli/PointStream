"""Verify the native neural stream and extract the twelve chosen RGB displays."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from src.pointstream.tracked_patch import yuv_to_rgb


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(a):
    assert sha(a.archive.read_bytes())==a.archive_sha256
    review=json.loads((a.review/"review.json").read_text())
    ticks=[row["frame_index"] for row in review["frames"]]
    with tarfile.open(a.archive) as archive:
        report=json.load(archive.extractfile("publish/result.json"))
        native=report["native"]
        assert report["source_sha256"]==review["source_video_sha256"]
        assert native["accepted"] and native["identical_frame_hashes"] and report["frames"]==240
        bitstream=archive.extractfile("publish/stream.psdc").read()
        assert len(bitstream)==native["bytes"] and sha(bitstream)==native["container_sha256"]
        width,height=native["size"]
        yuv=np.frombuffer(archive.extractfile("publish/decoded.yuv").read(),np.uint8).reshape(240,height*3//2,width)
        assert [sha(f) for f in yuv]==report["yuv_frames"]
    a.output.mkdir(parents=True,exist_ok=False)
    frames=[]
    for t in ticks:
        rgb=cv2.resize(yuv_to_rgb(yuv[t],width,height,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
        path=a.output/f"tick-{t:03}.png"
        assert cv2.imwrite(str(path),rgb[:,:,::-1],[cv2.IMWRITE_PNG_COMPRESSION,3])
        assert np.array_equal(cv2.imread(str(path))[:,:,::-1],rgb)
        frames.append(dict(tick=t,path=path.name,png_sha256=sha(path.read_bytes()),rgb_sha256=sha(rgb.tobytes())))
    result=dict(mode=f"DCVC-UF Q{native['qp']} {width}x{height}",bytes=len(bitstream),
                bitstream_sha256=native["container_sha256"],frames=frames,source_ticks=240,
                source_sha256=review["source_video_sha256"],fps=review["source_video_fps"],
                native_archive_sha256=a.archive_sha256,dimensions=[width,height],cv2=cv2.__version__)
    (a.output/"arm.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","review","output"):
        p.add_argument(n,type=Path)
    p.add_argument("archive_sha256")
    main(p.parse_args())
