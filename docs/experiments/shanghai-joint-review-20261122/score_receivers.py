"""Measure source-reviewed masks and official ball crops on genuine receiver frames."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"al028-neural-display-20261122"))
from score_bodies import EXPECTED, WEIGHTS, load_rgb, quality, sha, tensor  # noqa: E402
from score_review import crop, regions  # noqa: E402
from tools.datasets.al028_review_import import read_review  # noqa: E402


def measure(args):
    review = read_review(args.review)
    labels = json.loads(args.annotation.read_text())
    assert labels["schema"] == "pointstream-reviewed-player-official-ball-v1"
    assert labels["source_sha256"] == review["source_video_sha256"]
    assert labels["official_ball_sha256"] == review["official_ball_sha256"]
    assert labels["fps"] == [30000,1001] and labels["review_sha256"] == review["review_sha256"]
    if not args.qa_only and "QA" in labels["reviewer"].upper():
        raise ValueError("Synthetic reviewer fixture cannot enter the results")
    assert sha(WEIGHTS)==EXPECTED
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    net=quality.load_dists(WEIGHTS,"cpu")
    bank=json.loads(args.arm.read_text())
    assert bank["source_sha256"]==review["source_video_sha256"]
    assert bank["fps"]==[30000,1001] and bank["source_ticks"]==240
    assert bank["bytes"]>0 and len(bank["bitstream_sha256"])==64
    frames={r["tick"]:r for r in bank["frames"]}
    assert sorted(frames)==[r["frame_index"] for r in review["frames"]]
    scores=[]
    by_tick=[]
    start=time.process_time()
    for src,label in zip(review["frames"],labels["frames"],strict=True):
        tick=label["tick"]
        assert src["frame_index"]==tick and src["source_sha256"]==label["source_png_sha256"]
        refpath=args.review/src["source"]
        assert sha(refpath)==src["source_sha256"]
        original=load_rgb(refpath)
        arm=frames[tick]
        path=args.arm.parent/arm["path"]
        assert sha(path)==arm["png_sha256"]
        display=load_rgb(path)
        assert display.shape==original.shape==(1080,1920,3)
        assert hashlib.sha256(display.tobytes()).hexdigest()==arm["rgb_sha256"]
        values={}
        for identifier,rect,mask in regions(label,src):
            target=tensor(crop(original,rect))
            prediction=tensor(crop(display,rect))
            support=torch.from_numpy(np.ascontiguousarray(mask))[None,None]
            with torch.inference_mode():
                value=quality.masked_dists(net,prediction,target,support).item()
                if not scores:
                    self_check=quality.masked_dists(net,target,target,support).item()
                    assert abs(self_check)<1e-5
            assert np.isfinite(value)
            values[identifier]=value
            scores.append(dict(tick=tick,region=identifier,dists=value))
        people=[v for k,v in values.items() if k!="ball"]
        if len(people)==2 and "ball" in values:
            by_tick.append(dict(tick=tick,players=mean(people),ball=values["ball"],foreground=(mean(people)+values["ball"])/2))
        if tick==review["frames"][0]["frame_index"]:
            assert (time.process_time()-start)*12*1.5+60<650
    return dict(source_sha256=bank["source_sha256"],review_sha256=review["review_sha256"],
                annotation_sha256=sha(args.annotation),official_ball_sha256=review["official_ball_sha256"],
                backbone_sha256=EXPECTED,arm=bank["mode"],arm_manifest_sha256=sha(args.arm),
                bitstream_sha256=bank["bitstream_sha256"],bytes=bank["bytes"],
                kbps=bank["bytes"]*8*(30000/1001)/240/1000,source_ticks=240,
                qa_only=args.qa_only,scores=scores,paired_ticks=by_tick,
                mean_paired={k:mean(r[k] for r in by_tick) for k in ("players","ball","foreground")} if by_tick else {},
                cpu_seconds=time.process_time()-start)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("review","annotation","arm","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--qa-only",action="store_true")
    args=p.parse_args()
    record=measure(args)
    with args.out.open("x") as file:
        json.dump(record,file,indent=2)
        file.write("\n")
    print(record["arm"],record["kbps"],len(record["paired_ticks"]),record["mean_paired"])
