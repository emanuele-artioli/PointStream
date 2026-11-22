"""Show actual per-component delivered bytes across three complete receivers."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[2]/"experiments"


def read(path):
    return json.loads(path.read_text())


def main():
    serve=read(EXP/"joint-trajectory-20261122/result.json")["streams"]["20"]
    al=read(HERE/"al028.json")
    re=read(EXP/"reference-transfer-al028-20261122/wire-price-and-quality.json")
    med=read(EXP/"med-mixed-receiver-20261122/000-packets.json")
    m=next(r for r in med["rates"] if r["crf"]==med["selected_crf"])
    rows=[
        ("Nadal serve",4.8,serve["bytes"],serve["reference_bytes"],serve["base_bytes"],serve["pose_bytes"],serve["ball_bytes"]),
        ("Alcaraz rally",9.6,al["total"],re["actual_original_png_init_bytes"],al["parts"]["base"],al["parts"]["poses"],al["parts"]["trajectory"]),
        ("Medvedev rally",4.8,m["joint_bytes"],m["init_bytes"],m["base_bytes"],m["pose_bytes"],m["ball_bytes"]),
    ]
    fig,ax=plt.subplots(figsize=(6.4,2.65),layout="constrained")
    colors=["#506882","#d39127","#228778","#dedde2"]
    names=["Coded video","Player poses","Ball","Framing"]
    hatches=["","///","..","\\"]
    packed=[]
    for y,(shot,duration,total,init,base,poses,ball) in enumerate(rows):
        parts=[base,poses,ball,total-base-poses-ball]
        assert all(v>0 for v in parts)
        cost=total*8/duration/1000
        cold=(total+init)*8/duration/1000
        packed.append(dict(shot=shot,seconds=duration,changing_bytes=total,initialization_bytes=init,parts=dict(zip(names,parts,strict=True)),warm_kbps=cost,cold_kbps=cold))
        left=0
        for v,c,h in zip(parts,colors,hatches,strict=True):
            dx=v*8/duration/1000
            ax.barh(y,dx,left=left,color=c,edgecolor="#243340",linewidth=.45,hatch=h,height=.45)
            left+=dx
        ax.text(25.1,y,f"{cost:5.1f}",ha="center",va="center",fontsize=9)
        ax.text(30.5,y,f"{cold:5.1f}",ha="center",va="center",fontsize=9)
    ax.text(25.1,-.48,"Warm",ha="center",fontweight="bold",fontsize=9)
    ax.text(30.5,-.48,"Cold",ha="center",fontweight="bold",fontsize=9)
    ax.set(yticks=range(3),yticklabels=[r[0] for r in rows],ylim=(2.5,-.75),xlim=(0,32.5),xticks=[0,5,10,15,20],
           xlabel="Bitrate in kbps over the displayed rally")
    ax.spines[["top","left","right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0,20)
    ax.tick_params(axis="y",length=0)
    handles=[Patch(facecolor=c,edgecolor="#243340",linewidth=.45,hatch=h,label=name) for c,h,name in zip(colors,hatches,names,strict=True)]
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.45,.98),frameon=False,ncol=4,fontsize=8,handlelength=1.4,columnspacing=.9)
    fig.savefig(HERE/"stream-cost.pdf",metadata={"CreationDate":None})
    (HERE/"rows.json").write_text(json.dumps(packed,indent=2)+"\n")


if __name__=="__main__":
    main()
