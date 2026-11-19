"""Plot the frozen single-rally comparisons from the verified numeric summary."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot(summary, output):
    rows=json.loads(summary.read_text())['rows']
    fig, axes=plt.subplots(1,3,sharex=True,sharey=True,figsize=(11.7,4.7))
    for ax, region, title in zip(axes, ('p1','p2','ball'), ('Player 1','Player 2','Ball crop'),strict=True):
        for prefix,label,color,style,marker in [('plain-','Plain AV1','#4c5969','-','o'),('roi-','Player-and-ball ROI AV1','#738391','--','s'),
                                              ('ball','AV1 + tracked ball tiles','#117b72','-','o')]:
            pts=sorted((r['total_kbps'],r['dists'][region]) for r in rows if r['arm'].startswith(prefix))
            ax.plot(*zip(*pts),color=color,linestyle=style,marker=marker,markersize=4,label=label)
        for arm,label,color,marker in [('original','Generated players + ball, original base','#aa4e22','D'),
                                       ('refined','Generated players + ball, GrabCut base','#bb3845','P'),
                                       ('clean','Generated players + ball, broad-mask base','#9a668a','x')]:
            r=next(r for r in rows if r['arm']==arm)
            ax.scatter(r['total_kbps'],r['dists'][region],s=65,color=color,marker=marker,label=label,zorder=5)
        ax.set_title(title,fontsize=12,fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks([125,250,500,1000,2000])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.minorticks_off()
        ax.grid(axis='y',color='#d9dee0',linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('DISTS (lower is better)')
    fig.text(.5,.295,'Complete delivery rate (kbps, log scale)',ha='center',fontsize=10)
    fig.suptitle('Foreground quality on the Nadal–Kyrgios development rally',x=.075,ha='left',fontsize=16,fontweight='bold')
    fig.text(.075,.87,'12 source-reviewed frames  ·  4.8-second cold start  ·  players scored on body masks, ball on a 64 × 64 crop',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,.025))
    fig.subplots_adjust(left=.075,right=.99,top=.755,bottom=.4,wspace=.20)
    for ext in ('png','pdf'):
        fig.savefig(output.with_suffix('.'+ext),bbox_inches='tight',dpi=165)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    plot(args.summary,args.output)
