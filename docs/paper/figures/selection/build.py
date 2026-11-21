"""Build compatible rate–quality plots and measured bandwidth choices from receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[4]
ITEM = Path(__file__).resolve().parent
REGIONS = ('p1', 'p2', 'ball')
TITLES = ('Player 1 body', 'Player 2 body', 'Ball crop')
COLORS = {'plain': '#536474', 'roi': '#8b65a1', 'ball': '#007e80', 'original': '#d27e25',
          'clean': '#bb5663', 'refined': '#ca3b41'}
NAMES = {'plain': 'SVT-AV1', 'roi': 'Player-and-ball ROI AV1', 'ball': 'AV1 + tracked ball tiles',
         'original': 'Generated players + ball, original base', 'clean': 'Generated players + ball, broad cleanup',
         'refined': 'Generated players + ball, tighter cleanup'}


def read(relative, ledger):
    path = ROOT / relative
    raw = path.read_bytes()
    ledger.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)


def family(arm):
    for name in ('plain', 'roi', 'ball'):
        if arm.startswith(name):
            return name
    return arm


def choices(rows, region):
    """Choose from actual stored encodes whose complete byte rate fits the budget."""
    result = []
    best = None
    for row in sorted(rows, key=lambda r: (r['total_kbps'], r['arm'])):
        if best is None or row['dists'][region] < best['dists'][region]:
            best = row
            result.append(dict(minimum_budget_kbps=row['total_kbps'], arm=row['arm'],
                               dists=row['dists'][region], family=family(row['arm'])))
    return result


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dce1e3', linewidth=.7)
    ax.set_xscale('log')
    ax.set_xticks([125, 250, 500, 1000, 2000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(118, 2200)


def save(fig, output, name):
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def cold_plot(rows, selected, output):
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.7), sharex=True, sharey=True)
    for ax, region, title in zip(axes, REGIONS, TITLES, strict=True):
        for name in ('plain', 'roi', 'ball'):
            points = sorted((r for r in rows if family(r['arm']) == name), key=lambda r: r['total_kbps'])
            ax.plot([r['total_kbps'] for r in points], [r['dists'][region] for r in points],
                    color=COLORS[name], marker='o', markersize=4, lw=1.3, label=NAMES[name])
        for name, marker in [('original', 'D'), ('clean', 'x'), ('refined', 'P')]:
            row, = [r for r in rows if r['arm'] == name]
            ax.scatter(row['total_kbps'], row['dists'][region], color=COLORS[name], marker=marker,
                       s=50, zorder=5, label=NAMES[name])
        picked = selected[region]
        xs = [v['minimum_budget_kbps'] for v in picked] + [2200]
        ys = [v['dists'] for v in picked] + [picked[-1]['dists']]
        ax.step(xs, ys, where='post', color='#24292d', alpha=.6, linestyle=':', linewidth=1.2,
                label='Best stored decode within budget')
        style(ax)
        ax.set_title(title, fontsize=12, pad=10)
    axes[0].set_ylabel('DISTS (lower is better)')
    fig.suptitle('Measured codec choices across the bitrate range', x=.075, ha='left', fontsize=16, fontweight='bold')
    fig.text(.075, .875, 'Nadal–Kyrgios development rally · 240 frames at 50 fps · full 4.8-second cold-start delivery', fontsize=10)
    fig.text(.5, .285, 'Complete delivery bitrate (kbps, log scale)', ha='center', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.075, right=.99, top=.76, bottom=.40, wspace=.18)
    save(fig, output, 'rate-quality')


def policy_plot(selected, output):
    fig, ax = plt.subplots(figsize=(11.7, 3.5))
    for y, (region, title) in enumerate(zip(REGIONS, TITLES, strict=True)):
        picked = selected[region]
        for i, row in enumerate(picked):
            start = row['minimum_budget_kbps']
            end = picked[i + 1]['minimum_budget_kbps'] if i + 1 < len(picked) else 2200
            ax.fill_betweenx([y-.28, y+.28], start, end, color=COLORS[row['family']], linewidth=0)
            ax.plot([start, start], [y-.28, y+.28], color='white', lw=.6)
    style(ax)
    ax.grid(False)
    ax.set_yticks(range(3), TITLES)
    ax.invert_yaxis()
    ax.set_xlabel('Available bitrate (kbps, log scale)', labelpad=10)
    ax.set_title('Which stored decode gives the lowest object distortion?', loc='left', fontsize=15,
                 fontweight='bold', pad=44)
    fig.text(.18, .83, 'Each choice uses measured complete bytes and the named object metric.', fontsize=10)
    ax.legend(handles=[Patch(color=COLORS[n], label=NAMES[n]) for n in ('plain','roi','ball')],
              loc='lower center', bbox_to_anchor=(.5,-.66), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.18,right=.99,top=.70,bottom=.36)
    save(fig, output, 'bandwidth-choice')


def low_plot(dcvc, historical, output):
    points = sorted(dcvc, key=lambda r: r[1])
    fig, axes = plt.subplots(1,2,figsize=(8.4,3.7),sharex=True)
    player = historical['hybrid']['dists']['foreground']
    whole = historical['hybrid']['dists']['whole']
    for ax, title, col, quality in zip(axes, ('Player region','Whole frame'), (5,3), (player,whole), strict=True):
        ax.plot([p[1] for p in points],[p[col] for p in points], color='#536474',marker='o',lw=1.5,label='DCVC-UF, measured encodes')
        ax.scatter(historical['rate_kbps'],quality,s=70,color='#007e80',marker='D',zorder=4,
                   label='PointStream, two-hour modeled reuse')
        ax.set_title(title,fontsize=12)
        ax.set_xlabel('Bitrate (kbps)')
        ax.set_xlim(6,30)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#dce1e3',lw=.7)
    axes[0].set_ylabel('DISTS (lower is better)')
    fig.suptitle('Earlier low-rate player comparison', x=.09,ha='left',fontsize=15,fontweight='bold')
    fig.text(.09,.87,'Alcaraz–Hurkacz 028 · 80 sampled ticks from 240 frames at 25 fps',fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(.5,.01))
    fig.subplots_adjust(left=.09,right=.99,top=.73,bottom=.29,wspace=.23)
    save(fig,output,'historical-low-rate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ITEM)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for source in json.loads((ITEM/'sources.json').read_text()):
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256'], source['path']
    ledger=[]
    summary=read('docs/experiments/clean-court-regional-screen-20261118/summary.json',ledger)
    control=read('docs/experiments/clean-court-regional-screen-20261118/control-scores.json',ledger)
    matched=read('docs/experiments/matched-ball-rate-20261122/scores.json',ledger)
    cost=read('docs/experiments/matched-ball-rate-20261122/render-result.json',ledger)
    control_inputs=read('docs/experiments/clean-court-regional-screen-20261118/control-inputs.json',ledger)
    matched_inputs=read('docs/experiments/matched-ball-rate-20261122/crop-inputs.json',ledger)
    prior=read('docs/experiments/receiver-replay-20261118/selected-rate-support/prior-dcvc.json',ledger)
    dcvc=read('docs/experiments/w7-specs/dcvc-points-al028-scale2.json',ledger)
    historical=read('docs/experiments/w9-inputs/w7-repaired-2h.json',ledger)['tier1']['windows'][0]
    assert historical['window']=='al028'
    assert all(control_inputs[k]==matched_inputs[k] for k in ('source','labels','regions'))
    assert control['crops_sha256']==control_inputs['zip_sha256']
    assert matched['crops_sha256']==matched_inputs['zip_sha256']
    assert control['metric']==matched['metric'] and control['quality_file_sha256']==matched['quality_file_sha256']
    rows=summary['rows']+[dict(arm='ball-matched50',total_kbps=cost['candidate_kbps'],dists=matched['means']['matched'])]
    assert len({r['arm'] for r in rows})==len(rows)
    assert all(np.isfinite(r['total_kbps']) and r['total_kbps']>0 for r in rows)
    selected={r:choices(rows,r) for r in REGIONS}
    data=dict(scope='One opened cold-start rally, 12 reviewed native RGB instants, three object metrics.',
              protocol=dict(frames=240,fps=50,seconds=4.8,review_ticks=list(range(10,240,20)),
                            player_support='reviewed body mask, bbox plus 32 pixels context',
                            ball_support='64px crop around source-reviewed center',
                            rate='complete stored delivery, cold reference images paid once'),
              source_files=ledger,rows=rows,budget_choices=selected,
              cold_dcvc_excluded=dict(job=prior['historical_job'],reason=prior['scope'],
                                     scored_ticks=prior['scored_ticks']),
              historical=dict(source='alcaraz_hurkacz/028',frames=240,fps=25,scored_frames=80,
                              modeled_kbps=historical['rate_kbps'],dcvc_points=dcvc,
                              player_dists=historical['hybrid']['dists']['foreground'],whole_dists=historical['hybrid']['dists']['whole']))
    (args.output/'data.json').write_text(json.dumps(data,indent=2)+'\n')
    cold_plot(rows,selected,args.output)
    policy_plot(selected,args.output)
    low_plot(dcvc,historical,args.output)
    print(json.dumps(dict(points=len(rows),cold_dcvc_compatible=False,budget_choice_counts={r:len(selected[r]) for r in REGIONS},output=str(args.output))))


if __name__=='__main__':
    main()
