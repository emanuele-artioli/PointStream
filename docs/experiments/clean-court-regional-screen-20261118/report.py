"""Join verified transmission costs to single-rally regional scores without extrapolation."""
import argparse
import json
from pathlib import Path

import numpy as np
from src.pointstream.evaluation import anchor_sensitivity

REGIONS = ('p1','p2','ball')


def main(folder):
    def read(name):
        return json.loads((folder/name).read_text())

    controls, clean=read('control-scores.json'),read('clean-scores.json')
    budgets, clean_data=read('control-inputs.json'),read('clean-inputs.json')
    assert controls['crops_sha256']==budgets['zip_sha256'] and clean['crops_sha256']==clean_data['zip_sha256']
    assert controls['quality_file_sha256']==clean['quality_file_sha256'] and controls['metric']==clean['metric']
    assert budgets['regions']==clean_data['regions']
    original_receipt = read('joint-composition.json')
    for row in original_receipt['rows']:
        t=str(row['tick'])
        assert budgets['arms']['original'][t]==row['original_rgb_sha256'] and clean_data['arms']['clean'][t]==row['clean_rgb_sha256']
    refined=read('refined-scores.json')
    ball=read('ball-only-scores.json')
    upper=read('upper-scores.json')
    extra_inputs={'refined': read('refined-inputs.json'),'ball':read('ball-only-inputs.json'), 'upper':read('upper-inputs.json')}
    for tag, scores in [('refined', refined), ('ball',ball), ('upper',upper)]:
        assert scores['crops_sha256']==extra_inputs[tag]['zip_sha256'] and scores['metric']==controls['metric']
        assert scores['quality_file_sha256']==controls['quality_file_sha256']
        assert extra_inputs[tag]['source']==budgets['source'] and extra_inputs[tag]['labels']==budgets['labels']
        assert extra_inputs[tag]['regions']==budgets['regions']
    for row in read('refined-composition.json')['rows']:
        assert extra_inputs['refined']['arms']['refined'][str(row['tick'])]==row['clean_rgb_sha256']
    cost={}
    delivery=json.loads((folder.parent/'clean-court-grabcut-20261119/delivery-receipt.json').read_text())
    names={'original':'original', 'broad':'clean','grabcut':'refined'}
    for r in delivery['rows']:
        cost[names[r['name']]]=r['total_bytes']/600
        assert cost[names[r['name']]]==r['kbps']
    ball_receipt=json.loads((folder.parent/'ball-only-screen-20261119/receipt.json').read_text())
    for r in ball_receipt['rows']:
        cost['ball'+str(r['crf'])]=r['total_bytes']/600
        assert np.isclose(cost['ball'+str(r['crf'])],r['total_kbps'])
    for name,r in (budgets['arms'] | extra_inputs['upper']['arms']).items():
        if name!='original':
            assert np.isclose(r['complete_kbps'],r['file_bytes']/600)
            cost[name]=r['file_bytes']/600
    means={}
    for log in (controls,clean,refined,ball,upper):
        for name in log['means']:
            out={}
            for region in REGIONS:
                rows=[r for r in log['rows'] if r['arm']==name and r['region']==region]
                assert [r['t'] for r in rows]==list(range(10,240,20))
                out[region]=sum(r['dists'] for r in rows)/12
                assert np.isclose(out[region],log['means'][name][region])
            means[name]=out
    rows=[dict(arm=k,total_kbps=cost[k],dists=v) for k,v in means.items()]
    comparisons={}
    for name in ('original','clean','refined'):
        comparisons[name]={}
        for family in ('plain','roi'):
            anchors=sorted([k for k in means if k.startswith(family+'-')],key=lambda x:cost[x])
            lower=[k for k in anchors if cost[k]<=cost[name]]
            upper=[k for k in anchors if cost[k]>=cost[name]]
            comparisons[name][family]=dict(lower_anchor=lower[-1] if lower else None,upper_anchor=upper[0] if upper else None,
                lower_rate_anchors_beating_candidate_on_all_three=[k for k in lower if all(means[k][r]<means[name][r] for r in REGIONS)])
    curves={}
    for region in REGIONS:
        candidate=[(cost[k],-means[k][region]) for k in ball['means']]
        anchor=[(cost[k],-means[k][region]) for k in means if k.startswith(('plain-','roi-'))]
        curves[region]=anchor_sensitivity(candidate, anchor)
    report=dict(scope='12 development ticks of one match; no population inference or generated-joint BD-rate',
                cpu_only=True,rows=rows,rate_support=comparisons,ball_channel_bd_rate=curves)
    out=folder/'summary.json'
    with out.open('w') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    print(json.dumps(report['rate_support']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
