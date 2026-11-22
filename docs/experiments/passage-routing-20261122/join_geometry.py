"""Join completed geometry and partial smoke spans to frozen motion and camera labels."""
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def main():
    original=json.loads((P/'routing.json').read_text())
    records=[]
    for name in ('rallies','crowd_point','crowd_interlude'):
        path=P/(name+'-result.json')
        if not path.exists():
            continue
        data=json.loads(path.read_text())
        for key,record in data['records'].items():
            frames=[]
            for part in record['parts']:
                for tick in range(part['start'],part['end']):
                    motion=next(x['route'] for x in original['motion'] if x['start']<=tick<x['end'])
                    label=next(x['label'] for x in original['human'] if x['start']<=tick<x['end'])
                    frames.append((part['predicted'],motion,label))
            if len(frames)!=record['frames']:
                raise ValueError('geometry span mismatch')
            eligible=sum(g=='court' for g,_,_ in frames)
            point=sum(p=='point' for _,p,_ in frames)
            both=sum(g=='court' and p=='point' for g,p,_ in frames)
            records.append({'interval':key,'sampled_frames':len(frames),'start':record['parts'][0]['start'],
                            'end':record['parts'][-1]['end'],'geometry_court_frames':eligible,
                            'motion_point_frames':point,'both_frames':both,
                            'human_court_frames':sum(h=='court' for _,_,h in frames),
                            'human_crowd_frames':sum(h=='crowd' for _,_,h in frames),
                            'human_unreviewed_frames':sum(h=='unreviewed' for _,_,h in frames),
                            'partial':name=='crowd_interlude'})
    (P/'geometry-joined.json').write_text(json.dumps({'fps':50,'records':records},indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
