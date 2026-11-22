"""Eight-window equal-match descriptive comparison at 33 kbps (no extrapolation)."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TARGET = 33.0
MATCHES = {'al': ('al028', 'al034', 'al043'), 'med': ('med019', 'med024'), 'nad': ('nad000', 'nad001', 'nad002')}


def interpolate(points):
    pts = sorted((float(x), float(y)) for x, y in points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= TARGET <= x1 and x1 > x0:
            f = math.log(TARGET / x0) / math.log(x1 / x0)
            return y0 + f * (y1 - y0)
    return None


def main():
    windows = {w['window']: w for w in json.loads((HERE / 'nine-summary.json').read_text())['windows']}
    values = {}
    missing = {}
    for match, names in MATCHES.items():
        for name in names:
            w = windows[name]
            av1 = [x for pts in w['av1'].values() if (x := interpolate(pts)) is not None]
            row = {'PointStream': interpolate(w['pointstream']), 'ordinary AV1': min(av1) if av1 else None, 'DCVC-UF': interpolate(w['dcvc'])}
            values[name] = row
            absent = [key for key, value in row.items() if value is None]
            if absent:
                missing[name] = absent
    output = {'rate_kbps': TARGET, 'per_window': values, 'missing': missing, 'method': 'linear quality interpolation in log rate within each measured scale family; lowest supported ordinary AV1 family; no extrapolation; average windows within each match, then equal-weight matches'}
    if missing:
        (HERE/'at33-summary.json').write_text(json.dumps(output, indent=2)+'\n')
        print('missing', missing)
        return
    methods = list(next(iter(values.values())))
    per_match = {match: {m: float(np.mean([values[n][m] for n in names])) for m in methods} for match, names in MATCHES.items()}
    overall = {m: float(np.mean([per_match[k][m] for k in MATCHES])) for m in methods}
    output.update(per_match=per_match, equal_match_mean=overall)
    (HERE/'at33-summary.json').write_text(json.dumps(output, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    xs = np.arange(4)
    colors = ['#0072B2', '#D55E00', '#009E73']
    for i, (method, color) in enumerate(zip(methods, colors)):
        ys = [per_match[k][method] for k in MATCHES] + [overall[method]]
        ax.bar(xs + (i-1)*.23, ys, width=.22, label=method, color=color)
    ax.set_xticks(xs, ['Alcaraz–Hurkacz\n3 clips', 'Medvedev–Struff\n2 clips', 'Nadal–Kyrgios\n3 clips', 'Equal-match\nmean'])
    ax.set_ylabel('Player-region DISTS ↓')
    ax.set_title('33 kbps: sampled player regions')
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/'at33.pdf', bbox_inches='tight')
    fig.savefig(HERE/'at33.png', dpi=160, bbox_inches='tight')
    print(json.dumps(output['equal_match_mean']))


if __name__ == '__main__':
    main()
