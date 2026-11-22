"""Same seven W7 windows and equal match weights across each supported rate."""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = {'al': ('al028', 'al034', 'al043'), 'med': ('med019', 'med024'), 'nad': ('nad000', 'nad002')}


def interp(points, rate):
    points = sorted((float(a), float(b)) for a, b in points)
    for (a, x), (b, y) in zip(points, points[1:]):
        if a <= rate <= b and b > a:
            return x + (y-x) * math.log(rate/a) / math.log(b/a)
    return None


def pooled(windows, method, rate):
    scores = []
    for names in MATCHES.values():
        match = []
        for name in names:
            row = windows[name]
            if method == 'ordinary AV1':
                choices = [v for points in row['av1'].values() if (v := interp(points, rate)) is not None]
                value = min(choices) if choices else None
            else:
                value = interp(row['pointstream' if method == 'PointStream' else 'dcvc'], rate)
            if value is None:
                return None
            match.append(value)
        scores.append(float(np.mean(match)))
    return float(np.mean(scores))


def main():
    windows = {w['window']: w for w in json.loads((HERE/'nine-summary.json').read_text())['windows']}
    rates = np.geomspace(7, 100, 1200)
    methods = ('PointStream', 'ordinary AV1', 'DCVC-UF')
    scores = {method: [pooled(windows, method, float(r)) for r in rates] for method in methods}
    common = [all(scores[m][i] is not None for m in methods) for i in range(len(rates))]
    output = {'method': 'log-rate interpolation inside each measured scale-family only, best supported ordinary family, equally average windows per match then three matches (NAD001 omitted from all methods); 80 sampled ticks/window; 2-hour modeled asset reuse', 'points_33': {m: pooled(windows, m, 33.) for m in methods}, 'common_support_sampled_kbps': [float(rates[i]) for i, b in enumerate(common) if b]}
    (HERE/'pooled-curve-summary.json').write_text(json.dumps(output, indent=2)+'\n')
    if not all(value is not None for value in output['points_33'].values()):
        print('33 kbps incomplete', output['points_33'])
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for method, color, marker in zip(methods, ['#0072B2','#D55E00','#009E73'], ['o','s','^']):
        y = np.array([np.nan if v is None else v for v in scores[method]])
        ax.plot(rates, y, label=method, color=color, lw=2)
        if output['points_33'][method] is not None:
            ax.scatter([33], [output['points_33'][method]], marker=marker, color=color, s=28, zorder=3)
    if any(common):
        ax.fill_between(rates, 0, 1, where=common, transform=ax.get_xaxis_transform(), color='#777777', alpha=.14, label='All three supported')
    else:
        ax.text(.98, .96, 'DCVC-UF: no 7-window common support', ha='right', va='top', transform=ax.transAxes, fontsize=8)
    ax.axvline(33, color='#555555', lw=.8, ls=':')
    finite = [float(rates[i]) for i in range(len(rates)) if any(scores[m][i] is not None for m in methods)]
    ax.set(xscale='log', xlim=(min(finite)*.97, max(finite)*1.03), xlabel='Total rate (kbps); PointStream assets modeled over 2 h reuse', ylabel='Player-region DISTS ↓', title='Seven player windows: equal-match average')
    ticks = [v for v in (20, 25, 30, 33, 40, 50, 60) if min(finite)*.97 <= v <= max(finite)*1.03]
    ax.set_xticks(ticks, labels=[str(v) for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(.02, .03, 'Dots: interpolated at 33 kbps', transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE/'pooled-curve.pdf', bbox_inches='tight')
    fig.savefig(HERE/'pooled-curve.png', dpi=180, bbox_inches='tight')
    print('33 kbps',output['points_33'])


if __name__ == '__main__':
    main()
