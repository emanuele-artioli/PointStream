"""Render independently accepted repaired W7 results; no model or metric is run.

python -m experiments.tennis.w7_repaired_figure --input FILE --sha256 HASH --out-dir NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(args.sha256) != 64 or any(ch not in '0123456789abcdef' for ch in args.sha256):
        raise ValueError('expected full lowercase SHA256')
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError('input checksum mismatch')
    result = json.loads(raw)
    data = [result['tier1'], result['tier2']]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                                'svg.fonttype': 'none', 'svg.hashsalt': 'pointstream-w7-repaired-20261031'})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.25), sharey=True, layout='constrained')
    y_positions = [11, 10, 9, 8, 7, 6, 3.5, 2.5, 1.5]
    names = ['al028', 'al034', 'al043', 'med000', 'med024', 'med019', 'nad000', 'nad001', 'nad002']
    rows = {r['window']: r for tier in data for r in tier['windows']}
    labels = [f"{n}  ({len(rows[n]['anchors_used'])}/4)" for n in names]
    values = [rows[n]['gap'][metric]['delta'] for n in names for metric in ('lpips', 'dists')]
    bounds = [x for tier in data for metric in ('lpips', 'dists')
              for x in (tier['pooled'][metric]['lo'], tier['pooled'][metric]['hi'])]
    left, right = min(0.0, *values, *bounds), max(0.0, *values, *bounds)
    pad = max((right - left) * .12, .01)
    for ax, metric in zip(axes, ('lpips', 'dists')):
        ax.axvline(0, color='0.3', linewidth=.9, linestyle='--', zorder=1)
        ax.axhline(4.25, color='0.82', linewidth=.7)
        for y, n in zip(y_positions, names):
            ax.plot(rows[n]['gap'][metric]['delta'], y, 'o', color='0.38', markersize=3.8, zorder=3)
        for y, tier_result in ((5, data[0]), (.4, data[1])):
            s = tier_result['pooled'][metric]
            ax.plot([s['lo'], s['hi']], [y, y], color='black', linewidth=1.4, zorder=4)
            ax.plot([s['lo'], s['hi']], [y, y], '|', color='black', markersize=7, markeredgewidth=1)
            ax.plot(s['mean'], y, 'D', color='black', markersize=4.5, zorder=5)
        ax.set_title(metric.upper(), fontweight='bold', fontsize=9)
        ax.set_xlim(left - pad, right + pad)
        ax.set_xlabel('Player gap (anchor − hybrid)')
        ax.grid(axis='x', color='0.9', linewidth=.6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-.2, 11.55)
        ax.set_yticks(y_positions + [5, .4], labels + ['Tier 1 mean (n=6)', 'Tier 2 mean (n=3)'])
    axes[0].set_ylabel('Window (resolved anchor families / 4)')
    fig.legend(handles=[Line2D([0], [0], marker='o', color='0.38', linestyle='', markersize=4, label='Window'),
                        Line2D([0], [0], marker='D', color='black', linewidth=1.2, markersize=4, label='Mean and 95% t interval')],
               loc='outside lower center', ncol=2, frameon=False, fontsize=8)
    fig.savefig(out / 'w7-repaired.pdf', metadata={'Creator': 'PointStream W7 repaired', 'CreationDate': None, 'ModDate': None})
    fig.savefig(out / 'w7-repaired.svg', metadata={'Creator': 'PointStream W7 repaired', 'Date': None})
    fig.savefig(out / 'w7-repaired.png', dpi=180, metadata={'Software': 'PointStream W7 repaired'})
    plt.close(fig)


if __name__ == '__main__':
    main()
