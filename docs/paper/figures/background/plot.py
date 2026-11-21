"""Plot the preserved known-court-pixel pilot."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def aggregate(rows, key):
    scores = [row['scores'][key] for row in rows]
    return sum(value['sse'] for value in scores) / sum(value['components'] for value in scores)


def plot(source, destination):
    data = json.loads(source.read_text())
    rows = data['rows']
    assert len(rows) == 8
    available = [row for row in rows if row['condition'] == 'available']
    withheld = [row for row in rows if row['condition'] == 'half-withheld']
    assert len(available) == len(withheld) == 4
    baseline = aggregate(available, 'spatial_all')
    assert baseline == aggregate(withheld, 'spatial_all')
    values = [baseline, aggregate(available, 'hybrid_all'), aggregate(withheld, 'hybrid_all')]
    coverage = [100 * sum(r['covered_pixels'] for r in group) / sum(r['mask_pixels'] for r in group)
                for group in (available, withheld)]

    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = [2, 1, 0]
    labels = ['Spatial inpainting', 'Donors + hole filling\nObserved donors',
              'Donors + hole filling\nHalf of observations withheld']
    colors = ['#6e6e6e', '#0072b2', '#56b4e9']
    bars = ax.barh(y, values, color=colors, height=.57)
    for i, (bar, value) in enumerate(zip(bars, values, strict=True)):
        extra = '' if i == 0 else f'   ({coverage[i-1]:.1f}% donor coverage)'
        ax.text(value + 3, bar.get_y() + bar.get_height()/2, f'{value:.1f}{extra}',
                va='center', fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlabel('RGB mean squared error over the entire hidden region (lower is better)')
    ax.set_xlim(0, 220)
    ax.grid(axis='x', color='#d8dfe2', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=10)
    fig.suptitle('Recovering artificially hidden court pixels', x=.03, y=.98, ha='left',
                 fontweight='bold', fontsize=14)
    fig.text(.03, .87, 'Four frames from one development rally; same hidden pixels in every row', fontsize=10)
    fig.text(.03, .025, 'The last row simulates missing coverage. Spatial inpainting fills donor gaps.', fontsize=9)
    fig.subplots_adjust(left=.34, right=.97, bottom=.22, top=.76)
    for ext in ('pdf', 'png'):
        fig.savefig(destination.with_suffix('.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    plot(args.source, args.destination)
