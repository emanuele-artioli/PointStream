"""Plot the preserved known-court-pixel pilot."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def aggregate(rows, key):
    scores = [row['scores'][key] for row in rows]
    return sum(value['sse'] for value in scores) / sum(value['components'] for value in scores)


def plot(source, destination):
    data = json.loads(source.read_text())
    rows = data['rows']
    assert len(rows) == 8
    conditions = ('available', 'half-withheld')
    groups = [[row for row in rows if row['condition'] == condition] for condition in conditions]
    assert all(len(group) == 4 for group in groups)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    x = np.arange(2)
    labels = ('Observed donors', 'Half withheld')
    colors = ('#466379', '#1b8077', '#bd7442')
    for i, (key, label, color) in enumerate([
        ('spatial_all', 'Spatial inpaint', colors[0]),
        ('hybrid_all', 'Aligned donors + hole fill', colors[1]),
    ]):
        values = [aggregate(group, key) for group in groups]
        bars = axes[0].bar(x + (i-.5)*.29, values, .29, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+5, f'{value:.1f}',
                         ha='center', va='bottom', fontsize=9)
    axes[0].set_title('Entire hidden region', loc='left', fontweight='bold')
    axes[0].set_ylabel('RGB mean squared error')
    axes[0].set_ylim(0, 225)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, loc='upper right', fontsize=8)
    for i, (key, label, color) in enumerate([
        ('spatial_seen', 'Spatial inpaint', colors[0]),
        ('temporal_seen', 'Aligned donors only', colors[2]),
        ('hybrid_seen', 'Aligned donors + hole fill', colors[1]),
    ]):
        values = [aggregate(group, key) for group in groups]
        bars = axes[1].bar(x + (i-1)*.22, values, .22, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(bar.get_x()+bar.get_width()/2, value+5, f'{value:.1f}',
                         ha='center', va='bottom', fontsize=9)
    coverage = [100 * sum(r['covered_pixels'] for r in group)/sum(r['mask_pixels'] for r in group)
                for group in groups]
    axes[1].set_title('Pixels observed in donor frames', loc='left', fontweight='bold')
    axes[1].set_xticks(x, [f'{s}\n{v:.1f}% coverage' for s, v in zip(labels, coverage, strict=True)])
    axes[1].set_ylim(0, 275)
    axes[1].legend(frameon=False, loc='upper left', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#d8dfe2', zorder=0, linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    fig.suptitle('Reconstructing known court pixels', x=.08, y=.99, ha='left',
                 fontweight='bold', fontsize=14)
    fig.text(.08, .91, 'Artificial occlusions on four frames of one development rally',
             ha='left', fontsize=10)
    fig.text(.08, .025, 'Right condition simulates missing donor coverage by withholding half the test region.',
             fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.80, wspace=.25)
    for ext in ('pdf', 'png'):
        fig.savefig(destination.with_suffix('.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    plot(a.source, a.destination)
