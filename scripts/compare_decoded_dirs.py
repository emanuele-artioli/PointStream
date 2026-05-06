#!/usr/bin/env python3
"""Compare two decoded output directories (framewise mean pixel difference)."""
from pathlib import Path
import numpy as np
import cv2
import sys


def load_frames(dir_path):
    p = Path(dir_path)
    if not p.exists():
        return []
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in {'.png','.jpg','.jpeg'}])
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        frames.append(img.astype(np.float32))
    return frames


def mean_frame_diff(frames_a, frames_b):
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return None
    means = []
    for i in range(n):
        a = frames_a[i]
        b = frames_b[i]
        if a.shape != b.shape:
            # try resize b to a
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = np.abs(a - b)
        means.append(float(diff.mean()))
    return means


def find_latest_decoded_root(outputs_root='outputs'):
    root = Path(outputs_root)
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    decoded_dir = latest / 'decoded' / '0001'
    if decoded_dir.exists():
        return decoded_dir
    # fallback: search for any decoded dir
    for d in sorted(root.rglob('decoded')):
        maybe = d / '0001'
        if maybe.exists():
            return maybe
    return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: compare_decoded_dirs.py <decoded_dir_a> <decoded_dir_b>')
        print('Or: pass keyword "latest" for auto-detecting most recent decoded run')
        sys.exit(2)
    a = sys.argv[1]
    b = sys.argv[2]
    if a == 'latest':
        a = find_latest_decoded_root() or ''
    if b == 'latest':
        b = find_latest_decoded_root() or ''
    if not a or not b:
        print('Could not resolve decoded directories')
        sys.exit(2)
    fa = load_frames(a)
    fb = load_frames(b)
    means = mean_frame_diff(fa, fb)
    if means is None:
        print('No frames found in one of the directories')
        sys.exit(2)
    for i, m in enumerate(means):
        print(f'frame_{i:02d}: mean_diff={m:.3f}')
    print(f'average_mean_diff={sum(means)/len(means):.3f}')
