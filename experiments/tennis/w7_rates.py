"""W7: per-window rate inputs derived from the generalised stages (CPU, no quality).

`rates(transport)` -> the `--rates` file of `k6b_real_base_judge` for manifest windows (keypoint packet, bank and adapter terms);
`window_state(...)` -> the `--window-state` entry of the w1_* anchor stages. Policy (docs/experiments/w7-untouched.md, 2026-10-30):
the reference recipe is `window_frame0` (two crops per window, coded and billed as before). The old bank charge was the PNG bytes
of 12 crops over 7200 s; a window has two crops, so its bank term is its own PNG bytes times `BANK_CROPS / 2` (a 12-crop bank of
crops of the same size) over 7200 s, to keep the charge comparable to the old windows. This scaling is a reading to be confirmed.
"""
from __future__ import annotations

from typing import Any

AMORTISED_SECONDS = 7200
BANK_CROPS = 12
#: adapter + guider bytes billed per match (k6b-players.md, 2026-10-25 accounting), amortised over 7200 s
ADAPTER_BYTES = {'alcaraz_hurkacz': 1_488_252, 'medvedev_struff': 1_495_592}
START_BASE = {4: 60, 2: 67, 1: 70}  # first CRF probes only; never read as a result


def kbps(n_bytes: float, seconds: float) -> float:
    return n_bytes * 8 / seconds / 1000


def rates(transport: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in transport['rows']:
        fps = row['keypoints']['contract']['fps'][0]
        crops = len(row['reference_files'])
        if crops == 0:
            raise ValueError(f"{row['window']}: no reference crops")
        bank_bytes = row['references']['png_bytes'] * BANK_CROPS / crops
        out[row['window']] = {'keypoint_kbps': kbps(row['keypoints']['bytes'], 240 / fps),
                              'match_bank_kbps_7200': kbps(bank_bytes, AMORTISED_SECONDS),
                              'adapter_guider_kbps_7200': kbps(ADAPTER_BYTES[row['match']], AMORTISED_SECONDS),
                              # worst-case sensitivity: the window's own crops billed over the 240-frame window alone
                              'bank_worst_case_kbps': kbps(row['references']['png_bytes'], 240 / fps)}
    return out


def window_state(base_rows: list[dict[str, Any]], transport_row: dict[str, Any], name: str) -> dict[str, Any]:
    """`w1_roi.window_from_spec` state: plain CRF 63 base bytes per scale (native base manifest), packet sizes (transport)."""
    ivf = {int(r['scale']): int(r['ivf']['bytes']) for r in base_rows if r['window'] == name}
    if sorted(ivf) != [2, 4, 8]:
        raise ValueError(f'{name}: native base needs scales 8, 4 and 2')
    return {'control_ivf_bytes': ivf, 'start_base': dict(START_BASE), 'kps1_bytes': int(transport_row['keypoints']['kps1_bytes']),
            'kps2_bytes': int(transport_row['keypoints']['bytes']), 'hybrid_archive_sha': None}
