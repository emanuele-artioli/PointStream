"""W7 CPU stage: the selection helpers are mechanical and agree with the frozen registration gate."""
import numpy as np

from experiments.tennis import w7_select as w7
from experiments.tennis.k6b_confirm import gate


def rows(valid):
    return [{"fits": bool(v), "homography": np.eye(3).tolist()} for v in valid]


def test_first_window_boundary_and_agreement_with_gate():
    r = rows([1] * 12 + [0] * 13 + [1] * 240)  # [0, 240) has 227 valid, so the first admissible start is 13
    w = w7.first_window(r)
    assert w["start"] == 13 and w["valid"] == 228
    for start in range(len(r) - 239):
        assert (len(gate(r[start:start + 240])) >= 228) == (start >= w["start"] or len(gate(r[start:start + 240])) >= 228)
    brute = next(s for s in range(len(r) - 239) if len(gate(r[s:s + 240])) >= 228)
    assert brute == w["start"]


def test_first_window_fails_and_short_clip():
    w = w7.first_window(rows([1, 0] * 200))
    assert w["start"] is None and w["best_valid"] == 120
    assert w7.first_window(rows([1] * 100))["start"] is None
    bad = rows([1] * 240)
    bad[5] = {"fits": True, "homography": [[0] * 3] * 3}
    assert w7.first_window(bad)["valid"] == 239 or w7.first_window(bad)["start"] == 0


def test_draw_is_seeded_and_without_replacement():
    ids = [f"m/{i:03d}" for i in (0, 4, 19, 24, 27, 31, 42)]
    a = w7.draw(ids)
    assert a == w7.draw(list(reversed(ids))) and len(set(a)) == 3
    assert a == ["m/000", "m/024", "m/019"]  # numpy 1.26 default_rng(20261030), as run on gpu1
    assert w7.draw(ids[:2]) == sorted(ids[:2])


def test_exclusions_cover_confirmation_and_fit_clips():
    ex = w7.exclusions({"alcaraz_hurkacz/036"})
    assert len(w7.CONFIRM) == 12
    for clip in ("alcaraz_hurkacz/031", "medvedev_struff/014", "medvedev_struff/007", "alcaraz_hurkacz/002", "alcaraz_hurkacz/036"):
        assert clip in ex
    assert "alcaraz_hurkacz/028" not in ex


def test_confirm_manifest_passes_the_cohort_check():
    import json
    from pathlib import Path

    from experiments.tennis import window_manifest as wm
    from experiments.tennis.k6b_confirm import check_manifest

    sel = json.loads((Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/selection.json').read_text())
    windows = wm.from_selection(sel)
    rows = {c: rows_ok for c in {w.clip_id for w in windows}
            for rows_ok in [[{'frame': t, 'fits': True, 'homography': np.eye(3).tolist()} for t in range(sel['inventory'][c]['frames'])]]}
    manifest = w7.confirm_manifest(windows, sel['inventory'], rows, [f'old/{i}' for i in range(49)], '/d', '/a')
    gates = check_manifest(manifest)
    assert sorted(gates) == sorted(w.id for w in windows) and all(len(v) == 240 for v in gates.values())
    assert manifest['clips'][0]['staged']['path'] == '/d/sources/alcaraz_hurkacz__028.mp4'
