from experiments.tennis import w7_tier2_select as select


def test_draw_obeys_split_length_and_is_reproducible():
    # A shot crossing the split belongs to neither half.
    cuts = [100, 400, 700, 57_700, 57_800, 58_100, 58_400]
    text = "".join(f"frame:0 pts_time:{n / 50}\nlavfi.scene_score=0.3\n" for n in cuts)
    fit = select.draw_shots(text, "fit")
    evaluation = select.draw_shots(text, "eval")
    assert fit == select.draw_shots(text, "fit")
    assert all(b <= select.SPLIT and b - a >= 240 for a, b in fit)
    assert all(a >= select.SPLIT and b - a >= 240 for a, b in evaluation)
    assert (57_700, 57_800) not in fit + evaluation
    assert (100, 400) in fit and (57_800, 58_100) in evaluation

