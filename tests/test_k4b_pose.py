import json

import numpy as np
import pytest

from experiments.tennis import k4b_pose as kp
from experiments.tennis.k4b_features import key
from src.segmentation.masks import ClipMasks, compress, decompress


def test_crop_grows_by_a_fifth_of_the_height_and_stays_in_the_frame():
    assert kp.crop_box((100., 200., 140., 300.)) == (80, 180, 160, 320)
    assert kp.crop_box((0., 1000., 30., 1080.)) == (0, 984, 46, 1080)


def test_magnify_reports_its_factors_and_caps_the_long_side():
    image = np.zeros((100, 50, 3), np.uint8)
    assert kp.magnify(image, 1)[1] == (1., 1.)
    out, factors = kp.magnify(image, 4)
    assert out.shape[:2] == (400, 200) and factors == (4., 4.)
    out, factors = kp.magnify(np.zeros((700, 300, 3), np.uint8), 4)
    assert max(out.shape[:2]) == 1280 and factors[1] == pytest.approx(1280 / 700)
    assert kp.bucket(out) == 1280 and kp.bucket(np.zeros((10, 10, 3))) == 64


def test_oks_is_one_on_itself_zero_when_missed_and_undefined_without_confident_reference():
    ref = np.zeros((17, 3))
    ref[:, 0], ref[:, 1], ref[:, 2] = np.arange(17), 5., .9
    assert kp.oks(ref.tolist(), ref.tolist(), 400.) == pytest.approx(1.)
    assert kp.oks(None, ref.tolist(), 400.) == 0.
    shifted = ref.copy()
    shifted[:, 0] += 10
    assert 0 < kp.oks(shifted.tolist(), ref.tolist(), 400.) < kp.oks(shifted.tolist(), ref.tolist(), 4000.) < 1
    ref[:, 2] = .1
    assert kp.oks(ref.tolist(), ref.tolist(), 400.) is None


def test_far_is_the_smaller_median_height_and_only_a_tall_near_player_is_reduced():
    boxes = {"a": {0: (0, 0, 10, 300), 8: (0, 0, 10, 310), 9: (0, 0, 10, 290)}, "b": {0: (0, 0, 10, 80), 1: (0, 0, 10, 90)}}
    role = kp.roles(boxes)
    assert role == {"a": "near", "b": "far"}
    plan = kp.synthetic_plan(boxes, role)
    assert plan is not None and plan["player"] == "a" and plan["frames"] == [0, 8]
    assert plan["reduce"] == pytest.approx(85 / 300)
    assert kp.roles({"a": boxes["a"]}) == {"a": "single"}
    short = {"a": {0: (0, 0, 10, 150)}, "b": {0: (0, 0, 10, 80)}}
    assert kp.synthetic_plan(short, kp.roles(short)) is None


class _Tensor:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Result:
    def __init__(self, boxes, kps):
        self.boxes = type("B", (), {"xyxy": _Tensor(boxes)})()
        self.keypoints = type("K", (), {"data": _Tensor(kps)})()


class _Bright:
    """A perfect stand-in detector: the bright pixels' box, keypoints spread inside it."""

    def predict(self, images, **_):
        out = []
        for image in images:
            ys, xs = np.nonzero(image[:, :, 0] > 200)
            if not len(xs):
                out.append(_Result(np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32)))
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            k = np.zeros((1, 17, 3), np.float32)
            k[0, :, 0] = (x0 + x1) / 2
            k[0, :, 1] = np.linspace(y0 + 1, y1 - 2, 17)
            k[0, :, 2] = .9
            out.append(_Result(np.array([[x0, y0, x1, y1]], np.float32), k))
        return out


class _Poses:
    device = "cpu"
    models = {"yolo11x-pose": _Bright()}


def _window(tmp_path):
    clip = ClipMasks(("person", "player", "racket"), 540, 960, 25.)
    frames, dets = [], []
    for t in range(12):
        near = np.zeros((540, 960), bool)
        near[300:420, 400 + t:440 + t] = True  # 240 px tall at 1080p
        far = np.zeros_like(near)
        far[80:110, 600:612] = True  # 60 px
        clip.add(t, "person", 1, near, .9)
        clip.add(t, "player", 7, near, .8)
        clip.add(t, "person", 2, far, .9)
        rgb = np.zeros((1080, 1920, 3), np.uint8)
        rgb[np.repeat(np.repeat(near | far, 2, 0), 2, 1)] = 255
        frames.append(rgb)
        dets.append({"boxes": [[(400 + t) * 2, 600, (440 + t) * 2, 840]], "kps": [[[(420 + t) * 2, 700, .9]] * 17]})
    features = {"aliases": {key("person", 1): key("person", 1), key("player", 7): key("person", 1),
                            key("person", 2): key("person", 2)}}
    out = tmp_path / "publish" / "m__c@5"
    out.mkdir(parents=True)
    clip.save(out / "sam-masks.rle")
    (out / "features.json").write_text(json.dumps(features))
    players = [key("person", 1), key("person", 2)]
    result = kp.analyse_window(_Poses(), lambda t: frames[t], clip, features, players, dets, [5], out)
    (out / "pose-crops.json.zst").write_bytes(compress(json.dumps(result).encode()))
    return out, players, result


def test_window_fits_every_setting_and_the_validator_recomputes_it(tmp_path):
    out, players, result = _window(tmp_path)
    assert result["roles"] == {players[0]: "near", players[1]: "far"}
    assert len(result["rows"]) == 24 and result["crops"]["crop"] == 24 * 3 + 2 * 3
    for row in result["rows"]:
        assert set(row["settings"]) == set(kp.SETTINGS)
        assert all(s["fitted"] and s["inside"] == 1. for name, s in row["settings"].items() if name != "whole" or row["role"] == "near")
        assert row["settings"]["whole"]["fitted"] is (row["role"] == "near")
    synthetic = result["synthetic"]["rows"]
    assert [r["t"] for r in synthetic] == [0, 8] and all(v > .5 for r in synthetic for v in r["oks"].values())
    assert sorted(p.name for p in out.glob("review-*")) == ["review-t005-far.png", "review-t005-near.png"]
    assert kp.recompute(out, players) == []
    summary = kp.summarize([result], {})
    assert summary["table"]["crop1"]["far"]["fitted_share"] == 1. and summary["table"]["whole"]["far"]["fitted_share"] == 0.
    assert summary["chosen_scale"] == 1


def test_validator_catches_edited_keypoints(tmp_path):
    out, players, _ = _window(tmp_path)
    result = json.loads(decompress((out / "pose-crops.json.zst").read_bytes()))
    result["rows"][3]["settings"]["crop2"]["kps"][0][0] += 400
    result["synthetic"]["rows"][0]["oks"]["synthetic4"] = .123
    (out / "pose-crops.json.zst").write_bytes(compress(json.dumps(result).encode()))
    problems = kp.recompute(out, players)
    assert any("crop2: fit" in p for p in problems) and any("oks4" in p for p in problems)


def test_local_dilation_matches_k4s_whole_frame_dilation():
    from experiments.tennis import k4

    rng = np.random.default_rng(0)
    mask = np.zeros((1080, 1920), bool)
    mask[3:60, 1890:1917] = True
    mask[500:700, 900:960] = rng.random((200, 60)) > .3
    points = np.column_stack([rng.uniform(-20, 1940, 4000), rng.uniform(-20, 1100, 4000), np.ones(4000)])
    grown = k4.dilate(mask, k4.KP_DILATE_1080)
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, 1919)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, 1079)
    assert (kp.within(mask, points) == grown[ys, xs]).all()
