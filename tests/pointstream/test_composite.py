import numpy as np

from src.pointstream import composite as cp


def square_classes(h: int = 64, w: int = 80) -> np.ndarray:
    classes = np.zeros((h, w), np.uint8)
    classes[20:40, 20:30] = cp.HAND
    classes[20:40, 30:45] = cp.OBJECT
    return classes


def test_alpha_is_one_on_f_and_falls_over_the_ring() -> None:
    classes = square_classes()
    alpha, nearest = cp.alpha_and_classes(classes)
    f = classes > 0
    assert (alpha[f] == 1).all()
    assert alpha[30, 19] < 1 and alpha[30, 19] > alpha[30, 15] > 0  # 1 and 5 px left of F
    assert alpha[30, 20 - cp.RING - 1] == 0 and alpha[0, 0] == 0
    assert nearest[30, 15] == cp.HAND and nearest[30, 50] == cp.OBJECT
    assert (nearest[alpha == 0] == 0).all() and (nearest[f] == classes[f]).all()


def test_empty_foreground_shows_only_background() -> None:
    alpha, nearest = cp.alpha_and_classes(np.zeros((8, 8), np.uint8))
    assert not alpha.any() and not nearest.any()


def test_composite_blends_by_alpha() -> None:
    fg = np.full((1, 2, 2, 3), 200, np.uint8)
    bg = np.full((1, 2, 2, 3), 100, np.uint8)
    alpha = np.array([[[1.0, 0.0], [0.5, 0.25]]], np.float32)
    out = cp.composite(fg, bg, alpha)
    assert out[0, :, :, 0].tolist() == [[200, 100], [150, 125]]


def test_push_pull_keeps_known_pixels_and_fills_smoothly() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    known = np.zeros((37, 53), bool)
    known[10:20, 15:30] = True
    out = cp.push_pull(image, known)
    assert (out[known] == image[known]).all()
    flat = np.full_like(image, 90)
    filled = cp.push_pull(np.where(known[..., None], flat, image), known)
    assert (filled == 90).all()  # a constant known region extrapolates to the constant
    far = filled[~known].astype(int)
    assert far.min() >= 0 and far.max() <= 255


def test_foreground_input_size_and_kept_region() -> None:
    frames = np.random.default_rng(1).integers(0, 256, (2, 64, 96, 3), dtype=np.uint8)
    fg = np.zeros((2, 64, 96), bool)
    fg[:, 24:40, 40:56] = True
    small = cp.foreground_input(frames, fg, 2, (48, 32))
    assert small.shape == (2, 32, 48, 3)
    full = cp.foreground_input(frames, fg, 1, (96, 64))
    keep = cp.dilate(fg[0], cp.keep_radius(1))
    assert (full[0][keep] == frames[0][keep]).all()
    outside = ~cp.dilate(fg[0], cp.keep_radius(1) + 2)
    assert not (full[0][outside] == frames[0][outside]).all()


def test_apply_slots_only_touches_its_class() -> None:
    classes = square_classes()
    _, nearest = cp.alpha_and_classes(classes)
    up = np.full((64, 80, 3), 10, np.uint8)
    out = cp.apply_slots(up[None], nearest[None], {cp.HAND: lambda x, m: np.full_like(x, 99)})[0]
    assert (out[nearest == cp.HAND] == 99).all()
    assert (out[nearest != cp.HAND] == 10).all()
