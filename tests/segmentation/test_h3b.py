import math

import numpy as np

from experiments.visor import h3b
from experiments.visor import pose_coding as pc

SQUARE = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
SQUARE_FACES = np.array([[0, 1, 2], [0, 2, 3]], np.int32)


def test_raster_draws_the_nearer_face_last():
    points = np.concatenate([SQUARE, SQUARE + 5])
    faces = np.concatenate([SQUARE_FACES, SQUARE_FACES + 4])
    depth = np.array([1.0] * 4 + [2.0] * 4)  # the second square is farther
    ids = h3b.raster(points, depth, faces, (0, 0, 50, 50))
    assert ids[20, 20] in (0, 1)  # both cover it; the nearer wins
    assert ids[33, 33] in (2, 3)  # only the farther one
    assert ids[5, 5] == -1 and ids[45, 45] == -1


def test_barycentric_reproduces_pixel_positions():
    box = (5, 5, 40, 40)
    ids = h3b.raster(SQUARE, np.ones(4), SQUARE_FACES, box)
    ys, xs, f, w = h3b.barycentric(ids, SQUARE, SQUARE_FACES, box)
    assert len(ys) > 300 and np.allclose(w.sum(1), 1.0)
    back = (w[:, :, None] * SQUARE[SQUARE_FACES[f]]).sum(1)
    assert np.abs(back - np.stack([xs + box[0], ys + box[1]], 1)).max() < 1e-6


def test_render_carries_colours_by_surface_correspondence():
    rng = np.random.default_rng(0)
    box = (0, 0, 60, 60)
    source = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    region = np.zeros((60, 60), bool)
    region[8:34, 8:34] = True
    objects = np.zeros_like(region)
    z = np.ones(4)
    ids_u = h3b.raster(SQUARE, z, SQUARE_FACES, box)
    zmap = h3b.depth_map(ids_u, SQUARE, z, SQUARE_FACES, box)
    # The same pose: every covered pixel shows itself.
    covered, colours, stats = h3b.render(source, region, objects, ids_u, SQUARE, SQUARE, z, zmap, SQUARE_FACES, box,
                                         h3b.nearest_lookup(region))
    assert (colours[covered] == source[covered]).all() and stats["outside"] == 0 and stats["occluded"] == 0
    # Shifted 7 px right: the pixel at x shows the source's pixel at x − 7.
    moved = SQUARE + [7.0, 0.0]
    ids_c = h3b.raster(moved, z, SQUARE_FACES, box)
    covered, colours, _ = h3b.render(source, region, objects, ids_c, moved, SQUARE, z, zmap, SQUARE_FACES, box, h3b.nearest_lookup(region))
    assert (colours[15:25, 20:30] == source[15:25, 13:23]).all()
    # Objects in front are never covered.
    objects[15:20, 20:25] = True
    covered, _, _ = h3b.render(source, region, objects, ids_c, moved, SQUARE, z, zmap, SQUARE_FACES, box, h3b.nearest_lookup(region))
    assert not covered[15:20, 20:25].any() and covered[22, 22]


def test_render_samples_more_pixels_than_remap_takes_in_one_column():
    big = np.array([[5.0, 5.0], [245.0, 5.0], [245.0, 245.0], [5.0, 245.0]])
    box = (0, 0, 250, 250)
    rng = np.random.default_rng(3)
    source = rng.integers(0, 255, (250, 250, 3), dtype=np.uint8)
    region = np.ones((250, 250), bool)
    z = np.ones(4)
    ids = h3b.raster(big, z, SQUARE_FACES, box)
    zmap = h3b.depth_map(ids, big, z, SQUARE_FACES, box)
    covered, colours, stats = h3b.render(source, region, np.zeros_like(region), ids, big, big, z, zmap, SQUARE_FACES, box,
                                         h3b.nearest_lookup(region))
    assert stats["covered"] > 40000 and (colours[covered] == source[covered]).all()


def test_render_takes_the_nearest_region_pixel_outside_the_region():
    box = (0, 0, 60, 60)
    source = np.zeros((60, 60, 3), np.uint8)
    source[:, :20] = 200  # the region's colour
    region = np.zeros((60, 60), bool)
    region[:, :20] = True
    z = np.ones(4)
    ids = h3b.raster(SQUARE, z, SQUARE_FACES, box)  # spans x 10..30: half outside the region
    zmap = h3b.depth_map(ids, SQUARE, z, SQUARE_FACES, box)
    covered, colours, stats = h3b.render(source, region, np.zeros_like(region), ids, SQUARE, SQUARE, z, zmap, SQUARE_FACES,
                                         box, h3b.nearest_lookup(region))
    assert stats["outside"] > 0 and (colours[covered] == 200).all()


def test_coding_combo_finds_h2s_choice():
    combo = h3b.coding_combo({"choices": {"wilor": {"100": {"key": "gauss:33.3|15Hz-linear|full|q8|previous"}}}}, "wilor")
    assert pc.combo_key(combo) == "gauss:33.3|15Hz-linear|full|q8|previous" and combo["subspace"] is None


def test_kid_is_near_zero_for_one_distribution_and_positive_for_two():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 16))
    y = rng.normal(size=(400, 16))
    z = rng.normal(0.5, 1.0, size=(400, 16))
    same = h3b.kid_from(*h3b.kid_kernels(x, y), rng, subset=200, subsets=20)
    other = h3b.kid_from(*h3b.kid_kernels(x, z), rng, subset=200, subsets=20)
    assert abs(same) < 0.05 < other


def test_flicker_is_zero_when_the_method_changes_like_the_source():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
    mask = np.ones((20, 20), bool)
    assert h3b.flicker(a, b, a, b, mask) < 1e-4
    assert h3b.flicker(a, a, a, b, mask) > 10  # a frozen method on a changing source


def test_equal_quality_inside_and_beyond_the_curve():
    svt = [{"kbps": 100.0, "lpips": 0.1}, {"kbps": 10.0, "lpips": 0.3}]
    inside = h3b.equal_quality(svt, 0.2, 2.0)
    assert inside["where"] == "inside" and math.isclose(inside["saving"], 1 - 2.0 / math.sqrt(1000.0))
    assert h3b.equal_quality(svt, 0.05, 2.0)["where"] == "better than every CRF"
    assert h3b.equal_quality(svt, 0.4, 2.0)["where"] == "worse than every CRF"


def test_square_crop_pads_outside_the_frame():
    image = np.full((50, 50, 3), 9, np.uint8)
    crop = h3b.square_crop(image, np.array([0.0, 0.0, 20.0, 10.0]), 1.5, 30)
    assert crop.shape == (30, 30, 3) and crop[0, 0].tolist() == [0, 0, 0] and crop[-1, -1].tolist() == [9, 9, 9]


def test_variants_hold_every_composite_once():
    assert len(h3b.VARIANTS) == len(set(h3b.VARIANTS)) == 4 + 2 * (3 + 4)
    assert set(h3b.JUDGED) - {"source"} <= set(h3b.VARIANTS)
