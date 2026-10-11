import math

import numpy as np
import pytest

from experiments.visor import h3c
from experiments.visor import h3c_bits as bits
from src.codecs.dcvc_uf_worker import display_schedule


@pytest.mark.parametrize("count", [1, 2, 9, 10, 48, 240])
def test_bit_schedule_is_the_workers(count):
    assert bits.schedule(count) == display_schedule(count, bits.FRAME_DELAY)


def test_z_bits_are_shared_by_the_y_cells_beneath_and_conserved():
    rng = np.random.default_rng(0)
    y = rng.random((68, 120))
    z = rng.random((17, 30))
    cells = bits.spread_z(y, z)
    assert math.isclose(cells.sum(), y.sum() + z.sum())
    assert np.allclose(cells[:4, :4] - y[:4, :4], z[0, 0] / 16)


def test_region_bits_spread_each_cell_over_its_pixels_and_frames():
    cells = np.ones((68, 120))  # 1080 rows pad to 1088: 68 cells of 16
    region = np.zeros((2, 1080, 1920), bool)
    assert bits.region_bits(cells, region) == 0.0
    region[:] = True
    # Every real pixel: all cells but the bottom row, which is half padding.
    assert math.isclose(bits.region_bits(cells, region), 67 * 120 + 120 * 8 / 16)
    region[1] = False  # one of the two frames
    assert math.isclose(bits.region_bits(cells, region), (67 * 120 + 120 * 8 / 16) / 2)
    with pytest.raises(ValueError):
        bits.region_bits(np.ones((10, 10)), region)


def test_nal_bytes_split_by_each_nals_share():
    nals = [{"nal_bytes": 100, "sps_bytes": 10}, {"nal_bytes": 50, "sps_bytes": 0}]
    estimate = [{"estimated_bits": 1000.0, "estimated_region_bits": 250.0}, {"estimated_bits": 400.0, "estimated_region_bits": 400.0}]
    out = h3c.nal_region_bits(nals, estimate, 160 + h3c.CONTAINER_HEADER)
    header = 8 * h3c.CONTAINER_HEADER * 650 / 1400
    assert math.isclose(out["region_bits"], 8 * 110 * 0.25 + 8 * 50 + header)
    assert out["bits"] == 8 * (160 + h3c.CONTAINER_HEADER) and math.isclose(out["estimate_over_actual"], 1400 / (8 * 160))
    with pytest.raises(RuntimeError):
        h3c.nal_region_bits(nals[:1], estimate, 100)


def test_downscale_and_crop_keep_the_4_2_0_layout():
    w, h = 1920, 1080
    plane = np.zeros((h * 3 // 2, w), np.uint8)
    plane[:h] = 100
    plane[h:h + h // 4] = 50  # U
    plane[h + h // 4:] = 200  # V
    sw, sh = h3c.scaled_size(w, h, 4)
    small = h3c.downscale_planes(plane, w, h, sw, sh)
    assert small.shape == (sh * 3 // 2, sw)
    assert (small[:sh] == 100).all() and (small[sh:sh + sh // 4] == 50).all() and (small[sh + sh // 4:] == 200).all()
    crop = h3c.crop_planes(small, sw, sh, 0, 0, h3c.CROP)  # 480×272 into 512×512: edges replicated
    assert crop.shape == (h3c.CROP * 3 // 2, h3c.CROP)
    c = h3c.CROP
    assert (crop[:c] == 100).all() and (crop[c:c + c // 4] == 50).all() and (crop[c + c // 4:] == 200).all()
    mask = np.ones((sh, sw), bool)
    keep = h3c.crop_mask(mask, 0, 0, c)
    assert keep.shape == (c, c) and keep[:sh, :sw].all() and not keep[sh:].any() and not keep[:, sw:].any()


def test_segment_box_is_even_centred_and_inside():
    union = np.zeros((1080, 1920), bool)
    union[900:1000, 1800:1900] = True  # near the bottom-right corner
    x, y = h3c.segment_box(union, 512)
    assert x % 2 == 0 and y % 2 == 0 and x + 512 <= 1920 and y + 512 <= 1080 and (x, y) == (1408, 568)
    union[:] = False
    union[500:540, 900:960] = True
    x, y = h3c.segment_box(union, 512)
    assert abs(x + 256 - 929.5) <= 2 and abs(y + 256 - 519.5) <= 2
    assert h3c.segment_box(np.ones((272, 480), bool), 512) == (0, 0)


def test_choose_segments_skips_handless_segments_and_keeps_the_fullest():
    pixels = np.zeros(240, int)
    pixels[0:48] = 5
    pixels[96:144] = 50
    pixels[192:240] = 20
    assert h3c.choose_segments(pixels, None) == [0, 96, 192]
    assert h3c.choose_segments(pixels, 2) == [96, 192]


def test_bd_rate_on_lpips_fronts():
    anchor = [{"kbps": r, "lpips": q} for r, q in ((1.0, 0.4), (10.0, 0.2), (100.0, 0.1))]
    assert math.isclose(h3c.bd_rate(anchor, anchor, "lpips")["bd"], 0.0, abs_tol=1e-12)
    half = [{**p, "kbps": p["kbps"] / 2} for p in anchor]
    assert math.isclose(h3c.bd_rate(half, anchor, "lpips")["bd"], -0.5)
    better = [{"kbps": 5.0, "lpips": 0.05}, {"kbps": 50.0, "lpips": 0.01}]
    assert h3c.bd_rate(better, anchor, "lpips") == {"bd": None, "where": "test better than every anchor point"}


def test_the_component_never_sees_a_pilot_participant():
    assert {h3c.participant(p + "_0") for p in h3c.PILOT} == set(h3c.EXCLUDED)


def test_decision_reads_the_upper_bound_and_the_component():
    def arm(mean, below, ref=None):
        return {"scored": 4, "unclear_items": 0, "mean_bd": mean, "items_below_zero": below, "mean_bd_vs_reference": ref}

    out = h3c.decide({"upper-mse": arm(0.05, 1), "upper-lpips": arm(-0.4, 4), "component-lpips": arm(-0.2, 3, -0.3)})
    assert out["upper_bound"] == "passes" and out["better_beta"] == "lpips" and out["component"]["component-lpips"]["adopted"]
    out = h3c.decide({"upper-mse": arm(0.05, 1), "upper-lpips": arm(0.0, 2)})
    assert out["upper_bound"] == "fails"
    out = h3c.decide({"upper-mse": arm(-0.05, 3), "upper-lpips": arm(0.2, 0), "component-mse": arm(-0.2, 3, -0.05)})
    assert out["upper_bound"] == "unclear" and not out["component"]["component-mse"]["adopted"]
