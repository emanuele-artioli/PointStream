from fractions import Fraction

import numpy as np

from src.pointstream import ball
from src.pointstream.bitstream import Container, Tag


def flight(frames: int = 40, gap: tuple[int, int] = (15, 18)) -> ball.Track:
    track: ball.Track = []
    x, y, vx, vy = 20.0, 60.0, 4.0, -3.0
    for t in range(frames):
        if gap[0] <= t < gap[1]:
            track.append(None)
        else:
            track.append((int(round(x)), int(round(y))))
        x, y, vy = x + vx, y + vy, vy + 0.2
    return track


def test_exp_golomb_round_trips_signed_values_at_every_order():
    w = ball.BitWriter()
    values = [0, 1, -1, 2, -7, 40, -129, 1000]
    for k in range(4):
        for v in values:
            w.se(v, k)
    r = ball.BitReader(w.tobytes())
    assert [r.se(k) for k in range(4) for _ in values] == values * 4
    assert sum(ball.se_bits(v, 2) for v in values) == len(_bits(values, 2))


def _bits(values: list[int], k: int) -> list[int]:
    w = ball.BitWriter()
    for v in values:
        w.se(v, k)
    return w.bits


def test_payload_round_trips_positions_runs_and_appearance():
    track = flight()
    app = ball.Appearance((200, 210, 90), 3, 5, 2)
    payload = ball.encode(track, app, 200, 120)
    assert ball.decode(payload, len(track), 200, 120) == (track, app)
    assert len(payload) < 40  # 37 visible frames: header, runs, two absolutes, small residuals
    assert ball.decode(ball.encode(track, None, 200, 120), len(track), 200, 120) == ([None] * len(track), None)
    assert len(ball.encode(track, None, 200, 120)) == 1


def test_payload_handles_a_window_that_starts_hidden_and_ends_visible():
    track: ball.Track = [None, None, (5, 5), (6, 7), None, (100, 3)]
    app = ball.Appearance((1, 2, 3), 0, 0, 3)
    assert ball.decode(ball.encode(track, app, 128, 64), len(track), 128, 64) == (track, app)


def test_velocity_is_central_inside_a_run_one_sided_at_its_ends_and_zero_alone():
    v = ball.velocities([(0, 0), (2, 0), (6, 0), None, (9, 9)])
    assert v.tolist() == [[2, 0], [3, 0], [4, 0], [0, 0], [0, 0]]


def test_a_streak_spreads_the_same_ink_as_the_disk_along_the_velocity():
    dy, dx = np.mgrid[-20:21, -20:21].astype(np.float32)
    v = np.array([12.0, 0.0], np.float32)
    disk = ball.coverage(dx, dy, v, 3.0, 0.0, 0.0)
    streak = ball.coverage(dx, dy, v, 3.0, 1.0, 0.0)
    assert abs(disk.sum() - streak.sum()) < 0.02 * disk.sum()
    assert disk.max() == 1.0 and streak.max() <= 0.6  # about the diameter over the length
    assert np.ptp(np.nonzero(streak.sum(0))[0]) > np.ptp(np.nonzero(disk.sum(0))[0]) + 10


def test_the_fit_recovers_the_appearance_that_drew_the_source():
    rng = np.random.default_rng(1)
    bg = rng.normal(80, 4, (120, 200, 3)).clip(0, 255).astype(np.uint8)
    track = flight()
    vel = ball.velocities(track)
    true = ball.Appearance((230, 220, 100), 3, 4, 3)
    source = np.stack([ball.draw(bg, p, vel[t], true) if p else bg for t, p in enumerate(track)])
    decode = np.repeat(bg[None], len(track), 0)
    crops = ball.Crops.build(track, source, vel)
    dec = [[ball.crops_of(decode, g["frames"], g["centres"], g["half"]) for g in crops.groups]]
    streak = ball.fit(crops, dec, ball.grid("streak"))[0]
    assert streak["appearance"] == true
    assert abs(streak["delta_se"] + streak["base_se"]) < 1e-3 * streak["base_se"]
    disk = ball.fit(crops, dec, ball.grid("disk"))[0]
    assert disk["appearance"].exposure == 0 and disk["delta_se"] > streak["delta_se"]
    # A decode that already holds the ball: nothing to add, the mode is off.
    same = [[ball.crops_of(source, g["frames"], g["centres"], g["half"]) for g in crops.groups]]
    assert ball.fit(crops, same, ball.grid("disk"))[0]["appearance"] is None


def test_the_ball_segment_is_counted_in_the_container():
    c = Container(200, 120, Fraction(30), 40)
    payload = ball.encode(flight(), ball.Appearance((1, 2, 3), 0, 0, 3), 200, 120)
    c.add(Tag.BALL, payload)
    back = Container.unpack(c.pack())
    assert back.segments == [(Tag.BALL, payload)]
    assert back.breakdown()["ball"] == 2 + len(payload)


def test_per_frame_payload_round_trips_and_steady_frames_are_cheap():
    track = flight()
    apps: list[ball.Appearance | None] = [None if p is None else ball.Appearance((200 + t % 3, 210, 90), 3, 5 - t % 2, 2)
                                          for t, p in enumerate(track)]
    payload = ball.encode_frames(track, apps, 200, 120)
    assert ball.decode_frames(payload, len(track), 200, 120) == (track, apps)
    assert len(payload) < len(ball.encode(track, apps[0], 200, 120)) + 37 * 12 // 8  # under 12 bits a frame for small changes
    assert ball.decode_frames(ball.encode_frames([None] * 5, [None] * 5, 200, 120), 5, 200, 120) == ([None] * 5, [None] * 5)


def test_the_per_frame_fit_refines_jittered_positions_and_recovers_each_frames_appearance():
    rng = np.random.default_rng(2)
    bg = rng.normal(80, 4, (120, 200, 3)).clip(0, 255).astype(np.uint8)
    true = flight()
    vel = ball.velocities(true)
    apps = [None if p is None else ball.Appearance((230 - 2 * t, 220, 100 + t), 4 + t % 2, 0, 3) for t, p in enumerate(true)]
    source = np.stack([ball.render(bg, t, true, vel, apps) for t in range(len(true))])
    start: ball.Track = [None if p is None else (p[0] + int(rng.integers(-3, 4)), p[1] + int(rng.integers(-3, 4))) for p in true]
    decode = np.repeat(bg[None], len(true), 0)
    crops = ball.FrameCrops.build(start, source, [decode, source])
    got = ball.fit_frames(crops, ball.grid("disk"), (120, 200))
    assert got[0]["track"] == true
    assert [a and (a.radius, a.exposure) for a in got[0]["appearances"]] == [a and (a.radius, a.exposure) for a in apps]
    assert all(a is None or max(abs(x - y) for x, y in zip(a.colour, b.colour)) <= 2
               for a, b in zip(got[0]["appearances"], apps) if b is not None)
    assert got[1]["track"] == [None] * len(true)  # the decode holds the ball already: nothing sent
    payload = ball.encode_frames(got[0]["track"], got[0]["appearances"], 200, 120)
    assert ball.decode_frames(payload, len(true), 200, 120) == (got[0]["track"], got[0]["appearances"])


def test_a_patch_is_what_draw_changes():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (60, 80, 3), np.uint8)
    app = ball.Appearance((250, 240, 30), 5, 4, 2)
    (y0, y1, x0, x1), pixels = ball.patch(frame, (3, 30), np.array([9.0, -2.0]), app)
    drawn = ball.draw(frame, (3, 30), np.array([9.0, -2.0]), app)
    assert x0 == 0 and np.array_equal(drawn[y0:y1, x0:x1], pixels)
    outside = drawn.copy()
    outside[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    assert np.array_equal(outside, frame)
