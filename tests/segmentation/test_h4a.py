import json
import struct

import numpy as np
import pytest

from experiments.visor import h4a


def pinhole(t: int, points: np.ndarray) -> np.ndarray:
    """A camera at the origin looking down +z, f = 500 px (the world is the camera frame)."""
    return 500 * points[:, :2] / points[:, 2:3] + 700


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rot_z(deg: float) -> np.ndarray:
    m = np.eye(4)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m[:2, :2] = [[c, -s], [s, c]]
    return m


CUBE = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-0.03, 0.03)])


def carried_track(frames: int = 60, start: int = 10, stop: int = 50, slip_deg: float = 0.0) -> h4a.Track:
    """At rest at 0.5 m, then carried by the right wrist (5 mm/frame along x) from ``start`` to ``stop``; the
    left wrist stays put. ``slip_deg`` turns the object in the hand per frame."""
    wrist_r = np.stack([translation(0.005 * min(max(t - start, 0), stop - start), 0.0, 0.42) for t in range(frames)])
    grip = translation(0.0, 0.0, 0.08)
    poses = np.stack([wrist_r[t] @ grip @ rot_z(slip_deg * min(max(t - start, 0), stop - start)) for t in range(frames)])
    wrists = {"right": wrist_r, "left": np.stack([translation(-0.2, 0, 0.4)] * frames)}
    return h4a.Track(CUBE, poses, np.ones(frames, bool), np.stack([np.eye(4)] * frames), wrists, pinhole)


def test_quaternion_matrix_is_a_rotation_and_rejects_bad_norms():
    m = h4a.quat_matrix([np.cos(0.3), 0, 0, np.sin(0.3)], [1, 2, 3])
    assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3)) and np.allclose(m[:3, 3], [1, 2, 3])
    assert h4a.rotation_deg(np.eye(4), m) == pytest.approx(np.degrees(0.6))
    with pytest.raises(ValueError):
        h4a.quat_matrix([1, 1, 0, 0], [0, 0, 0])
    assert np.allclose(h4a.rigid_inverse(m) @ m, np.eye(4))


def test_glb_vertices_reads_position():
    verts = np.arange(12, dtype="<f4").reshape(4, 3)
    binary = verts.tobytes()
    doc = {"nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": 4}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}]}
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    blob = struct.pack("<I", len(text)) + b"JSON" + text + struct.pack("<I", len(binary)) + b"BIN\0" + binary
    data = b"glTF" + struct.pack("<II", 2, 12 + len(blob)) + blob
    assert np.array_equal(h4a.glb_vertices(data), verts)
    doc["nodes"][0]["translation"] = [0, 0, 1]
    with pytest.raises(ValueError):
        text2 = json.dumps(doc).encode()
        text2 += b" " * (-len(text2) % 4)
        blob2 = struct.pack("<I", len(text2)) + b"JSON" + text2 + struct.pack("<I", len(binary)) + b"BIN\0" + binary
        h4a.glb_vertices(b"glTF" + struct.pack("<II", 2, 12 + len(blob2)) + blob2)


def test_moving_segments_dilate_and_merge():
    d = np.zeros(40)
    d[10:13] = 1.0
    d[19] = 1.0  # dilated to 16..22, touching 7..15
    d[35:] = 1.0
    assert h4a.moving_segments(d, 0.5, 3) == [(7, 22), (32, 39)]
    assert h4a.moving_segments(np.full(5, np.nan)) == []


def test_a_carried_object_is_held_by_the_grip_and_not_the_world():
    track = carried_track()
    (a, b), = h4a.moving_segments(track.displacement_mm())
    frames = list(range(a, b + 1))
    grip = h4a.replay(track, frames, "grip", 2.0)
    world = h4a.replay(track, frames, "world", 2.0)
    hybrid = h4a.replay(track, frames, "hybrid", 2.0)
    assert len(grip) == 1 and grip[0]["kind"] == "grip-right" and grip[0]["cut"]
    assert grip[0]["life_s"] == pytest.approx(len(frames) / h4a.FPS)
    assert len(world) > 10 and all(r["kind"] == "world" for r in world)
    assert len(hybrid) == 1 and sum(r["held"] + 1 for r in hybrid) == len(frames)
    for rows in (grip, world, hybrid):  # every scored frame is a refresh or held, once
        assert sum(r["held"] + 1 for r in rows) == len(frames)


def test_slip_in_the_hand_forces_grip_refreshes_and_shows_in_the_spread():
    track = carried_track(slip_deg=2.0)
    disp = track.displacement_mm()
    (a, b), = h4a.moving_segments(disp)
    grip = h4a.replay(track, list(range(a, b + 1)), "grip", 2.0)
    assert len(grip) > 3
    spread = h4a.grip_spread(track, a, b, disp)
    assert spread is not None and spread["hand"] == "right" and spread["rot_max_deg"] > 30
    still = h4a.grip_spread(carried_track(), a, b, carried_track().displacement_mm())
    assert still is not None and still["rot_max_deg"] < 1e-6 and still["trans_max_mm"] < 1e-6
    assert still["world_trans_median_mm"] > 10


def test_a_missing_hand_ends_a_grip_life():
    track = carried_track()
    track.W["right"][30] = np.nan
    (a, b), = h4a.moving_segments(track.displacement_mm())
    grip = h4a.replay(track, list(range(a, b + 1)), "grip", 2.0)
    assert [r["frame"] for r in grip][:2] == [a, 30]


def unit(scored: int, lives: list[tuple[str, float, int]], frames: int = 150) -> tuple[dict, dict]:
    rows = [(tau, coder, kind, held) for tau in h4a.TAUS for coder, kind, held in
            [(c, k, h) for c, k, h in lives]]
    a = {"refresh_tau": np.array([r[0] for r in rows], np.float32),
         "refresh_coder": np.array([h4a.CODERS.index(r[1]) for r in rows], np.int8),
         "refresh_kind": np.array([h4a.PREDICTORS.index(r[2]) for r in rows], np.int8),
         "refresh_held": np.array([r[3] for r in rows], np.int32),
         "refresh_life_s": np.array([(r[3] + 1) / h4a.FPS for r in rows], np.float32),
         "refresh_cut": np.zeros(len(rows), bool), "rest_px": np.array([0.2, 0.3], np.float32)}
    meta = {"frames": frames, "scored_frames": scored, "moving_objects": 1, "spreads": [],
            "visible_centroid_inside": [True]}
    return meta, a


def test_decision_needs_held_frames_long_lives_and_the_grip():
    # 60 scored frames: the hybrid holds them in two grip lives; world refreshes every 3 frames.
    good = unit(60, [("hybrid", "grip-right", 29), ("hybrid", "grip-right", 29)]
                + [("world", "world", 2)] * 20 + [("grip", "grip-right", 29)] * 2)
    s = h4a.summarize([good])
    assert s["hybrid@2"]["held_share"] == pytest.approx(58 / 60)
    assert s["hybrid@2"]["frames_by_kind_share"]["grip-right"] == 1.0
    assert h4a.verdict(s)["verdict"] == "passes"
    assert h4a.unit_checks([good])["every_scored_frame_is_held_or_a_refresh_once"]
    # Long lives, but the world alone does as well: (c) fails.
    worldly = unit(60, [("hybrid", "world", 29)] * 2 + [("world", "world", 29)] * 2 + [("grip", "grip-right", 29)] * 2)
    v = h4a.verdict(h4a.summarize([worldly]))
    assert v["verdict"] == "unclear" and not v["at_2px"]["checks"]["c_grip_explains"]
    short = unit(60, [("hybrid", "grip-right", 1)] * 30 + [("world", "world", 0)] * 60 + [("grip", "grip-right", 1)] * 30)
    assert h4a.verdict(h4a.summarize([short]))["verdict"] == "fails"


def test_units_round_trip_through_a_published_tar(tmp_path):
    meta, a = unit(60, [("hybrid", "grip-right", 59), ("world", "world", 59), ("grip", "grip-right", 59)])
    meta["clip"] = "clip-000001.tar"
    publish = tmp_path / "publish"
    h4a.save_unit(publish, "clips", "clip-000001", a, meta)
    import tarfile
    with tarfile.open(tmp_path / "published.tar", "w") as tar:
        tar.add(publish, "publish")
    (back_meta, back), = h4a.units_from_tar(tmp_path / "published.tar")
    assert back_meta["clip"] == "clip-000001.tar" and np.array_equal(back["refresh_held"], a["refresh_held"])
