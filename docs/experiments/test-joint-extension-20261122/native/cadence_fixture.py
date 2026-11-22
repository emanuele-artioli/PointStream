"""Verify the named 60-fps package using real SVT/dav1d frames, outside any TEST outcome."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct

import numpy as np
from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis.player_requalification import array_sha, sha256
from src.codecs import svtav1


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    yy, xx = np.indices((96, 160))
    planes = []
    for t in range(240):
        y = np.uint8(16 + (xx + 2 * yy + t) % 201)
        uv = np.full((48, 160), 128, dtype=np.uint8)
        planes.append(np.concatenate([y, uv]))
    source = a.out / "fixture.yuv"
    source.write_bytes(np.stack(planes).tobytes())
    e, d = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    encode = svtav1.code(
        source,
        a.out,
        width=160,
        height=96,
        fps=Fraction(60),
        frames=240,
        crf=50,
        preset=4,
        threads=1,
        full_range=False,
        encoder=e["path"],
        decoder=d["path"],
        timeout=180,
    )
    base = Path(encode["stream"]).read_bytes()
    packet._ivf(base, 160, 96, (60, 1))
    original = (a.existing / "players.pscp").read_bytes()
    pieces, unused, decoded = packet.unpack(original)
    profile = packet._embedded_profile(original)
    profile["recipe"] = packet.RECIPE_60
    carrier = packet.build(
        profile, base, pieces["keypoints"], {k: pieces[k] for k in ("ref0", "ref1")}
    )
    old = (a.existing / "rally.psj").read_bytes()
    jp = packet._embedded_profile(old)
    segments, _ = packet.wire.unpack(old, jp)
    initial = (a.existing / "initial.psr").read_bytes()
    cp = packet._embedded_profile(initial)
    refs, _ = packet.wire.unpack(initial, cp)
    runtime = json.loads(manifest.encode(packet.wire.sha(carrier)))
    del runtime["carrier_sha256"], runtime["composition"]
    cache = packet.wire.pack(
        dict(cp, runtime=runtime), [(k, "reference", v) for k, v in refs.items()]
    )
    jp = dict(
        jp,
        player={k: v for k, v in profile.items() if k != "resources"},
        cache_sha256=packet.wire.sha(cache),
    )
    joint = packet.wire.pack(
        jp,
        [
            ("base", "frame", base),
            ("poses", "frame", segments["poses"]),
            ("trajectory", "frame", segments["trajectory"]),
        ],
    )
    for name, b in (("rally.psj", joint), ("initial.psr", cache), ("players.pscp", carrier)):
        (a.out / name).write_bytes(b)
    c = mixed.decode(
        a.out / "rally.psj", a.out / "initial.psr", a.out / "players.pscp", packet.wire.sha(carrier)
    )
    assert (
        c["wire"]["active_ticks"] == [0, 240]
        and c["base"] == base
        and c["profile"]["recipe"]["fps"] == [60, 1]
    )
    raw = np.memmap(encode["decoded"], np.uint8, "r", shape=(240, 144, 160))
    grid = [array_sha(mixed.canvas(raw[t], t, c)) for t in (0, 39, 119, 239)]
    assert len(set(grid)) == 4
    bad = bytearray(base)
    struct.pack_into("<II", bad, 16, 25, 1)
    try:
        packet.build(profile, bytes(bad), pieces["keypoints"], refs)
    except ValueError:
        rejected = True
    else:
        raise AssertionError("mismatched native timebase accepted")
    out = dict(
        scope="interface qualification using a deterministic synthetic 60-fps base and preexisting development conditioning; no independent event result",
        fps=[60, 1],
        source=sha256(source),
        encoder=e,
        decoder=d,
        ivf=sha256(Path(encode["stream"])),
        native_decoded=sha256(Path(encode["decoded"])),
        ticks=240,
        duration_seconds=4,
        paid_roles=c["wire"],
        decoded_checks=grid,
        wrong_25_fps_rejected=rejected,
    )
    a.result.write_text(json.dumps(out, indent=2) + "\n")
    print(out["duration_seconds"], out["ivf"], out["wrong_25_fps_rejected"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("out", "existing", "result"):
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())
