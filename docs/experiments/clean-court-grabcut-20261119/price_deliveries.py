"""Price complete paired deliveries under one measured macOS CPU decoder profile."""

import hashlib
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from experiments.tennis import common_player_ball as ball
from experiments.tennis import common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifests
from experiments.tennis.player_requalification import array_sha
from src.codecs import svtav1

ROOT = Path(
    "/Users/manu/Datasets/pointstream-submission-20261118/selected-rate-support-linux-png/packets/48"
)
OUT = Path("/private/tmp/ps-clean-court-deliveries-20261119")
BASES = {
    "original": Path("/private/tmp/ps-clean-court-grabcut-20261119/full/plain/stream.ivf"),
    "broad": Path("/private/tmp/ps-clean-court-rally-20261118/full/clean/stream.ivf"),
    "grabcut": Path("/private/tmp/ps-clean-court-grabcut-20261119/full/clean/stream.ivf"),
}
DAV1D = "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    start, begin = time.monotonic(), cpu()
    OUT.mkdir(exist_ok=False)
    original = (ROOT / "players.pscp").read_bytes()
    side = (ROOT / "ball.sidecar").read_bytes()
    require(
        sha(original) == "144bd0870dd37b6b586663473d12a50bdb78015626661eccaa7a427fb60c9b87",
        "original player carrier",
    )
    require(
        sha(side) == "616f63b0f3f2ca84e3eb87645d1a0877d29a6aa7130d986c31a3a4f0a7b3b2d2",
        "frozen local ball sidecar",
    )
    manifests.verify(ROOT / "receiver.manifest", sha(original))
    sections, _, original_decoded = player.unpack(original)
    positions, tile, canvas = ball._open(original, side, DAV1D)
    require(
        canvas == (1920, 1080)
        and len(positions) == 240
        and sum(p is not None for p in positions) == 195,
        "ball coverage",
    )
    positions_data = b"".join(ball.PAIR.pack(*(ball.ABSENT if p is None else p)) for p in positions)
    ffmpeg, dav1d = svtav1.tool("/opt/homebrew/bin/ffmpeg"), svtav1.tool("/opt/homebrew/bin/dav1d")
    local = dict(
        platform="macOS",
        numpy=np.__version__,
        opencv=cv2.__version__,
        opencv_build_sha256=sha(cv2.getBuildInformation().encode()),
        native_ffmpeg=ffmpeg,
        tile_dav1d=dav1d,
    )
    rows = []
    for name, base_path in BASES.items():
        base = base_path.read_bytes()
        if name == "original":
            require(base == sections["base"], "same native original base")
        profile = player._embedded_profile(original)
        profile["resources"]["native_decoder"] = ffmpeg["sha256"]
        png_carrier = player.build(
            profile,
            base,
            sections["keypoints"],
            {k: v for k, v in sections.items() if k.startswith("ref")},
        )
        optimized = player.optimize_lossless_carrier(png_carrier)
        require(len(optimized) <= len(png_carrier), "complete carrier grew")
        parts, roles, decoded = player.unpack(optimized)
        require(
            sum(roles.values()) == len(optimized)
            and parts["base"] == base
            and parts["keypoints"] == sections["keypoints"],
            "changed base or keypoints",
        )
        info = player.inspect(optimized)
        require(
            info["required_resources"]["native_decoder"] == ffmpeg["sha256"]
            and info["required_resources"]["reference_decoder"] == local["opencv_build_sha256"],
            "wrong local native/reference pin",
        )
        units = {}
        for unit, runs in decoded.items():
            a, b = original_decoded[unit][0], runs[0]
            require(len(runs) == 1 and a["seed"] == b["seed"], "changed runs or seed")
            for key in ("reference", "boxes", "kp", "t"):
                require(np.array_equal(a[key], b[key]), "changed conditioning " + key)
            units[unit] = {
                key: array_sha(np.asarray(b[key])) for key in ("reference", "boxes", "kp", "t")
            }
        rebound = ball.pack(optimized, positions, tile, decoder_sha256=dav1d["sha256"])
        pos2, tile2, size2 = ball._open(optimized, rebound, dav1d["sha256"])
        require(pos2 == positions and tile2 == tile and size2 == canvas, "changed ball payload")
        companion = json.loads(manifests.encode(sha(optimized)))
        companion.update(
            profile="common-tennis-macos-cpu-v1",
            native_ffmpeg=ffmpeg["sha256"],
            numpy=local["numpy"],
            opencv_build_sha256=local["opencv_build_sha256"],
        )
        manifest = json.dumps(companion, sort_keys=True, separators=(",", ":")).encode("ascii")
        folder = OUT / name
        folder.mkdir()
        files = {"players.pscp": optimized, "receiver.manifest": manifest, "ball.sidecar": rebound}
        for filename, blob in files.items():
            path = folder / filename
            path.write_bytes(blob)
            require(path.read_bytes() == blob, "published bytes changed")
        require(json.loads(manifest)["carrier_sha256"] == sha(optimized), "manifest identity")
        total = sum(map(len, files.values()))
        rows.append(
            dict(
                name=name,
                base_path=str(base_path),
                base_bytes=len(base),
                base_sha256=sha(base),
                carrier_bytes=len(optimized),
                manifest_bytes=len(manifest),
                sidecar_bytes=len(rebound),
                total_bytes=total,
                kbps=total / 600,
                role_bytes=roles,
                components={k: dict(bytes=len(v), sha256=sha(v)) for k, v in parts.items()},
                conditioning=units,
                files={k: dict(bytes=len(v), sha256=sha(v)) for k, v in files.items()},
            )
        )
    require(
        all(
            r["components"]["ref0"] == rows[0]["components"]["ref0"]
            and r["components"]["ref1"] == rows[0]["components"]["ref1"]
            for r in rows
        ),
        "reference encodings differ",
    )
    result = dict(
        input_carrier_sha256=sha(original),
        input_sidecar_sha256=sha(side),
        input_manifest_sha256=sha((ROOT / "receiver.manifest").read_bytes()),
        input_linux_native_decoder=manifests.NATIVE,
        input_linux_ball_decoder=DAV1D,
        local_cpu_profile=local,
        rows=rows,
        shared_positions_sha256=sha(positions_data),
        shared_tile_sha256=sha(tile),
        placements=195,
        frames=240,
        seconds=4.8,
        script_sha256=sha(Path(__file__).read_bytes()),
        cpu_seconds=cpu() - begin,
        wall_seconds=time.monotonic() - start,
        scope="complete serialized local CPU profile and exact reference/geometry bindings; no new learned rendering or source-denied full decoder",
    )
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([(r["name"], r["total_bytes"], r["kbps"]) for r in rows]))


if __name__ == "__main__":
    main()
