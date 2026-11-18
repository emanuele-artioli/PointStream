"""Price the approved PNG carrier, Linux manifest and paid ball on three native bases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifests

LINUX_DAV1D = "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("references", "bases", "ledger", "publication", "prior", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    if np.__version__ != "1.26.4":
        raise ValueError("unexpected Linux sidecar dtype/runtime binding")
    publication = json.loads(args.publication.read_text())
    logged = json.loads(args.ledger.read_text())["carriers"]["png"]
    prior = json.loads(args.prior.read_text())
    original = (args.references / "qpng.pscp").read_bytes()
    if (
        publication["quality"] != "png"
        or sha(original) != publication["files"]["qpng.pscp"]
        or sha(original) != logged["sha256"]
    ):
        raise ValueError("selected reference carrier identity mismatch")
    manifests.verify(args.references / "qpng.manifest", sha(original))
    sections, _, _ = player.unpack(original)
    profile = player.inspect(original)["profile"]
    if (
        profile["resources"]["native_decoder"] != manifests.NATIVE
        or profile["recipe"]["reference_codec"] != "png"
        or {name: sha(blob) for name, blob in sections.items()} != logged["sections"]
    ):
        raise ValueError("selected Linux profile or actual section hashes changed")
    profile["coded_size"] = [960, 544]
    args.output.mkdir(exist_ok=False, parents=True)
    rows = []
    for old in prior["rows"]:
        crf = old["crf"]
        native = (args.bases / str(crf) / "native.pscp").read_bytes()
        side = (args.bases / str(crf) / "ball.sidecar").read_bytes()
        if sha(native) != old["native_sha256"] or sha(side) != old["sidecar_sha256"]:
            raise ValueError("original native or side channel changed")
        positions, tile, _ = ball._open(native, side, prior["decoder"]["sha256"])
        base = player.unpack(native)[0]["base"]
        carrier = player.build(
            profile,
            base,
            sections["keypoints"],
            {k: v for k, v in sections.items() if k.startswith("ref")},
        )
        parts, roles, _ = player.unpack(carrier)
        assert parts == {**sections, "base": base} and sum(roles.values()) == len(carrier)
        new_side = ball.pack(carrier, positions, tile, decoder_sha256=LINUX_DAV1D)
        out = args.output / str(crf)
        out.mkdir()
        for filename, blob in [
            ("players.pscp", carrier),
            ("receiver.manifest", manifests.encode(sha(carrier))),
            ("ball.sidecar", new_side),
        ]:
            (out / filename).write_bytes(blob)
            assert (out / filename).read_bytes() == blob
        mb = manifests.verify(out / "receiver.manifest", sha(carrier))["manifest_bytes"]
        pos2, tile2, size2 = ball._open(
            (out / "players.pscp").read_bytes(), (out / "ball.sidecar").read_bytes(), LINUX_DAV1D
        )
        assert pos2 == positions and tile2 == tile and size2 == (1920, 1080) and mb == 625
        bstats = ball.inspect(carrier, new_side, decoder_sha256=LINUX_DAV1D)
        total = bstats["total_bytes"] + mb
        assert total == sum(f.stat().st_size for f in out.iterdir())
        rows.append(
            dict(
                crf=crf,
                base_ivf_bytes=len(base),
                base_ivf_sha256=sha(base),
                role_bytes=roles,
                sections={k: {"bytes": len(v), "sha256": sha(v)} for k, v in parts.items()},
                player_carrier_bytes=len(carrier),
                manifest_bytes=mb,
                sidecar=bstats,
                player_total_bytes=len(carrier) + mb,
                joint_total_bytes=total,
                player_kbps=(len(carrier) + mb) / 600,
                joint_kbps=total / 600,
                sha256={f.name: sha(f.read_bytes()) for f in out.iterdir()},
                ball_tile_sha256=sha(tile),
            )
        )
    result = dict(
        source_sha256=prior["source_sha256"],
        selected_carrier_sha256=sha(original),
        seconds=4.8,
        frames=240,
        policy="Linux lossless PNG, Linux FFmpeg and Linux dav1d sidecar, one cold receiver manifest per carrier",
        linux_native_ffmpeg_sha256=manifests.NATIVE,
        linux_tile_dav1d_sha256=LINUX_DAV1D,
        numpy=np.__version__,
        code_sha256=sha(Path(__file__).read_bytes()),
        rows=rows,
    )
    (args.output / "rates.json").write_text(json.dumps(result, indent=2) + "\n")
    print([(r["crf"], r["player_kbps"], r["joint_kbps"]) for r in rows])


if __name__ == "__main__":
    main()
