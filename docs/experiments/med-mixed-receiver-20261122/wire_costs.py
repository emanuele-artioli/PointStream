"""Recover per-component delivered bytes from three immutable, nested native packets."""

import argparse
import json
from pathlib import Path

from experiments.tennis import common_player_packet as packet

HERE = Path(__file__).resolve().parent


def main(a):
    results = []
    for shot in ("000", "019", "024"):
        r = json.loads((HERE / (shot + "-packets.json")).read_text())
        w = next(x for x in r["rates"] if x["crf"] == r["selected_crf"])
        path = a.packet_root / ("ps-med" + shot + "-paid-mixed-donor-20261122") / f"crf{w['crf']}"
        payload = (path / "rally.psj").read_bytes()
        init = (path / "initial.psr").read_bytes()
        assert (
            packet.wire.sha(payload) == w["joint_sha256"]
            and packet.wire.sha(init) == w["init_sha256"]
        )
        sections, roles = packet.wire.unpack(payload, packet._embedded_profile(payload))
        assets, init_roles = packet.wire.unpack(init, packet._embedded_profile(init))
        b = dict(
            base_bytes=len(sections["base"]),
            pose_bytes=len(sections["poses"]),
            ball_bytes=len(sections["trajectory"]),
            init_bytes=len(init),
            joint_bytes=len(payload),
        )
        assert all(b[k] == w[k] for k in b)
        b["joint_header_and_framing_bytes"] = roles["window"]
        b["reference_bytes"] = init_roles["reference"]
        b["initialization_framing_and_resource_ids"] = init_roles["window"]
        b["reference_sections"] = {k: len(v) for k, v in assets.items()}
        assert sum(
            b[k]
            for k in ["base_bytes", "pose_bytes", "ball_bytes", "joint_header_and_framing_bytes"]
        ) == len(payload)
        assert sum(b["reference_sections"].values()) == b["reference_bytes"] and b[
            "reference_bytes"
        ] + b["initialization_framing_and_resource_ids"] == len(init)
        results.append(
            dict(
                shot=shot,
                source=r["source_sha256"],
                wire=w["joint_sha256"],
                cache=w["init_sha256"],
                **b,
                native_fps=50,
                frames=240,
                active_frames=120,
                active=r["active"],
                warm_kbps=w["kbps_cached"],
                cold_kbps=w["kbps_cold"],
            )
        )
    a.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
