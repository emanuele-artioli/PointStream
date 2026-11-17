"""Prepare authenticated, unreleased JPEG-reference pilot inputs on CPU."""
from __future__ import annotations

import argparse
import copy
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.jobs.inbox import validate_spec
from experiments.tennis.k6b_generate import sha256
from experiments.tennis.k6b_rate_streams import decode_reference, encode_reference
from experiments.tennis.player_requalification import array_sha
from experiments.tennis.player_requalification_specs import build_specs
from experiments.tennis.w8_payload import unpack_window

PACKET = "eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4"
PROFILE = "bec98d67280e9f777a0d3cc110baa51ae2bf2c876aefe6b247d9691432f61107"
ROOT = Path("docs/experiments/player-requalification-20261116")
REMOTE = "/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-pilot-20261117"


def prepare(source: Path, output: Path) -> dict[str, Any]:
    packet, profile_path = source / "packet-eff1aca8.psw8", source / "profile-bec98d67.json"
    if sha256(packet) != PACKET or sha256(profile_path) != PROFILE:
        raise ValueError("packet or profile identity mismatch")
    profile = json.loads(profile_path.read_text())
    sections, _ = unpack_window(packet.read_bytes(), profile)
    original_config = json.loads((ROOT / "config.json").read_text())
    original_units = {unit["unit"]: unit for unit in original_config["units"]}
    if set(original_units) != {u["id"] for u in profile["units"]}:
        raise ValueError("packet unit mismatch")
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for quality in (70, 40):
        folder = output / f"q{quality}"
        publish = folder / "publish"
        publish.mkdir(parents=True)
        config = copy.deepcopy(original_config)
        config["units"] = []
        rows = []
        for index in (1, 0):
            binding = profile["units"][index]
            name = binding["id"]
            unit = copy.deepcopy(original_units[name])
            source_npz = source / f"{name}.npz"
            if sha256(source_npz) != unit["sha256"]:
                raise ValueError("source conditioning mismatch")
            old_ref = sections[binding["reference"]]
            meta, image = decode_reference(old_ref)
            with np.load(source_npz, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            if not np.array_equal(image, arrays["reference"]):
                raise ValueError("packet and conditioning reference mismatch")
            clean_meta = {k: v for k, v in meta.items() if k not in ("codec", "quality", "color")}
            encoded = encode_reference(image, clean_meta, quality)
            decoded_meta, decoded = decode_reference(encoded)
            if decoded_meta != {**clean_meta, "codec": ".jpg", "quality": quality, "color": "RGB8"}:
                raise ValueError("reference metadata mismatch")
            ref_path = folder / f"{name}.ref1"
            ref_path.write_bytes(encoded)
            arrays["reference"] = decoded
            target = publish / f"{name}.npz"
            np.savez_compressed(target, **arrays)
            unit["sha256"] = sha256(target)
            unit["review_label"] = f"Player {index + 1}"
            config["units"].append(unit)
            rows.append({"unit": name, "review_label": unit["review_label"],
                         "source_npz_sha256": sha256(source_npz), "source_rgb_sha256": array_sha(image),
                         "old_ref1_sha256": __import__("hashlib").sha256(old_ref).hexdigest(),
                         "old_ref1_bytes": len(old_ref), "encoded_ref1_sha256": sha256(ref_path),
                         "encoded_ref1_bytes": len(encoded), "decoded_rgb_sha256": array_sha(decoded),
                         "new_npz_sha256": unit["sha256"], "new_npz_bytes": target.stat().st_size})
        archive = folder / "conditioning.tar"
        with tarfile.open(archive, "w") as tar:
            for path in sorted(publish.iterdir()):
                tar.add(path, arcname=f"publish/{path.name}")
        provenance = {"quality": quality, "packet_sha256": PACKET, "profile_sha256": PROFILE, "units": rows}
        config["reference_pilot"] = provenance
        config["conditioning"]["provenance"] = "Authenticated packet references JPEG encoded and decoded; all other conditioning retained. Appearance pilot, not complete codec."
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        inventory = json.loads((ROOT / "inputs.json").read_text())
        conditioning = next(x for x in inventory["inputs"] if x["name"] == "conditioning")
        conditioning.update(path=f"{REMOTE}/q{quality}/conditioning.tar", sha256=sha256(archive))
        inventory_path = folder / "inputs.json"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        # Placeholder deadline must be replaced after release, before validation/dispatch.
        spec = build_specs(config_path, inventory_path, f"{REMOTE}/q{quality}/config.json", "2099-01-01T00:00:00Z")["tennis_only"]
        spec["smoke"].update(seconds=300, representative_basis="Player 2: one complete 24-tick context with the same JPEG reference and production settings as full.")
        spec["full"]["seconds"] = 450
        spec["budget_seconds"] = 900
        validate_spec(spec)
        spec_path = folder / "spec-unreleased.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        summaries[str(quality)] = {**provenance, "archive_sha256": sha256(archive),
                                   "archive_bytes": archive.stat().st_size,
                                   "config_sha256": sha256(config_path), "spec_sha256": sha256(spec_path)}
    (output / "manifest.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
    print(f"Prepared unreleased pilot at {args.output}")


if __name__ == "__main__":
    main()
