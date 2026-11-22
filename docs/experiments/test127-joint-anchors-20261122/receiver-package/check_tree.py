"""Exercise the paid Sydney conditioning path from a single extracted source tree."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def check(tree, carrier, manifest, specs, pair):
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise RuntimeError("run this probe with Python -I -B")
    sys.path.insert(0, str(tree.resolve(strict=True)))
    import cv2
    import numpy as np
    from experiments.tennis import common_receiver_generate as gen
    from experiments.tennis import common_receiver_isolation
    from experiments.tennis import common_receiver_job as job
    from experiments.visor import b1

    cv2.setNumThreads(1)
    profile = job.packet._embedded_profile(carrier.read_bytes())
    if profile["recipe"]["fps"] != [60, 1]:
        raise ValueError("Sydney source must use its original 60-fps cadence")
    result = {
        "producer_hashes": gen.producer_sources(),
        "roles": {},
        "fps": profile["recipe"]["fps"],
    }
    for spec_file in specs:
        spec = json.loads(spec_file.read_text())
        args = spec["arguments"]
        unit = args[args.index("--unit") + 1]
        expected = args[args.index("--carrier-sha256") + 1]
        binding = job.manifest.verify(manifest, expected)
        if binding["generator_env"] != spec["environment"]["sha256"]:
            raise ValueError("environment identity differs from the paid manifest")
        by_name = {s["name"]: s for s in spec["staged_inputs"]}
        if (
            binding["manifest_sha256"] != by_name["manifest"]["sha256"]
            or expected != by_name["carrier"]["sha256"]
        ):
            raise ValueError("submitted inputs differ from the sender")
        count = spec["scale"]["ticks"]["full"]
        prefix = spec["scale"]["ticks"]["smoke"]
        if (count, prefix) != (240, 64):
            raise ValueError("wrong Sydney source cadence or pilot length")
        role = {}
        stages = [prefix, count]
        if pair:
            from experiments.tennis import reference_pair_validation
            from experiments.jobs import inbox

            pair_spec = json.loads(pair.read_text())
            staged = [dict(row) for row in pair_spec["staged_inputs"]]
            for item in staged:
                if item["name"] in {"carrier", "manifest"}:
                    item["path"] = str({"carrier": carrier, "manifest": manifest}[item["name"]])
            os.environ["PS_STAGE"] = "smoke" if unit == "match127_near" else "full"
            command = inbox.phase_command(
                pair_spec, os.environ["PS_STAGE"], staged, python=sys.executable
            )

            def value(flag):
                return command[command.index(flag) + 1]

            if int(value("--pair-stage")) != (1 if unit == "match127_near" else 2):
                raise ValueError("unbound pilot stage")
            selection = argparse.Namespace(
                unit=value("--unit"),
                comparison=value("--comparison"),
                control=Path(value("--control")),
                q90=Path(value("--q90")),
                control_manifest=Path(value("--control-manifest")),
                q90_manifest=Path(value("--q90-manifest")),
                weight=[Path(command[i + 1]) for i, v in enumerate(command) if v == "--weight"],
                deny=[Path(command[i + 1]) for i, v in enumerate(command) if v == "--deny"],
            )
            reference_pair_validation.select(selection)
            if (selection.frames, selection.unit, selection.carrier_sha256) != (24, unit, expected):
                raise ValueError("24-tick stage selected an unexpected role or carrier")
            stages.insert(0, 24)
        for n in stages:
            ((actual, seed, arrays),) = gen.conditioning(carrier, expected, n, [unit])
            poses = arrays["pose"]
            if (
                actual != unit
                or poses.shape != (n, 784, 512, 3)
                or poses.dtype != np.uint8
                or not np.array_equal(arrays["t"], np.arange(n))
                or arrays["box"].shape != (n, 4)
                or arrays["reference"].shape != poses.shape[1:]
                or not np.any(poses)
                or not np.any(arrays["reference"])
            ):
                raise ValueError("incomplete native conditioning")
            digests = {
                k: hashlib.sha256(np.ascontiguousarray(v)).hexdigest() for k, v in arrays.items()
            }
            if n == count:
                for shorter, original in role.items():
                    if original["seed"] != seed or original["reference"] != digests["reference"]:
                        raise ValueError("reference or seed changed with the selected interval")
                    for key in ("pose", "box", "t"):
                        chunk = np.ascontiguousarray(arrays[key][: int(shorter)])
                        if hashlib.sha256(chunk).hexdigest() != original[key]:
                            raise ValueError("conditioning prefix mismatch for " + key)
            role[str(n)] = {**digests, "seed": seed}
            del poses, arrays
        result["roles"][unit] = role
    if set(result["roles"]) != {"match127_far", "match127_near"}:
        raise ValueError("two distinct player units are required")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"experiments", "src", "tools"} and getattr(
            module, "__file__", None
        ):
            if (
                not Path(module.__file__)
                .resolve(strict=True)
                .is_relative_to(tree.resolve(strict=True))
            ):
                raise ValueError("project module imported from outside extracted tree: " + name)
    result["required_modules"] = [m.__name__ for m in (b1, job, gen, common_receiver_isolation)]
    result["observed_local_runtime"] = {
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    if pair:
        result["required_modules"].append(reference_pair_validation.__name__)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tree", type=Path)
    p.add_argument("carrier", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("specs", nargs=2, type=Path)
    p.add_argument("--pair", type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.tree, a.carrier, a.manifest, a.specs, a.pair)))
