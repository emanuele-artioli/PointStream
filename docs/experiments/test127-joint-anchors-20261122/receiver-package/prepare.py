"""Prepare a clean Sydney two-player source archive and validate it without registering a job."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
NATIVE = HERE.parents[1] / "test-joint-extension-20261122" / "native"
SELECTION = ("experiments", "src", "scripts", "tools")
ENTRY = "experiments.tennis.common_receiver_job"
# Loading a site directory without processing .pth files prevents editable installs from supplying missing code.
BOOT = "import os,runpy,sys;sys.path.extend(sys.argv.pop(1).split(os.pathsep));"
RUN_SCRIPT = BOOT + "s=sys.argv.pop(1);sys.argv[0]=s;runpy.run_path(s,run_name='__main__')"
RUN_MODULE = (
    BOOT
    + "sys.path.insert(0,sys.argv.pop(1));m=sys.argv.pop(1);sys.argv[0]=m;runpy.run_module(m,run_name='__main__')"
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare(a):
    source = a.source_tree.resolve(strict=True)
    if a.output.resolve().is_relative_to(source):
        raise ValueError("write the generated source archive outside the source checkout")
    if a.deadline:
        deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
        if not deadline.tzinfo or deadline <= datetime.now(timezone.utc):
            raise ValueError("specs require a future UTC deadline with offset")
    sys.path.insert(0, str(source))
    from experiments.jobs import fleet, inbox
    from experiments.tennis import w8_strict_source

    a.output.mkdir(exist_ok=False, parents=True)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    archive, sha, metadata = fleet._build_snapshot(source, include_paths=SELECTION)
    target = a.output / "source.tar"
    shutil.move(archive, target)
    if head != metadata["git_head"] or digest(target) != sha:
        raise ValueError("source archive identity")
    isolated = [sys.executable, "-I", "-S", "-B", "-c"]
    packages = os.pathsep.join(dict.fromkeys(sysconfig.get_path(k) for k in ("purelib", "platlib")))
    specs = [NATIVE / ("match127-generate-" + r + ".json") for r in ("far", "near")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    receipt = {
        "source_revision": metadata["git_head"],
        "source_tar_sha256": sha,
        "tracked_paths": SELECTION,
        "template_sha256": {p.name: digest(p) for p in specs},
        "registered_jobs": 0,
        "gpu_seconds": 0,
    }
    if a.pair:
        pair_path = NATIVE / "match127-role-pilot-24.json"
        pair_contents = subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "show",
                head + ":docs/experiments/test-joint-extension-20261122/native/" + pair_path.name,
            ]
        )
        if pair_contents != pair_path.read_bytes():
            raise ValueError("pair request differs from its committed source revision")
        receipt["pair_spec_sha256"] = digest(pair_path)
    terminal = json.loads((NATIVE / "match127-generator-terminal-20261122.json").read_text())
    for role, path in zip(("far", "near"), specs, strict=True):
        frozen = inbox.validate_spec(
            json.loads(path.read_text()), now=terminal[role]["started_unix"]
        )
        if inbox.digest(frozen) != terminal[role]["spec_sha256"]:
            raise ValueError("receiver template differs from the registered job")
    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch)
        if a.failed_archive:
            hashes = {r["snapshot"] for r in terminal.values()}
            registered = json.loads((HERE / "original-source.json").read_text())
            if (
                hashes != {registered["source_sha256"]}
                or digest(a.failed_archive) != registered["snapshot_sha256"]
                or terminal["near"]["spec_sha256"] != registered["spec_sha256"]
            ):
                raise ValueError("original failed archive identity")
            old = temp / "old"
            if (
                inbox.extract_source(a.failed_archive, old, registered)
                != registered["source_sha256"]
            ):
                raise ValueError("original extracted source identity")
            negative = subprocess.run(
                [*isolated, RUN_MODULE, packages, str(old), ENTRY, "run", "--help"],
                cwd=old,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            if (
                negative.returncode == 0
                or "ModuleNotFoundError: No module named 'experiments.visor'" not in negative.stderr
            ):
                raise ValueError(
                    f"original entry returned {negative.returncode}; {negative.stderr[-1200:]}"
                )
            receipt["original_failure"] = {
                "source_tar_sha256": digest(a.failed_archive),
                "missing_module": "experiments.visor",
            }
        outer = temp / "outer"
        receipt["extracted_source_identity"] = inbox.extract_source(target, outer, metadata)
        inner = temp / "inner"
        inner_manifest = w8_strict_source.stage_source(outer, inner)
        w8_strict_source.verify_source(inner, inner_manifest)
        receipt["worker_python_manifest_sha256"] = inner_manifest["manifest_sha256"]
        receipt["worker_python_file_count"] = len(inner_manifest["files"])
        for tree in (outer, inner):
            probe = subprocess.run(
                [*isolated, RUN_MODULE, packages, str(tree), ENTRY, "run", "--help"],
                cwd=tree,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=45,
            )
            if not all(
                flag in probe.stdout for flag in ("--carrier", "--run-length", "--unit", "--weight")
            ):
                raise ValueError("receiver CLI failed")
            if a.pair:
                paired = subprocess.check_output(
                    [
                        *isolated,
                        RUN_MODULE,
                        packages,
                        str(tree),
                        "experiments.tennis.reference_pair_validation",
                        "run",
                        "--help",
                    ],
                    cwd=tree,
                    env=env,
                    text=True,
                    timeout=45,
                )
                if "sydney-roles" not in paired:
                    raise ValueError("Sydney pair is unavailable in this source revision")
            report = subprocess.check_output(
                [
                    *isolated,
                    RUN_SCRIPT,
                    packages,
                    str(HERE / "check_tree.py"),
                    str(tree),
                    str(a.carrier.resolve(strict=True)),
                    str(a.manifest.resolve(strict=True)),
                    *[str(s) for s in specs],
                    *(["--pair", str(NATIVE / "match127-role-pilot-24.json")] if a.pair else []),
                ],
                cwd=tree,
                env=env,
                text=True,
                timeout=120,
            )
            checked = json.loads(report)
            if tree == outer:
                receipt["conditioning"] = checked
            elif checked != receipt["conditioning"]:
                raise ValueError("outer and source-denied worker trees disagree")
    if a.deadline:
        for path in specs:
            spec = json.loads(path.read_text())
            spec["deadline"] = deadline.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if spec["entrypoint"] != ["-m", ENTRY] or spec["validator"] != [
                "{python}",
                "-m",
                ENTRY,
                "validate-stage",
            ]:
                raise ValueError("unexpected role entrypoint or stage validator")
            inbox.validate_spec(spec)
            save(a.output / path.name, spec)
    save(a.output / "preflight.json", receipt)
    print(
        json.dumps(
            {
                "output": str(a.output),
                "source_tar_sha256": sha,
                "roles": list(receipt["conditioning"]["roles"]),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-tree", "carrier", "manifest", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument(
        "--failed-archive",
        type=Path,
        help="verify the exact original regression against its recorded archive SHA",
    )
    p.add_argument(
        "--pair",
        action="store_true",
        help="also verify both real 24-tick role selections and their conditioning",
    )
    p.add_argument(
        "--deadline", help="write both future-window specs, changing only the expired deadline"
    )
    prepare(p.parse_args())
