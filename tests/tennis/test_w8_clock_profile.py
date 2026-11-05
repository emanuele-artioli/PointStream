"""Verify opt-in clock grants and receipt binding without GPU or model inputs."""

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.tennis import w8_clock_profile as clock


def receipt():
    parents = []
    for i, path in enumerate(clock.PATHS):
        parents.append(
            {
                "path": str(path),
                "device": 23,
                "inode": 100 + i,
                "uid": 0,
                "mode": stat.S_IFDIR | 0o755 if i == 0 else stat.S_IFREG | 0o644,
                "sha256": None if i == 0 else "a" * 64,
            }
        )
    encoded = (
        json.dumps({"profile": clock.PROFILE, "objects": parents}, sort_keys=True) + "\n"
    ).encode()
    return {
        "profile": clock.PROFILE,
        "parent_receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "objects": [
            {
                "parent": p,
                "child": dict(p, uid=65534),
                "rights": "READ_DIR recursive" if i == 0 else "READ_FILE",
            }
            for i, p in enumerate(parents)
        ],
    }


def test_clock_receipt_rejects_wider_or_unbound_grants():
    value = receipt()
    clock.validate_receipt(value)
    value["objects"][0]["rights"] = "READ_FILE|READ_DIR"
    with pytest.raises(ValueError, match="identity differs"):
        clock.validate_receipt(value)
    value = receipt()
    value["objects"][1]["parent"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="binding differs"):
        clock.validate_receipt(value)
    value = receipt()
    value["objects"][1]["child"]["inode"] += 1
    with pytest.raises(ValueError, match="identity differs"):
        clock.validate_receipt(value)


def test_restrict_is_opt_in_read_only_and_restores_common_policy():
    observed = []

    def old(path, output, abi):
        return 4

    def apply(paths, output):
        observed.extend(fake.allowed_rights(p, output, 4) for p in paths)
        return 4, 123

    fake = SimpleNamespace(allowed_rights=old, restrict=apply, READ_DIR=8)
    assert clock.restrict(fake, list(clock.PATHS), Path("/output")) == (4, 123)
    assert observed == [8, 4, 4] and fake.allowed_rights is old

    def fail(paths, output):
        raise OSError("failure")

    fake.restrict = fail
    with pytest.raises(OSError):
        clock.restrict(fake, list(clock.PATHS), Path("/output"))
    assert fake.allowed_rights is old


def test_child_rejects_protected_alias_and_parent_sha_tamper(tmp_path, monkeypatch):
    value = receipt()
    path = tmp_path / "receipt.json"
    parent = {"profile": clock.PROFILE, "objects": [r["parent"] for r in value["objects"]]}
    path.write_text(json.dumps(parent, sort_keys=True) + "\n")
    monkeypatch.setattr(
        clock,
        "_public_identity",
        lambda p, **kw: next(r["child"] for r in value["objects"] if r["child"]["path"] == str(p)),
    )
    protected = tmp_path / "protected"
    protected.write_bytes(b"synthetic")
    result, proof = clock.prepare_child(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), [protected]
    )
    assert result == list(clock.PATHS)
    clock.validate_receipt(proof)
    with pytest.raises(ValueError, match="hash differs"):
        clock.prepare_child(path, "0" * 64, [protected])
    fake_stat = protected.stat()
    value["objects"][0]["child"]["device"] = fake_stat.st_dev
    value["objects"][0]["child"]["inode"] = fake_stat.st_ino
    value["objects"][0]["parent"]["device"] = fake_stat.st_dev
    value["objects"][0]["parent"]["inode"] = fake_stat.st_ino
    path.write_text(
        json.dumps(
            {"profile": clock.PROFILE, "objects": [r["parent"] for r in value["objects"]]},
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="protected alias"):
        clock.prepare_child(path, hashlib.sha256(path.read_bytes()).hexdigest(), [protected])


def test_actual_generator_binding_uses_manifest_and_rejects_tampering(tmp_path):
    from experiments.tennis import w8_strict_host_cost_job as job

    relative = "experiments/tennis/w8_clock_profile.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("synthetic source")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    files = {}
    job.add_clock_source(files, tmp_path, {"files": {relative: sha}})
    assert files["code:experiments.tennis.w8_clock_profile"] == {"path": str(source), "sha256": sha}
    source.write_text("tampered source")
    with pytest.raises(ValueError, match="source identity differs"):
        job.add_clock_source({}, tmp_path, {"files": {relative: sha}})


def test_actual_post_constructor_guard_binds_opt_in_helper_path_and_hash(tmp_path):
    import ast
    import inspect
    from experiments.tennis import w8_receiver_generate as receiver

    tree = ast.parse(inspect.getsource(receiver.run_isolated))
    function = tree.body[0]
    statements = function.body
    first = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "clock_module" for t in node.targets)
    )
    # Execute the actual post-constructor selection and unchanged comparison body.
    guard = compile(
        ast.fix_missing_locations(ast.Module(body=statements[first : first + 4], type_ignores=[])),
        "<actual-guard>",
        "exec",
    )
    path_loop = next(
        node
        for node in statements
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(key, path)"
        and ast.unparse(node.iter) == "paths.items()"
    )
    hash_guard = compile(
        ast.fix_missing_locations(ast.Module(body=[path_loop], type_ignores=[])),
        "<actual-asset-guard>",
        "exec",
    )
    helper = tmp_path / "w8_clock_profile.py"
    aa = tmp_path / "__init__.py"
    helper.write_text("helper source")
    aa.write_text("aa source")
    files = {
        "code:experiments.tennis.w8_clock_profile": {
            "path": str(helper),
            "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        },
        "code:animate_anyone": {
            "path": str(aa),
            "sha256": hashlib.sha256(aa.read_bytes()).hexdigest(),
        },
    }
    paths = {name: Path(item["path"]) for name, item in files.items()}
    fake_modules = {
        "experiments.tennis.w8_clock_profile": SimpleNamespace(__file__=str(helper)),
        "animate_anyone": SimpleNamespace(__file__=str(aa)),
    }
    env = {
        "Path": Path,
        "sys": SimpleNamespace(modules=fake_modules),
        "paths": paths,
        "isolation": {"clock_profile": {"profile": clock.PROFILE}},
        "config": {"files": files},
        "fixed": SimpleNamespace(file_sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()),
        "aa_verified": {str(aa.resolve()): hashlib.sha256(aa.read_bytes()).hexdigest()},
    }
    exec(hash_guard, env)
    exec(guard, env)
    wrong = tmp_path / "wrong.py"
    wrong.write_text("helper source")
    fake_modules["experiments.tennis.w8_clock_profile"].__file__ = str(wrong)
    with pytest.raises(ValueError, match="undeclared executing source module"):
        exec(guard, env)
    fake_modules["experiments.tennis.w8_clock_profile"].__file__ = str(helper)
    helper.write_text("tampered helper")
    with pytest.raises(ValueError, match="executing source changed"):
        exec(guard, env)
    with pytest.raises(ValueError, match="asset identity mismatch"):
        exec(hash_guard, env)
    fake_modules["experiments.tennis.w8_clock_profile"].__file__ = ""
    with pytest.raises(ValueError, match="module is not loaded or has no origin"):
        exec(guard, env)
    del fake_modules["experiments.tennis.w8_clock_profile"]
    with pytest.raises(ValueError, match="module is not loaded"):
        exec(guard, env)


def test_candidate_spec_changes_only_clock_opt_in_and_deadline():
    from experiments.tennis import w8_strict_host_cost_job as job

    root = Path(__file__).resolve().parents[2] / "docs/experiments/w8-inputs/receiver-20261101"
    old = json.loads((root / "w8-declared-aa-requalification-gpu5-spec-20261105.json").read_text())
    new = json.loads((root / "w8-clock-qualified-cost-gpu5-spec-20261105.json").read_text())
    assert new["clock_policy_profile"] == clock.PROFILE
    assert new["clock_reference_audit_sha256"] == job.CLOCK_AUDIT_SHA
    assert new["arguments"] == old["arguments"] + ["--clock-profile"]
    for value in (old, new):
        value.pop("deadline", None)
        value.pop("deadline_epoch", None)
    new.pop("clock_policy_profile")
    new.pop("clock_reference_audit_sha256")
    new["arguments"].remove("--clock-profile")
    assert new == old


def test_retained_generator_config_is_joined_to_source_manifest(tmp_path):
    from experiments.tennis import w8_strict_host_cost_job as job

    name = "experiments.tennis.w8_clock_profile"
    relative = name.replace(".", "/") + ".py"
    entry = {"path": str(tmp_path / relative), "sha256": "a" * 64}
    config = {"allow_paths": [str(tmp_path)], "files": {f"code:{name}": entry}}
    origins = {name: relative}
    files = {relative: "a" * 64}
    job.check_clock_source_binding(config, origins, files)
    entry["path"] = str(tmp_path / "other.py")
    with pytest.raises(ValueError, match="config and source manifest"):
        job.check_clock_source_binding(config, origins, files)
    entry["path"] = str(tmp_path / relative)
    entry["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="config and source manifest"):
        job.check_clock_source_binding(config, origins, files)
    entry["sha256"] = files[relative]
    with pytest.raises(ValueError, match="config and source manifest"):
        job.check_clock_source_binding(config, {}, files)
