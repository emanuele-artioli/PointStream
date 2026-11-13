"""Opt-in study contracts; no native tools, models, GPU or fleet execution."""

import datetime
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.jobs import inbox
from experiments.tennis import k5d_svt_oracle as oracle, k5d_svt_profile as profile

IDENTITY = {"source_sha256": "a" * 64, "snapshot_sha256": "b" * 64, "environment_sha256": "c" * 64}
REQUIRED = {
    "qualification16": (),
    "cost48": ("qualification16",),
    "confirmation292": ("qualification16", "cost48"),
    "confirmation586": ("qualification16", "cost48", "confirmation292"),
    "study": ("qualification16", "cost48", "confirmation292", "confirmation586"),
}
SPANS = {
    "qualification16": ("tracknet/game10/Clip3", 16),
    "cost48": ("tracknet/game10/Clip3", 48),
    "confirmation292": ("tracknet/game9/Clip5", 292),
    "confirmation586": ("tracknet/game10/Clip12", 586),
    "study": ("tracknet/game10/Clip12", 586),
}


def asset(tmp_path, name, value):
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(value))
    return {
        "local": str(path),
        "remote": "/home/itec/emanuele/Datasets/pointstream-data/checks/" + path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def gate(stage):
    clip, frames = SPANS[stage]
    return {
        "approved": True,
        "purpose": "SVT-only-study-stage",
        "execution_identity": IDENTITY,
        "selection_sha256": oracle.fleet.SOURCE_SHA,
        "profile": profile.STUDY_PROFILE,
        "stage": stage,
        "clip": clip,
        "frames": frames,
        "evidence_sha256": {name: "d" * 64 for name in REQUIRED[stage]},
        "complete_forecast_fits": True,
        "cost_margin": 1.5,
        "failure_reserve_seconds": 30,
    }


def spec_scope(stage):
    clip, frames = SPANS[stage]
    return {
        "k5d_evaluation_profile": profile.STUDY_PROFILE,
        "k5d_study_stage": stage,
        "k5d_study_clip": clip,
        "k5d_study_frames": frames,
    }


@pytest.mark.parametrize("stage", profile.STUDY_STAGES)
def test_stage_decision_is_source_bound_and_prospective(tmp_path, stage):
    entry = asset(tmp_path, "gate", gate(stage))
    args = SimpleNamespace(study_gate=entry["local"], study_gate_sha256=entry["sha256"])
    assert oracle.study_admission(args, IDENTITY, spec_scope(stage))["stage"] == stage
    with pytest.raises(ValueError, match="source-bound"):
        oracle.study_admission(args, {**IDENTITY, "source_sha256": "e" * 64}, spec_scope(stage))


@pytest.mark.parametrize(
    "change",
    [
        {"approved": False},
        {"clip": "tracknet/game10/Clip3"},
        {"frames": 291},
        {"evidence_sha256": {}},
        {"complete_forecast_fits": False},
        {"cost_margin": 1.49},
        {"failure_reserve_seconds": 0},
        {"cost_margin": True},
    ],
)
def test_confirmation_cannot_skip_actual_cost_approval(tmp_path, change):
    value = {**gate("confirmation292"), **change}
    entry = asset(tmp_path, "gate", value)
    args = SimpleNamespace(study_gate=entry["local"], study_gate_sha256=entry["sha256"])
    with pytest.raises(ValueError):
        oracle.study_admission(args, IDENTITY, spec_scope("confirmation292"))


def candidate(tmp_path, monkeypatch, stage="cost48"):
    monkeypatch.setattr(oracle, "source_admission", lambda *a: None)
    assets = {name: asset(tmp_path, name, {}) for name in sorted(oracle.ASSETS)}
    monkeypatch.setattr(oracle, "REFERENCE_SHA", assets["ada_reference"]["sha256"])
    decision = asset(tmp_path, "study_gate", gate(stage))
    clip, frames = SPANS[stage]
    fidelity = None
    if stage in ("confirmation292", "confirmation586", "study"):
        value = {
            "schema_version": 1,
            "passed": True,
            "scope": "fidelity-only",
            "archive_sha256": oracle.fleet.EXPECTED["archive"],
            "calibration_job_id": "actual",
            "evidence_sha256": "e" * 64,
            "checks": [
                "source_transform",
                "native_and_wire",
                "receiver_parity",
                "perceptual_metrics",
                "full_cost",
                "pair_cost",
            ],
            "purpose": "first-292-pair-v1"
            if stage == "confirmation292"
            else "full-retrospective-benchmark",
            "clip": clip,
            "max_frames": frames,
            "pair_cost_bound_sha256": "f" * 64,
        }
        fidelity = asset(tmp_path, "study_fidelity_gate", value)
    base = json.loads(oracle.cf.BASE.read_text())
    kwargs = dict(
        assets=assets,
        identity=IDENTITY,
        deadline=(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=40)
        ).isoformat(),
        unshare_sha256="a" * 64,
        smoke_seconds=100,
        budget_seconds=1000,
        study_stage=stage,
        clip=clip,
        frames=frames,
        study_gate=decision,
        study_fidelity_gate=fidelity,
        full_seconds=5 if stage == "qualification16" else 600,
    )
    return base, kwargs, oracle.spec(base, **kwargs)


@pytest.mark.parametrize("stage", profile.STUDY_STAGES)
def test_constructor_keeps_real16_gold_smoke_and_declared_full_context(
    tmp_path, monkeypatch, stage
):
    base, kwargs, result = candidate(tmp_path, monkeypatch, stage)
    assert result["k5d_evaluation_profile"] == profile.STUDY_PROFILE
    assert "--image-ckpt" in base["arguments"] and "--image-ckpt" not in result["arguments"]
    staged = result["staged_inputs"]
    for phase in ("smoke", "full"):
        command = inbox.phase_command(result, phase, staged, "/python")
        args = oracle.parser().parse_args(command[3:])
        monkeypatch.setenv("PS_STAGE", phase)
        if phase == "full" and stage == "qualification16":
            assert args.action == "noop"
        else:
            scope = oracle.execution_scope(args, result)
            assert scope["stage"] == ("qualification16" if phase == "smoke" else stage)
            assert (args.clip, args.frames) == (
                SPANS["qualification16"] if phase == "smoke" else SPANS[stage]
            )
    assert result["contention"]["resume_attempts"] == 0
    assert not {"image_ckpt", "video_ckpt"} & {x["name"] for x in staged}
    assert result["validator"] == [
        "{python}",
        "-m",
        "experiments.tennis.k5d_svt_oracle",
        "validate",
    ]
    # Scope alone never grants a later stage, even if the saved command is altered.
    args.study_stage = "cost48"
    args.clip = SPANS["cost48"][0]
    args.frames = 48
    monkeypatch.setenv("PS_STAGE", "smoke")
    with pytest.raises(ValueError, match="dispatched"):
        oracle.execution_scope(args, result)


@pytest.mark.parametrize(
    "change",
    [
        {"feasibility_no_metrics": True},
        {"isolated_receiver": False},
        {"same_job_receiver": False},
        {"cost_pilot": True},
        {"benchmark": False},
        {"start_frame": 1},
    ],
)
def test_full_encoder_never_weakens_receiving_scoring_or_context(change):
    args = SimpleNamespace(
        study_stage="confirmation292",
        clip=SPANS["confirmation292"][0],
        frames=292,
        start_frame=0,
        feasibility_no_metrics=False,
        isolated_receiver=True,
        same_job_receiver=True,
        cost_pilot=False,
        benchmark=True,
    )
    assert oracle.enc.study_scope(args)["full_context"] is True
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        oracle.enc.study_scope(args)


def validator_fixture(tmp_path, monkeypatch, stage):
    clip, frames = SPANS[stage]
    scope = profile.scope(stage, clip, frames)
    context = {
        "execution_identity": IDENTITY,
        "job_id": "actual",
        "stage": "full",
        "gpu": {"uuid": "GPU-real"},
    }
    points = []
    for p in (48, 55, 62):
        receiver = {
            "decoded_yuv_sha256": hashlib.sha256(b"yuv").hexdigest(),
            "rendered_rgb_sha256": hashlib.sha256(b"rgb").hexdigest(),
        }
        point = {"point": p, "receiver": receiver, "streak": {"guard_off": True}}
        points.append(point)
        directory = tmp_path / f"svt-{p}"
        evidence = directory / "cpu-receiver-evidence"
        evidence.mkdir(parents=True)
        for name, value in [
            ("packet.psbl", b"wire"),
            ("native.ivf", b"native"),
            ("native.yuv", b"yuv"),
            ("rendered.rgb", b"rgb"),
        ]:
            (evidence / name).write_bytes(value)
        (directory / "stream.psbl").write_bytes(b"wire")
        (directory / "native.bin").write_bytes(b"native")
    absent, excluded = (
        (0, 0) if stage == "cost48" else (30, 1) if stage == "confirmation292" else (5, 0)
    )
    labels = [0] * absent + [-1] * excluded + [1] * (frames - absent - excluded)
    qualification = {
        "frames": frames,
        "qualified": False,
        "rows": [{"label_kind": k} for k in labels],
        "identities": {
            key: {"sha256": value}
            for key, value in [
                ("exposure", oracle.fleet.EXPOSURE_SHA),
                ("manifest", oracle.fleet.SOURCE_SHA),
                ("archive", oracle.fleet.EXPECTED["archive"]),
            ]
        },
    }
    names = [
        "source.oracle_total",
        "encoder.fit_streak",
        "source.rgb_to_yuv",
        "source.yuv_to_rgb_floor",
        "model.wasb_init",
        "model.balltrack_init",
        "model.lpips_init",
        "model.dists_init",
    ]
    for p in (48, 55, 62):
        names += [
            f"native.svt.{p}.total",
            f"receiver.svt.{p}",
            f"encoder.svt.{p}.guard",
            f"metrics.svt.{p}.baseline",
            f"judges.svt.{p}.baseline",
        ]
    files = {
        "pilot.json": {
            "clip": clip,
            "identities": {"lpips_backbone": {}, "dists_backbone": {}},
            "points": points,
        },
        "svt-oracle.json": {
            "profile": profile.STUDY_PROFILE,
            "execution_identity": IDENTITY,
            "producer_context": context,
            "points": [list(p) for p in profile.POINTS],
            "receiver_config_sha256": "cfg",
            "study_scope": scope,
        },
        "resources.json": {"resources": {"svt": {}}},
        "source-qualification.json": qualification,
        "svt-resource.json": {"setup": {"gpu_uuid": "GPU-real"}},
        "timing.json": {
            "events": [
                {
                    "name": n,
                    "completed": True,
                    "wall_seconds": 0.1,
                    "self_cpu_seconds": 0.1,
                    "waited_children_cpu_seconds": 0.0,
                }
                for n in names
            ]
        },
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(
        oracle.enc,
        "validate",
        lambda *a, **kw: {"frames": frames, "points": 3, "full_metrics": True},
    )
    monkeypatch.setattr(oracle.oracle, "validate_report", lambda *a: None)
    memory_calls = []
    monkeypatch.setattr(profile, "validate_memory", lambda *a, **kw: memory_calls.append(kw))
    receiver_calls = []
    monkeypatch.setattr(
        oracle.isolated, "verify_receipt", lambda *a, **kw: receiver_calls.append(kw)
    )
    return scope, context, qualification, memory_calls, receiver_calls


@pytest.mark.parametrize("stage", ["cost48", "confirmation292", "confirmation586"])
def test_fresh_byte_joins_and_failed_source_coverage_are_retained(tmp_path, monkeypatch, stage):
    scope, context, qualification, memory, receivers = validator_fixture(
        tmp_path, monkeypatch, stage
    )
    result = oracle.validate_output(
        tmp_path, IDENTITY, producer_context=context, expected_reference=None, study_scope=scope
    )
    assert result["frames"] == scope["frames"] and result["source_task_qualified"] is False
    assert result["source_failures_retained"] and memory == [
        {"expected_profile": profile.STUDY_PROFILE}
    ]
    assert len(receivers) == 3 and all(x["require_cpu_only"] for x in receivers)
    if stage == "confirmation292":
        assert result["label_counts"] == {"absent": 30, "visible": 261, "excluded": 1}
    if stage == "confirmation586":
        assert result["label_counts"] == {"absent": 5, "visible": 581, "excluded": 0}
    with pytest.raises(ValueError, match="no archived gold"):
        oracle.validate_output(
            tmp_path, IDENTITY, producer_context=context, expected_reference={}, study_scope=scope
        )
    (tmp_path / "svt-48/stream.psbl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receiver bytes"):
        oracle.validate_output(
            tmp_path, IDENTITY, producer_context=context, expected_reference=None, study_scope=scope
        )


def test_absent_excluded_frames_cannot_be_removed_or_replaced(tmp_path, monkeypatch):
    scope, context, qualification, _, _ = validator_fixture(
        tmp_path, monkeypatch, "confirmation292"
    )
    qualification["rows"][0]["label_kind"] = 1
    (tmp_path / "source-qualification.json").write_text(json.dumps(qualification))
    with pytest.raises(ValueError, match="absence/exclusion"):
        oracle.validate_output(
            tmp_path, IDENTITY, producer_context=context, expected_reference=None, study_scope=scope
        )


@pytest.mark.parametrize(
    "phase,stage",
    [
        ("smoke", "confirmation292"),
        ("full", "cost48"),
        ("full", "confirmation292"),
        ("full", "confirmation586"),
    ],
)
def test_real_run_wiring_uses_only16_gold_and_full_target_fidelity(
    tmp_path, monkeypatch, phase, stage
):
    import sys

    _, _, specification = candidate(tmp_path, monkeypatch, stage)
    command = inbox.phase_command(specification, phase, specification["staged_inputs"], "/python")
    args = oracle.parser().parse_args(command[3:])
    directory = tmp_path / phase
    directory.mkdir()
    (directory / "dispatch.json").write_text("{}")
    (tmp_path / "spec.json").write_text(json.dumps(specification))
    monkeypatch.setenv("PS_STAGE", phase)
    monkeypatch.setenv("PS_STAGE_DIR", str(directory))
    (tmp_path / "scratch").mkdir()
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-real")
    monkeypatch.setattr(oracle.cp, "execution_identity", lambda: IDENTITY)
    monkeypatch.setattr(oracle, "study_admission", lambda *a: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "Ada"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(profile, "admit", lambda *a: {"profile": profile.STUDY_PROFILE})
    monkeypatch.setattr(profile, "validate_memory", lambda *a, **kw: None)
    monkeypatch.setattr(oracle, "MemoryTimer", lambda *a: SimpleNamespace(observe=lambda _: None))
    monkeypatch.setattr(oracle.witness, "context", lambda *a: {"execution_identity": IDENTITY})
    data = SimpleNamespace(
        output=tmp_path / "output",
        archive_sha256=oracle.fleet.EXPECTED["archive"],
        manifest="manifest",
        exposure="exposure",
        fidelity_gate=args.fidelity_gate,
        fidelity_gate_sha256=args.fidelity_gate_sha256,
    )
    monkeypatch.setattr(oracle.scale, "data_args", lambda *a: data)
    references, comparisons, admissions, encodings = [], [], [], []
    monkeypatch.setattr(oracle, "reference_projection", lambda p: references.append(p) or {})
    monkeypatch.setattr(oracle, "compare_projection", lambda *a: comparisons.append(a))
    monkeypatch.setattr(oracle.enc, "fidelity_gate", lambda *a, **kw: admissions.append((a, kw)))

    def encode(actual, **kwargs):
        encodings.append(kwargs)
        actual.output.mkdir()
        for name in ("pilot.json", "source-qualification.json"):
            (actual.output / name).write_text("{}")

    monkeypatch.setattr(oracle.enc, "run", encode)
    validations = []
    monkeypatch.setattr(oracle, "validate_output", lambda *a, **kw: validations.append(kw))
    oracle.run(args)
    target = "qualification16" if phase == "smoke" else stage
    assert (data.clip, data.study_stage) == (SPANS[target][0], target)
    assert data.cost_pilot is (target in ("qualification16", "cost48"))
    assert data.benchmark is not data.cost_pilot
    assert encodings == [{"point_profile": profile.STUDY_PROFILE}]
    assert bool(references) is (target == "qualification16")
    assert bool(comparisons) is (target == "qualification16")
    expected_purpose = (
        "48-frame-cost-pilot"
        if data.cost_pilot
        else "first-292-pair-v1"
        if target == "confirmation292"
        else "full-retrospective-benchmark"
    )
    assert admissions[0][1] == {
        "purpose": expected_purpose,
        "clip": data.clip,
        "frames": args.frames,
    }
    assert str(admissions[0][0][0]) == (
        args.fidelity_gate if data.cost_pilot else args.study_fidelity_gate
    )
    assert validations[0]["study_scope"] == profile.scope(target, data.clip, args.frames)
    assert (data.output / "svt-reference.json").exists() is (target == "qualification16")
