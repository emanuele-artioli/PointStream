"""Contract tests only: synthetic receipts do not qualify a model or GPU."""
import argparse
import copy
import json

import numpy as np
import pytest

from experiments.tennis import player_requalification as p


@pytest.fixture
def sample(tmp_path, monkeypatch):
    names = {f"weight-{i}": "" for i in range(8)}
    files = {}
    for i, name in enumerate(names):
        path = tmp_path / name
        path.write_text(name)
        names[name] = p.sha256(path)
        files[name] = {"path": str(path), "sha256": names[name]}
    base, match = tmp_path / "base", tmp_path / "match"
    base.write_text("base")
    match.write_text("match")
    monkeypatch.setattr(p, "PINNED_WEIGHTS", names)
    monkeypatch.setattr(p, "BASE_SHA", p.sha256(base))
    unit = tmp_path / "unit.npz"
    pose = np.zeros((24, p.GEN_H, p.GEN_W, 3), np.uint8)
    pose[:, 10, 10, 0] = 255
    np.savez_compressed(unit, reference=np.zeros((p.GEN_H, p.GEN_W, 3), np.uint8), pose=pose,
                        t=np.arange(24), run=np.zeros(24, dtype=np.int64), box=np.tile([1., 2., 100., 150.], (24, 1)))
    cfg = {"schema": 1, "scope": "paired-player-context-development", "settings": p.SETTINGS.copy(),
           "source": {"group": "match-a", "window": "window-a", "source_video_sha256": "a" * 64,
                      "exposure": "opened-development", "native_fps": [30000, 1001], "width": 1920,
                      "height": 1080, "provenance": "declared source"},
           "conditioning": {"provenance": "declared packet-derived pose and granted reference",
                            "not_complete_packet": True, "not_independent_source_decoder": True},
           "units": [{"unit": "window_player", "path": str(unit), "sha256": p.sha256(unit), "run": 0, "start_t": 0}],
           "weights": files, "tennis_checkpoint": {"path": str(base), "sha256": p.BASE_SHA},
           "match_checkpoint": {"path": str(match), "sha256": p.sha256(match), "base_sha256": p.BASE_SHA,
                                "match": "match-a", "bits": 4}}
    config = tmp_path / "config.json"

    def write(value):
        config.write_text(json.dumps(value))
        return config, p.sha256(config)

    return cfg, write, tmp_path


def test_config_rejects_wrong_hash_lineage_and_fps(sample):
    cfg, write, _ = sample
    path, digest = write(cfg)
    assert p.load_config(path, digest) == cfg
    with pytest.raises(ValueError, match="config hash"):
        p.load_config(path, "0" * 64)
    for mutate in (lambda c: c["match_checkpoint"].update(base_sha256="0" * 64),
                   lambda c: c["match_checkpoint"].update(match="other"),
                   lambda c: c["match_checkpoint"].update(bits=0),
                   lambda c: c["source"].update(native_fps=[0, 1]),
                   lambda c: c["settings"].update(cfg=float("nan")),
                   lambda c: c["weights"]["weight-0"].update(sha256="f" * 64)):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError):
            p.load_config(*write(bad))


def test_profile_wiring_and_context_pairing(sample):
    cfg, write, _ = sample
    path, digest = write(cfg)
    cfg = p.load_config(path, digest)
    plan, arrays = p.make_plan(cfg, digest, 1)
    assert list(arrays[0]["t"]) == list(range(24))
    assert p.profile_inputs(cfg, "original") == (None, None)
    tennis, base = p.profile_inputs(cfg, "tennis_only")
    assert str(tennis) == cfg["tennis_checkpoint"]["path"] and base is None
    match, base = p.profile_inputs(cfg, "match_adapted")
    assert str(match) == cfg["match_checkpoint"]["path"] and str(base) == str(tennis)
    receipts = [{"profile": profile, "plan": copy.deepcopy(plan), "plan_sha256": p.digest_json(plan)} for profile in p.PROFILES]
    p.compare_receipts(receipts)
    receipts[1]["plan"]["units"][0]["pose_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unequal conditioning"):
        p.compare_receipts(receipts)


def test_bad_ticks_and_stale_output_are_rejected_before_torch(sample, monkeypatch):
    cfg, write, tmp = sample
    unit = cfg["units"][0]
    with np.load(unit["path"]) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["t"][3] = arrays["t"][2]
    np.savez_compressed(unit["path"], **arrays)
    unit["sha256"] = p.sha256(p.Path(unit["path"]))
    with pytest.raises(ValueError, match="ticks"):
        p.make_plan(cfg, "a" * 64, 1)
    arrays["t"][3] = 3
    np.savez_compressed(unit["path"], **arrays)
    unit["sha256"] = p.sha256(p.Path(unit["path"]))
    path, digest = write(cfg)
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "100")
    monkeypatch.setenv("PS_STAGE_DIR", str(tmp))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp))
    (tmp / "publish").mkdir()
    with pytest.raises(FileExistsError):
        p.run(argparse.Namespace(config=path, config_sha256=digest, profile="original", max_units=1))


def test_validation_rejects_missing_or_duplicate_ticks_and_crops(sample):
    cfg, write, tmp = sample
    _, digest = write(cfg)
    plan, arrays = p.make_plan(cfg, digest, 1)
    output = tmp / "validated"
    output.mkdir()
    unit = plan["units"][0]
    path = output / f"{unit['unit']}@original.npz"
    crops = np.zeros((24, p.GEN_H, p.GEN_W, 3), np.uint8)
    crops[:, :, :p.GEN_W // 2, 0] = 255
    crops[:, 0, 0, 1] = np.arange(24)
    data = {"t": arrays[0]["t"], "box": arrays[0]["box"], "crops": crops}
    sidecar = {"plan_sha256": p.digest_json(plan), "profile": "original", "unit": unit["unit"],
               "reference_sha256": unit["reference_sha256"], "pose_sha256": unit["pose_sha256"],
               "profiled_render_seconds": 2., "kernels": {"kernel_launches": 4}}
    models = {name: {"missing": 0, "unexpected": 0} for name in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider")}
    models.update(profile="original", cfg=p.CFG, motion_module_keys=101)
    receipt = {"schema": 1, "scope": "24-native-frame-context-only", "plan": plan, "plan_sha256": p.digest_json(plan),
               "profile": "original", "fit_sha256": None, "base_sha256": None,
               "device": {"name": "synthetic", "capability": [8, 9], "cuda": "synthetic"},
               "on_cuda": {key: True for key in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider")},
               "gpu_memory_limit_mib": 100, "gpu_peak_mib": 20, "model_record": models}

    def save():
        np.savez_compressed(path, **data)
        sidecar["output_sha256"] = p.sha256(path)
        sidecar["distinct_frame_count"] = len({p.array_sha(c) for c in data["crops"]})
        path.with_suffix(".json").write_text(json.dumps(sidecar))
        receipt["outputs"] = [sidecar]
        (output / "receipt.json").write_text(json.dumps(receipt))

    save()
    p.validate_output(output, cfg, digest, "original", 1)
    models["vae"]["unexpected"] = 1
    save()
    with pytest.raises(ValueError, match="loading keys"):
        p.validate_output(output, cfg, digest, "original", 1)
    models["vae"]["unexpected"] = 0
    models["vae"]["mismatched"] = 1
    save()
    with pytest.raises(ValueError, match="loading keys"):
        p.validate_output(output, cfg, digest, "original", 1)
    models["vae"]["mismatched"] = 0
    data["t"] = data["t"].copy()
    data["t"][1] = 0
    save()
    with pytest.raises(ValueError, match="ticks"):
        p.validate_output(output, cfg, digest, "original", 1)
    data["t"] = arrays[0]["t"]
    data["crops"] = crops.copy()
    data["crops"][1] = data["crops"][0]
    save()
    p.validate_output(output, cfg, digest, "original", 1)  # repeated frames may be legitimate
    data["crops"] = crops[:-1]
    save()
    with pytest.raises(ValueError, match="crops"):
        p.validate_output(output, cfg, digest, "original", 1)


def test_synthetic_orchestration_seals_archive_and_preserves_inputs(sample, monkeypatch):
    """Fake model and profiler exercise plumbing only; this is not GPU evidence."""
    import sys
    import tarfile
    import types

    from experiments.audit import env_smoke

    cfg, write, tmp = sample
    path, digest = write(cfg)
    stage, scratch = tmp / "stage", tmp / "scratch"
    stage.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(scratch))
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "100")
    before = p.sha256(p.Path(cfg["units"][0]["path"]))
    settings = {}
    fake_cuda = types.SimpleNamespace(is_available=lambda: True,
        get_device_properties=lambda _: types.SimpleNamespace(name="synthetic", major=8, minor=9, total_memory=1000 * 2**20),
        set_per_process_memory_fraction=lambda value: settings.update(memory_fraction=value), max_memory_allocated=lambda: 20 * 2**20)
    fake_torch = types.SimpleNamespace(cuda=fake_cuda, __version__="synthetic", version=types.SimpleNamespace(cuda="synthetic"),
                                       set_num_threads=lambda n: settings.update(threads=n))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(env_smoke, "model_on_cuda", lambda _: True)
    monkeypatch.setattr(env_smoke, "profile_cuda", lambda fn: (fn(), {"kernel_launches": 3}))

    class FakeRenderer:
        def __init__(self, weights, profile, device, scratch_dir, fit=None, base=None, bits=0, cfg=None):
            settings.update(profile=profile, device=device, fit=fit, base=base, bits=bits, cfg=cfg)
            self.modules = {name: object() for name in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider")}
            self.record = {name: {"missing": 0, "unexpected": 0} for name in self.modules}
            self.record.update(profile="original", cfg=cfg, motion_module_keys=101)
            if settings.get("bad_model"):
                self.record["vae"]["unexpected"] = 1

        def render(self, reference, pose, steps, seed):
            settings["renders"] = settings.get("renders", 0) + 1
            settings.update(reference_hash=p.array_sha(reference), pose_hash=p.array_sha(pose), steps=steps, seed=seed)
            output = np.zeros((24, p.GEN_H, p.GEN_W, 3), dtype=np.uint8)
            output[:, :, :p.GEN_W // 2, 0] = 255
            return output  # deliberately identical: valid static frames

    monkeypatch.setattr(p, "Renderer", FakeRenderer)
    p.run(argparse.Namespace(config=path, config_sha256=digest, profile="original", max_units=1))
    plan, _ = p.make_plan(cfg, digest, 1)
    assert settings["threads"] == 1 and settings["steps"] == 30 and settings["seed"] == plan["units"][0]["seed"]
    assert settings["reference_hash"] == plan["units"][0]["reference_sha256"]
    assert settings["pose_hash"] == plan["units"][0]["pose_sha256"]
    assert p.sha256(p.Path(cfg["units"][0]["path"])) == before
    with tarfile.open(stage / "published.tar") as archive:
        archive.extractall(tmp / "unpacked", filter="data")
    p.validate_output(tmp / "unpacked" / "publish", cfg, digest, "original", 1)
    # Fleet republishes the scratch tree before invoking its validator.
    from experiments.jobs.staging import publish_scratch

    publish_scratch(scratch, stage)
    timing = json.loads((stage / "worker_timing.json").read_text())
    assert timing["archive_sha256"] == p.sha256(stage / "published.tar")
    assert timing["worker_wall_seconds"] > 0 and timing["worker_cpu_seconds"] > 0
    command = ["python", "-m", "experiments.tennis.player_requalification", "run", "--config", str(path),
               "--config-sha256", digest, "--profile", "original", "--max-units", "1"]
    (stage / "dispatch.json").write_text(json.dumps({"command": command}))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp / "validation.json"))
    monkeypatch.setattr(sys, "argv", ["player_requalification", "validate-stage"])
    p.main()
    assert json.loads((tmp / "validation.json").read_text())["passed"] is True
    command[2] = "experiments.tennis.k6b_generate"
    (stage / "dispatch.json").write_text(json.dumps({"command": command}))
    with pytest.raises(ValueError, match="entrypoint"):
        p.main()
    bad_stage, bad_scratch = tmp / "bad-stage", tmp / "bad-scratch"
    bad_stage.mkdir()
    bad_scratch.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(bad_stage))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(bad_scratch))
    settings["bad_model"] = True
    renders_before = settings["renders"]
    with pytest.raises(ValueError, match="loading keys"):
        p.run(argparse.Namespace(config=path, config_sha256=digest, profile="original", max_units=1))
    assert settings["renders"] == renders_before
    assert not (bad_stage / "published.tar").exists()


def test_staged_paths_are_portable_and_reject_missing_extra_or_escape(sample):
    import shutil

    cfg, write, tmp = sample
    bindings_a = []
    bindings_b = []
    for root_name in ("root-a", "root-b"):
        root = tmp / root_name
        root.mkdir()
        conditioning = root / "conditioning"
        conditioning.mkdir()
        shutil.copyfile(cfg["units"][0]["path"], conditioning / "unit.npz")
        bindings = [["conditioning", str(conditioning)]]
        for i, (_, item) in enumerate(cfg["weights"].items()):
            destination = root / f"w{i}"
            shutil.copyfile(item["path"], destination)
            bindings.append([f"w{i}", str(destination)])
        for alias, key in (("base", "tennis_checkpoint"), ("match", "match_checkpoint")):
            destination = root / alias
            shutil.copyfile(cfg[key]["path"], destination)
            bindings.append([alias, str(destination)])
        if root_name == "root-a":
            bindings_a = bindings
        else:
            bindings_b = bindings
    cfg["units"][0]["path"] = "staged:conditioning/unit.npz"
    for i, item in enumerate(cfg["weights"].values()):
        item["path"] = f"staged:w{i}"
    cfg["tennis_checkpoint"]["path"] = "staged:base"
    cfg["match_checkpoint"]["path"] = "staged:match"
    path, digest = write(cfg)
    a = p.load_config(path, digest, bindings_a)
    b = p.load_config(path, digest, bindings_b)
    assert p.make_plan(a, digest, 1)[0] == p.make_plan(b, digest, 1)[0]
    with pytest.raises(ValueError, match="missing staged"):
        p.load_config(path, digest, bindings_a[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        p.load_config(path, digest, bindings_a + [bindings_a[0]])
    with pytest.raises(ValueError, match="extra staged"):
        p.load_config(path, digest, bindings_a + [["extra", str(tmp)]])
    bad = copy.deepcopy(cfg)
    bad["units"][0]["path"] = "staged:conditioning/../base"
    with pytest.raises(ValueError, match="invalid staged path"):
        p.load_config(*write(bad), bindings_a)
    bad["units"][0]["path"] = "staged:conditioning/escape"
    (p.Path(bindings_a[0][1]) / "escape").symlink_to(tmp / "root-a" / "base")
    with pytest.raises(ValueError, match="escapes root"):
        p.load_config(*write(bad), bindings_a)


def test_motion_lengths_and_contiguous_runs(sample, monkeypatch):
    cfg, write, _ = sample
    monkeypatch.setattr(p, 'GEN_H', 4)
    monkeypatch.setattr(p, 'GEN_W', 3)
    cfg.update(schema=2, scope='player-motion-development', native_frames={'smoke': 44, 'full': 240})
    cfg.pop('match_checkpoint')
    cfg['units'][0]['review_label'] = 'Player 2'
    unit = cfg['units'][0]
    arrays = {'reference': np.zeros((4, 3, 3), np.uint8), 'pose': np.ones((240, 4, 3, 3), np.uint8),
              't': np.arange(240), 'run': np.zeros(240, np.int64), 'box': np.tile([0, 0, 3, 4], (240, 1))}

    def save():
        np.savez_compressed(unit['path'], **arrays)
        unit['sha256'] = p.sha256(p.Path(unit['path']))

    save()
    loaded = p.load_config(*write(cfg))
    for length in (44, 240):
        plan, data = p.make_plan(loaded, 'a' * 64, 1, length)
        assert plan['native_frames'] == length and len(data[0]['pose']) == length
    assert p.make_plan(loaded, 'a' * 64, 1)[0]['native_frames'] == 240
    assert p.make_plan(loaded, 'a' * 64, 1, 44)[0]['units'][0]['review_label'] == 'Player 2'
    for label in (None, '', '  ', 'x' * 81, 'bad\nlabel', 5):
        bad = copy.deepcopy(loaded)
        if label is None:
            bad['units'][0].pop('review_label')
        else:
            bad['units'][0]['review_label'] = label
        with pytest.raises(ValueError, match='review label'):
            p.make_plan(bad, 'a' * 64, 1, 44)
    for length in (None, 24, 45):
        with pytest.raises(ValueError):
            p.selected_frames(loaded, length, required=True)
    with pytest.raises(ValueError):
        p.selected_frames({'schema': 1}, 44, required=True)
    with pytest.raises(ValueError):
        p.require_profile(loaded, 'original')
    p.require_profile(loaded, 'tennis_only')
    arrays['t'][20:] += 1
    save()
    with pytest.raises(ValueError, match='consecutive'):
        p.make_plan(cfg, 'a' * 64, 1, 44)
    arrays['t'] = np.arange(240)
    arrays['run'][20:] = 1
    save()
    with pytest.raises(ValueError, match='one run'):
        p.make_plan(cfg, 'a' * 64, 1, 44)


def test_motion_crop_coverage_and_cli():
    rng = np.random.default_rng(7)
    crops = rng.integers(0, 256, (44, p.GEN_H, p.GEN_W, 3), dtype=np.uint8)
    assert p.complete_crops(crops, 44)
    assert not p.complete_crops(crops[:-1], 44)
    assert not p.complete_crops(crops, 240)
    args = p.argument_parser().parse_args(['run', '--config', '/a', '--config-sha256', 'x', '--profile',
                                           'tennis_only', '--max-units', '2', '--native-frames', '44'])
    assert args.native_frames == 44


def test_motion_single_render_and_receipt(sample, monkeypatch):
    import sys
    import types

    from experiments.audit import env_smoke

    cfg, write, tmp = sample
    monkeypatch.setattr(p, 'GEN_H', 4)
    monkeypatch.setattr(p, 'GEN_W', 4)
    cfg.update(schema=2, scope='player-motion-development', native_frames={'smoke': 44, 'full': 240})
    cfg.pop('match_checkpoint')
    cfg['units'][0]['review_label'] = 'Player 2'
    unit = cfg['units'][0]
    np.savez_compressed(unit['path'], reference=np.zeros((4, 4, 3), np.uint8),
                        pose=np.ones((44, 4, 4, 3), np.uint8), t=np.arange(44),
                        run=np.zeros(44, np.int64), box=np.tile([0, 0, 4, 4], (44, 1)))
    unit['sha256'] = p.sha256(p.Path(unit['path']))
    path, digest = write(cfg)
    stage, scratch = tmp / 'stage2', tmp / 'scratch2'
    stage.mkdir()
    scratch.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setenv('PS_GPU_MEMORY_LIMIT_MIB', '100')
    cuda = types.SimpleNamespace(is_available=lambda: True,
        get_device_properties=lambda _: types.SimpleNamespace(name='fake', major=8, minor=9, total_memory=1000 * 2**20),
        set_per_process_memory_fraction=lambda _: None, max_memory_allocated=lambda: 20 * 2**20)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda, __version__='fake',
                        version=types.SimpleNamespace(cuda='fake'), set_num_threads=lambda _: None))
    monkeypatch.setattr(env_smoke, 'model_on_cuda', lambda _: True)
    monkeypatch.setattr(env_smoke, 'profile_cuda', lambda fn: (fn(), {'kernel_launches': 1}))
    calls = []

    class FakeRenderer:
        def __init__(self, weights, profile, device, scratch_dir, fit=None, base=None, bits=0, cfg=None):
            self.modules = {name: object() for name in ('vae', 'image_encoder', 'reference_unet', 'denoising_unet', 'pose_guider')}
            self.record = {name: {'missing': 0, 'unexpected': 0} for name in self.modules}
            self.record.update(profile='original', cfg=cfg, motion_module_keys=101,
                               fit={'checkpoint_sha256': p.sha256(fit), 'base': None})
            assert base is None and bits == 0

        def render(self, reference, pose, steps, seed):
            calls.append(len(pose))
            result = np.zeros((len(pose), 4, 4, 3), np.uint8)
            result[:, :, :2, 0] = 255
            return result

    monkeypatch.setattr(p, 'Renderer', FakeRenderer)
    args = argparse.Namespace(config=path, config_sha256=digest, profile='tennis_only', max_units=1, native_frames=44)
    p.run(args)
    assert calls == [44]
    receipt = json.loads((scratch / 'publish' / 'receipt.json').read_text())
    assert (receipt['schema'], receipt['scope'], receipt['native_frames']) == (2, 'player-motion-development', 44)
    p.validate_output(scratch / 'publish', cfg, digest, 'tennis_only', 1, 44)
    receipt['native_frames'] = 240
    (scratch / 'publish' / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt'):
        p.validate_output(scratch / 'publish', cfg, digest, 'tennis_only', 1, 44)


def test_motion_smoke_cost_gate():
    cfg = {'schema': 2, 'units': [{}]}
    receipt = {'native_frames': 44, 'load_seconds': 20.0}
    timing = {'worker_wall_seconds': 100.0}
    dispatch = {'campaign_clock': {'budget_end': 2000.0}}
    result = p.smoke_cost_gate(cfg, receipt, timing, dispatch, 100.0)
    assert result['forecast_seconds'] == pytest.approx(1.25 * (20 + 80 * 240 / 44) + 90)
    assert result['campaign_remaining_seconds'] == 1900
    assert result['available_after_validator_reserve_seconds'] == 1840
    for bad_cfg, bad_receipt, bad_timing, bad_dispatch, now in (
        ({'schema': 2, 'units': [{}, {}]}, receipt, timing, dispatch, 100.0),
        (cfg, {'native_frames': 240, 'load_seconds': 20}, timing, dispatch, 100.0),
        (cfg, {'native_frames': 44, 'load_seconds': float('nan')}, timing, dispatch, 100.0),
        (cfg, {'native_frames': 44, 'load_seconds': 101}, timing, dispatch, 100.0),
        (cfg, receipt, {'worker_wall_seconds': float('inf')}, dispatch, 100.0),
        (cfg, receipt, {'worker_wall_seconds': 300}, dispatch, 100.0),
        (cfg, receipt, timing, {}, 100.0),
        (cfg, receipt, timing, {'campaign_clock': {'budget_end': float('nan')}}, 100.0),
        (cfg, receipt, timing, {'campaign_clock': {'budget_end': 140}}, 100.0),
        (cfg, receipt, timing, {'campaign_clock': {'budget_end': 500}}, 100.0),
    ):
        with pytest.raises(ValueError):
            p.smoke_cost_gate(bad_cfg, bad_receipt, bad_timing, bad_dispatch, now)


def test_motion_config_requires_one_unit(sample):
    cfg, write, _ = sample
    cfg.update(schema=2, scope='player-motion-development', native_frames={'smoke': 44, 'full': 240})
    cfg.pop('match_checkpoint')
    cfg['motion_admission_policy'] = 'unknown'
    with pytest.raises(ValueError, match='unknown motion'):
        p.load_config(*write(cfg))
    cfg.pop('motion_admission_policy')
    cfg['units'].append(dict(cfg['units'][0], unit='other'))
    with pytest.raises(ValueError, match='exactly one'):
        p.load_config(*write(cfg))


def test_motion_admission_policy_observed_cost_regression():
    cfg = {'schema': 2, 'units': [{}]}
    receipt = {'native_frames': 44, 'load_seconds': 22.933354850}
    timing = {'worker_wall_seconds': 255.548335404}
    dispatch = {'campaign_clock': {'budget_end': 3000.0}}
    with pytest.raises(ValueError, match='forecast'):
        p.smoke_cost_gate(cfg, receipt, timing, dispatch, 100.0)
    cfg['motion_admission_policy'] = 'extended-1800-v1'
    result = p.smoke_cost_gate(cfg, receipt, timing, dispatch, 100.0)
    assert result['forecast_seconds'] == pytest.approx(1704.6779246)
    assert result['full_stage_cap_seconds'] == 1800 and result['policy'] == 'extended-1800-v1'
    with pytest.raises(ValueError, match='campaign'):
        p.smoke_cost_gate(cfg, receipt, timing, {'campaign_clock': {'budget_end': 150.0}}, 100.0)
    cfg['motion_admission_policy'] = 'unknown'
    with pytest.raises(ValueError, match='unknown motion'):
        p.smoke_cost_gate(cfg, receipt, timing, dispatch, 100.0)
    with pytest.raises(ValueError, match='unknown motion'):
        p.selected_frames({**cfg, 'scope': 'player-motion-development',
                           'native_frames': {'smoke': 44, 'full': 240}}, 44)
