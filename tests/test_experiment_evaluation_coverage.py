import pytest
from pathlib import Path

from src.experiment_evaluation import evaluate_run_summary, _normalize_evaluation_metrics

def test_normalize_evaluation_metrics():
    assert _normalize_evaluation_metrics(None) == ["psnr"]
    assert _normalize_evaluation_metrics("none") == []
    assert _normalize_evaluation_metrics("psnr,ssim") == ["psnr", "ssim"]
    assert _normalize_evaluation_metrics(["vmaf", "psnr"]) == ["vmaf", "psnr"]
    
    with pytest.raises(ValueError):
        _normalize_evaluation_metrics("none,psnr")
        
    with pytest.raises(ValueError):
        _normalize_evaluation_metrics("invalid_metric")
        
    with pytest.raises(ValueError):
        _normalize_evaluation_metrics(123)

def test_evaluate_run_summary_missing_files(tmp_path: Path):
    summary = {
        "source_uri": "does_not_exist.mp4",
        "decoded_uri": "does_not_exist_either.mp4",
        "evaluation": {
            "sizes_bytes": {
                "transport_total": 500,
                "source": 1000,
            }
        }
    }
    
    metrics = evaluate_run_summary(
        summary=summary,
        experiment_dir=tmp_path,
        max_frames=10,
        metrics=["psnr", "ssim", "vmaf"],
    )
    
    assert metrics["sizes_bytes"]["transport_savings_percent"] == 50.0
    assert metrics["psnr_num_frames"] == 0
    assert metrics["ssim_num_frames"] == 0
    assert metrics["vmaf_num_frames"] == 0

def test_evaluate_run_summary_with_dummy_images(monkeypatch, tmp_path: Path):
    # Create dummy files
    src = tmp_path / "source.mp4"
    src.write_text("dummy")
    dec = tmp_path / "decoded.mp4"
    dec.write_text("dummy")
    
    summary = {
        "source_uri": str(src),
        "decoded_uri": str(dec),
    }

    # Mock subprocess for ffmpeg/ffprobe (PSNR, SSIM, and VMAF are all
    # computed by shelling out to the system ffmpeg binary, mirroring how
    # opencv-python's bundled ffmpeg lacks an AV1 decoder and cannot read
    # real decoder output).
    import subprocess
    class DummyProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = "fake stdout"
            self.stderr = ""

    def fake_run(*args, **kwargs):
        # Depending on args, write to stats_file or log_file if they are passed in filter_complex
        cmd = args[0] if args else kwargs.get("args", [])
        filter_complex = ""
        for i, c in enumerate(cmd):
            if c == "-filter_complex" and i + 1 < len(cmd):
                filter_complex = cmd[i+1]

        if "psnr=stats_file=" in filter_complex:
            path = filter_complex.split("stats_file=")[1]
            Path(path).write_text("n:1 mse_avg:4.22 psnr_avg:40.00\n")
        elif "ssim=stats_file=" in filter_complex:
            path = filter_complex.split("stats_file=")[1]
            Path(path).write_text("n:1 Y:0.99 U:0.99 V:0.99 All:0.990000 (20.0)\n")
        elif "libvmaf=log_path=" in filter_complex:
            path = filter_complex.split("log_path=")[1].split(":")[0]
            import json
            Path(path).write_text(json.dumps({
                "pooled_metrics": {"vmaf": {"mean": 95.5}},
                "frames": [{"vmaf": 95.5}, {"vmaf": 95.5}]
            }))

        return DummyProcess()
        
    monkeypatch.setattr(subprocess, "run", fake_run)
    
    import shutil
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/fake_" + x)
    
    metrics = evaluate_run_summary(
        summary=summary,
        experiment_dir=tmp_path,
        metrics=["psnr", "ssim", "vmaf"],
    )
    
    assert metrics["psnr_mean"] == 40.0
    assert metrics["ssim_mean"] == 0.99
    assert metrics["vmaf_mean"] == 95.5
