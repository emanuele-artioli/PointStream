def test_k8b_decode_keeps_all_50fps_frames(tmp_path):
    import shutil
    import subprocess

    import pytest

    from experiments.tennis import k8b

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    if subprocess.run([ffmpeg, "-version"], capture_output=True).returncode:
        pytest.skip("ffmpeg cannot start")
    video = tmp_path / "source.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x64:r=50",
                    "-frames:v", "10", "-c:v", "ffv1", str(video)], check=True)
    decoded = subprocess.run(k8b.decode_command(ffmpeg, video, 0, 10), capture_output=True, check=True)
    assert len(decoded.stdout) == 10 * 1920 * 1080 * 3 // 2
