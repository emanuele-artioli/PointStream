"""Boundary audit uses optional model libraries, but never model weights."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from experiments.tennis.player_tensor_preflight import audit, fixture  # noqa: E402


def test_processor_boundary_and_asymmetric_fixture() -> None:
    a = fixture()
    assert tuple(a[0, 0]) == (241, 37, 13)
    assert tuple(a[-1, 0]) == (9, 67, 229)
    result = audit()
    assert not result["cuda_initialized"]
    assert result["pipeline_source_verification"] == "declared-only"
    assert result["comparisons"]["clip_float32"]["different_values"] == 0
    assert result["comparisons"]["old_direct_clip_vs_square"]["different_values"] > 0


def test_rejects_wrong_pipeline_source(tmp_path) -> None:
    path = tmp_path / "pipeline.py"
    path.write_text("not the pinned source")
    with pytest.raises(ValueError, match="SHA256"):
        audit(path)
