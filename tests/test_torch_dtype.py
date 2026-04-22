from __future__ import annotations

import pytest
import torch

from src.shared import torch_dtype as td


class _FakeSample:
    def __add__(self, other):
        _ = other
        return self

    def sum(self):
        return self

    def item(self):
        return 0.0


@pytest.fixture(autouse=True)
def clear_cuda_dtype_cache() -> None:
    td._is_cuda_dtype_supported.cache_clear()


def test_parse_gpu_dtype_env_handles_missing_and_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POINTSTREAM_GPU_DTYPE", raising=False)
    assert td.parse_gpu_dtype_env() is None

    monkeypatch.setenv("POINTSTREAM_GPU_DTYPE", "   ")
    assert td.parse_gpu_dtype_env() is None


def test_parse_gpu_dtype_env_parses_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POINTSTREAM_GPU_DTYPE", "Fp16")
    assert td.parse_gpu_dtype_env() == torch.float16


def test_parse_gpu_dtype_env_warns_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POINTSTREAM_GPU_DTYPE", "invalid-dtype")
    with pytest.warns(RuntimeWarning, match="Unsupported POINTSTREAM_GPU_DTYPE"):
        assert td.parse_gpu_dtype_env() is None


def test_is_cuda_dtype_supported_returns_false_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: False)
    assert td._is_cuda_dtype_supported("float16", 0) is False


def test_is_cuda_dtype_supported_returns_false_for_unknown_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: True)
    assert td._is_cuda_dtype_supported("not_a_dtype", 0) is False


def test_is_cuda_dtype_supported_returns_true_when_tensor_ops_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(td.torch, "ones", lambda *args, **kwargs: _FakeSample())
    assert td._is_cuda_dtype_supported("float16", 0) is True


def test_is_cuda_dtype_supported_returns_false_when_ops_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: True)

    def _raise(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("boom")

    monkeypatch.setattr(td.torch, "ones", _raise)
    assert td._is_cuda_dtype_supported("float16", 0) is False


def test_resolve_torch_dtype_for_device_returns_float32_on_cpu() -> None:
    assert td.resolve_torch_dtype_for_device("cpu") == torch.float32


def test_resolve_torch_dtype_for_device_uses_supported_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "parse_gpu_dtype_env", lambda env_var="POINTSTREAM_GPU_DTYPE": torch.bfloat16)
    monkeypatch.setattr(td, "_is_cuda_dtype_supported", lambda dtype_name, device_index: dtype_name == "bfloat16")

    assert td.resolve_torch_dtype_for_device("cuda:0") == torch.bfloat16


def test_resolve_torch_dtype_for_device_warns_when_requested_not_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td, "parse_gpu_dtype_env", lambda env_var="POINTSTREAM_GPU_DTYPE": torch.bfloat16)
    monkeypatch.setattr(td, "_is_cuda_dtype_supported", lambda dtype_name, device_index: dtype_name == "float16")

    with pytest.warns(RuntimeWarning, match="not allowed"):
        assert td.resolve_torch_dtype_for_device("cuda:0", allowed_cuda={torch.float16}) == torch.float16


def test_resolve_torch_dtype_for_device_falls_back_when_candidate_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td, "parse_gpu_dtype_env", lambda env_var="POINTSTREAM_GPU_DTYPE": torch.float16)

    def _support(dtype_name: str, device_index: int) -> bool:
        _ = device_index
        return dtype_name == "bfloat16"

    monkeypatch.setattr(td, "_is_cuda_dtype_supported", _support)

    with pytest.warns(RuntimeWarning) as caught:
        resolved = td.resolve_torch_dtype_for_device(
            "cuda:0",
            default_cuda=torch.float16,
            allowed_cuda={torch.bfloat16},
        )
    assert resolved == torch.bfloat16
    messages = [str(item.message) for item in caught]
    assert any("not allowed" in message for message in messages)
    assert any("Falling back to bfloat16" in message for message in messages)


def test_resolve_torch_dtype_for_device_falls_back_to_float32_when_none_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td, "parse_gpu_dtype_env", lambda env_var="POINTSTREAM_GPU_DTYPE": torch.float16)
    monkeypatch.setattr(td, "_is_cuda_dtype_supported", lambda dtype_name, device_index: False)

    with pytest.warns(RuntimeWarning, match="Falling back to float32"):
        assert td.resolve_torch_dtype_for_device("cuda:0", default_cuda=torch.float16) == torch.float32


def test_is_cuda_device_usable_handles_device_type_and_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert td.is_cuda_device_usable("cpu") is False

    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: False)
    assert td.is_cuda_device_usable("cuda:0") is False


def test_is_cuda_device_usable_uses_dtype_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(td, "_is_cuda_dtype_supported", lambda dtype_name, device_index: True)
    assert td.is_cuda_device_usable("cuda:1") is True
