"""Generative dispatch follows declared capabilities, never a class or name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE, CONDITION_MASK
from src.contracts.conditioning import ConditioningBundle
from src.contracts.errors import ConfigValueError, MissingConditioningError, UnsupportedCapabilityError
from src.pipeline.reconstruction import DevicePolicy, GeneratorRef, dispatch, is_out_of_memory


def _crop(value: int = 50) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _bundle(*, mask: np.ndarray | None = None) -> ConditioningBundle:
    return ConditioningBundle(
        appearance=_crop(9),
        mask=mask,
        bbox=(0, 0, 8, 8),
        object_id="player",
        frame_index=0,
    )


class AnimateAnyoneStrategy:
    """Named like the pre-rewrite class the compositor used to isinstance."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
        self.calls.append(f"frame:{device}")
        return _crop(11)

    def generate_sequence(self, conditioning, *, seed, device, params):  # noqa: ANN001
        self.calls.append(f"sequence:{device}")
        return [_crop(22) for _ in conditioning]


class CheapPix2Pix:
    """A name that must not block sequence dispatch when the capability is set."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
        self.calls.append("frame")
        return _crop(33)

    def generate_sequence(self, conditioning, *, seed, device, params):  # noqa: ANN001
        self.calls.append("sequence")
        return [_crop(44) for _ in conditioning]


def test_a_temporal_name_without_the_capability_is_driven_per_frame() -> None:
    backend = AnimateAnyoneStrategy()
    ref = GeneratorRef(backend=backend, name="animate-anyone")
    bundles = (_bundle(), _bundle())
    crops, _ = dispatch(ref, bundles, seed=1)
    assert backend.calls == ["frame:cpu", "frame:cpu"]
    assert all(int(frame[0, 0, 0]) == 11 for frame in crops)


def test_sequence_capability_dispatches_even_when_the_class_is_not_the_old_one() -> None:
    backend = CheapPix2Pix()
    ref = GeneratorRef(
        backend=backend,
        capabilities=frozenset({CAP_TEMPORAL_SEQUENCE}),
        name="pix2pix",
    )
    crops, _ = dispatch(ref, (_bundle(), _bundle()), seed=1)
    assert backend.calls == ["sequence"]
    assert all(int(frame[0, 0, 0]) == 44 for frame in crops)


def test_declared_temporal_without_generate_sequence_is_a_lying_registry_entry() -> None:
    class _FrameOnly:
        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            return _crop()

    ref = GeneratorRef(
        backend=_FrameOnly(),
        capabilities=frozenset({CAP_TEMPORAL_SEQUENCE}),
        name="broken-temporal",
    )
    with pytest.raises(UnsupportedCapabilityError, match="generate_sequence"):
        dispatch(ref, (_bundle(),), seed=1)


def test_dispatch_does_not_inspect_class_names() -> None:
    import ast

    source = Path(__file__).resolve().parents[2] / "src/pipeline/reconstruction/dispatch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"__class__", "__name__"}:
            raise AssertionError(f"dispatch inspects {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "isinstance":
                args = node.args
                if len(args) >= 2 and isinstance(args[1], ast.Name):
                    assert args[1].id == "SequenceGenerator", (
                        "dispatch may only isinstance the SequenceGenerator protocol, "
                        f"not {args[1].id}"
                    )
    text = source.read_text(encoding="utf-8")
    assert "CAP_TEMPORAL_SEQUENCE" in text
    assert "supports_sequence" in text


def test_missing_required_conditioning_fails_at_dispatch() -> None:
    class _NeedsMask:
        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            return _crop()

    ref = GeneratorRef(
        backend=_NeedsMask(),
        requires=frozenset({CONDITION_MASK}),
        name="seg-controlnet",
    )
    with pytest.raises(MissingConditioningError, match="mask"):
        dispatch(ref, (_bundle(mask=None),), seed=1)


def test_empty_bundles_are_rejected() -> None:
    class _Gen:
        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            return _crop()

    with pytest.raises(ConfigValueError, match="nothing to draw"):
        dispatch(GeneratorRef(backend=_Gen()), (), seed=1)


def test_oom_policy_retries_on_the_fallback_device() -> None:
    seen: list[str] = []

    def _op(device: str) -> str:
        seen.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "ok"

    policy = DevicePolicy(preferred="cuda", fallback="cpu", allow_fallback=True)
    result, decision = policy.run(_op)
    assert result == "ok"
    assert seen == ["cuda", "cpu"]
    assert decision.fell_back
    assert decision.device == "cpu"


def test_oom_policy_does_not_swallow_a_real_bug() -> None:
    def _op(device: str) -> str:
        raise RuntimeError("shape mismatch on " + device)

    policy = DevicePolicy(preferred="cuda", fallback="cpu")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        policy.run(_op)


def test_oom_policy_without_fallback_propagates() -> None:
    def _op(device: str) -> str:
        raise RuntimeError("out of memory")

    policy = DevicePolicy(preferred="cuda", fallback="cpu", allow_fallback=False)
    with pytest.raises(RuntimeError, match="out of memory"):
        policy.run(_op)


def test_is_out_of_memory_does_not_treat_every_runtime_error_as_oom() -> None:
    assert is_out_of_memory(RuntimeError("CUDA out of memory"))
    assert not is_out_of_memory(RuntimeError("invalid homography"))
