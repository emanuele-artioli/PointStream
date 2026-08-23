"""Every registered backend constructs, or fails with a stated reason."""

from __future__ import annotations

from typing import Any

from experiments.probe.construct import construct_all, construct_one, stated_reason
from src.contracts.registry import BackendSpec, Registry


def test_stated_reason_rejects_an_empty_attribute_error() -> None:
    """A wrapper bug is not a limitation."""
    assert stated_reason(AttributeError(""), axis="detector", name="yolo") is None
    assert stated_reason(AttributeError("foo"), axis="detector", name="yolo") is None


def test_stated_reason_accepts_a_licence_block() -> None:
    message = "Construction is refused until that stack is licence-cleared."
    assert stated_reason(RuntimeError(message), axis="generator", name="mofa-video") == message


def test_stated_reason_names_the_sam3_attention_gap() -> None:
    reason = stated_reason(
        AttributeError("module 'torch.nn' has no attribute 'attention'"),
        axis="detector",
        name="sam3",
    )
    assert reason is not None
    assert "torch.nn.attention" in reason
    assert "2.2.2" in reason


def test_construct_one_records_a_stated_refusal() -> None:
    registry: Registry[object] = Registry("toy")

    def factory(**_kwargs: Any) -> None:
        raise RuntimeError("not bundled: Stability AI weights")

    registry.register(
        BackendSpec(name="blocked", target="experiments.probe.construct:stated_reason")
    )
    spec = registry.spec("blocked")
    original_build = registry.build

    def fake_build(name: str, **kwargs: Any) -> Any:
        del name, kwargs
        return factory()

    registry.build = fake_build  # type: ignore[method-assign]
    try:
        record = construct_one(registry, spec)
    finally:
        registry.build = original_build  # type: ignore[method-assign]
    assert record.ok is False
    assert record.reason is not None
    assert "not bundled" in record.reason


def test_every_registered_backend_constructs_or_states_why() -> None:
    report = construct_all()
    assert report.records, "no backends were registered"
    assert not report.failures, (
        "backend(s) failed construction without a stated reason: "
        + "; ".join(
            f"{item.axis}/{item.name} ({item.exception_type})" for item in report.failures
        )
    )
    refused = [item for item in report.records if not item.ok]
    names = {(item.axis, item.name) for item in refused}
    assert ("generator", "mofa-video") in names
    assert ("detector", "rf-detr") in names
    assert ("detector", "sam3") in names
    assert ("segmenter", "sam3") in names
    for item in refused:
        assert item.reason, item
        assert item.reason.strip()
