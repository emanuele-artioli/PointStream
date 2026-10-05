import pytest

from demo.experiments.checkpoint_contract import hand_checkpoint_contract


@pytest.mark.parametrize("model,channels", [("pix2pix", 3), ("spade", 4)])
def test_latest_trained_format_requires_matching_factory(model, channels):
    checkpoint = {"model": model, "factory": "factory002", "state_dict": {"weight": 1}}
    architecture, outputs, state = hand_checkpoint_contract(
        checkpoint, expected_factory="factory002"
    )
    assert (architecture, outputs, state) == (model, channels, {"weight": 1})
    for factory in (None, "factory001"):
        with pytest.raises(ValueError, match="factory"):
            hand_checkpoint_contract(checkpoint, expected_factory=factory)


def test_legacy_format_remains_explicitly_supported():
    assert hand_checkpoint_contract({"model_state_dict": {"enc1.0.weight": 1}, "out_channels": 4})[
        :2
    ] == ("spade", 4)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {},
        {"state_dict": {}},
        {"model_state_dict": {}},
        {"model_state_dict": {"x": 1}, "out_channels": True},
    ],
)
def test_unknown_or_empty_checkpoint_rejected(checkpoint):
    with pytest.raises(ValueError):
        hand_checkpoint_contract(checkpoint)
