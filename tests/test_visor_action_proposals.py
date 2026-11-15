import pytest
from src.segmentation import visor
from tools.datasets.visor_action_proposals import decide, filtered


def annotation(name):
    return {'id': 1, 'name': name, 'segments': [[[0, 0], [10, 0], [10, 10], [0, 10]]]}


def test_action_policy_is_source_specific_and_does_not_invent_motion():
    assert decide(annotation('right hand'), 'P35_105')[0]
    assert decide(annotation('saucepan'), 'P35_105')[0]
    assert not decide(annotation('cloth'), 'P35_105')[0]
    assert not decide(annotation('sink'), 'P12_101')[0]
    assert decide(annotation('tap'), 'P12_101')[0]
    assert not decide(annotation('oven'), 'P11_103')[0]
    assert decide(annotation('pan'), 'P11_103')[0]
    assert decide(annotation('novel object'), 'P35_105')[1].startswith('UNRESOLVED')
    with pytest.raises(ValueError):
        decide(annotation('pan'), 'new-video')


def test_filter_keeps_all_instance_dispositions_and_unknowns_are_not_truth():
    frame = visor.Frame(1, (annotation('right hand'), annotation('sink'), annotation('tap')), ())
    union, rows = filtered(frame, 'P12_101', dense=False)
    assert union.shape == (1080, 1920) and union.any()
    assert len(rows) == 3 and sum(r['keep_candidate'] for r in rows) == 2
    unknown, rows = filtered(None, 'P35_105', dense=False)
    assert not unknown.any() and not rows
