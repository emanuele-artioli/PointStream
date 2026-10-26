import numpy as np

from experiments.tennis.k6b_receiver_silhouette import project, select_mask


def test_paid_pose_selects_player_over_larger_bystander():
    player = np.zeros((20, 20), bool)
    player[2:6, 2:6] = True
    bystander = np.zeros_like(player)
    bystander[8:20, 8:20] = True
    pose = np.array([[3, 3, 1], [4, 4, 1], [10, 10, 0]])
    found = select_mask([('person', bystander, .99), ('person', player, .3)], pose, (20, 20))
    assert np.array_equal(found, player)
    assert not select_mask([], pose, (20, 20)).any()


def test_projection_clips_crop_without_inventing_support():
    mask = np.ones((4, 4), bool)
    full = project(mask, np.array([-2., -2., 4., 4.]))
    assert full.shape == (1080, 1920)
    assert full.sum() == 4
    assert full[:2, :2].all()


def test_transport_root_accepts_both_native_publication_layouts(tmp_path):
    from experiments.tennis.k6b_receiver_silhouette import transport_root
    for name, relative in [('direct', ''), ('wrapped', 'publish')]:
        extracted = tmp_path / name
        root = extracted / relative
        (root / 'packets').mkdir(parents=True)
        (root / 'transport.json').write_text('{}')
        assert transport_root(extracted) == root
