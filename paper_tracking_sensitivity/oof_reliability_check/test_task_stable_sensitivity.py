import numpy as np

from task_stable_sensitivity import pair_geometry, held_geometry, binary_ba


def data():
    z, y, ports = [], [], []
    for port, offset in enumerate([-2., 0., 2.]):
        for label, center in [(0, 1.), (1, -1.)]:
            for _ in range(5):
                z.append([center, offset])
                y.append(label)
                ports.append(port)
    return np.array(z), np.array(y), np.array(ports)


def test_port_motion_orthogonal_to_task_does_not_become_task_drift():
    z, y, ports = data()
    g = pair_geometry(z, y, ports, 0, 1)
    np.testing.assert_allclose(g['direction'], [1., 0.])
    assert g['source_gap'] == 2.
    assert g['port_sd_along_task'] == 0.
    assert g['total_port_rms_over_gap'] > 0.


def test_source_port_weighting_is_not_sample_prevalence_weighting():
    z, y, ports = data()
    z[:, 0] += ports * .2
    g = pair_geometry(z, y, ports, 0, 1)
    duplicate = np.flatnonzero((ports == 0) & (y == 0))
    ids = np.r_[np.arange(len(z)), np.tile(duplicate, 20)]
    g2 = pair_geometry(z[ids], y[ids], ports[ids], 0, 1)
    np.testing.assert_allclose(g['delta'], g2['delta'])
    np.testing.assert_allclose(g['midpoint'], g2['midpoint'])


def test_held_reversal_not_fixed_by_using_target_labels():
    z, y, ports = data()
    g = pair_geometry(z, y, ports, 0, 1)
    z[:, 0] *= -1
    result = held_geometry(z, y, 0, 1, g)
    assert result['auc'] == 0.
    assert result['fixed_midpoint_ba'] == 0.
    assert result['reversed']
    assert result['target_gap_ratio'] == -1.


def test_rotation_preserves_geometry_and_held_scores():
    z, y, ports = data()
    rotation = np.array([[.6, -.8], [.8, .6]])
    a = pair_geometry(z, y, ports, 0, 1)
    b = pair_geometry(z @ rotation, y, ports, 0, 1)
    np.testing.assert_allclose(a['total_port_rms_over_gap'], b['total_port_rms_over_gap'])
    np.testing.assert_allclose(held_geometry(z, y, 0, 1, a)['auc'],
                               held_geometry(z @ rotation, y, 0, 1, b)['auc'])


def test_pair_accuracy_is_class_balanced():
    labels = np.r_[np.zeros(9, dtype=int), 1]
    assert binary_ba(labels, np.ones(10), 0, 1) == .5
