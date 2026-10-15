import numpy as np
import pandas as pd
from run_relative_observation_trial import axis_features, knowledge


def test_relative_angle_invariance_to_common_rotation_and_axis_reversal():
    f = pd.DataFrame({'relative_status': ['ok', 'ok', 'missing'], 'body_axis_deg': [10., 190., np.nan],
                      'range_axis_deg': [30., 30., np.nan]})
    a = axis_features(f)
    np.testing.assert_allclose(a['relative'][0], a['relative'][1], atol=1e-12)
    np.testing.assert_allclose(a['relative'][2], 0)
    f[['body_axis_deg', 'range_axis_deg']] += 57.
    np.testing.assert_allclose(a['relative'], axis_features(f)['relative'], atol=1e-12)


def test_interaction_is_image_dependent_and_unavailable_rows_are_zero():
    f = pd.DataFrame({'relative_status': ['ok', 'missing'], 'body_axis_deg': [0., np.nan], 'range_axis_deg': [0., np.nan]})
    axis = axis_features(f)
    visual = np.ones((2, 8))
    one, scales = knowledge('relative_visual_interaction', np.zeros((2, 2)), axis, visual)
    two, _ = knowledge('relative_visual_interaction', np.zeros((2, 2)), axis, visual*2)
    assert one.shape == (2, 21) and scales.shape == (21,)
    np.testing.assert_allclose(two[0, 5:], 2*one[0, 5:])
    np.testing.assert_allclose(one[1], 0)
