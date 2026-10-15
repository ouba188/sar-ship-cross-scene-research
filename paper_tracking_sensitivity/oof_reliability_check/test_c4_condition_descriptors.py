import numpy as np

from c4_condition_descriptors import FEATURES, describe


def test_blank_keeps_missing_shape_but_valid_zero_contrast():
    result = describe(np.zeros((2, 128, 128), dtype=np.uint8))
    assert np.isnan(result['visible_footprint_proxy_fraction'])
    assert np.isnan(result['elongation_proxy'])
    assert result['proxy_missing']
    assert result['contrast_proxy'] == 0
    assert not result['contrast_proxy_missing']


def test_bright_rectangle_known_area_and_rotation_invariance():
    pair = np.zeros((2, 128, 128), dtype=np.uint8)
    pair[:, 60:68, 40:88] = 200
    result = describe(pair)
    assert result['visible_footprint_proxy_fraction'] == 8*48/(128*128)
    assert result['elongation_proxy'] > 5
    assert result['center_excess_proxy'] > 0
    for k in range(1, 4):
        rotated = describe(np.rot90(pair, k, axes=(-2, -1)))
        for name in FEATURES:
            np.testing.assert_allclose(result[name], rotated[name], atol=1e-10)


def test_invalid_image_does_not_become_valid_zero_proxy():
    pair = np.ones((2, 128, 128))
    pair[0, 0, 0] = np.nan
    result = describe(pair)
    assert all(result[name+'_missing'] for name in FEATURES)
