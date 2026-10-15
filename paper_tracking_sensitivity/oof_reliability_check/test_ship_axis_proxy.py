import numpy as np
from ship_axis_proxy import measure_axis, axial_difference, combine_axes, world_axis


def test_simple_rotated_target_and_blank_rejection():
    yy, xx = np.mgrid[:128, :128]-63.5
    for angle in [0, 25, 70, 110, 160]:
        rad = np.deg2rad(angle)
        along = xx*np.cos(rad)+yy*np.sin(rad)
        across = -xx*np.sin(rad)+yy*np.cos(rad)
        image = 220*np.exp(-.5*((along/10)**2+(across/2)**2))
        axis = measure_axis(image)
        assert axis['accepted']
        assert axial_difference(axis['angle_pixel_deg'], angle) < 1.
    assert not measure_axis(np.zeros((128, 128)))['accepted']


def test_isotropic_bright_target_is_not_a_reliable_axis():
    yy, xx = np.mgrid[:128, :128]-63.5
    assert not measure_axis(200*np.exp(-(xx*xx+yy*yy)/32))['accepted']


def test_axial_mean_wraps_without_bow_stern_assumption():
    angle, difference = combine_axes(179, 1)
    assert min(abs(angle), abs(angle-180)) < 1e-8
    assert abs(difference-2) < 1e-8
    assert axial_difference(5, 185) == 0


def test_georeferencing_accounts_for_image_y_sign():
    geo = dict(transform=[500000, 10, 0, 1000000, 0, -10], epsg=32631)
    east = world_axis(geo, 0, 0, 0)['axis_true_north_deg']
    north_south = world_axis(geo, 0, 0, 90)['axis_true_north_deg']
    assert axial_difference(east, 90) < .001
    assert axial_difference(north_south, 0) < .001
