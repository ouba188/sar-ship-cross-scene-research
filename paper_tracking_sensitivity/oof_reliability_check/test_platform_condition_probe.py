import numpy as np
import pandas as pd
from run_platform_condition_probe import platform_pair_features


def test_platform_interactions_are_differences_of_joint_observations():
    # Platform switches from A to C while t changes from -1 to +1.
    conditions=np.array([[33.,0.,0.,0.],[43.,0.,0.,1.]])
    pairs=pd.DataFrame({'obs_a':[0],'obs_b':[1]})
    x=platform_pair_features(None,conditions,pairs)
    np.testing.assert_allclose(x,[[2.,0.,0.,0.,1.,1.,1.]])


def test_same_platform_can_have_angle_interaction_but_no_platform_switch():
    conditions=np.array([[38.,0.,0.,1.],[43.,0.,0.,1.]])
    pairs=pd.DataFrame({'obs_a':[0],'obs_b':[1]})
    np.testing.assert_allclose(platform_pair_features(None,conditions,pairs),[[1.,1.,0.,0.,0.,1.,1.]])
