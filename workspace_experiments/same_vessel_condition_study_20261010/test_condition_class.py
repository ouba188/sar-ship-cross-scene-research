import unittest
import pandas as pd
from condition_class import match_anchors, factor_masks


class MatchingTests(unittest.TestCase):
    def test_metadata_only_nearest_no_reuse_and_time_limit(self):
        p = pd.DataFrame(dict(pair_id=[1, 2, 3, 4, 5, 6], a=[0, 0, 0, 0, 9, 0],
            gap_bin=['30-90d']*5+['>90d'], gap_days=[40., 43., 39., 88., 41., 100.],
            outcome=[1, 0, 1, 0, 1, 0]))
        exposed, control = p.pair_id.isin([1, 2, 5, 6]), p.pair_id.isin([3, 4])
        m = match_anchors(p, exposed, control)
        self.assertEqual(m[['pair_id_exposed', 'pair_id_control']].values.tolist(), [[1, 3]])
        p.outcome = 1-p.outcome
        pd.testing.assert_frame_equal(m, match_anchors(p, exposed, control))

    def test_unknown_pass_is_not_control(self):
        p = pd.DataFrame(dict(group=['unmatched', 'SP_near', 'SP_far', 'SP_far'],
            same_port=[True]*4, same_platform=[True]*4, same_pass=[True, True, True, False],
            delta_incidence=[0., 2., 5., 1.], delta_relative_direction=[0., 10., 25., 30.]))
        masks = factor_masks(p)
        self.assertEqual(masks['incidence'][0].tolist(), [False, False, True, False])
        self.assertEqual(masks['incidence'][1].tolist(), [False, True, False, False])
        self.assertEqual(masks['pass'][0].tolist(), [False, False, False, True])


if __name__ == '__main__':
    unittest.main()
