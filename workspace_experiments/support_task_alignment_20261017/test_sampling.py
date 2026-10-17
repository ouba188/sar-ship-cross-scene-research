import unittest
import numpy as np
import pandas as pd
from sampling import draw_like, legal_pool, training_schedule


class SamplingTest(unittest.TestCase):
    def test_class_counts_and_no_replacement(self):
        labels = np.repeat([0, 2, 5], 12)
        original = np.array([0, 12, 24, 1, 13, 25, 2, 14, 26, 3, 15, 27, 4, 16, 5, 6])
        actual = draw_like(original, np.arange(36), labels, np.random.default_rng(12))
        np.testing.assert_array_equal(labels[actual], labels[original])
        self.assertEqual(len(set(actual)), 16)
        self.assertNotEqual(set(actual), set(original))

    def test_fixed_prior_pool_excludes_other_observations_of_same_ship(self):
        frame = pd.DataFrame(dict(sample_id=list('abcdef'), acquisition=['A','B','B','C','B','B'],
                                 mmsi=['q','x','p','p','z','q']), index=np.arange(6))
        episode = dict(q=np.array([0]), prior_rows=np.array([3]), s=np.array([1,4]), support_acquisition='B')
        pool = legal_pool(dict(train=np.arange(6)), episode, 'fit', frame)
        self.assertEqual(set(pool), {1,4})

    def test_full_class_pool_does_not_claim_order_changes_as_new_members(self):
        original = np.arange(16)
        labels = np.repeat([0,1], 8)
        result = draw_like(original, original, labels, np.random.default_rng(9))
        self.assertEqual(set(result), set(original))

    def test_episode_schedule_replays_legacy_rng(self):
        families, choices = training_schedule(2, 27)
        rng = np.random.default_rng(20261019)
        for step in range(200):
            np.testing.assert_array_equal(families[step], rng.choice(27, 2, replace=False))
        self.assertEqual(choices.shape, (200,2))
        self.assertTrue(np.all((choices >= 0) & (choices < 5)))


if __name__ == '__main__':
    unittest.main()
