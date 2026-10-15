import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import condition_kernel_residual as core
from condition_coverage import make_episodes


class CoverageTests(unittest.TestCase):
    def test_32_support_training_slices(self):
        torch.set_num_threads(1)
        meta = pd.DataFrame(dict(class_id=np.arange(48)%8, mmsi=np.arange(48)))
        episode = dict(tag='synthetic', support=np.arange(32), query=np.arange(32, 48),
            phi=torch.cat([torch.randn(48, 512)/512**.5, torch.ones(48, 1)], 1), prior=torch.randn(48, 8))
        with patch.object(core, 'STEPS', 1):
            model, trace = core.fit(torch.randn(48, 32), meta, [episode], 'test')
        self.assertTrue(np.isfinite(trace[0]['query_loss']))
        self.assertGreater(float(model.map.weight.detach().abs().sum()), 0)

    def test_episode_budgets_and_identity_exclusions(self):
        torch.set_num_threads(1)
        meta = pd.DataFrame(dict(port=np.repeat(['A', 'B', 'C'], 48), mmsi=np.arange(144), class_id=np.arange(144)%8))
        episodes, _ = make_episodes(meta, np.random.default_rng(3).normal(size=(144, 16)).astype(np.float32))
        self.assertEqual(len(episodes), 9)
        for ep in episodes:
            self.assertEqual(len(ep['support']), 32)
            self.assertEqual(len(ep['query']), 16)
            self.assertFalse(set(ep['support'])&set(ep['query']))
            self.assertFalse(set(meta.iloc[ep['reference']].port)&{ep['port']})


if __name__ == '__main__':
    unittest.main()
