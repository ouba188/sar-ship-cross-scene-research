import unittest
import numpy as np
from channel_operator import loading_to_channels, build_blocks


class OperatorTests(unittest.TestCase):
    def test_channel_trace_matches_direct_perturbation(self):
        rng = np.random.default_rng(42)
        model = {'r18_components': rng.normal(size=(4, 9)),
                 'r18_directions': rng.normal(size=(4, 2)),
                 'r18_scale': rng.uniform(.1, 2, 9)}
        dx = rng.normal(size=(7, 9))
        direct = (dx/model['r18_scale'])@model['r18_components'].T/2@model['r18_directions']
        np.testing.assert_allclose(dx@loading_to_channels(model, 'r18'), direct, atol=1e-12)

    def test_gate_affects_real_block_and_receives_gradient(self):
        import torch
        torch.set_num_threads(1)
        block = build_blocks()[1]
        torch.manual_seed(42)
        x = torch.rand(2, 512, 4, 4)
        gate = torch.ones(512, requires_grad=True)
        ref = block(x)
        actual = block(x*gate[None, :, None, None])
        torch.testing.assert_close(ref, actual, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertGreater(float(gate.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in block.parameters()))
        with torch.no_grad():
            gate[:64] = .5
        changed = block(x*gate[None, :, None, None])
        self.assertGreater(float((changed-ref).abs().max()), .001)


if __name__ == '__main__':
    unittest.main()
