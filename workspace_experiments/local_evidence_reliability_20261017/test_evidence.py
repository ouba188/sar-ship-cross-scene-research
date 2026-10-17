import unittest
import numpy as np
import torch
from local_evidence import EvidenceReader, retrieval_loss
from extract_quality import measurements


class EvidenceTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(4)
        body, token, pos, quality = torch.randn(5, 128), torch.randn(5, 12, 64), torch.randn(5, 12, 16), torch.randn(5, 12, 10)
        pos[:, :, 2:4] = 1
        return body, token, pos, quality

    def test_shared_initialization_and_psd(self):
        inputs = self.inputs()
        values = []
        for arm in ('uniform', 'static', 'dynamic'):
            torch.manual_seed(5)
            model = EvidenceReader(arm)
            raw, weights, _ = model(*inputs)
            values.append(raw)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(raw@raw.T).min()), -1e-6)
            torch.testing.assert_close(weights.sum(1), torch.ones(5))
            self.assertTrue((weights[:, -1] > 0).all())
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[0], values[2])

    def test_unavailable_regions_use_null_not_zero_sea(self):
        inputs = self.inputs()
        inputs[2][:, :, 2:4] = 0
        raw, weights, _ = EvidenceReader('dynamic')(*inputs)
        self.assertTrue((raw == 0).all())
        torch.testing.assert_close(weights[:, -1], torch.ones(5))

    def test_frozen_override_really_removes_instance_gate(self):
        model = EvidenceReader('dynamic')
        inputs = self.inputs()
        with torch.no_grad():
            model.gate.weight.fill_(.2)
        _, _, correction = model(*inputs)
        _, weights, _ = model(*inputs, gate_override=correction.mean(0))
        torch.testing.assert_close(weights, weights[:1].expand_as(weights))

    def test_retrieval_support_count_and_missing_inputs(self):
        h = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        loss = retrieval_loss(h, torch.tensor([0, 1, 0]), 2, torch.ones(3, dtype=torch.bool))
        repeated = torch.cat([h[:1], h[:1], h[1:]])
        same = retrieval_loss(repeated, torch.tensor([0, 0, 1, 0]), 3, torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(loss, same)
        zero = retrieval_loss(h, torch.tensor([0, 1, 0]), 2, torch.tensor([False, True, True]))
        self.assertEqual(float(zero), 0.)

    def test_uniform_radiometry_and_both_polarizations(self):
        image = np.ones((3, 3))*4
        body = np.ones((3, 3), bool)
        weights = np.ones((2, 9))/9
        features, means, _ = measurements(image, body, weights)
        np.testing.assert_allclose(features[:, 0], np.log(2))
        np.testing.assert_allclose(features[:, 1], 0)
        np.testing.assert_allclose(means, 4)


if __name__ == '__main__':
    unittest.main()
