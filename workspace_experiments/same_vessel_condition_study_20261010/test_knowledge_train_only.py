"""Small synthetic checks; no optimizer steps or experiment data are used."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import io
import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from knowledge_train_only import (
    KnowledgeStudent, SAROnlyPredictor, attribute_loss, phi,
    prepare_attributes, sample_weights,
)
from target_response_adapter import residual_head


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def attribute_fixture(self):
        meta = pd.DataFrame(dict(mmsi=[10, 10, 10, 20, 30, 30, 40],
                                 class_id=[0, 0, 0, 0, 1, 1, 1]))
        values = np.exp(np.array([[2., 1.]]*3+[[4., 2.]]+[[6., 3.]]*2+[[0., 0.]]))
        values[-1] = np.nan
        valid = np.array([True]*6+[False])
        probability = sample_weights(meta)
        probability /= probability.sum()
        return meta, values, valid, probability

    def student_fixture(self):
        torch.manual_seed(31415)
        raw = torch.randn(96, 40)
        center, scale = raw.mean(0), raw.std(0, unbiased=False).clamp_min(1e-5)
        x = (raw-center)/scale
        y = torch.arange(len(x))%8
        model = KnowledgeStudent(x, y, np.ones(len(x)), dimensions=12)
        return model, raw, x, y, center, scale

    def test_statistics_balance_classes_and_ships_not_repeated_rows(self):
        meta, values, valid, probability = self.attribute_fixture()
        result = prepare_attributes(values, valid, meta, probability)
        np.testing.assert_allclose(result['log_mean'], [4.5, 2.25])
        np.testing.assert_allclose(result['log_scale']**2, [2.75, .6875])
        np.testing.assert_allclose(result['class_log_means'][:2], [[3., 1.5], [6., 3.]])
        np.testing.assert_array_equal(result['valid'], valid)
        # The shared sampler's expected auxiliary weights give .25/.25/.5 per ship.
        effective = probability*result['importance']
        np.testing.assert_allclose(effective, [1/12]*3+[.25, .25, .25, 0], rtol=1e-6)
        np.testing.assert_allclose(effective@result['real'], [0., 0.], atol=1e-7)
        self.assertTrue(np.isfinite(result['real']).all())
        self.assertTrue(np.isfinite(result['control']).all())
        np.testing.assert_array_equal(result['real'][~valid], 0)
        np.testing.assert_array_equal(result['control'][~valid], 0)

    def test_invalid_values_do_not_change_statistics_or_control_targets(self):
        meta, values, valid, probability = self.attribute_fixture()
        baseline = prepare_attributes(values, valid, meta, probability)
        values[-1] = [-99., np.inf]
        values.flags.writeable = False
        valid.flags.writeable = False
        probability.flags.writeable = False
        changed = prepare_attributes(values, valid, meta, probability)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], changed[key])
        np.testing.assert_array_equal(valid, [True]*6+[False])

    def test_conflicting_valid_ship_dimensions_are_not_silently_averaged(self):
        meta, values, valid, probability = self.attribute_fixture()
        values[1, 0] *= 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            prepare_attributes(values, valid, meta, probability)

    def test_attribute_gradient_reaches_frame_and_missing_rows_keep_class_gradient(self):
        model, _, x, y, _, _ = self.student_fixture()
        z, scores = model(x)
        scores.retain_grad()
        predictions = model.attribute_head(z)
        predictions.retain_grad()
        valid = torch.arange(len(x))%3 != 0
        targets = torch.randn(len(x), 2)
        targets[~valid] = float('nan')
        auxiliary = attribute_loss(predictions, targets, valid, torch.ones(len(x)))
        gradient = torch.autograd.grad(auxiliary, model.frame, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.)
        classification = F.mse_loss(scores, 2*F.one_hot(y, 8).float()-1)
        (classification+.1*auxiliary).backward()
        self.assertTrue(torch.isfinite(model.frame.grad).all())
        self.assertGreater(float(model.attribute_head.weight.grad.norm()), 0.)
        self.assertTrue((scores.grad[~valid].abs().sum(1) > 0).all())
        torch.testing.assert_close(predictions.grad[~valid], torch.zeros_like(predictions.grad[~valid]))

    def test_all_missing_batch_has_finite_zero_auxiliary_gradient(self):
        prediction = torch.randn(5, 2, requires_grad=True)
        loss = attribute_loss(prediction, torch.full((5, 2), float('nan')),
                              torch.zeros(5, dtype=torch.bool), torch.zeros(5))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_deleted_aux_head_and_serialized_deployment_need_only_sar(self):
        model, raw, x, y, center, scale = self.student_fixture()
        case_state = model.state_dict()
        with torch.no_grad():
            kernel, expected_prior = model(x)
        del model.attribute_head
        deployed = SAROnlyPredictor.from_student(model, case_state, center, scale)
        self.assertFalse(any('attribute' in key for key in deployed.state_dict()))
        self.assertEqual(len(list(deployed.parameters())), 0)
        prior, actual_kernel = deployed(raw)
        torch.testing.assert_close(prior, expected_prior, rtol=0, atol=0)
        torch.testing.assert_close(actual_kernel, kernel, rtol=0, atol=0)
        support, query = slice(0, 64), slice(64, None)
        beta = residual_head(phi(kernel[support]), expected_prior[support], y[support], balanced=True)[0]
        expected = expected_prior[query]+phi(kernel[query])@beta
        established = deployed.establish_support(raw[support], y[support])
        # Different GEMM batch sizes may round the score's last bits differently.
        torch.testing.assert_close(deployed.predict(raw[query], established), expected, rtol=1e-5, atol=1e-6)
        stream = io.BytesIO()
        torch.save(deployed.state_dict(), stream)
        stream.seek(0)
        restored = SAROnlyPredictor(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored.predict(raw[query], established),
                                   deployed.predict(raw[query], established), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
