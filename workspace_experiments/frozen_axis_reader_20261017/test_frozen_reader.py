"""Focused checks for a frozen axis reader and an independent common reader."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_reader as frozen


def fixture():
    generator = torch.Generator().manual_seed(20261017)
    n, ns, nc = 10, 6, 8
    features = {}
    for prefix in ('', 'common_'):
        features[prefix+'body'] = torch.randn(n, 128, generator=generator)
        features[prefix+'tokens'] = torch.randn(n, 12, 64, generator=generator)
        positions = torch.randn(n, 12, 16, generator=generator)
        positions[:, :, 2] = .8
        positions[:, :, 3] = 1.
        features[prefix+'positions'] = positions
        features[prefix+'valid'] = torch.ones(n, dtype=torch.bool)
    ids = np.array([8, 1, 7, 0, 5, 3, 9, 2, 6, 4])
    labels = torch.empty(n, dtype=torch.long)
    labels[ids] = torch.tensor([0, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    z = torch.randn(n, 5, generator=generator, dtype=torch.float64)
    prior = .3*torch.randn(n, nc, generator=generator, dtype=torch.float64)
    task = dict(
        ns=ns, ids=ids, s=ids[:ns], q=ids[ns:],
        ss=z[:ns]@z[:ns].T, qs=z[ns:]@z[:ns].T,
        root=torch.tensor([.6, .6, 1.2, 1.2, 1.2, 1.2],
                          dtype=torch.float64).sqrt(),
        desired=2*torch.eye(nc, dtype=torch.float64)[labels[ids[:ns]]]-1,
        prior_s=prior[:ns], prior_q=prior[ns:],
        base=torch.randn(n-ns, nc, generator=generator,
                         dtype=torch.float64).softmax(-1),
        old=torch.randn(n-ns, nc, generator=generator,
                        dtype=torch.float64).softmax(-1))
    return features, task, labels


class FrozenReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20261017)
        self.features, self.task, self.labels = fixture()
        self.old = frozen.EvidenceReader('static')
        self.new = frozen.EvidenceReader('static')
        self.axis = frozen.freeze_axis(self.old, self.features)

    def assert_nonzero(self, value):
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(float(value.detach().abs().sum()), 1e-10)

    def test_frozen_axis_has_no_gradient_and_new_task_loss_backpropagates(self):
        self.assertFalse(self.old.training)
        self.assertFalse(self.axis.requires_grad)
        self.assertIsNone(self.axis.grad_fn)
        self.assertTrue(all(not p.requires_grad for p in self.old.parameters()))
        loss = frozen.task_loss(
            self.new, self.features, self.task, self.labels, self.axis)[0]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.old.parameters()))
        gradients = [p.grad.flatten() for p in self.new.parameters()
                     if p.grad is not None]
        self.assertTrue(gradients)
        self.assert_nonzero(torch.cat(gradients))

    def test_one_update_keeps_old_state_and_raw_exactly_unchanged(self):
        old_state = {name: value.clone() for name, value in self.old.state_dict().items()}
        old_raw = self.axis.clone()
        common_before = frozen.all_raw(self.new, self.features, common=True)
        parameters = [p for p in self.new.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        optimizer.zero_grad(set_to_none=True)
        frozen.task_loss(
            self.new, self.features, self.task, self.labels, self.axis)[0].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, initial)
                            for p, initial in zip(parameters, before)))
        self.assert_nonzero(frozen.all_raw(self.new, self.features, common=True)-common_before)
        for name, value in self.old.state_dict().items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(value, old_state[name], atol=0, rtol=0)
        torch.testing.assert_close(self.axis, old_raw, atol=0, rtol=0)
        torch.testing.assert_close(
            frozen.all_raw(self.old, self.features), old_raw, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in self.old.parameters()))

    def test_readers_have_independent_parameter_objects_and_storage(self):
        old_parameters = dict(self.old.named_parameters())
        new_parameters = dict(self.new.named_parameters())
        self.assertEqual(old_parameters.keys(), new_parameters.keys())
        for name, old_parameter in old_parameters.items():
            with self.subTest(parameter=name):
                new_parameter = new_parameters[name]
                self.assertIsNot(old_parameter, new_parameter)
                self.assertNotEqual(old_parameter.data_ptr(), new_parameter.data_ptr())

    def test_no_fixed_axis_matches_legacy_margin_loss(self):
        expected = frozen.loss_parts(
            self.new, self.features, self.task, self.labels, False, True)
        actual = frozen.task_loss(
            self.new, self.features, self.task, self.labels, fixed_axis=None)
        self.assertEqual(len(actual), len(expected))
        for name, value, reference in zip(
                ('loss', 'brier', 'retrieval', 'margin', 'probability', 'features'),
                actual, expected):
            with self.subTest(part=name):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)

    def test_prediction_gradient_crosses_common_ss_qs_and_refit(self):
        ids, ns = self.task['ids'], self.task['ns']
        common = frozen.read(self.new, self.features, ids, True)[0]
        solve = torch.linalg.solve
        captured = []

        def capture_solve(matrix, rhs):
            solution = solve(matrix, rhs)
            captured.append((matrix, solution))
            return solution

        with mock.patch.object(torch.linalg, 'solve', side_effect=capture_solve):
            probability, _, _ = frozen.predict(
                self.axis[ids], self.features['valid'][ids], self.task,
                common, self.features['common_valid'][ids], axis_weight=1.)
        self.assertEqual(len(captured), 1)
        matrix, solution = captured[0]
        # Prediction risk alone excludes gradients supplied by the retrieval loss.
        risk = frozen.prior.old.risk(probability, self.labels[self.task['q']])
        matrix_grad, solution_grad, total_grad = torch.autograd.grad(
            risk, (matrix, solution, common), retain_graph=True)
        self.assert_nonzero(matrix_grad)
        self.assert_nonzero(solution_grad)
        ss_grad, = torch.autograd.grad(
            matrix, common, grad_outputs=matrix_grad, retain_graph=True)
        self.assert_nonzero(ss_grad[:ns])
        torch.testing.assert_close(ss_grad[ns:], torch.zeros_like(ss_grad[ns:]),
                                   atol=0, rtol=0)
        # The remaining route is QS; query features never enter the SS solve.
        qs_grad = total_grad-ss_grad
        self.assert_nonzero(qs_grad[:ns])
        self.assert_nonzero(qs_grad[ns:])
        risk.backward()
        self.assert_nonzero(self.new.content.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.old.parameters()))


if __name__ == '__main__':
    unittest.main()
