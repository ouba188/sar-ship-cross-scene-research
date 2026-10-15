"""CPU checks for actual readout mathematics and supported data-leak boundaries."""

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'oof_reliability_check'))
from strong_single_r50_task import (
    SingleR50, balanced_loss, episode_rows, geometry, grouped_split,
    support_scores, target_design, valid_identity, fit_source_prior,
)


class HeadTests(unittest.TestCase):
    def test_dual_matches_independent_weighted_primal(self):
        rng = np.random.default_rng(12)
        x, q = rng.normal(size=(17, 11)), rng.normal(size=(9, 11))
        y = np.array([0]*9+[2]*5+[7]*3)
        counts = np.bincount(y, minlength=8)
        weights = len(y)/(3*counts[y])
        desired = 2*np.eye(8)[y]-1
        beta = np.linalg.solve(x.T@(weights[:, None]*x)+8*np.eye(x.shape[1]),
                               x.T@(weights[:, None]*desired))
        actual = support_scores(torch.tensor(x), torch.tensor(y), torch.tensor(q))
        np.testing.assert_allclose(actual.numpy(), q@beta, atol=1e-11)

    def test_gradients_reach_support_and_query(self):
        torch.manual_seed(2)
        raw = torch.randn(80, 12, requires_grad=True)
        phi = geometry(raw)
        labels = torch.arange(64) % 8
        scores = support_scores(phi[:64], labels, phi[64:])
        balanced_loss(scores, torch.arange(16) % 8).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad[:64].norm()), 0.)
        self.assertGreater(float(raw.grad[64:].norm()), 0.)

    def test_support_order_and_query_chunk_invariance(self):
        torch.manual_seed(5)
        phi = geometry(torch.randn(88, 12))
        y = torch.arange(64) % 8
        expected = support_scores(phi[:64], y, phi[64:])
        order = torch.randperm(64)
        actual = support_scores(phi[order], y[order], phi[64:])
        torch.testing.assert_close(expected, actual, atol=4e-6, rtol=4e-6)
        split = torch.cat([support_scores(phi[:64], y, q) for q in phi[64:].split(5)])
        torch.testing.assert_close(expected, split, atol=4e-6, rtol=4e-6)

    def test_normalization_and_trainable_scope(self):
        phi = geometry(torch.randn(4, 2048))
        torch.testing.assert_close(phi[:, :-1].square().sum(1), torch.full((4,), 32.))
        torch.testing.assert_close(phi[:, -1], torch.ones(4))
        net = SingleR50(pretrained=False)
        net.train()
        self.assertEqual(net.prefix[0].in_channels, 2)
        self.assertTrue(all(not p.requires_grad for p in net.prefix.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.layer4.parameters()))
        self.assertFalse(any(m.training for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        self.assertFalse(hasattr(net, 'second_backbone'))

    def test_actual_checkpointed_tail_receives_gradient(self):
        torch.manual_seed(15)
        net = SingleR50(pretrained=False).train()
        phi = net(torch.randn(3, 2, 32, 32), microbatch=2)
        prior = net.prior(phi)
        scores = support_scores(phi[:2], torch.tensor([0, 1]), phi[2:],
                                support_prior=prior[:2], query_prior=prior[2:])
        (balanced_loss(scores, torch.tensor([0]))+.25*balanced_loss(prior[2:], torch.tensor([0]))).backward()
        self.assertGreater(float(net.layer4[0].conv1.weight.grad.norm()), 0.)
        self.assertGreater(float(net.prior.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in net.prefix.parameters()))

    def test_absent_support_classes_keep_distinct_source_evidence(self):
        support = torch.tensor([[1., 0.], [1., 0.]])
        query = torch.tensor([[.5, 1.]])
        y = torch.tensor([0, 1])
        sp = torch.zeros(2, 8)
        qp = torch.zeros(1, 8)
        qp[0, 6], qp[0, 7] = .7, -.3
        zero = support_scores(support, y, query)
        self.assertEqual(float(zero[0, 6]), float(zero[0, 7]))
        actual = support_scores(support, y, query, support_prior=sp, query_prior=qp)
        self.assertAlmostEqual(float(actual[0, 6]-actual[0, 7]), 1., places=6)

    def test_prior_fit_matches_balanced_sklearn_and_ignores_held_rows(self):
        from sklearn.linear_model import RidgeClassifier
        rng = np.random.default_rng(73)
        raw = rng.normal(size=(96, 10)).astype(np.float32)
        labels = np.arange(96) % 8
        rows = np.arange(72)
        phi = geometry(torch.tensor(raw)).numpy().astype(float)
        expected = RidgeClassifier(alpha=1., class_weight='balanced').fit(phi[rows, :-1], labels[rows])
        weights = fit_source_prior(raw, labels, rows)
        np.testing.assert_allclose(phi@weights.T, expected.decision_function(phi[:, :-1]), atol=1e-10)
        labels[72:] = 7-labels[72:]
        raw[72:] *= 100
        np.testing.assert_array_equal(weights, fit_source_prior(raw, labels, rows))


class LeakageTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({'port': np.repeat(['A', 'B', 'C', 'D'], 150),
                              'mmsi': [str(i+1000) for i in range(600)],
                              'product': [f'p{i}' for i in range(600)]})
        return frame

    def test_whole_port_and_shared_vessel_product_purge(self):
        frame = self.frame()
        rows = np.arange(len(frame))
        train, val, _ = grouped_split(frame, rows, validation_ports=1)
        frame.loc[train[0], 'mmsi'] = frame.loc[val[0], 'mmsi']
        frame.loc[train[1], 'product'] = frame.loc[val[1], 'product']
        frame.loc[train[2], 'mmsi'] = ''
        frame.loc[val[2], 'mmsi'] = ''
        new_train, new_val, audit = grouped_split(frame, rows, validation_ports=1)
        self.assertNotIn(train[0], new_train)
        self.assertNotIn(train[1], new_train)
        self.assertIn(train[2], new_train)
        np.testing.assert_array_equal(val, new_val)
        self.assertFalse(set(frame.iloc[new_train].port) & set(frame.iloc[new_val].port))
        for col in ('mmsi', 'product'):
            values = valid_identity(frame[col])
            self.assertFalse((set(values.iloc[new_train]) & set(values.iloc[new_val])) - {''})
        self.assertEqual(audit['identity_purge']['mmsi']['removed'], 1)

    def test_labels_do_not_change_split(self):
        frame = self.frame()
        frame['class_id'] = np.arange(len(frame)) % 8
        first = grouped_split(frame, np.arange(580), validation_ports=1)
        frame['class_id'] = 7-frame.class_id
        second = grouped_split(frame, np.arange(580), validation_ports=1)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue((first[0] < 580).all())

    def test_episode_disjointness(self):
        support, query = episode_rows(np.arange(180), np.random.default_rng(10))
        self.assertEqual(len(support), 64)
        self.assertEqual(len(query), 32)
        self.assertFalse(np.intersect1d(support, query).size)

    def test_target_query_labels_are_not_read(self):
        support = np.tile(np.arange(64), (5, 1))
        masks = np.tile(np.arange(80) >= 64, (5, 1))
        allowed = dict(source_rows=np.arange(40), rows=np.arange(80), ids=np.arange(80).astype(str),
                       support=support, support_labels=support % 8, eval_masks=masks)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'target.npz'
            np.savez(path, **allowed, query_labels=np.zeros(80))
            first = target_design(path)
            np.savez(path, **allowed, query_labels=np.ones(80)*7)
            second = target_design(path)
            self.assertNotIn('query_labels', first)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            masks[0, 0] = True
            np.savez(path, **allowed)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                target_design(path)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
