"""Independent numerical checks for axis-preserving common evidence addition."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.special import softmax
import torch

LEGACY = Path(__file__).resolve().parent.parent/'axis_access_margin_20261017'
sys.path.insert(0, str(LEGACY))
import model

spec = importlib.util.spec_from_file_location('axis_margin_run_tests', LEGACY/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    rng = np.random.default_rng(20261017)
    ns, nq, nc = 6, 4, 8
    z = rng.normal(size=(ns+nq, 5))
    axis = torch.tensor(rng.normal(size=(ns+nq, 4)), dtype=torch.float64)
    common = torch.tensor(rng.normal(size=(ns+nq, 3)), dtype=torch.float64)
    axis[ns:] += 1.7
    common[ns:] -= .8
    av = torch.tensor([True, False, True, True, False, True,
                       True, False, True, False])
    cv = torch.tensor([False, True, True, False, True, True,
                       True, True, False, False])
    labels = torch.tensor([0, 0, 1, 2, 3, 4, 0, 1, 2, 3])
    task = dict(ns=ns, ids=np.arange(ns+nq), s=np.arange(ns), q=np.arange(ns, ns+nq))
    arrays = dict(ss=z[:ns]@z[:ns].T, qs=z[ns:]@z[:ns].T,
                  root=np.sqrt(np.array([.6, .6, 1.2, 1.2, 1.2, 1.2])),
                  desired=2*np.eye(nc)[labels[:ns].numpy()]-1,
                  prior_s=rng.normal(size=(ns, nc))*.3,
                  prior_q=rng.normal(size=(nq, nc))*.3,
                  base=softmax(rng.normal(size=(nq, nc)), axis=1),
                  old=softmax(rng.normal(size=(nq, nc)), axis=1))
    task.update({key: torch.tensor(value, dtype=torch.float64)
                 for key, value in arrays.items()})
    return z, axis, common, av, cv, labels, task


def primal_probability(z, axis, common, av, cv, task, axis_weight):
    ns = task['ns']
    blocks = [z]
    for raw, valid, weight in ((axis, av, axis_weight), (common, cv, .5)):
        values, mask = raw.numpy(), valid.numpy()
        centered = (values-values[:ns][mask[:ns]].mean(axis=0))*mask[:, None]
        normalized = centered/np.sqrt(1+(centered**2).sum(axis=1, keepdims=True))
        blocks.append(np.sqrt(weight)*normalized)
    design = np.concatenate(blocks, axis=1)
    support, query = design[:ns], design[ns:]
    weights = task['root'].numpy()**2
    residual = (task['desired']-task['prior_s']).numpy()
    # Solve in feature space, independently of the production dual kernel solve.
    coefficient = np.linalg.solve(
        support.T@(weights[:, None]*support)+8*np.eye(design.shape[1]),
        support.T@(weights[:, None]*residual))
    probability = .5*(task['base'].numpy()+softmax(
        task['prior_q'].numpy()+query@coefficient, axis=1))
    active = ((av[ns:] & (av[:ns].sum() >= 2)) |
              (cv[ns:] & (cv[:ns].sum() >= 2))).numpy()
    return np.where(active[:, None], probability, task['old'].numpy())


class AxisPreservingAdditionTests(unittest.TestCase):
    def test_axis_one_common_half_matches_independent_weighted_primal(self):
        z, axis, common, av, cv, _, task = fixture()
        expected = primal_probability(z, axis, common, av, cv, task, 1.)
        actual, _, _ = model.predict(axis, av, task, common, cv, axis_weight=1.)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=0)
        old_half = primal_probability(z, axis, common, av, cv, task, .5)
        self.assertGreater(float(np.max(np.abs(expected-old_half))), 1e-5)

    def test_zero_common_with_valid_axes_recovers_legacy_axis_one(self):
        _, axis, common, _, _, _, task = fixture()
        av = torch.ones(len(axis), dtype=torch.bool)
        common = torch.zeros_like(common)
        expected, legacy_h = model.prior.old.predict(axis, av, task, 1.)
        for common_available in (False, True):
            with self.subTest(common_available=common_available):
                cv = torch.full_like(av, common_available)
                actual, h, valid = model.predict(
                    axis, av, task, common, cv, axis_weight=1.)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
                torch.testing.assert_close(h[:, :axis.shape[1]], legacy_h, atol=0, rtol=0)
                self.assertEqual(int(torch.count_nonzero(h[:, axis.shape[1]:])), 0)
                self.assertTrue(bool(valid.all()))

    def test_default_preserves_legacy_half_axis_half_common(self):
        z, axis, common, av, cv, _, task = fixture()
        expected = primal_probability(z, axis, common, av, cv, task, .5)
        default = model.predict(axis, av, task, common, cv)
        explicit = model.predict(axis, av, task, common, cv, axis_weight=.5)
        np.testing.assert_allclose(default[0].numpy(), expected, atol=1e-12, rtol=0)
        for actual, reference in zip(default, explicit):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        legacy_p, legacy_h = model.prior.old.predict(axis, av, task, 1.)
        default_p, default_h, _ = model.predict(axis, av, task)
        torch.testing.assert_close(default_p, legacy_p, atol=0, rtol=0)
        torch.testing.assert_close(default_h, legacy_h, atol=0, rtol=0)

    def test_loss_and_evaluate_forward_axis_weight(self):
        z, axis, common, av, cv, labels, task = fixture()
        features = dict(valid=av, common_valid=cv)

        def read_stub(reader, feature_map, ids, use_common=False):
            return (common if use_common else axis), None, None

        with mock.patch.object(model, 'read', side_effect=read_stub):
            for weight in (.5, 1.):
                with self.subTest(axis_weight=weight):
                    parts = model.loss_parts(
                        None, features, task, labels, True, True, axis_weight=weight)
                    expected = primal_probability(z, axis, common, av, cv, task, weight)
                    np.testing.assert_allclose(parts[4].numpy(), expected, atol=1e-12, rtol=0)
                    torch.testing.assert_close(
                        parts[0], parts[1]+.05*parts[2]+.05*parts[3], atol=0, rtol=0)
                    evaluated = runner.evaluate(
                        None, features, [task], labels, True, True, axis_weight=weight)
                    for key, value in zip(
                            ('objective', 'brier', 'retrieval_ce', 'margin_loss'), parts[:4]):
                        self.assertAlmostEqual(evaluated[key], float(value), places=12)

    def test_predict_ignores_truth_and_task_query_labels(self):
        _, axis, common, av, cv, labels, task = fixture()
        expected = model.predict(axis, av, task, common, cv, axis_weight=1.)
        for shift in (0, 3):
            with self.subTest(label_shift=shift):
                query_labels = (labels[task['q']]+shift) % 8
                labeled_task = dict(task, truth=query_labels, querylabels=query_labels,
                                    query_labels=query_labels)
                actual = model.predict(axis, av, labeled_task, common, cv, axis_weight=1.)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
