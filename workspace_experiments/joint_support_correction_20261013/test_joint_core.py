"""Bounded integration checks for fold-local correction and refitted gates."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

import joint_core as j


def ridge(kernel, prior, labels):
    # Equivalent weighted dual solve without the production square-root factors.
    counts = np.bincount(labels, minlength=8)
    weights = len(labels) / (np.count_nonzero(counts) * counts[labels])
    target = 2 * np.eye(8)[labels] - 1 - prior
    return np.linalg.solve(
        kernel + np.diag((len(labels) / 8.) / weights), target)


def transform(raw, model):
    cols = j.c.COLS
    x = raw[:, cols]
    moved = x + model['image_mean_d'] + (x - model['image_mean_x']) @ model['image_coef']
    centered = moved - moved.mean(axis=0)
    factor = np.linalg.norm(x - x.mean(axis=0)) / np.linalg.norm(centered)
    result = raw.copy()
    result[:, cols] = x.mean(axis=0) + factor * centered
    return result


class JointCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261013)
        self.n = 16
        cols = np.asarray(j.c.COLS)
        width = int(cols.max()) + 2
        self.raw = np.zeros((self.n, width))
        self.raw[:, cols[:3]] = rng.normal(size=(self.n, 3))
        self.raw[:4, cols[0]] += 5.
        self.raw[4:8, cols[1]] -= 3.
        self.raw[:, -1] = 1.
        self.prior = rng.normal(scale=.4, size=(self.n, 8))
        self.base_prior = rng.normal(scale=.5, size=(self.n, 8))
        base_features = rng.normal(size=(self.n, 5))
        self.kb = base_features @ base_features.T
        self.labels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
        self.groups = np.repeat(np.arange(8), 2)
        self.parts = []
        for repeat in range(3):
            order = np.arange(8) if repeat == 0 else rng.permutation(8)
            for fold, held_groups in enumerate(np.split(order, 4)):
                val = np.flatnonzero(np.isin(self.groups, held_groups))
                train = np.setdiff1d(np.arange(self.n), val)
                self.parts.append(dict(repeat=repeat, fold=fold,
                                       train=train.tolist(), validation=val.tolist()))
        self.model = dict(image_coef=np.zeros((len(cols), len(cols))),
                          image_mean_x=np.zeros(len(cols)),
                          image_mean_d=np.zeros(len(cols)))
        self.model['image_coef'][:3, :3] = np.array(
            [[.9, .8, -.4], [-.5, -.7, .6], [.4, -.2, .3]])
        self.model['image_mean_x'][:3] = [.2, -.4, .1]
        self.model['image_mean_d'][:3] = [.5, -.2, .4]

    def inputs(self, model=None):
        return (self.raw, self.prior, self.kb, self.base_prior, self.labels,
                self.groups, self.model if model is None else model, self.parts)

    def reference_oof(self, model):
        result = [np.empty((3, self.n, 8)) for _ in range(4)]
        for part in self.parts:
            r, train, val = part['repeat'], part['train'], part['validation']
            raw, changed = self.raw[train], transform(self.raw[train], model)
            result[0][r, val] = self.prior[val]
            result[1][r, val] = self.raw[val] @ raw.T @ ridge(
                raw @ raw.T, self.prior[train], self.labels[train])
            result[2][r, val] = self.raw[val] @ changed.T @ ridge(
                changed @ changed.T, self.prior[train], self.labels[train])
            ab = ridge(self.kb[np.ix_(train, train)],
                       self.base_prior[train], self.labels[train])
            result[3][r, val] = softmax(
                self.base_prior[val] + self.kb[np.ix_(val, train)] @ ab, axis=1)
        return [value.reshape(-1, 8) for value in result]

    def test_zero_map_initial_equals_raw_oof_and_closed_form(self):
        model = {key: np.zeros_like(value) for key, value in self.model.items()}
        actual, labels, _ = j.paired_oof(*self.inputs(model))
        expected = self.reference_oof(model)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual[2], actual[1], rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(labels, np.tile(self.labels, 3))

    def test_nonzero_map_uses_train_center_energy_and_raw_validation(self):
        before = self.raw.copy()
        actual, _, _ = j.paired_oof(*self.inputs())
        for got, want in zip(actual, self.reference_oof(self.model)):
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(self.raw, before)
        train, val = self.parts[0]['train'], self.parts[0]['validation']
        changed = transform(self.raw[train], self.model)
        alpha = ridge(changed @ changed.T, self.prior[train], self.labels[train])
        expected = self.raw[val] @ changed.T @ alpha
        full_changed = transform(self.raw, self.model)
        wrong_train = full_changed[train]
        wrong_full = self.raw[val] @ wrong_train.T @ ridge(
            wrong_train @ wrong_train.T, self.prior[train], self.labels[train])
        wrong_validation = transform(self.raw[val], self.model) @ changed.T @ alpha
        # Negative controls make sure this fixture distinguishes the actual risks.
        full_gap = np.max(np.abs(expected - wrong_full))
        validation_gap = np.max(np.abs(expected - wrong_validation))
        self.assertGreater(full_gap, 1e-3)
        self.assertGreater(validation_gap, 1e-3)
        print(f'\nNegative-control gaps: full-support={full_gap:.6g}, '
              f'transformed-validation={validation_gap:.6g}', flush=True)

    def test_combined_refits_initial_oof_and_shrinks_gate_at_prediction(self):
        expected = self.reference_oof(self.model)
        labels = np.tile(self.labels, 3)
        with patch.object(j.utility, 'fit_gate', wraps=j.utility.fit_gate) as fit:
            fitted, _, _ = j.fit_pair(*self.inputs())
        self.assertEqual(fit.call_count, 2)
        expected_gates = {}
        for call, arm, index in zip(fit.call_args_list, ('light', 'combined'), (1, 2)):
            prior, residual, base, y, tau, mode = call.args
            for got, want in ((prior, expected[0]), (residual, expected[index]),
                              (base, expected[3])):
                np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(y, labels)
            self.assertEqual((tau, mode), (.03, 'joint'))
            gate, _ = j.utility.fit_gate(
                expected[0], expected[index], expected[3], labels, .03, 'joint')
            expected_gates[arm] = gate
            np.testing.assert_allclose(fitted['gate_' + arm], gate, rtol=1e-7, atol=1e-8)
        gate_gap = np.max(np.abs(expected_gates['combined'] - expected_gates['light']))
        self.assertGreater(gate_gap, 1e-4)

        query = self.raw[[1, 6, 12]].copy()
        query[:, np.asarray(j.c.COLS)[:3]] += [.2, -.3, .1]
        changed = transform(self.raw, self.model)
        raw_residual = query @ self.raw.T @ ridge(
            self.raw @ self.raw.T, self.prior, self.labels)
        initial_residual = query @ changed.T @ ridge(
            changed @ changed.T, self.prior, self.labels)
        np.testing.assert_allclose(query @ fitted['raw_weight'], raw_residual,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(query @ fitted['initial_weight'], initial_residual,
                                   rtol=1e-10, atol=1e-10)
        prior, base = self.prior[:3], self.base_prior[:3]
        actual = j.scores(base, prior, raw_residual, initial_residual,
                          initial_residual, fitted['gate_light'], fitted['gate_combined'])
        for arm, residual in (('light', raw_residual), ('combined', initial_residual)):
            gamma = 1 + .25 * (expected_gates[arm] - 1)
            want = .5 * (softmax(base, axis=1) + softmax(prior + gamma * residual, axis=1))
            np.testing.assert_allclose(actual[arm], want, rtol=1e-9, atol=1e-10)
        unshrunk = .5 * (softmax(base, axis=1) + softmax(
            prior + expected_gates['combined'] * initial_residual, axis=1))
        reused = .5 * (softmax(base, axis=1) + softmax(
            prior + (1 + .25 * (expected_gates['light'] - 1)) * initial_residual, axis=1))
        self.assertGreater(np.max(np.abs(actual['combined'] - unshrunk)), 1e-5)
        self.assertGreater(np.max(np.abs(actual['combined'] - reused)), 1e-5)
        print(f'\nIndependent raw/initial gate difference: {gate_gap:.6g}', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
