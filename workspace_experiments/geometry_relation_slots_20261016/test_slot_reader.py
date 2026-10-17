"""CPU-only checks of task centering, full ridge integration and runner inputs."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_slots as runner
from slot_reader import SlotReader, centered_features, predict_increment


def numpy_softmax(x):
    value = np.exp(x - x.max(axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def numpy_coefficients(kernel, prior, labels):
    # Solve the unweighted normal equation with lambda / sample_weight on
    # the diagonal, independently of the production square-root formulation.
    classes, counts = np.unique(labels, return_counts=True)
    counts_by_class = dict(zip(classes, counts))
    weight = np.array([len(labels) / (len(classes) * counts_by_class[y]) for y in labels])
    rhs = 2 * np.eye(prior.shape[1])[labels] - 1 - prior
    return np.linalg.solve(kernel + np.diag(8 / weight), rhs)


class SlotReaderTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.nq, self.nc = 5, 3, 8
        self.rng = torch.Generator().manual_seed(702)
        self.ss_features = self.rand(self.ns, 6)
        self.qs_features = self.rand(self.nq, 6)
        self.ss = self.ss_features @ self.ss_features.T
        self.qs = self.qs_features @ self.ss_features.T
        self.ps, self.pq = self.rand(self.ns, self.nc), self.rand(self.nq, self.nc)
        self.y = torch.tensor([0, 0, 1, 2, 2])
        self.bp = self.rand(self.nq, self.nc).softmax(-1)
        coef = runner._coefficients(self.ss, self.ps, self.y)
        self.old = .5 * (self.bp + (self.pq + self.qs @ coef).softmax(-1))
        self.raw = self.rand(self.ns + self.nq, 7)
        self.valid = torch.tensor([True, True, False, True, True, True, False, True])

    def rand(self, *shape):
        return torch.randn(shape, generator=self.rng, dtype=torch.float64)

    def predict(self, raw=None, valid=None, coefficient_fn=None):
        return predict_increment(self.ss, self.qs, self.ps, self.pq, self.y,
                                 self.bp, self.old,
                                 self.raw if raw is None else raw,
                                 self.valid if valid is None else valid,
                                 runner._coefficients if coefficient_fn is None else coefficient_fn)

    def test_support_only_centering_excludes_invalid_supports_and_queries(self):
        psi = centered_features(self.raw, self.valid, self.ns)
        expected_mean = self.raw[[0, 1, 3, 4]].mean(0)
        expected = (self.raw - expected_mean) * self.valid[:, None]
        torch.testing.assert_close(psi, expected, atol=0, rtol=0)
        changed = self.raw.clone()
        changed[2] += 999
        changed[self.ns + 1] -= 999
        changed[-1] += 123
        result = centered_features(changed, self.valid, self.ns)
        torch.testing.assert_close(result[:self.ns + 2], psi[:self.ns + 2], atol=0, rtol=0)
        torch.testing.assert_close(result[-1] - psi[-1], torch.full((7,), 123., dtype=torch.float64))
        torch.testing.assert_close(psi[:self.ns][self.valid[:self.ns]].mean(0),
                                   torch.zeros(7, dtype=torch.float64), atol=1e-15, rtol=0)

    def test_constant_descriptor_zero_and_baseline_probability(self):
        raw = torch.full_like(self.raw, 3.25)
        probability, kernel, psi = self.predict(raw=raw)
        torch.testing.assert_close(psi, torch.zeros_like(raw), atol=0, rtol=0)
        torch.testing.assert_close(kernel, self.qs, atol=0, rtol=0)
        torch.testing.assert_close(probability, self.old, atol=0, rtol=0)

    def test_zero_or_one_valid_support_returns_exact_baseline(self):
        for count in (0, 1):
            with self.subTest(count=count):
                valid = torch.ones_like(self.valid)
                valid[:self.ns] = False
                valid[:count] = True
                probability, kernel, psi = self.predict(valid=valid)
                torch.testing.assert_close(probability, self.old, atol=0, rtol=0)
                torch.testing.assert_close(psi, torch.zeros_like(self.raw), atol=0, rtol=0)
                torch.testing.assert_close(kernel, self.qs, atol=0, rtol=0)

    def test_invalid_query_exact_baseline_even_when_support_solve_changes(self):
        captured = []

        def coefficients(kernel, prior, labels):
            captured.append(kernel.detach().clone())
            return runner._coefficients(kernel, prior, labels)

        probability, kernel, _ = self.predict(coefficient_fn=coefficients)
        self.assertGreater(float((captured[0] - self.ss).norm()), .1)
        torch.testing.assert_close(kernel[1], self.qs[1], atol=0, rtol=0)
        naive = .5 * (self.bp + (self.pq + kernel @ runner._coefficients(
            captured[0], self.ps, self.y)).softmax(-1))
        self.assertGreater(float((naive[1] - self.old[1]).norm()), 1e-6)
        torch.testing.assert_close(probability[1], self.old[1], atol=0, rtol=0)
        self.assertGreater(float((probability[0] - self.old[0]).norm()), 1e-6)

    def test_independent_numpy_ridge_and_fixed_additive_strength(self):
        raw, valid = self.raw.numpy(), self.valid.numpy()
        psi = (raw - raw[:self.ns][valid[:self.ns]].mean(axis=0)) * valid[:, None]
        ss = self.ss.numpy() + .75 * psi[:self.ns] @ psi[:self.ns].T
        qs = self.qs.numpy() + .75 * psi[self.ns:] @ psi[:self.ns].T
        coef = numpy_coefficients(ss, self.ps.numpy(), self.y.numpy())
        expected = .5 * (self.bp.numpy() + numpy_softmax(self.pq.numpy() + qs @ coef))
        expected[~valid[self.ns:]] = self.old.numpy()[~valid[self.ns:]]
        actual, kernel, _ = self.predict()
        np.testing.assert_allclose(kernel.numpy(), qs, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)

    def test_support_and_query_gradients_flow_and_match_finite_difference(self):
        raw = self.raw.clone().requires_grad_(True)
        probability, _, psi = self.predict(raw=raw)
        psi.retain_grad()
        loss = -probability[0, 2].log() - probability[2, 1].log()
        loss.backward()
        for sl in (slice(0, self.ns), slice(self.ns, None)):
            self.assertTrue(torch.isfinite(raw.grad[sl]).all())
            self.assertGreater(float(raw.grad[sl].norm()), 1e-6)
            self.assertGreater(float(psi.grad[sl].norm()), 1e-6)
        torch.testing.assert_close(raw.grad[~self.valid], torch.zeros_like(raw.grad[~self.valid]), atol=0, rtol=0)
        for row in (0, self.ns):
            plus, minus = self.raw.clone(), self.raw.clone()
            plus[row, 0] += 1e-5
            minus[row, 0] -= 1e-5
            pp, _, _ = self.predict(raw=plus)
            pm, _, _ = self.predict(raw=minus)
            finite = (-pp[0, 2].log() - pp[2, 1].log() + pm[0, 2].log() + pm[2, 1].log()) / 2e-5
            torch.testing.assert_close(raw.grad[row, 0], finite, atol=2e-8, rtol=2e-5)

    def make_runner_data(self):
        n = self.ns + self.nq
        geometry = self.rand(n, 7)
        geometry[:, 3:6] = self.valid[:, None].double()
        d = dict(img_tokens=self.rand(n, 4, 4, 64), ship_tokens=self.rand(n, 4, 4, 64),
                 img_areas=torch.ones(n, 4, 4, dtype=torch.float64),
                 ship_areas=torch.ones(n, 4, 4, dtype=torch.float64), geometry=geometry,
                 valid=self.valid, labels=torch.cat((self.y, torch.tensor([2, 0, 1]))))
        item = dict(s=torch.arange(self.ns), q=torch.arange(self.ns, n),
                    ss=self.ss, qs=self.qs, prior_s=self.ps, prior_q=self.pq,
                    sy=self.y, base_prob=self.bp, old_prob=self.old)
        return {"tensors": d}, item

    def test_runner_query_labels_unused_and_query_batch_independent(self):
        data, item = self.make_runner_data()
        model = SlotReader("geometry").double().eval()
        with patch.object(runner, "DEVICE", "cpu"):
            first, _, _ = runner.prediction(model, data, item)
            data["tensors"]["labels"][item["q"]] = 7
            changed, _, _ = runner.prediction(model, data, item)
            single, _, _ = runner.prediction(model, data, item, torch.tensor([0]))
        torch.testing.assert_close(first, changed, atol=0, rtol=0)
        torch.testing.assert_close(first[:1], single, atol=1e-12, rtol=1e-12)

    def test_reader_parameters_receive_gradient_through_full_solve(self):
        for mode in ("spatial", "conditional", "geometry"):
            with self.subTest(mode=mode):
                data, item = self.make_runner_data()
                model = SlotReader(mode).double()
                with patch.object(runner, "DEVICE", "cpu"):
                    probability, _, _ = runner.prediction(model, data, item)
                (-probability[0, 2].log() - probability[2, 1].log()).backward()
                for parameter in model.relation.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.norm()), 0)
                if mode != "spatial":
                    self.assertGreater(float(model.condition.weight.grad.norm()), 0)

    def test_runner_ship45_rolls_slots_negative_one_not_positive(self):
        data, _ = self.make_runner_data()
        d = data["tensors"]
        model = SlotReader("geometry").double().eval()
        with torch.no_grad():
            model.condition.weight.copy_(self.rand(16, 8))
        geometry = d["geometry"].clone()
        geometry[:, 0] += np.pi / 4
        expected = model(d["ship_tokens"].roll(-1, 2), d["ship_areas"].roll(-1, 2), geometry)
        actual = runner.descriptor(model, d, slice(None), "ship45")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        wrong = model(d["ship_tokens"].roll(1, 2), d["ship_areas"].roll(1, 2), geometry)
        self.assertGreater(float((actual - wrong).detach().norm()), 1e-4)

    def test_integration_check_handles_pool_without_eligible_fit_tasks(self):
        # Shanghai's real B32 inventory has no eligible geometry tasks. The
        # integration-check phase must record that skip, as training does.
        with patch.object(runner, "fit_inventory", return_value=([], {}, [], {"tasks": 0})):
            result = runner.check({})
        self.assertIsInstance(result, dict)
        self.assertTrue(result["unavailable"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
