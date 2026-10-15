"""Bounded CPU synthetic checks; no data loading, training, or artifact writes.

Run: python -B body_context_gate_test.py
"""

import math
import unittest

import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from body_context_gate import Gate, _coefficients, readout
from full_readout import infer_with_local
from source_context_pilot import coefficients as numpy_coefficients
from unified_head_core import class_weights


def fixture():
    rng = np.random.default_rng(20261015)

    def unit(width):
        x = rng.normal(size=(29, width))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    data = dict(base=np.c_[math.sqrt(32.) * unit(128), np.ones(29)],
                expert=np.c_[4. * unit(128), 4. * unit(128), np.ones(29)],
                prior_b=rng.normal(0., .4, (29, 8)),
                prior_e=rng.normal(0., .4, (29, 8)),
                relation=rng.normal(0., .7, (29, 28)),
                weights=np.linspace(.2, 1.8, 28),
                labels=np.full(29, -1, dtype=np.int64),
                q=np.array([22, 4, 17, 0, 13], dtype=np.int64),
                s=np.array([21, 3, 19, 1, 15, 9, 11, 7, 25], dtype=np.int64))
    # Imbalanced support with missing classes; query labels are deliberately unknown.
    data["labels"][data["s"]] = [0, 0, 0, 0, 2, 2, 5, 5, 7]
    return data, unit(64), unit(4)


def numpy_readout(data):
    return infer_with_local(
        data["prior_b"], data["prior_e"], data["base"], data["expert"],
        data["q"], data["s"], data["labels"], local=data.get("local"),
        fraction=.25, relation=data.get("relation"), weights=data.get("weights"))


def primal_readout(data):
    """Independent weighted feature-space regression, without support kernels."""
    z = data["expert"]
    if data.get("local") is not None:
        z = np.c_[z[:, :128], math.sqrt(.75) * z[:, 128:256],
                  z[:, 256:], 2. * data["local"]]
    if data.get("relation") is not None:
        multiplier = data.get("weights")
        multiplier = 1. if multiplier is None else multiplier
        z = np.c_[math.sqrt(.75) * z,
                  .5 * data["relation"] * np.sqrt(multiplier)]
    s, q = data["s"], data["q"]
    y = data["labels"][s]
    counts = {int(c): int(np.sum(y == c)) for c in set(y)}
    w = np.array([len(y) / (len(counts) * counts[int(c)]) for c in y])
    target = 2. * np.eye(data["prior_b"].shape[1])[y] - 1.
    probs = []
    for x, prior in ((data["base"], data["prior_b"]), (z, data["prior_e"])):
        xs = x[s]
        head = np.linalg.solve(xs.T @ (w[:, None] * xs) + 8. * np.eye(x.shape[1]),
                               xs.T @ (w[:, None] * (target - prior[s])))
        probs.append(softmax(prior[q] + x[q] @ head, axis=1))
    return .5 * (probs[0] + probs[1])


class BodyContextGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.blas_limit = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.blas_limit.restore_original_limits()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.arrays, body, context = fixture()
        self.data = {k: torch.from_numpy(v.copy()) for k, v in self.arrays.items()}
        self.body = torch.from_numpy(body)
        self.context = torch.from_numpy(context)

    def test_zero_init_additive_kernel_and_parameter_counts(self):
        additive = torch.cat((self.body, self.context), dim=1) * math.sqrt(.5)
        for mode, count in (("static", 64), ("conditional", 320)):
            with self.subTest(mode=mode):
                gate = Gate(mode).double()
                self.assertEqual(sum(p.numel() for p in gate.parameters()), count)
                self.assertEqual(hasattr(gate, "W"), mode == "conditional")
                self.assertTrue(all(torch.count_nonzero(p) == 0 for p in gate.parameters()))
                local = gate(self.body, self.context)
                self.assertEqual(local.shape, (29, 132))
                torch.testing.assert_close(local @ local.T, additive @ additive.T,
                                           atol=1e-14, rtol=1e-14)
                torch.testing.assert_close(readout(**self.data, local=local),
                                           readout(**self.data, local=additive),
                                           atol=1e-14, rtol=1e-14)

    def test_gate_formula_and_gate_input_only_override(self):
        shuffled = self.context.flip(0)
        for mode in ("static", "conditional"):
            gate = Gate(mode).double()
            with torch.no_grad():
                gate.c.copy_(torch.linspace(-.4, .4, 64))
                if mode == "conditional":
                    gate.W.copy_(torch.linspace(-.8, .9, 256).reshape(64, 4))
            actual = gate(self.body, self.context, shuffled)
            logits = gate.c.detach().numpy()
            if mode == "conditional":
                logits = shuffled.numpy() @ gate.W.detach().numpy().T + logits
            u = self.body.numpy() * (1. + .25 * np.tanh(logits))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            expected = np.c_[.5 * self.body.numpy(), math.sqrt(.5) * self.context.numpy(), .5 * u]
            np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-14, atol=1e-14)
            original = gate(self.body, self.context)
            torch.testing.assert_close(actual[:, :68], original[:, :68], rtol=0, atol=0)
            if mode == "static":
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
            else:
                self.assertGreater((actual[:, 68:] - original[:, 68:]).abs().max().item(), 1e-4)

    def test_balanced_coefficients_against_numpy_and_independent_system(self):
        for y in (np.array([0, 0, 0, 0, 2, 2, 5, 5, 7]),
                  np.array([6, 6, 6]), np.array([7])):
            with self.subTest(labels=y.tolist()):
                x = self.arrays["base"][:len(y)]
                prior = self.arrays["prior_e"][:len(y)]
                kernel = x @ x.T
                w = np.array([len(y) / (len(set(y)) * np.sum(y == c)) for c in y])
                np.testing.assert_allclose(class_weights(y), w, rtol=0, atol=0)
                expected = numpy_coefficients(kernel, prior, y, 8.)
                # Alternative normal equations: (K + ridge / sample_weight) a = residual.
                independent = np.linalg.solve(kernel + np.diag(8. / w),
                                              2. * np.eye(8)[y] - 1. - prior)
                actual = _coefficients(torch.from_numpy(kernel), torch.from_numpy(prior),
                                       torch.from_numpy(y)).numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual, independent, rtol=1e-12, atol=1e-12)

    def test_full_numpy_parity_and_independent_primal(self):
        gate = Gate().double()
        with torch.no_grad():
            gate.W.copy_(torch.linspace(-.7, .8, 256).reshape(64, 4))
            gate.c.copy_(torch.linspace(-.2, .3, 64))
        additive = torch.cat((self.body, self.context), dim=1) * math.sqrt(.5)
        for local in (None, self.body, self.context, additive,
                      gate(self.body, self.context).detach()):
            for relation_mode in ("plain", "fixed", "learned"):
                with self.subTest(local_dim=None if local is None else local.shape[1],
                                  relation=relation_mode):
                    data = dict(self.data, local=local)
                    if relation_mode == "plain":
                        data["relation"] = None
                    if relation_mode != "learned":
                        data["weights"] = None
                    arrays = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                              for k, v in data.items()}
                    actual = readout(**data).numpy()
                    np.testing.assert_allclose(actual, numpy_readout(arrays), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(actual, primal_readout(arrays), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(actual.sum(1), 1., rtol=1e-14, atol=1e-14)

    def assert_directional_gradients(self, objective, variables):
        loss = objective()
        gradients = torch.autograd.grad(loss, variables)
        generator = torch.Generator().manual_seed(710)
        for value, gradient in zip(variables, gradients):
            direction = torch.randn(value.shape, generator=generator, dtype=value.dtype)
            direction /= direction.norm()
            analytic = (gradient * direction).sum().item()
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 1e-10)
            original = value.detach().clone()
            epsilon = 1e-5
            try:
                with torch.no_grad():
                    value.copy_(original + epsilon * direction)
                    plus = objective().item()
                    value.copy_(original - epsilon * direction)
                    minus = objective().item()
            finally:
                with torch.no_grad():
                    value.copy_(original)
            numerical = (plus - minus) / (2. * epsilon)
            np.testing.assert_allclose(analytic, numerical, atol=3e-9, rtol=3e-5)

    def test_full_readout_float_input_gradients(self):
        data = dict(self.data, local=torch.cat((self.body, self.context), dim=1))
        variables = [data[name].requires_grad_() for name in
                     ("base", "expert", "prior_b", "prior_e", "relation", "weights", "local")]

        def objective():
            prob = readout(**data)
            return -torch.log(prob[torch.arange(5), torch.tensor([0, 2, 5, 7, 1])]).mean()

        self.assert_directional_gradients(objective, variables)

    def test_gate_gradients_through_full_support_solve(self):
        for mode in ("static", "conditional"):
            for nonzero in (False, True):
                with self.subTest(mode=mode, nonzero=nonzero):
                    gate = Gate(mode).double()
                    if nonzero:
                        with torch.no_grad():
                            for parameter in gate.parameters():
                                parameter.copy_(torch.linspace(-.3, .4, parameter.numel()).reshape_as(parameter))
                    body = self.body.clone().requires_grad_()
                    context = self.context.clone().requires_grad_()
                    gate_context = self.context.flip(0).clone().requires_grad_()

                    def objective():
                        local = gate(body, context, gate_context)
                        prob = readout(**self.data, local=local)
                        return -torch.log(prob[torch.arange(5), torch.tensor([0, 2, 5, 7, 1])]).mean()

                    variables = [*gate.parameters(), body, context]
                    if mode == "conditional" and nonzero:
                        variables.append(gate_context)
                    self.assert_directional_gradients(objective, variables)

    def test_compact_tasks_and_dtype(self):
        selected = torch.cat((self.data["s"], self.data["q"]))
        data = dict(self.data, local=Gate().double()(self.body, self.context).detach())
        compact = {k: v[selected] if k not in ("q", "s", "weights") else v
                   for k, v in data.items()}
        compact["s"] = torch.arange(9)
        compact["q"] = torch.arange(9, 14)
        expected = readout(**compact)
        unused = torch.ones(29, dtype=torch.bool)
        unused[selected] = False
        for key in ("base", "expert", "prior_b", "prior_e", "relation", "local"):
            data[key] = data[key].clone()
            data[key][unused] = float("nan")
        torch.testing.assert_close(readout(**data), expected, rtol=1e-14, atol=1e-14)
        single = {k: v.float() if v.is_floating_point() else v for k, v in compact.items()}
        actual = readout(**single)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
        self.assertEqual(Gate()(self.body.float(), self.context.float()).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
