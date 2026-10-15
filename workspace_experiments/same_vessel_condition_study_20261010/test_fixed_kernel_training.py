import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_explicit_candidate as training


class TinyCandidate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(5, 3)
        self.head = torch.nn.Linear(3, 8)
        self.calls = []

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        prior = self.head(z)
        self.calls.append((z.detach().clone(), prior.detach().clone()))
        return z, prior


class FixedKernelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        generator = torch.Generator().manual_seed(43)
        rng = np.random.default_rng(47)
        self.all_x = torch.randn(512, 5, generator=generator)
        self.all_y = torch.arange(512) % 8
        order = rng.permutation(512)
        self.x = self.all_x[order[:192]]
        self.y = self.all_y[order[:192]]
        self.roles = dict(val_new_ship=order[192:352], val_new_time=order[352:])
        self.validation = {}
        for role in self.roles:
            indices = rng.permutation(160)
            self.validation[role] = [(indices[:64], indices[64:])]
        self.episodes = []
        for _ in range(2):
            indices = rng.permutation(192)
            self.episodes.append((indices[:64], indices[64:160]))
        self.kernel_fit = torch.randn(192, 3, generator=generator).requires_grad_()
        self.kernel_all = torch.randn(512, 3, generator=generator).requires_grad_()
        self.original_fit = self.kernel_fit.detach().clone()
        self.original_all = self.kernel_all.detach().clone()
        self.target = dict(
            eligible=torch.ones(192, dtype=torch.bool),
            lower=torch.zeros(192, 8),
            upper=torch.ones(192, 8),
            mask=torch.ones(192, 8, dtype=torch.bool),
        )
        anchors = np.arange(32)
        self.graph = dict(
            anchor=anchors,
            donor=self.x[anchors] + .25,
            probability=np.full(32, 1 / 32),
            reliability=torch.ones(32),
        )

    def run_training(self, fixed=False, explicit_none=False, gradients=False):
        with torch.random.fork_rng():
            torch.manual_seed(53)
            model = TinyCandidate()
        recorded = dict(phi=[], heads=[], scores=[], metrics=[], gradients=[])
        real_phi = training.phi
        real_head = training.residual_head
        real_mse = training.balanced_mse
        real_metrics = training.metrics

        def observe_phi(z):
            design = real_phi(z)
            recorded['phi'].append((z.detach().clone(), z.requires_grad, design))
            return design

        def observe_head(design, prior, labels, **kwargs):
            result = real_head(design, prior, labels, **kwargs)
            recorded['heads'].append((design, prior, labels.clone(), result[0], kwargs))
            return result

        def observe_mse(scores, labels):
            loss = real_mse(scores, labels)
            recorded['scores'].append(scores.detach().clone())
            if gradients:
                # Differentiate only the actual episode loss through its support prior.
                # Supervised loss and the direct query-prior path cannot satisfy this.
                prior = recorded['heads'][-1][1]
                support_grad = torch.autograd.grad(
                    loss, prior, retain_graph=True, allow_unused=True,
                )[0]
                prior_grad = None
                if support_grad is not None:
                    prior_grad = torch.autograd.grad(
                        prior, model.head.weight, grad_outputs=support_grad,
                        retain_graph=True, allow_unused=True,
                    )[0]
                recorded['gradients'].append((support_grad, prior_grad))
            return loss

        def observe_metrics(prediction, truth):
            recorded['metrics'].append((prediction.copy(), truth.copy()))
            return real_metrics(prediction, truth)

        kwargs = {}
        if fixed:
            kwargs.update(kernel_fit=self.kernel_fit, kernel_all=self.kernel_all)
        elif explicit_none:
            kwargs.update(kernel_fit=None, kernel_all=None)
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(training, 'STEPS', 2))
            stack.enter_context(patch.object(training, 'phi', side_effect=observe_phi))
            stack.enter_context(patch.object(training, 'residual_head', side_effect=observe_head))
            stack.enter_context(patch.object(training, 'balanced_mse', side_effect=observe_mse))
            stack.enter_context(patch.object(training, 'metrics', side_effect=observe_metrics))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            _, trace, choice = training.train(
                model, self.x, self.y, torch.zeros(192, 8),
                torch.ones(192, dtype=torch.bool), self.target,
                np.full(192, 1 / 192), self.graph, self.episodes,
                self.validation, self.all_x, self.all_y, self.roles,
                59, 'task', 1., output_dir=Path(directory), **kwargs,
            )
            checkpoint = torch.load(
                Path(directory) / 'task_59.pt', map_location='cpu', weights_only=True,
            )
        recorded.update(model=model, trace=trace, choice=choice, checkpoint=checkpoint)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(recorded['phi']), 6)
        self.assertEqual(len(recorded['heads']), 4)
        self.assertEqual(len(recorded['scores']), 2)
        return recorded

    def assert_design(self, design, kernel):
        torch.testing.assert_close(design[:, :-1], kernel)
        torch.testing.assert_close(design[:, -1], torch.ones(len(kernel)))

    def test_training_uses_fixed_support_and_query_kernels_with_learned_priors(self):
        result = self.run_training(fixed=True)
        for step, (support, query) in enumerate(self.episodes):
            support_phi = result['phi'][2 + 2 * step]
            query_phi = result['phi'][3 + 2 * step]
            torch.testing.assert_close(support_phi[0], self.original_fit[support])
            torch.testing.assert_close(query_phi[0], self.original_fit[query])
            self.assertFalse(support_phi[1])
            self.assertFalse(query_phi[1])
            design, prior, labels, beta, options = result['heads'][2 + step]
            self.assert_design(design, self.original_fit[support])
            learned_prior = result['model'].calls[2 + step][1]
            torch.testing.assert_close(prior, learned_prior[128:192])
            torch.testing.assert_close(labels, self.y[support])
            self.assertTrue(options['balanced'])
            torch.testing.assert_close(
                result['scores'][step],
                learned_prior[192:288] + query_phi[2].detach() @ beta.detach(),
            )

    def test_validation_uses_global_fixed_kernel_at_role_indices(self):
        result = self.run_training(fixed=True)
        for index, (role, ids) in enumerate(self.roles.items()):
            support, query = self.validation[role][0]
            kernel = self.original_all[ids]
            torch.testing.assert_close(result['phi'][index][0], kernel)
            design, prior, labels, beta, options = result['heads'][index]
            self.assert_design(design, kernel[support])
            learned_prior = result['model'].calls[index][1]
            torch.testing.assert_close(prior, learned_prior[support])
            torch.testing.assert_close(labels, self.all_y[ids][support])
            self.assertTrue(options['balanced'])
            query_design = result['phi'][index][2][query]
            expected = (learned_prior[query] + query_design @ beta).argmax(1).numpy()
            predicted, truth = result['metrics'][2 * index + 1]
            np.testing.assert_array_equal(predicted, expected)
            np.testing.assert_array_equal(truth, self.all_y[ids][query].numpy())

    def test_episode_gradient_reaches_support_prior_but_not_fixed_kernels(self):
        result = self.run_training(fixed=True, gradients=True)
        self.assertEqual(len(result['gradients']), 2)
        for support_grad, prior_grad in result['gradients']:
            for grad in (support_grad, prior_grad):
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.norm()), 0.)
        self.assertIsNone(self.kernel_fit.grad)
        self.assertIsNone(self.kernel_all.grad)
        torch.testing.assert_close(self.kernel_fit, self.original_fit, rtol=0, atol=0)
        torch.testing.assert_close(self.kernel_all, self.original_all, rtol=0, atol=0)

    def test_default_uses_learned_embeddings_and_matches_explicit_none(self):
        result = self.run_training()
        for index in range(2):
            learned_z, _ = result['model'].calls[index]
            torch.testing.assert_close(result['phi'][index][0], learned_z)
        for step in range(2):
            learned_z, _ = result['model'].calls[2 + step]
            support_phi, query_phi = result['phi'][2 + 2 * step:4 + 2 * step]
            torch.testing.assert_close(support_phi[0], learned_z[128:192])
            torch.testing.assert_close(query_phi[0], learned_z[192:288])
            self.assertTrue(support_phi[1])
            self.assertTrue(query_phi[1])
        explicit = self.run_training(explicit_none=True)
        self.assertEqual(result['trace'], explicit['trace'])
        self.assertEqual(result['choice'], explicit['choice'])
        for snapshot in ('selected', 'last'):
            for name, value in result['checkpoint'][snapshot].items():
                torch.testing.assert_close(
                    value, explicit['checkpoint'][snapshot][name], rtol=0, atol=0,
                )


if __name__ == '__main__':
    unittest.main()
