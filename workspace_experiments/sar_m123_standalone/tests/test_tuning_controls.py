"""CPU regressions for opt-in tuning; no experiment or evaluation labels."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from sar_m123 import train_ops as op
from sar_m123.training_alignment import fit_metric


class TinyMetric(nn.Module):
    def __init__(self, initial=-.05):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(initial))

    def forward(self, x):
        return x*self.value


class TinyMatcher:
    device = 'cpu'

    def local(self, ids, metric):
        sign = torch.tensor(np.where(np.asarray(ids) % 2 == 0, 1., -1.), dtype=torch.float32)
        return metric(torch.stack((sign, -sign), 1))


def tiny_decision(local, ids):
    return local


class TuningControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def run_fit(self, stage, **options):
        rng = np.random.default_rng(17)
        # Rows outside fit/selection deliberately have unusable labels.
        y = np.array([0, 1, 0, 1, -1, 999])
        fit, val = np.array([0, 1]), np.array([2, 3])
        base = rng.normal(size=(6, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch.object(torch.optim, 'AdamW', wraps=torch.optim.AdamW) as optimizer:
                if stage == 'head':
                    x = rng.normal(size=(6, 2, 15)).astype(np.float32)
                    torch.manual_seed(17)
                    result = op.fit_head('tiny', op.JointHead(), x, base, y, fit, val, out, **options)
                    checkpoint, curve = 'tiny.pt', 'curve_tiny.csv'
                elif stage == 'gate':
                    raw = rng.normal(size=(6, 2, 185)).astype(np.float32)
                    dv, dh = (rng.normal(size=(6, 2)).astype(np.float32) for _ in range(2))
                    result = op.fit_gate('frequency', 17, raw, base, dv, dh,
                                         np.ones((6, 2), np.float32), y, fit, val, .5, out, **options)
                    checkpoint, curve = 'frequency.pt', 'curve_frequency.csv'
                else:
                    result = fit_metric(TinyMatcher(), TinyMetric(), tiny_decision,
                                        y, fit, val, out, **options)[:2]
                    checkpoint, curve = 'aligned_metric.pt', 'metric_curve.csv'
                optimizer.assert_called_once()
                kwargs = optimizer.call_args.kwargs
            return result, op.load_checkpoint(out/checkpoint), pd.read_csv(out/curve), kwargs

    def test_keyword_only_controls_and_original_step_defaults(self):
        for function, steps in ((op.fit_head, 200), (op.fit_gate, 100), (fit_metric, 100)):
            with self.subTest(function=function.__name__):
                params = inspect.signature(function).parameters
                self.assertEqual(params['steps'].default, steps)
                self.assertEqual(params['steps'].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name, default in (('lr', .003), ('weight_decay', .01)):
                    self.assertEqual(params[name].kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertEqual(params[name].default, default)
        temperature = inspect.signature(op.Matcher).parameters['temperature']
        self.assertEqual(temperature.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(temperature.default)
        batch_size = inspect.signature(fit_metric).parameters['batch_size']
        self.assertEqual(batch_size.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(batch_size.default, 24)

    def test_implicit_and_explicit_defaults_match_exactly(self):
        for stage in ('head', 'gate', 'metric'):
            with self.subTest(stage=stage):
                implicit = self.run_fit(stage)
                options = dict(batch_size=24) if stage == 'metric' else {}
                explicit = self.run_fit(stage, lr=.003, weight_decay=.01, **options)
                for actual, expected in zip(implicit[0], explicit[0]):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(implicit[1]['selected_step'], explicit[1]['selected_step'])
                for key, value in implicit[1]['state'].items():
                    torch.testing.assert_close(value, explicit[1]['state'][key], rtol=0, atol=0)
                pd.testing.assert_frame_equal(implicit[2], explicit[2], check_exact=True)
                self.assertEqual(implicit[3], dict(lr=.003, weight_decay=.01))
                expected_steps = ([0, 5, 10, 15, 20, 30, 50, 75, 100] if stage == 'metric'
                                  else list(range(0, 201 if stage == 'head' else 101, 5)))
                self.assertEqual(implicit[2].step.tolist(), expected_steps)

    def test_alternate_optimizer_options_are_applied(self):
        for stage in ('head', 'gate', 'metric'):
            with self.subTest(stage=stage):
                baseline = self.run_fit(stage, steps=10)
                for options in (dict(lr=.03), dict(weight_decay=.7)):
                    with self.subTest(options=options):
                        alternate = self.run_fit(stage, steps=10, **options)
                        self.assertEqual(alternate[3], dict(dict(lr=.003, weight_decay=.01), **options))
                        self.assertFalse(np.array_equal(baseline[2].loss.iloc[1:], alternate[2].loss.iloc[1:]))

    def test_metric_batching_preserves_full_objective_gradient_and_step(self):
        results = []
        original_step = torch.optim.AdamW.step
        for options in ({}, dict(batch_size=24), dict(batch_size=128)):
            calls, gradients, updated = [], [], []

            class RecordingMatcher(TinyMatcher):
                def local(self, ids, metric):
                    calls.append((torch.is_grad_enabled(), len(ids)))
                    return super().local(ids, metric)

            def record_step(optimizer, *args, **kwargs):
                parameter = optimizer.param_groups[0]['params'][0]
                gradients.append(parameter.grad.detach().clone())
                result = original_step(optimizer, *args, **kwargs)
                updated.append(parameter.detach().clone())
                return result

            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                with patch.object(torch.optim.AdamW, 'step', record_step):
                    fit_metric(RecordingMatcher(), TinyMetric(), tiny_decision,
                               np.arange(114) % 2, np.arange(53), np.arange(53, 114),
                               out, steps=1, **options)
                loss = pd.read_csv(out/'metric_curve.csv').loss.iloc[-1]
            self.assertEqual(len(gradients), 1)
            self.assertEqual(len(updated), 1)
            results.append((gradients[0], updated[0], loss, calls))
        for index, result in enumerate(results[1:], 1):
            tolerance = 0 if index == 1 else 1e-6
            torch.testing.assert_close(result[0], results[0][0], rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(result[1], results[0][1], rtol=tolerance, atol=tolerance)
            self.assertAlmostEqual(result[2], results[0][2], places=6)
        self.assertEqual(results[0][3], [(False, 48), (False, 13), (True, 24),
                                        (True, 24), (True, 5), (False, 48), (False, 13)])
        self.assertEqual(results[1][3], results[0][3])
        self.assertEqual(results[2][3], [(False, 61), (True, 53), (False, 61)])

    def test_metric_extended_schedule_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            metric = TinyMetric(-127.5)
            selected, gradient, _ = fit_metric(
                TinyMatcher(), metric, tiny_decision, np.array([0, 1, 0, 1]),
                np.array([0, 1]), np.array([2, 3]), out, 128, lr=1., weight_decay=0.)
            curve = pd.read_csv(out/'metric_curve.csv')
            self.assertEqual(curve.step.tolist(), [0, 5, 10, 15, 20, 30, 50, 75, 100, 125, 128])
            self.assertTrue((curve.J.iloc[:-1] == 0).all())
            self.assertEqual(curve.J.iloc[-1], 1.)
            self.assertEqual(selected, 128)
            self.assertGreater(gradient, 0.)
            checkpoint = op.load_checkpoint(out/'aligned_metric.pt')
            self.assertEqual(checkpoint['selected_step'], 128)
            torch.testing.assert_close(metric.value, checkpoint['state']['value'], rtol=0, atol=0)
            self.assertFalse(metric.value.requires_grad)
        for steps, tail in ((103, [100, 103]), (150, [125, 150]), (7, [5, 7])):
            with self.subTest(steps=steps):
                result = self.run_fit('metric', steps=steps)
                self.assertEqual(result[2].step.tolist()[-2:], tail)

    def test_matcher_temperature_default_parity_and_forwarding(self):
        rng = np.random.default_rng(29)
        features = dict(x=rng.normal(size=(3, 2, 20, 32)).astype(np.float32),
                        mass=np.ones((3, 2, 20), np.float32),
                        joint=rng.normal(size=(3, 2, 20, 96)).astype(np.float32))
        refs = np.broadcast_to(np.array([[1], [2]]), (3, 2, 1)).copy()
        scaler = dict(mu=np.zeros(96, np.float32), scale=np.ones(96, np.float32), active=np.ones(96, bool))
        torch.manual_seed(29)
        metric = op.PairMetric()
        outputs = []
        for options in ({}, dict(temperature=None), dict(temperature=.1), dict(temperature=.2)):
            with self.subTest(options=options):
                matcher = op.Matcher(features, refs, scaler, 'cpu', **options)
                with patch.object(op, 'sinkhorn', wraps=op.sinkhorn) as transport:
                    outputs.append(matcher.predict(np.array([0]), metric))
                self.assertEqual(transport.call_count, len(op.tokens.REGIONS))
                expected = {} if options.get('temperature') is None else dict(eps=options['temperature'])
                for call in transport.call_args_list:
                    self.assertEqual(call.kwargs, expected)
                    self.assertEqual(len(call.args), 3)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], outputs[2])
        self.assertFalse(np.array_equal(outputs[0], outputs[3]))


if __name__ == '__main__':
    unittest.main()
