"""Analytic checks for decoder dependence, cohort indexing, and vessel weighting."""

import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from content_condition_swap_diagnose import reconstruction_diagnostics
from relational_factors import FactorNet


class ToyEncoder(nn.Module):
    def __init__(self, branch, mode):
        super().__init__()
        self.branch, self.mode = branch, mode

    def forward(self, x):
        width = 128 if self.branch == 'stable' else 32
        result = x.new_zeros((len(x), width))
        if self.branch == 'stable':
            result[:, 0] = x[:, 0]
            if self.mode == 'ignore_condition':
                result[:, 1] = x[:, 1]
        elif self.mode == 'ignore_stable':
            result[:, :2] = x
        else:
            result[:, 0] = x[:, 1]
        return result


class ToyDecoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.gain = nn.Parameter(torch.tensor(1.))
        self.last_input = None

    def forward(self, z):
        self.last_input = z.detach().cpu().numpy().copy()
        if self.mode == 'ignore_condition':
            out = z[:, :2]
        elif self.mode == 'ignore_stable':
            out = z[:, 128:130]
        elif self.mode == 'interaction':
            out = torch.stack([z[:, 0], z[:, 0]*z[:, 128]], dim=1)
        elif self.mode == 'biased':
            out = torch.stack([2*z[:, 0], z[:, 128]], dim=1)
        else:
            out = torch.stack([z[:, 0], z[:, 128]], dim=1)
        return out*self.gain


class ToyNet(nn.Module):
    def __init__(self, mode='separated'):
        super().__init__()
        self.stable = ToyEncoder('stable', mode)
        self.condition = ToyEncoder('condition', mode)
        self.decoder = ToyDecoder(mode)


def cohort():
    x = np.array([[0, 0], [2, 2], [10, 1], [10, 3], [20, 5], [20, 9]], np.float32)
    meta = pd.DataFrame(dict(mmsi=[90, 91, 1, 1, 2, 2], class_id=[0]*6,
                             product=['fit', 'fit', 'a', 'b', 'a', 'b']))
    return x, meta, np.array([0, 1]), np.array([2, 3, 4, 5])


class ReconstructionTests(unittest.TestCase):
    def test_useful_condition_and_fit_only_mean_replacements(self):
        result = reconstruction_diagnostics(ToyNet(), *cohort())
        self.assertEqual(result['self_mse'], 0.)
        self.assertEqual(result['swap_mse'], 0.)
        self.assertEqual(result['condition_delta_mse'], 0.)
        self.assertEqual(result['condition_delta_zero_mse'], 5.)
        self.assertEqual(result['condition_delta_skill'], 100.)
        self.assertEqual(result['mean_condition_mse'], 10.5)
        self.assertEqual(result['mean_stable_mse'], 110.5)
        self.assertEqual(result['mean_condition_output_change_mse'], 10.5)
        self.assertEqual(result['mean_stable_output_change_mse'], 110.5)

    def test_ignored_condition_has_zero_skill_despite_perfect_self(self):
        result = reconstruction_diagnostics(ToyNet('ignore_condition'), *cohort())
        self.assertEqual(result['self_mse'], 0.)
        self.assertEqual(result['swap_mse'], 5.)
        self.assertEqual(result['condition_delta_mse'], 5.)
        self.assertEqual(result['condition_delta_skill'], 0.)
        self.assertEqual(result['mean_condition_mse'], 0.)
        self.assertEqual(result['mean_condition_output_change_mse'], 0.)
        self.assertGreater(result['mean_stable_output_change_mse'], 0.)

    def test_ignored_stable_is_visible_despite_perfect_swap_and_delta(self):
        result = reconstruction_diagnostics(ToyNet('ignore_stable'), *cohort())
        self.assertEqual(result['self_mse'], 0.)
        self.assertEqual(result['swap_mse'], 0.)
        self.assertEqual(result['condition_delta_skill'], 100.)
        self.assertEqual(result['mean_stable_mse'], 0.)
        self.assertEqual(result['mean_stable_output_change_mse'], 0.)
        self.assertGreater(result['mean_condition_output_change_mse'], 0.)

    def test_both_nonlinear_directions_are_evaluated_independently(self):
        x = np.array([[0, 0], [2, 3], [4, 5]], np.float32)
        meta = pd.DataFrame(dict(mmsi=[0, 1, 1], class_id=[0]*3, product=['f', 'a', 'b']))
        result = reconstruction_diagnostics(ToyNet('interaction'), x, meta, [0], [1, 2])
        self.assertEqual(result['swap_forward_mse'], 14.5)
        self.assertEqual(result['swap_reverse_mse'], 42.5)
        self.assertEqual(result['swap_mse'], 28.5)
        self.assertEqual(result['condition_delta_mse'], 12.)
        self.assertEqual(result['condition_delta_zero_mse'], 4.)
        self.assertEqual(result['condition_delta_skill'], -200.)

    def test_ships_have_equal_weight_despite_unequal_pair_counts(self):
        x = np.array([[0, 0], [1, 0], [1, 2], [1, 4], [3, 0], [3, 10]], np.float32)
        meta = pd.DataFrame(dict(mmsi=[0, 1, 1, 1, 2, 2], class_id=[0]*6,
                                 product=['f', 'a', 'b', 'c', 'a', 'b']))
        result = reconstruction_diagnostics(ToyNet('ignore_condition'), x, meta, [0], np.arange(1, 6))
        # Ship 1: mean pair error (2+8+2)/3=4; ship 2: 50.
        self.assertEqual(result['same_ship_pairs'], 4)
        self.assertEqual(result['condition_delta_zero_mse'], 27.)
        self.assertEqual(result['condition_delta_mse'], 27.)
        self.assertEqual(result['swap_mse'], 27.)
        biased = reconstruction_diagnostics(ToyNet('biased'), x, meta, [0], np.arange(1, 6))
        self.assertEqual(biased['self_mse'], 2.5)
        self.assertEqual(biased['swap_mse'], 2.5)
        self.assertEqual(biased['mean_stable_mse'], 2.5)
        self.assertAlmostEqual(biased['mean_condition_mse'], 50/3)

    def test_shuffled_global_positions_and_test_only_control_donors(self):
        x = np.array([[777, 888], [10, 1], [0, 0], [20, 5], [500, 700],
                      [10, 3], [2, 2], [20, 9]], np.float32)
        meta = pd.DataFrame(dict(mmsi=[1, 1, 90, 2, 1, 1, 91, 2], class_id=[0]*8,
                                 product=['c', 'a', 'f', 'a', 'd', 'b', 'f', 'b']),
                            index=np.arange(100, 108))
        model = ToyNet()
        result = reconstruction_diagnostics(model, x, meta, [6, 2], [7, 1, 5, 3])
        self.assertEqual(result['same_ship_pairs'], 2)
        self.assertEqual(result['directed_pairs'], 4)
        self.assertEqual(result['condition_delta_zero_mse'], 5.)
        self.assertEqual(result['mean_condition_mse'], 10.5)
        self.assertEqual(result['mean_stable_mse'], 110.5)
        self.assertEqual(result['control_directed_pairs'], 4)
        self.assertEqual(result['control_direction_coverage_fraction'], 1.)
        # Last decoder call is the donor control. Each donor is a different TEST ship.
        control = model.decoder.last_input
        for row in control:
            self.assertIn(row[128], [5., 9.] if row[0] == 10. else [1., 3.])
        self.assertGreater(result['same_class_other_ship_delta_mse'], 0.)
        self.assertEqual(result['control_matched_condition_delta_mse'], 0.)

    def test_partial_control_coverage_has_matched_denominator(self):
        x, meta, fit, test = cohort()
        x = np.concatenate([x, np.array([[30, 0], [30, 20]], np.float32)])
        meta = pd.concat([meta, pd.DataFrame(dict(mmsi=[3, 3], class_id=[1, 1], product=['a', 'b']))],
                         ignore_index=True)
        result = reconstruction_diagnostics(ToyNet(), x, meta, fit, np.r_[test, 6, 7])
        self.assertEqual(result['paired_ships'], 3)
        self.assertEqual(result['control_ships'], 2)
        self.assertEqual(result['control_directed_pairs'], 4)
        self.assertEqual(result['control_direction_coverage_fraction'], 2/3)
        self.assertEqual(result['control_ship_coverage_fraction'], 2/3)
        self.assertEqual(result['same_class_other_ship_delta_zero_mse'], 5.)
        self.assertEqual(result['condition_delta_zero_mse'], 70.)

    def test_unavailable_controls_do_not_borrow_fit_ships_or_other_classes(self):
        x, meta, fit, test = cohort()
        meta.loc[[4, 5], 'class_id'] = 1
        result = reconstruction_diagnostics(ToyNet(), x, meta, fit, test)
        self.assertEqual(result['control_directed_pairs'], 0)
        self.assertEqual(result['control_ships'], 0)
        self.assertTrue(np.isnan(result['same_class_other_ship_delta_skill']))
        self.assertTrue(np.isnan(result['control_matched_swap_mse']))

    def test_no_cross_product_pairs_and_zero_delta_are_explicit(self):
        x, meta, fit, test = cohort()
        meta.loc[test, 'product'] = 'a'
        result = reconstruction_diagnostics(ToyNet(), x, meta, fit, test)
        self.assertEqual(result['same_ship_pairs'], 0)
        self.assertEqual(result['pair_ship_coverage_fraction'], 0.)
        self.assertTrue(np.isnan(result['self_mse']))
        self.assertTrue(np.isnan(result['condition_delta_skill']))
        empty = reconstruction_diagnostics(ToyNet(), x, meta, fit, [])
        self.assertEqual(empty['test_ships'], 0)
        meta.loc[[3, 5], 'product'] = 'b'
        x[[3, 5]] = x[[2, 4]]
        zero = reconstruction_diagnostics(ToyNet(), x, meta, fit, test)
        self.assertEqual(zero['condition_delta_zero_mse'], 0.)
        self.assertEqual(zero['condition_delta_mse'], 0.)
        self.assertTrue(np.isnan(zero['condition_delta_skill']))

    def test_eval_no_grad_batched_and_original_model_state_preserved(self):
        x = np.column_stack([np.ones(318), np.arange(318)]).astype(np.float32)
        meta = pd.DataFrame(dict(mmsi=[0]*300+[1]*18, class_id=[0]*318,
                                 product=[str(i) for i in range(318)]))
        model = ToyNet().train()
        model.condition.eval()
        model.decoder.gain.grad = torch.tensor(7.)
        modes = [module.training for module in model.modules()]
        state = {key: value.clone() for key, value in model.state_dict().items()}
        calls = []
        def record(module, args, output):
            calls.append((module.training, torch.is_grad_enabled(), len(args[0])))
        hooks = [module.register_forward_hook(record) for module in (model.stable, model.condition, model.decoder)]
        try:
            result = reconstruction_diagnostics(model, x, meta, np.arange(300), np.arange(300, 318))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(result['directed_pairs'], 306)
        self.assertEqual(result['condition_delta_skill'], 100.)
        self.assertTrue(all(not training and not grad and count <= 256 for training, grad, count in calls))
        self.assertTrue(any(count == 256 for _, _, count in calls))
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(model.decoder.gain.requires_grad)
        self.assertEqual(float(model.decoder.gain.grad), 7.)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_real_factor_net_feature_contract(self):
        _, meta, fit, test = cohort()
        x = np.random.default_rng(19).normal(size=(6, 512)).astype(np.float32)
        model = FactorNet().eval()
        with torch.no_grad():
            expected = float((model(torch.from_numpy(x[test]))[-1]-torch.from_numpy(x[test])).square().mean())
        result = reconstruction_diagnostics(model, x, meta, fit, test)
        self.assertAlmostEqual(result['self_mse'], expected, places=6)
        self.assertTrue(np.isfinite(result['condition_delta_skill']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
