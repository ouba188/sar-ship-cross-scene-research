"""Focused response semantics and isolation checks; no research data or fits."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_condition_evidence as runner
from condition_evidence_response import (
    ARMS, design, response_target, clip_prediction, fit_model, predict_model, evaluate,
)
from incidence_transport import matched_pairs, pair_weights
from diagnose_condition_evidence import product_exclusion, accounting, aggregate, comparable_cells


def synthetic():
    rng = np.random.default_rng(1907)
    rows, context, values = [], [], []
    for ship in range(96):
        content, c = rng.normal(0, 8), rng.normal()
        for visit, angle in enumerate(rng.uniform(31, 45, 3)):
            rows.append(dict(obs_id=len(rows), mmsi=ship, class_id=ship % 4,
                port='p', platform='S1A', pass_name='Ascending',
                product=f's{ship}_{visit}', acq_time=f'2025-01-0{visit+1}',
                incidence_deg=angle))
            context.append(c)
            values.append(content + (.8 + .1 * c) * (angle - 38) / 5)
    meta = pd.DataFrame(rows)
    return meta, np.asarray(context), np.asarray(values), np.arange(216), np.arange(216, 288)


class ResponseTests(unittest.TestCase):
    def test_product_exclusion_removes_both_products_and_their_pairs(self):
        meta, context, values, fit, held = synthetic()
        pairs = matched_pairs(meta, fit)
        products = meta.iloc[[0, 4]]['product'].tolist()
        remaining, retained_pairs = product_exclusion(meta, fit, pairs, products)
        self.assertFalse(meta.iloc[remaining]['product'].isin(products).any())
        self.assertFalse(np.isin(retained_pairs, [0, 4]).any())
        self.assertTrue(np.isin(retained_pairs, remaining).all())
        self.assertEqual(len(remaining), len(fit) - 2)

    def test_group_contributions_sum_to_actual_global_gain(self):
        meta, context, values, fit, held = synthetic()
        pairs = matched_pairs(meta, held)
        rng = np.random.default_rng(13)
        errors = {arm: rng.uniform(.1, 2, len(pairs)) for arm in ARMS}
        total = aggregate(meta, pairs, errors)
        rows = pd.DataFrame(accounting(meta, pairs, errors, dict(name='synthetic')))
        for kind in ('class', 'port', 'ship'):
            for arm in ('anchor', 'capacity', 'sham'):
                self.assertAlmostEqual(rows.loc[rows.grouping == kind, 'global_mse_reduction_vs_' + arm].sum(),
                    total['mse_' + arm] - total['mse_condition'])

    def test_comparable_cells_do_not_compare_disjoint_port_class_cells(self):
        meta = pd.DataFrame(dict(port=['A']*6 + ['B']*2,
            class_id=[0]*4 + [1]*4, mmsi=[1, 1, 2, 2, 3, 3, 4, 4],
            incidence_deg=[31, 32, 31, 37, 31, 32, 31, 37]))
        pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        values = np.array([0., 1., 0., 3., 0., 10., 0., 30.])
        rows = comparable_cells(meta, pairs, values, 'synthetic')
        eligible = [r for r in rows if r['comparable']]
        self.assertEqual(len(eligible), 1)
        self.assertEqual((eligible[0]['port'], eligible[0]['class_id']), ('A', 0))
        self.assertEqual(eligible[0]['large_minus_near'], 2.)
        self.assertEqual(eligible[0]['common_ships'], 0)

    def fit_predict(self, mode, arm):
        meta, context, values, fit, held = synthetic()
        train = matched_pairs(meta, fit)
        pairs = matched_pairs(meta, held)
        directed = np.r_[pairs, pairs[:, ::-1]]
        a, b = directed.T
        angle = meta.incidence_deg.to_numpy()
        model = fit_model(meta, fit, train, values, context, values, mode, arm)
        prediction = predict_model(values[a], context[a], angle[a], angle[b], model)
        truth = response_target(values, directed, mode)
        return evaluate(meta, pairs, truth, prediction, model['constant'])

    def test_known_signed_response_generalizes_to_new_synthetic_ships(self):
        condition = self.fit_predict('signed', 'condition')
        capacity = self.fit_predict('signed', 'capacity')
        self.assertGreater(condition['skill_vs_constant'], 95)
        self.assertLess(condition['mse'], .1 * capacity['mse'])

    def test_known_condition_dependent_magnitude_is_learnable(self):
        condition = self.fit_predict('magnitude', 'condition')
        capacity = self.fit_predict('magnitude', 'capacity')
        self.assertGreater(condition['skill_vs_constant'], 85)
        self.assertLess(condition['mse'], .15 * capacity['mse'])

    def test_excluded_observations_do_not_change_any_fitted_arm(self):
        meta, context, values, fit, held = synthetic()
        pairs = matched_pairs(meta, fit)
        changed_meta, changed_values, changed_context = meta.copy(), values.copy(), context.copy()
        changed_values[held], changed_context[held] = 1e9, -1e9
        changed_meta.loc[held, 'class_id'] = 7
        changed_meta.loc[held, 'incidence_deg'] = -999
        for arm in ARMS:
            before = fit_model(meta, fit, pairs, values, context, values, 'signed', arm)
            after = fit_model(changed_meta, fit, pairs, changed_values, changed_context,
                              changed_values, 'signed', arm)
            for key in ('coefficient', 'center', 'scale', 'xcenter', 'xscale', 'constant'):
                np.testing.assert_array_equal(before[key], after[key])

    def test_prediction_has_no_b_evidence_or_truth_input(self):
        self.assertEqual(list(inspect.signature(predict_model).parameters),
                         ['anchor_a', 'context_a', 'angle_a', 'angle_b', 'model'])
        meta, context, values, fit, held = synthetic()
        model = fit_model(meta, fit, matched_pairs(meta, fit), values, context, values,
                          'signed', 'condition')
        a, b = held[0], held[1]
        angle = meta.incidence_deg.to_numpy()
        before = predict_model(values[[a]], context[[a]], angle[[a]], angle[[b]], model)
        values[b], context[b] = 123456, -123456
        meta.loc[b, 'class_id'] = 7
        after = predict_model(values[[a]], context[[a]], angle[[a]], angle[[b]], model)
        np.testing.assert_array_equal(before, after)

    def test_capacity_condition_have_equal_columns_not_equal_functions(self):
        u, v = np.array([-.3, 0., 2.]), np.array([.2, 1., -.4])
        a, b = np.array([32., 35., 39.]), np.array([37., 39., 42.])
        for mode in ('signed', 'magnitude', 'deterioration', 'error'):
            cap = design(u, v, a, b, mode, 'capacity')
            cond = design(u, v, a, b, mode, 'condition')
            self.assertEqual(cap.shape, (3, 6))
            self.assertEqual(cap.shape, cond.shape)
            np.testing.assert_array_equal(cap[:, :2], cond[:, :2])
            np.testing.assert_array_equal(design(u, v, a, b, mode, 'sham'), cond)
            np.testing.assert_array_equal(cap, design(u, v, a + 2, b - 1, mode, 'capacity'))
            self.assertFalse(np.array_equal(cond, design(u, v, a + 2, b - 1, mode, 'condition')))

    def test_magnitude_drop_and_error_are_different_targets(self):
        directed = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
        margin = np.array([1., 3., .5, -.5])
        np.testing.assert_array_equal(response_target(margin, directed, 'signed'), [2, -2, -1, 1])
        np.testing.assert_array_equal(response_target(margin, directed, 'magnitude'), [2, 2, 1, 1])
        np.testing.assert_array_equal(response_target(margin, directed, 'deterioration'), [0, 2, 1, 0])
        error = (margin < 0).astype(float)
        np.testing.assert_array_equal(response_target(error, directed, 'error'), [0, 0, 1, 0])
        np.testing.assert_array_equal(clip_prediction(np.array([-2., .3, 2.]), 'error'), [0, .3, 1])
        np.testing.assert_array_equal(clip_prediction(np.array([-2., .3, 2.]), 'magnitude'), [0, .3, 2])

    def test_deterioration_current_angle_terms_cannot_encode_direction_at_fixed_anchor(self):
        # Document a representational limit, not a demand to alter the model here.
        forward = design([1.], [.2], [32.], [42.], 'deterioration', 'condition')
        reverse = design([1.], [.2], [42.], [32.], 'deterioration', 'condition')
        np.testing.assert_array_equal(forward, reverse)

    def test_fit_evaluate_share_class_ship_pair_weights(self):
        meta = pd.DataFrame(dict(mmsi=[10, 10, 10, 11, 11, 20, 20],
            class_id=[0, 0, 0, 0, 0, 1, 1], incidence_deg=[31, 35, 40, 32, 42, 33, 41]))
        pairs = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [5, 6]])
        np.testing.assert_allclose(pair_weights(meta, pairs), [1/12, 1/12, 1/12, 1/4, 1/2])
        # Distinct ship outcomes expose accidental per-pair rather than per-ship means.
        error = np.array([0., 0., 0., 1., 1., .5, .5])
        model = fit_model(meta, np.arange(7), pairs, np.zeros(7), np.zeros(7),
                          error, 'error', 'anchor')
        self.assertAlmostEqual(model['constant'], .5)
        directed = np.r_[pairs, pairs[:, ::-1]]
        truth = response_target(error, directed, 'error')
        result = evaluate(meta, pairs, truth, np.zeros(10), 0.)
        self.assertAlmostEqual(result['target_mean'], .5)
        self.assertAlmostEqual(result['mse'], .375)

    def test_risk_anchors_and_candidate_pairs_ignore_held_truth(self):
        meta, context, values, fit, held = synthetic()
        rng = np.random.default_rng(11)
        features = rng.normal(size=(len(meta), 6))
        names = np.array([f'feature_{j}' for j in range(6)])
        scores = {31415: rng.normal(size=(len(meta), 8))}
        changed = meta.copy()
        changed.loc[held, 'class_id'] = (changed.loc[held, 'class_id'] + 2) % 8
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'OUT', Path(folder)):
            original = runner.make_tasks(meta, fit, held, features, names, scores)
            original_pairs = pd.read_csv(Path(folder) / 'source_confusion_pairs.csv')
            altered = runner.make_tasks(changed, fit, held, features, names, scores)
            pd.testing.assert_frame_equal(original_pairs, pd.read_csv(Path(folder) / 'source_confusion_pairs.csv'))
            for first, second in zip(original, altered):
                for key in ('name', 'family', 'seed', 'mode'):
                    self.assertEqual(first[key], second[key])
                np.testing.assert_array_equal(first['anchor'], second['anchor'])
                np.testing.assert_array_equal(first['context'], second['context'])
                np.testing.assert_array_equal(first['values'][fit], second['values'][fit])
            risk = [t for t in original if t['family'] == 'risk']
            ordered = np.sort(scores[31415], axis=1)
            for task in risk:
                np.testing.assert_array_equal(task['anchor'], ordered[:, -1] - ordered[:, -2])
                np.testing.assert_array_equal(task['context'], ordered[:, -1])
            changed_scores = {31415: scores[31415].copy()}
            changed_scores[31415][held] = rng.normal(50, 10, (len(held), 8))
            runner.make_tasks(changed, fit, held, features, names, changed_scores)
            pd.testing.assert_frame_equal(original_pairs, pd.read_csv(Path(folder) / 'source_confusion_pairs.csv'))


if __name__ == '__main__':
    unittest.main()
