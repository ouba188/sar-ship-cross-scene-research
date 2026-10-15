"""Small synthetic CPU tests; no real query labels and no CUDA execution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]/'oof_reliability_check'))
from strong_single_r50_task import support_scores
from strong_single_r50_task_diagnose import (
    ARMS, MODES, PATHS, check_diagonals, cross_predictions, pair_flow, run,
)
from unified_head_core import coefficients


class DiagnosisTests(unittest.TestCase):
    def arrays(self):
        rng = np.random.default_rng(9)
        phis = [torch.tensor(rng.normal(size=(90, 13)), dtype=torch.float32) for _ in range(2)]
        weights = [torch.tensor(rng.normal(size=(8, 13)), dtype=torch.float32) for _ in range(2)]
        supports = np.stack([rng.choice(90, 64, replace=False) for _ in range(5)])
        labels = np.tile(np.arange(64) % 8, (5, 1))
        labels[:, :16] = 0
        return phis, weights, supports, labels

    def test_four_cells_agree_with_independent_kernel_form(self):
        phis, weights, supports, labels = self.arrays()
        predictions = cross_predictions(phis, weights, supports, labels)
        prior = [F.linear(x, w).numpy().astype(float) for x, w in zip(phis, weights)]
        for draw, support in enumerate(supports):
            for index, (pi, ki) in enumerate(PATHS):
                x = phis[ki].numpy().astype(float)
                kernel = x@x[support].T
                alpha = coefficients(kernel[support], prior[pi][support], labels[draw], 8.)
                expected = (prior[pi]+kernel@alpha).argmax(1)
                np.testing.assert_array_equal(predictions[draw, index], expected)

    def test_replay_failure_receipt_preserves_original(self):
        phis, weights, supports, labels = self.arrays()
        predictions = cross_predictions(phis, weights, supports, labels)
        saved = dict(arms=np.array(ARMS), predictions=predictions[:, [0, 3]].copy())
        saved['predictions'][0, 0, 2] = (saved['predictions'][0, 0, 2]+1) % 8
        original = saved['predictions'].copy()
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder)/'replay.json'
            with self.assertRaises(AssertionError):
                check_diagonals(predictions, saved, receipt)
            status = json.loads(receipt.read_text())
            self.assertFalse(status['exact'])
            self.assertEqual(status['arms'][ARMS[0]]['mismatches_per_draw'], [1, 0, 0, 0, 0])
            np.testing.assert_array_equal(saved['predictions'], original)

    def test_unbalanced_class_rescue_accounting(self):
        truth = np.array([0, 0, 0, 0, 1, 1])
        reference = np.array([0, 0, 0, 1, 0, 0])
        candidate = np.array([1, 1, 0, 1, 1, 0])
        flow, classes = pair_flow(truth, candidate, reference)
        self.assertEqual((flow['rescues'], flow['harms']), (1, 2))
        self.assertAlmostEqual(flow['net_oa_pp'], -100/6)
        self.assertAlmostEqual(flow['net_ba_pp'], 0.)
        self.assertEqual([x['net_sign'] for x in classes], [-1, 1])

    def test_three_seed_report_and_exact_original_paths(self):
        phis, weights, supports, labels = self.arrays()
        masks = np.ones((5, 90), bool)
        for draw, support in enumerate(supports):
            masks[draw, support] = False
        data = dict(source_rows=np.arange(100), rows=np.arange(90), ids=np.arange(90).astype(str),
                    support=supports, support_labels=labels, eval_masks=masks)
        seeds = [101, 102, 103]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = np.empty((5, 2, 90), np.int8)
            for index in range(2):
                prior = F.linear(phis[index], weights[index])
                for draw, support in enumerate(supports):
                    original[draw, index] = support_scores(phis[index][support], torch.tensor(labels[draw]), phis[index],
                        support_prior=prior[support], query_prior=prior).argmax(1).numpy()
            for seed in seeds:
                run_folder = out/f'seed_{seed}'
                run_folder.mkdir()
                (run_folder/'COMPLETE.json').write_text('{}')
                np.savez(run_folder/'predictions.npz', predictions=original, arms=np.array(ARMS), **data)
                torch.save(dict(source_rows=data['source_rows'], frozen_prior={'weight': weights[0]},
                                prior={'weight': weights[1]}), run_folder/'final_tail.pt')
                for index, arm in enumerate(ARMS):
                    np.save(run_folder/f'{arm}_phi.npy', phis[index].numpy())
            reference = out/'reference.npz'
            np.savez(reference, predictions=original[:, 0:1], modes=np.array(['dual_refit_late']), **data)
            manifest = out/'target.csv'
            pd.DataFrame(dict(sample_id=data['ids'], class_id=np.arange(90) % 8)).to_csv(manifest, index=False)
            destination = run(out, seeds, reference, manifest)
            summary = pd.read_csv(destination/'neural_summary.csv')
            self.assertTrue((summary.neural_seeds == 3).all())
            np.testing.assert_allclose(summary.ba_seed_sd, 0.)
            self.assertEqual(len(pd.read_csv(destination/'metrics_by_seed_draw.csv')), 60)
            self.assertEqual(len(pd.read_csv(destination/'fixed_dual_by_draw.csv')), 5)
            self.assertTrue(json.loads((destination/'COMPLETE.json').read_text())['diagonal_replay_exact'])
            self.assertEqual(len(pd.read_csv(destination/'support_draw_means.csv')), 20)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
