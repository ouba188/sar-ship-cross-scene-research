"""Synthetic qualification, paired-pixel and matched-exposure regression tests."""
from argparse import Namespace
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from test_two_stage_comparison import make_records
from sar_port.two_stage.cli import (SupplementalInterventions, epoch_batches, protocol,
                                    train_stage1, load_checkpoint)
from sar_port.two_stage.data import prepare_record, collate
from sar_port.two_stage.training import intervention_inputs


def empirical_fixture(root):
    rows = make_records(root, controlled=True)
    donors = [r['id'] for r in rows[:2]]
    queries = [r['id'] for r in rows[2:4]]
    for i, row in enumerate(rows):
        row['leakage_controlled'] = False
        row['leakage_regularization_eligible'] = i in (2, 3)
        if i in (2, 3): row['background_donor_ids'] = donors
    cfg = dict(protocol='development_split', leakage_mode='empirical_regularization',
               donor_ids=donors, supplemental_batches=[queries, queries], seed=17,
               batch_size=4, stage1_epochs=1, allow_incomplete_supervision=False,
               loss=dict(hard_gate_after=0, warmup_steps=0, ramp_steps=1))
    samples = {i: prepare_record(r, root) for i, r in enumerate(rows[:4])}
    return rows, cfg, samples


class EmpiricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_dedicated_donor_schedule_preserves_joint_pixels_and_query_order(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, cfg, samples = empirical_fixture(Path(temp))
            for i in (0, 1): samples[i].images[1] = samples[i].images[0]+.25
            schedule = SupplementalInterventions(samples, rows, range(4), cfg)
            expected = list(epoch_batches(samples, list(range(4)), 4, 17, 0, schedule))
            torch.rand(100); np.random.rand(100)
            self.assertEqual(expected, list(epoch_batches(samples, list(range(4)), 4, 17, 0, schedule)))
            self.assertEqual(expected[-2:], [([2,3], True), ([2,3], True)])
            assignments = []
            for ids, extra in expected:
                if not extra: continue
                batch = collate([samples[i] for i in ids])
                assignments.append(schedule.apply(batch, ids, samples))
                changed, eligible = intervention_inputs(batch)
                self.assertTrue(eligible.all())
                q = batch['background'].bool()
                self.assertTrue(torch.equal(changed[~q], batch['images'][~q]))
                common = q[:,0] & q[:,1]
                torch.testing.assert_close((changed[:,1]-changed[:,0])[common],
                                           torch.full_like(changed[:,0][common], .25))
            self.assertEqual(assignments, [[0,0],[1,1]])
            self.assertEqual(schedule.counts, {2:2,3:2})
            # Next epoch begins at the next position; natural batches never consume it.
            batch = collate([samples[2], samples[3]])
            self.assertEqual(schedule.apply(batch, [2,3], samples), [0,0])

    def test_audit_and_split_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, cfg, samples = empirical_fixture(Path(temp))
            bad = deepcopy(cfg); bad['donor_ids'] += [rows[-1]['id']]
            with self.assertRaisesRegex(ValueError, 'training split'):
                SupplementalInterventions(samples, rows, range(4), bad)
            bad = deepcopy(cfg); bad['donor_ids'] += [rows[2]['id']]
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                SupplementalInterventions(samples, rows, range(4), bad)
            badrows = deepcopy(rows); badrows[2]['leakage_regularization_eligible'] = False
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                SupplementalInterventions(samples, badrows, range(4), cfg)
            samples[0].background.zero_()
            with self.assertRaisesRegex(ValueError, 'jointly trusted Q'):
                SupplementalInterventions(samples, rows, range(4), cfg)

    def test_external_views_reject_unqualified_or_nonlocal_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            rows, cfg, samples = empirical_fixture(Path(temp))
            batch = collate([samples[2], samples[3]])
            schedule = SupplementalInterventions(samples, rows, range(4), cfg)
            schedule.apply(batch, [2,3], samples)
            with patch('sar_port.two_stage.training.intervene_background', side_effect=AssertionError('legacy not used')):
                intervention_inputs(batch)
            for variant in ('outside', 'shape', 'finite', 'qualification', 'missing', 'ineligible'):
                bad = {k: v.clone() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                if variant == 'outside': bad['intervention_images'][0,0,0,31,31] += 1
                if variant == 'shape': bad['intervention_images'] = bad['intervention_images'][:1]
                if variant == 'finite': bad['intervention_images'][0,0,0,0,0] = float('nan')
                if variant == 'qualification': bad['pair_eligible'][0] = False
                if variant == 'missing': del bad['intervention_eligible']
                if variant == 'ineligible': bad['intervention_eligible'][0] = False
                with self.subTest(variant=variant), self.assertRaises(ValueError): intervention_inputs(bad)

    def test_empirical_all_losses_and_baseline_have_identical_exposure_without_control_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows, cfg, _ = empirical_fixture(root)
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(rows), encoding='utf8')
            args = Namespace(manifest=manifest, output=root/'empirical.pt')
            split = protocol(rows, cfg)
            torch.manual_seed(17)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'sar_port.two_stage.training.intervene_background', side_effect=AssertionError('no natural replacement')):
                train_stage1(args, cfg, rows, split, 'cpu')
            saved = load_checkpoint(args.output); report = saved['history'][0]
            self.assertEqual(saved['leakage_mode'], 'empirical_regularization')
            self.assertFalse(saved['incomplete_supervision'])
            self.assertTrue(all(not r['leakage_controlled'] for r in saved['supervision_metadata']))
            torch.testing.assert_close(saved['leakage_prior'], torch.tensor([.5,.5]))
            self.assertEqual(report['controlled_class_counts'], [0,0])
            self.assertEqual(report['regularization_class_counts'], [2,2])
            self.assertEqual(report['losses']['controlled_samples'], 0)
            for loss in ('cls', 'scene', 'invariance', 'leakage', 'discriminator'):
                self.assertGreater(report['losses'][loss], 0, loss)
            cfg['encoder'] = dict(gate_enabled=False); args.output = root/'baseline.pt'
            torch.manual_seed(17)
            with contextlib.redirect_stdout(io.StringIO()): train_stage1(args, cfg, rows, split, 'cpu')
            baseline = load_checkpoint(args.output)['history'][0]
            self.assertEqual(baseline['query_order'], report['query_order'])
            self.assertEqual(baseline['donor_assignments'], report['donor_assignments'])
            self.assertEqual(baseline['losses']['paired_samples'], 0)
            self.assertEqual(baseline['regularization_class_counts'], [0,0])


if __name__ == '__main__': unittest.main()
