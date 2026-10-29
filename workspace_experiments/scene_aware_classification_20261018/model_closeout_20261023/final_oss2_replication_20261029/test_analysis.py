"""Synthetic CPU checks only. Fixtures stay below this task's analysis/ directory."""
import importlib.abc
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'sar_m123'):
            raise AssertionError(f'CPU analysis attempted prohibited import: {fullname}')


sys.meta_path.insert(0, NoTorch())
import numpy as np
import analysis as a


def make_rows():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    passes = ['A', 'A', 'B', 'C', 'C', 'D', 'E', 'E']
    return [dict(id=str(i), label=y, class_name=a.CLASSES[y], original_class_id=a.CLASS_INDICES[y],
        mmsi=str(100+i), port='opensarship2', pass_group=passes[i],
        analysis_available='vv_only' if i == 4 else 'vh_only' if i == 5 else 'both')
        for i, y in enumerate(labels)]


def make_scores():
    old, new = np.full((8, 4), -2.), np.full((8, 4), -2.)
    old[np.arange(8), [1, 1, 2, 0, 0, 2, 0, 3]] = 2.
    new[np.arange(8), [0, 2, 2, 0, 1, 1, 3, 3]] = 2.
    return {(c, s): (new.copy() if c == 'full_adapt' else old.copy()) for c in a.CONDITIONS for s in a.SEEDS}


def fixture_gate(root):
    predictions, training, evaluation = a.input_paths(root)
    for path in predictions.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in (*a.COUNTS, 'references'):
        path = root/'manifests'/f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        a.base.dump(path, [])
    a.base.dump(root/'gpu_checks.json', dict(status='passed', class_indices=list(a.CLASS_INDICES)))
    for (condition, seed), path in training.items():
        arm, stage = condition.split('_')
        epochs, batch_size, steps = a.BUDGET[stage]
        path.parent.mkdir(parents=True, exist_ok=True)
        a.base.dump(path, dict(arm=arm, stage=stage, seed=seed, steps=steps, completed_epochs=epochs,
            batch_size=batch_size, source_count=478 if stage == 'fit' else 241, reference_count=32,
            classes=list(a.CLASSES), class_indices=list(a.CLASS_INDICES), validation_selection=False,
            frozen_unchanged=True, coefficient=a.COEFFICIENT if arm == 'full' else 0.,
            save_replay_error=0., native_cache_error=0.))
    for (condition, seed), path in evaluation.items():
        arm, stage = condition.split('_')
        path.parent.mkdir(parents=True, exist_ok=True)
        a.base.dump(path, dict(arm=arm, stage=stage, seed=seed, primary='evaluation378',
            checkpoint_selection=False, native_cache_error=0.))
    return predictions, training, evaluation


class AnalysisTests(unittest.TestCase):
    def temporary(self):
        folder = a.HERE/'analysis'
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='synthetic_', dir=folder)

    def test_ready_and_receipts_gate_before_any_prediction_read(self):
        with patch.object(a, 'require_receipts') as gate, patch.object(a.np, 'load') as loader:
            with self.assertRaisesRegex(PermissionError, 'READY'):
                a.load_inputs()
            gate.assert_not_called()
            loader.assert_not_called()
        with self.temporary() as directory:
            root = Path(directory)
            with patch.object(a.np, 'load') as loader:
                with self.assertRaisesRegex(ValueError, 'Incomplete inputs'):
                    a.load_inputs(root, ready=True)
                loader.assert_not_called()
            predictions, training, evaluation = fixture_gate(root)
            self.assertEqual([len(x) for x in (predictions, training, evaluation)], [12, 12, 12])
            self.assertEqual(sum(key[0].endswith('adapt') for key in evaluation), 6)
            with patch.object(a.np, 'load') as loader:
                a.require_receipts(root)
                loader.assert_not_called()
                path = training['full_adapt', 17]
                record = a.base.read_json(path)
                record['steps'] = 54
                a.base.dump(path, record)
                with self.assertRaisesRegex(ValueError, '96-step'):
                    a.load_inputs(root, ready=True)
                loader.assert_not_called()
            self.assertNotIn('torch', sys.modules)

    def test_npz_four_class_mapping_id_join_and_rejections(self):
        rows = make_rows()
        score = make_scores()['full_adapt', 17]
        with self.temporary() as directory:
            path = Path(directory)/'synthetic.npz'
            order = np.arange(8)[::-1]
            def save(values=score[order], labels=None, classes=a.CLASSES, ids=None):
                np.savez(path, score=values, y=np.array([rows[i]['label'] for i in order]) if labels is None else labels,
                         ids=np.array([str(i) for i in order]) if ids is None else ids, classes=np.array(classes))
            save()
            np.testing.assert_array_equal(a.load_score(path, rows), score)
            save(classes=tuple(reversed(a.CLASSES)))
            with self.assertRaisesRegex(ValueError, 'names/order'):
                a.load_score(path, rows)
            save(values=np.zeros((8, 8)))
            with self.assertRaisesRegex(ValueError, 'logits'):
                a.load_score(path, rows)
            save(labels=np.zeros(8, dtype=int))
            with self.assertRaisesRegex(ValueError, 'labels disagree'):
                a.load_score(path, rows)
            save(ids=np.array(['0']*8))
            with self.assertRaisesRegex(ValueError, 'IDs'):
                a.load_score(path, rows)
            save(values=np.full((8, 4), np.nan))
            with self.assertRaisesRegex(ValueError, 'finite'):
                a.load_score(path, rows)

    def test_four_class_brier_present_class_ba_and_order_sd(self):
        p = np.array([[.7, .1, .1, .1], [.1, .6, .2, .1], [.1, .7, .1, .1]])
        result = a.metrics(np.log(p), np.array([0, 0, 1]))
        self.assertAlmostEqual(result['OA'], 200/3)
        self.assertAlmostEqual(result['BA'], 75.)
        self.assertAlmostEqual(result['NLL'], -(2*np.log(.7)+np.log(.1))/3)
        self.assertAlmostEqual(result['Brier'], (.12+1.22+.12)/3)
        self.assertEqual(a.moments([1, 2, 3])['sd'], 1)
        self.assertIsNone(a.metrics(np.empty((0, 4)), np.array([], int))['BA'])
        with self.assertRaisesRegex(ValueError, 'N,4'):
            a.per_sample(np.zeros((2, 8)), np.array([0, 1]))

    def test_paired_events_directions_groups_and_before_after(self):
        rows, scores = make_rows(), make_scores()
        events = a.pair_events(rows, scores, 'plain_adapt', 'full_adapt')
        self.assertEqual([e['state'] for e in events[:4]], ['rescued', 'harmed', 'CC', 'WW'])
        self.assertEqual(events[0]['class_name'], 'bulk_carrier')
        self.assertEqual(events[1]['class_name'], 'general_cargo')
        self.assertEqual(events[0]['plain_fit_prediction'], 1)
        reverse = a.pair_events(rows, scores, 'full_adapt', 'plain_adapt')
        for forward, backward in zip(events, reverse):
            self.assertEqual(forward['fixed_rival'], backward['fixed_rival'])
            self.assertAlmostEqual(forward['delta_fixed_margin'], -backward['delta_fixed_margin'])
        results, tables, _ = a.summarize(rows, scores, repeats=20)
        self.assertEqual(set(results['comparisons']), set(a.COMPARISONS))
        primary = results['comparisons'][a.PRIMARY]
        self.assertEqual(primary['transitions']['rescued']['seed_events'], 6)
        self.assertEqual(primary['transitions']['rescued']['unique_ids'], 2)
        self.assertEqual(primary['pass_bootstrap']['pass_count'], 5)
        self.assertEqual({r['group'] for r in tables['groups'] if r['dimension'] == 'label'}, {'0', '1', '2', '3'})
        self.assertEqual(results['comparisons']['plain_adapt_minus_plain_fit']['overall']['delta_OA_mean'], 0.)

    def test_five_pass_bootstrap_matches_explicit_cluster_draws(self):
        rows = make_rows()
        y = np.array([r['label'] for r in rows])
        rng = np.random.default_rng(19)
        scores = {(c, s): rng.normal(size=(8, 4)) for c in a.CONDITIONS for s in a.SEEDS}
        result = a.pass_bootstrap(rows, scores, 'plain_adapt', 'full_adapt', repeats=80, seed=81)
        groups = [np.flatnonzero(np.array([r['pass_group'] for r in rows]) == name) for name in 'ABCDE']
        rng, brute, missing = np.random.default_rng(81), [], 0
        for _ in range(80):
            indices = np.concatenate([groups[i] for i in rng.integers(0, 5, 5)])
            missing += int(len(set(y[indices])) < 4)
            values = []
            for seed in a.SEEDS:
                old, new = [a.metrics(scores[c, seed][indices], y[indices]) for c in ('plain_adapt', 'full_adapt')]
                values.append([new[m]-old[m] for m in a.METRICS])
            brute.append(np.mean(values, axis=0))
        expected = np.quantile(brute, [.025, .975], axis=0)
        for i, metric in enumerate(a.METRICS):
            self.assertAlmostEqual(result['metrics'][metric]['low'], expected[0, i])
            self.assertAlmostEqual(result['metrics'][metric]['high'], expected[1, i])
        self.assertEqual(result['missing_class_replicates'], missing)
        self.assertGreater(missing, 0)

    def test_case_selection_both_missing_polarizations_and_reference_exclusion(self):
        rows, scores = make_rows(), make_scores()
        events = a.pair_events(rows, scores, 'plain_adapt', 'full_adapt')
        chosen = a.select_cases(events)
        self.assertEqual(len({e['id'] for e in chosen}), 8)
        self.assertTrue(set(a.STATES).issubset({e['selection_category'] for e in chosen}))
        self.assertTrue({'vv_only', 'vh_only'}.issubset({e['available'] for e in chosen}))
        refs = [dict(id=f'r{i}', label=i % 4, mmsi=str(500+i)) for i in range(32)]
        refs[0]['mmsi'] = rows[0]['mmsi']
        scores['plain_fit', 43][0, 3] = 20.
        case, group, roles = a.case_details(next(e for e in chosen if e['id'] == '0'), 0, rows[0], refs, scores)
        self.assertEqual(len(case['fixed_bank_ids']), 32)
        self.assertNotIn('r0', case['eligible_reference_ids'])
        self.assertIn('r3', case['rival_reference_ids'])
        self.assertEqual(case['reference_use']['plain'], 'NONE')
        self.assertTrue(all(role.startswith('FULL ') for role in roles[1:]))
        self.assertEqual(case['visual_status'], 'not_reviewed')
        self.assertEqual(len(group), 32)
        stable = {(c, s): scores['plain_adapt', s].copy() for c in a.CONDITIONS for s in a.SEEDS}
        no_flips = a.select_cases(a.pair_events(rows, stable, 'plain_adapt', 'full_adapt'))
        self.assertFalse(any(c['selection_category'] in ('rescued', 'harmed') for c in no_flips))

    def test_native_8_queries_16_panels_four_class_labels_and_pixel_scale(self):
        native = a.full.module_at('oss2_synthetic_native', a.PACKAGE/'sar_m123/input.py')
        with self.temporary() as directory:
            out = Path(directory)
            chip = out/'synthetic.png'
            a.base.Image.new('L', (37, 29), 128).save(chip)
            view = dict(image=str(chip), target_index=0,
                        detections=[dict(corners=[[5, 5], [15, 5], [15, 20], [5, 20]], confidence=.9)])
            rows, scores = make_rows(), make_scores()
            for row in rows:
                row['vv'] = None if row['analysis_available'] == 'vh_only' else view
                row['vh'] = None if row['analysis_available'] == 'vv_only' else view
            refs = [dict(id=f'r{i}', label=i % 4, mmsi=str(500+i), vv=view, vh=view) for i in range(32)]
            cases = a.gallery(out, dict(evaluation=rows, references=refs), scores,
                              a.pair_events(rows, scores, 'plain_adapt', 'full_adapt'), native)
            self.assertEqual(len(cases), 8)
            self.assertEqual(len(list((out/'cases').glob('*.png'))), 16)
            with a.base.Image.open(cases[0]['panels']['vv']) as panel:
                self.assertEqual(panel.getpixel((8, 63)), (128, 128, 128))
                self.assertEqual(panel.getpixel((44, 91)), (128, 128, 128))
                self.assertEqual(panel.getpixel((45, 91)), (255, 255, 255))
            sources = a.base.read_json(out/'cases/native_sources.json')
            self.assertEqual(len(sources), 8*2*33)
            self.assertTrue(all(r['pixel_scale'] == 1 and not r['contrast_stretch'] for r in sources))
            self.assertTrue(all(not r['plain_reference_effect'] for r in sources))
            self.assertTrue(any(not r['available'] for r in sources if r['role'] == 'QUERY'))
            page = (out/'cases/index.html').read_text(encoding='utf-8')
            self.assertIn('PLAIN has NO reference contribution', page)
            for name in a.CLASSES:
                self.assertIn(name, page)
            self.assertNotIn('fishing', page)
            self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
