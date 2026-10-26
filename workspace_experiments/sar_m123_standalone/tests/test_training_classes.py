"""Small synthetic regression tests for class-generic staged training."""
import functools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from sar_m123 import aligned, tokens, train_aligned as training, train_ops as op
from sar_m123.m3 import PairMetric, PolarHead, regional_descriptors
from sar_m123.training_alignment import FrozenDecision, fit_metric


def example(classes, sparse_local=False):
    rng = np.random.default_rng(71)
    rows = []
    for role in training.ROLES:
        for c in range(classes):
            for _ in range(8 if role == 'fit_train' else 1):
                i = len(rows)
                rows.append(dict(obs_id=i, mmsi=str(i), pass_group=role,
                                 role=role, class_id=c if role != 'evaluation' else -1))
    table = pd.DataFrame(rows)
    n = len(table)
    bank = np.arange(classes*8).reshape(classes, 8)
    refs = np.broadcast_to(bank[:, :2], (n, classes, 2)).copy()
    for i in bank.ravel():
        refs[i, i//8] = bank[i//8][bank[i//8] != i][:2]
    mass = rng.uniform(.2, 1, (n, 2, 20)).astype(np.float32)
    mass[-1, 1] = 0
    mass[-2] = 0
    mean = rng.normal(size=(n, 2, 20, 32)).astype(np.float32)
    joint = rng.normal(size=(n, 2, 20, 96)).astype(np.float32)
    mean[mass <= 0] = 0
    joint[mass <= 0] = 0
    data = dict(obs_id=table.obs_id.to_numpy(), bank=bank, refs=refs,
                raw=rng.normal(size=(n, 1538)).astype(np.float32),
                aux=rng.normal(size=(n, 18*classes+32)).astype(np.float32),
                x=tokens.unit(mean), mean=mean, mass=mass, joint=joint,
                e_image=rng.uniform(0, 1, (n, 2, 20, 24)).astype(np.float32),
                coverage=rng.uniform(0, 1, (n, 2, 20, 3)).astype(np.float32),
                covariance_mu=np.zeros(96, np.float32),
                covariance_scale=np.ones(96, np.float32),
                covariance_active=np.ones(96, bool))
    if sparse_local:
        extra = table.iloc[[0, classes*8]].copy()
        extra['obs_id'] = [n, n+1]
        extra['mmsi'] = [str(n), str(n+1)]
        table = pd.concat((table, extra), ignore_index=True)
        for key in (*training.FEATURES, 'raw', 'aux', 'refs'):
            data[key] = np.concatenate((data[key], data[key][:2]))
        data['obs_id'] = table.obs_id.to_numpy()
        data['joint'][-2:] = 0
        data['local_available'] = np.r_[np.ones(n, bool), False, False]
    return table, data


def covariance(data):
    return {k: data['covariance_'+k] for k in ('mu', 'scale', 'active')}


def frozen_fixture(folder, data, base, local):
    """Use nonzero heads/gate so score equality also exercises the gradient path."""
    torch.manual_seed(17)
    sc = op.scale_fit(op.local_raw(local, base), np.arange(len(base)))
    x = torch.tensor(op.transform(op.local_raw(local, base), sc))
    null = op.transform(np.zeros((1, 1, 15)), sc).reshape(15)
    delta = []
    for p, name in enumerate(('vv', 'vh_centered')):
        head = PolarHead(p, centered=bool(p), null=null)
        torch.save(dict(state=op.state(head)), folder/f'{name}.pt')
        delta.append(head(x).detach().numpy())
    desc = regional_descriptors(data, covariance(data))
    gr, av = op.gate_raw(desc, desc, data['refs'], base, *delta)
    gs = op.scale_fit(gr, np.arange(len(base)))
    gate = op.Gate(185, .4)
    with torch.no_grad():
        gate.net[-1].weight.normal_(std=.1)
    w = gate(torch.tensor(op.transform(gr, gs)), torch.tensor(av)).detach().numpy()
    torch.save(sc, folder/'M3_readout_scaler.pt')
    torch.save(dict(state=op.state(gate), scaler=gs, alpha=.4), folder/'frequency.pt')
    cached = dict(gate_raw=gr, available=av)
    return FrozenDecision(folder, cached, base, 'cpu'), base+w[:, :1]*delta[0]+w[:, 1:]*delta[1]


class TrainingClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_roles_bank_and_reference_class_order(self):
        for classes in (4, 6, 8):
            with self.subTest(classes=classes):
                table, data = example(classes)
                groups = training.validate_roles(table, data)
                self.assertEqual(len(groups['fit_train']), classes*8)
                self.assertEqual(len(groups['evaluation']), classes)
                training.validate_roles(table, data, classes=classes)
                with self.assertRaises(ValueError):
                    training.validate_roles(table, data, classes=classes+1)
                data['refs'][:, [-1, 0]] = data['refs'][:, [0, -1]]
                with self.assertRaisesRegex(ValueError, 'candidate class order'):
                    training.validate_roles(table, data)

    def test_eight_class_role_must_cover_highest_class(self):
        table, data = example(8)
        table.loc[table.role.eq('selection') & table.class_id.eq(7), 'class_id'] = 6
        with self.assertRaisesRegex(ValueError, 'selection needs all 8'):
            training.validate_roles(table, data)

    def test_local_availability_required_only_for_m3_rows(self):
        table, data = example(8, sparse_local=True)
        groups = training.validate_roles(table, data)
        for row in (data['bank'][7, 7], groups['utility_train'][0],
                    groups['selection'][0], groups['evaluation'][0]):
            with self.subTest(row=row):
                data['local_available'][row] = False
                with self.assertRaisesRegex(ValueError, 'local_available must cover'):
                    training.validate_roles(table, data)
                data['local_available'][row] = True

    def test_class_weights_and_metrics(self):
        for classes in (4, 6, 8):
            with self.subTest(classes=classes):
                count = np.arange(1, classes+1)
                y = np.repeat(np.arange(classes), count)
                ids = np.arange(len(y))
                wt = op.balanced_weights(np.r_[y, -1, 999], ids, 'cpu', classes=classes).numpy()
                np.testing.assert_allclose(wt.sum(), 1, atol=1e-7)
                class_mass = np.bincount(y, weights=wt)
                np.testing.assert_allclose(class_mass, .5*count/count.sum()+.5/classes)
                np.testing.assert_array_equal(wt, op.balanced_weights(y, ids, 'cpu').numpy())
                with self.assertRaises(AssertionError):
                    op.balanced_weights(y, ids[y != classes-1], 'cpu', classes=classes)
                # Evaluation can cover only a subset of the output classes.
                labels = np.array([0, classes-1])
                metrics = op.metrics(labels, np.zeros((2, classes), np.float32))
                self.assertAlmostEqual(metrics['Brier'], (classes-1)/classes)
                self.assertAlmostEqual(metrics['NLL'], np.log(classes), places=6)
                self.assertEqual(metrics['J'], float(op.exact_j(labels, np.zeros((2, classes)))))

    def test_competitors_include_all_classes(self):
        for classes in (4, 6, 8):
            with self.subTest(classes=classes):
                base = np.arange(classes, dtype=np.float32)[None]
                local = np.arange(classes*12, dtype=np.float32).reshape(1, classes, 2, 3, 2)
                raw = op.local_raw(local, base)
                np.testing.assert_array_equal(raw[..., :12], local.reshape(1, classes, 12))
                np.testing.assert_array_equal(raw[0, :-1, -1], np.arange(classes-1)-(classes-1))
                self.assertEqual(raw[0, -1, -1], 1)
                if classes == 4:
                    np.testing.assert_array_equal(raw, aligned.local_features(local, base))

    def test_gate_permutation_and_four_class_parity(self):
        for classes in (4, 8):
            with self.subTest(classes=classes):
                table, data = example(classes)
                rng = np.random.default_rng(93)
                base, dv, dh = (rng.normal(size=(len(table), classes)).astype(np.float32) for _ in range(3))
                desc = regional_descriptors(data, covariance(data))
                gr, av = op.gate_raw(desc, desc, data['refs'], base, dv, dh)
                self.assertEqual(gr.shape, (len(table), classes, 185))
                np.testing.assert_array_equal(av[-2:], [[0, 0], [1, 0]])
                order = np.arange(classes)[::-1]
                perm, _ = op.gate_raw(desc, desc, data['refs'][:, order], base[:, order], dv[:, order], dh[:, order])
                np.testing.assert_allclose(perm, gr[:, order], atol=1e-6)
                if classes == 4:
                    expected, flags = aligned.gate_features(desc, desc, data['refs'], base, dv, dh)
                    np.testing.assert_array_equal(gr, expected)
                    np.testing.assert_array_equal(av, flags)

    def test_base_dimensions_and_checkpoint_reload(self):
        for classes in (4, 8):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as folder:
                table, data = example(classes)
                groups = training.validate_roles(table, data)
                out = Path(folder)
                score = training.fit_base(data, table, groups, 17, out, steps=5)
                ck = op.load_checkpoint(out/'raw_summary.pt')
                self.assertEqual(ck['aux_dim'], 18*classes+32)
                self.assertEqual(ck['state']['0.weight'].shape, (128, 1570+18*classes))
                self.assertEqual(ck['state']['2.weight'].shape, (classes, 128))
                model = torch.nn.Sequential(torch.nn.Linear(1570+18*classes, 128),
                    torch.nn.GELU(), torch.nn.Linear(128, classes))
                model.load_state_dict(ck['state'])
                x = np.c_[training.scale_input(data['raw'], training.load_npz(out/'raw_scaler.npz')),
                          training.scale_input(data['aux'], training.load_npz(out/'summary_scaler.npz'))]
                np.testing.assert_array_equal(model(torch.tensor(x)).detach().numpy(), score)
                data['aux'] = data['aux'][:, :-1]
                with self.assertRaisesRegex(ValueError, 'Expected raw'):
                    training.fit_base(data, table, groups, 17, out, steps=5)

    def test_frozen_decision_parity_and_real_matcher_gradient(self):
        for classes in (4, 8):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as folder:
                table, data = example(classes)
                rng = np.random.default_rng(81)
                base = rng.normal(size=(len(table), classes)).astype(np.float32)
                local = rng.normal(size=(len(table), classes, 2, 3, 2)).astype(np.float32)
                decision, expected = frozen_fixture(Path(folder), data, base, local)
                ids = np.arange(len(table))
                np.testing.assert_allclose(decision(torch.tensor(local), ids).detach().numpy(),
                                           expected, atol=1e-6, rtol=1e-6)
                # Real PairMetric + transport must propagate through the frozen heads.
                matcher = op.Matcher(data, data['refs'], covariance(data), 'cpu')
                metric = PairMetric()
                ids = training.validate_roles(table, data)['utility_train'][:2]
                actual = matcher.local(ids, metric)
                self.assertEqual(tuple(actual.shape), (2, classes, 2, 3, 2))
                logits = decision(actual, ids)
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(table.iloc[ids].class_id.to_numpy()))
                loss.backward()
                for parameter in metric.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().max()), 0)

    def test_eight_class_npz_staged_training_ignores_evaluation_labels(self):
        self.check_staged_training(sparse_local=False)

    def test_eight_class_staged_training_skips_base_only_local_rows(self):
        self.check_staged_training(sparse_local=True)

    def check_staged_training(self, sparse_local):
        table, data = example(8, sparse_local=sparse_local)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, assets = root/'bundle', root/'assets'
            bundle.mkdir()
            initial = assets/'warm'/'seed17'
            initial.mkdir(parents=True)
            torch.manual_seed(17)
            torch.save(dict(state=op.state(PairMetric())), initial/'both_metric.pt')
            torch.save(covariance(data), initial/'metric_scaler.pt')
            np.savez_compressed(bundle/'features.npz', **data)
            fit_base, fit_head, fit_gate = training.fit_base, op.fit_head, op.fit_gate
            outputs, receipts = [], []
            # Five optimizer steps retain the real selection cadence and all stages.
            with patch.object(training, 'fit_base', functools.partial(fit_base, steps=5)), \
                 patch.object(op, 'fit_head', functools.partial(fit_head, steps=5)), \
                 patch.object(op, 'fit_gate', functools.partial(fit_gate, steps=5)), \
                 patch.object(training, 'fit_metric', functools.partial(fit_metric, steps=5)), \
                 patch.object(op.Matcher, 'predict', autospec=True, side_effect=op.Matcher.predict) as predict:
                for run in range(2):
                    table.loc[table.role.eq('evaluation'), 'class_id'] = -1 if run == 0 else 999
                    table.to_csv(bundle/'table.csv', index=False)
                    out = root/f'run{run}'
                    results = training.train(bundle, assets, out, seeds=(17,), classes=None if run == 0 else 8)
                    dest = out/'seed17'
                    outputs.append(training.load_npz(dest/'predictions.npz'))
                    receipts.append({k: v for k, v in results[0].items() if k != 'metric_seconds'})
                    metadata = json.loads((out/'training.json').read_text())
                    self.assertEqual(metadata['classes'], 8)
                    self.assertFalse(metadata['evaluation_labels_used'])
                    for stage in (dest/'warm', dest):
                        for name in ('vv.pt', 'vh_centered.pt', 'frequency.pt', 'M3_readout_scaler.pt'):
                            self.assertTrue((stage/name).is_file(), (stage, name))
                    self.assertTrue((dest/'aligned_metric.pt').is_file())
                    self.assertEqual(outputs[-1]['frequency'].shape, (len(table), 8))
                    np.testing.assert_array_equal(outputs[-1]['refs'], data['refs'])
                    ck = op.load_checkpoint(dest/'raw_summary.pt')
                    self.assertEqual(ck['state']['0.weight'].shape, (128, 1714))
                    self.assertEqual(ck['aux_dim'], 176)
                    metric_curve = pd.read_csv(dest/'metric_curve.csv')
                    self.assertGreaterEqual(len(metric_curve), 2)
                    self.assertTrue(np.isfinite(outputs[-1]['frequency']).all())
                expected_ids = np.flatnonzero(data.get('local_available', np.ones(len(table), bool)))
                for call in predict.call_args_list:
                    np.testing.assert_array_equal(call.args[1], expected_ids)
            if sparse_local:
                np.testing.assert_array_equal(outputs[0]['local_available'], data['local_available'])
                np.testing.assert_array_equal(outputs[0]['local'][~data['local_available']], 0)
            else:
                self.assertNotIn('local_available', outputs[0])
            self.assertEqual(receipts[0], receipts[1])
            for key in outputs[0]:
                np.testing.assert_array_equal(outputs[0][key], outputs[1][key], err_msg=key)

    def test_eight_class_nonzero_metric_optimization(self):
        table, data = example(8)
        groups = training.validate_roles(table, data)
        rng = np.random.default_rng(81)
        base = rng.normal(size=(len(table), 8)).astype(np.float32)
        local = rng.normal(size=(len(table), 8, 2, 3, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            decision, _ = frozen_fixture(out, data, base, local)
            metric = PairMetric()
            matcher = op.Matcher(data, data['refs'], covariance(data), 'cpu')
            fit, val = groups['utility_train'], groups['selection']
            y = table.class_id.to_numpy()
            selected, gradient, _ = fit_metric(matcher, metric, decision, y, fit, val, out, steps=5)
            self.assertGreater(gradient, 0)
            curve = pd.read_csv(out/'metric_curve.csv')
            self.assertEqual(curve.step.tolist(), [0, 5])
            self.assertTrue(np.isfinite(curve.iloc[-1]['loss']))
            ck = op.load_checkpoint(out/'aligned_metric.pt')
            self.assertEqual(ck['selected_step'], selected)
            for key, value in metric.state_dict().items():
                torch.testing.assert_close(value, ck['state'][key], rtol=0, atol=0)
            scores = decision(torch.tensor(matcher.predict(val, metric)), val).detach().numpy()
            self.assertAlmostEqual(float(op.exact_j(y[val], scores)), curve.J.max())


if __name__ == '__main__':
    unittest.main()
