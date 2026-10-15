import unittest
import numpy as np
import pandas as pd
import torch
from change_decomposition import (EvidenceDecomposition, acquisition_pairs, train_decomposition,
    relation_delta, relation_features, relation_episode_loss)
from paired_evidence_factors import support_operator, adapted_scores


def example():
    rng = np.random.default_rng(913)
    meta = pd.DataFrame([dict(mmsi=ship, class_id=ship%8, product=j, port=j%2)
                         for ship in range(24) for j in range(3)])
    return rng.normal(size=(len(meta), 32)), meta


class ChangeTests(unittest.TestCase):
    def test_initial_information_is_not_deleted(self):
        x, meta = example()
        model = EvidenceDecomposition(x, meta)
        out = model(torch.tensor(x, dtype=torch.float32))
        c, r = out['content_component'].detach().numpy(), out['residual_component'].detach().numpy()
        np.testing.assert_allclose(c+r, x, atol=2e-6)
        np.testing.assert_allclose((c*r).sum(1), 0, atol=3e-6)

    def test_distinct_products_and_ship_weights(self):
        x, meta = example()
        a, b, w, info = acquisition_pairs(x, meta)
        self.assertEqual(len(info), 24*3)
        np.testing.assert_allclose(w.sum(), 1)
        duplicated = pd.concat([meta, meta.iloc[:1]], ignore_index=True)
        a2, b2, w2, info2 = acquisition_pairs(np.r_[x, x[:1]], duplicated)
        np.testing.assert_allclose(a, a2)
        np.testing.assert_allclose(b, b2)
        np.testing.assert_allclose(w, w2)

    def test_excluded_rows_cannot_change_learning(self):
        x, meta = example()
        fit, val = np.arange(48), np.arange(48, 60)
        a = train_decomposition(x, meta, fit, val, 17, 'paired', steps=2, every=1)
        poisoned, mm = x.copy(), meta.copy()
        poisoned[60:] = 1e4
        mm.loc[60:, 'class_id'] = 7
        b = train_decomposition(poisoned, mm, fit, val, 17, 'paired', steps=2, every=1)
        for key in a[1]['last']:
            torch.testing.assert_close(a[1]['last'][key], b[1]['last'][key], atol=0, rtol=0)

    def test_relation_bounds_and_zero_start(self):
        rng = np.random.default_rng(51)
        f, p = rng.normal(size=(31, 7)), rng.normal(size=(31, 8))
        np.testing.assert_array_equal(relation_delta(f, p, np.zeros(28)), 0)
        delta = relation_delta(f*1e4, p*0, np.ones(28)*100)
        self.assertLessEqual(np.abs(delta).max(), .500001)
        np.testing.assert_allclose(delta.sum(1), 0, atol=1e-7)

    def test_relation_readout_includes_support_and_absent_classes(self):
        rng = np.random.default_rng(79)
        f, p, k = rng.normal(size=(12, 7)), rng.normal(size=(12, 8)), rng.normal(size=(12, 4))
        ys, yq = np.array([0, 1, 1, 2]), np.arange(8)
        op, present = support_operator(k[:4], k[4:], ys)
        coefficient = rng.normal(size=28).astype(np.float32)
        design = relation_features(f, p)
        fs = torch.tensor(design[:4], dtype=torch.float32, requires_grad=True)
        args = [torch.tensor(v, dtype=torch.float32) for v in (design[4:], p[:4], p[4:], op, present)]
        loss, adapted, baseline = relation_episode_loss(torch.tensor(coefficient), fs, *args,
                                                        torch.tensor(ys), torch.tensor(yq))
        d = relation_delta(f, p, coefficient)
        expected = adapted_scores(p[:4]+d[:4], p[4:]+d[4:], op, present, ys)
        np.testing.assert_allclose(adapted.detach(), expected, atol=5e-7)
        loss.backward()
        self.assertGreater(float(fs.grad.abs().sum()), 0)

    def test_ce_training_path_moves_class_relations(self):
        from change_decomposition import train_relation_correction
        rng = np.random.default_rng(21)
        features = rng.normal(size=(32, 7))
        prior = rng.normal(size=(32, 8))
        kernel = rng.normal(size=(32, 4))
        labels = np.arange(32)%8
        episodes = [(np.arange(8), np.arange(8, 32))]*2
        coef, _ = train_relation_correction(features, prior, labels, kernel, episodes, objective='ce')
        self.assertGreater(float(np.linalg.norm(coef)), 0)
        self.assertTrue(np.isfinite(coef).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
