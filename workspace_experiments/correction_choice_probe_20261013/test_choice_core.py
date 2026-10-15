"""Focused tests for utility direction, support evidence and role features."""

import unittest
import numpy as np
from choice_core import class_weights, gain_labels, support_profile, features, fit_probe, predict_probe


class ChoiceTests(unittest.TestCase):
    def test_gain_and_population_weights(self):
        y = np.array([0, 0, 0, 1])
        raw = np.eye(2)[[1, 0, 0, 1]]
        initial = np.eye(2)[[0, 0, 0, 0]]
        gain = gain_labels(raw, initial, y)
        np.testing.assert_array_equal(gain, [1, 0, 0, -1])
        self.assertAlmostEqual(class_weights(y) @ gain, -1/3)

    def test_oof_repeats_do_not_inflate_support(self):
        y = np.array([0, 0, 1, 1])
        prior = np.zeros((4, 8))
        r = 9 * np.eye(8)[[1, 0, 0, 1]]
        i = 9 * np.eye(8)[[0, 1, 1, 1]]
        bp = np.full((4, 8), 1/8)
        p = support_profile(prior, r, i, bp, y)
        triple = support_profile(*[np.tile(a, (3, 1)) for a in (prior, r, i, bp)], y)
        for key in p:
            np.testing.assert_allclose(p[key], triple[key])
        np.testing.assert_array_equal(p['counts'], [2, 2, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(p['mean'][1], 1/6)
        self.assertAlmostEqual(p['sd'][2]**2, .1)

    def test_role_invariance_and_signed_learning(self):
        rng = np.random.default_rng(17)
        r, i, b = [rng.dirichlet(np.ones(8), 30) for _ in range(3)]
        p = dict(mean=np.linspace(-.1,.1,8), sd=np.linspace(.1,.3,8), counts=np.arange(1,9))
        f = features(r, i, b, p)
        order = rng.permutation(8)
        g = features(r[:,order],i[:,order],b[:,order],{k:v[order] for k,v in p.items()})
        np.testing.assert_allclose(f,g,atol=1e-14)
        np.testing.assert_allclose(f[:4],features(r[:4],i[:4],b[:4],p))
        x = np.linspace(-2,2,80)[:,None]
        model = fit_probe(x,np.where(x[:,0]>0,1,-1),np.ones(80),1)
        prob = predict_probe(model,np.array([[-1.],[1.]]))
        self.assertLess(prob[0],.5)
        self.assertGreater(prob[1],.5)


if __name__ == '__main__':
    unittest.main()
