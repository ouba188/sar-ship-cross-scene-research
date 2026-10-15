"""Focused checks for SSL leakage, refitted priors, and PCA complement loss."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import unittest
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from backbone_ablation_core import single_transform, r50_views, AblationContext, TASK_ROOT, EPISODES
from protected_drift_activation import fit_source_heads
from unified_head_core import score, class_weights


class BackboneTests(unittest.TestCase):
    def test_real_cached_r50_views_retain_original_geometry(self):
        from pathlib import Path
        import json
        from run_protected_drift_activation import make_views
        from evaluate_joint_residual import views

        for task in ('self24ports/Busan', 'f0921', 'oss2'):
            with self.subTest(task=task):
                ctx = AblationContext(TASK_ROOT/task)
                if ctx.dataset == 'self24ports':
                    bank = np.asarray(np.load(ctx.parent/'target_bank.npy', mmap_mode='r')[:32])
                    rows = np.load(ctx.parent/'target_rows.npy')[:32]
                    raw = ctx.raw[rows]
                    extra = np.load(EPISODES/ctx.task/'matched_joint/joint_extra.npy', mmap_mode='r')[:32]
                    original = make_views(bank, raw, extra, ctx.geometry)['common'][True]
                    retained, complement = r50_views(raw, ctx.geometry['model'], ctx.geometry['stats'], bank, ctx.geometry)
                    changed = bank.copy()
                    changed[:, 128:192] += 100.
                    r2, c2 = r50_views(raw, ctx.geometry['model'], ctx.geometry['stats'], changed, ctx.geometry)
                    np.testing.assert_array_equal(retained, r2)
                    np.testing.assert_array_equal(complement, c2)
                    dr, dd = original['retained'], original['complement']
                else:
                    target = Path(json.loads((ctx.joint.parent/'protocol.json').read_text())['target'])
                    if ctx.dataset == 'oss2':
                        with np.load(target/'features.npz') as z:
                            raw = z['paired_r50'][:32]
                    else:
                        raw = np.load(target/'paired_r50.npy', mmap_mode='r')[:32]
                    extra = np.load(ctx.joint/'target_extra.npy', mmap_mode='r')[:32]
                    _, dr, dd = views(raw, extra, ctx.model, ctx.stats, True)
                    retained, complement = r50_views(raw, ctx.model, ctx.stats)
                np.testing.assert_allclose(retained, dr[:, :128], atol=1e-12, rtol=0)
                np.testing.assert_allclose(complement, dd[:, :2048], atol=1e-12, rtol=0)
                dual = .5*(dr@dr.T+dd@dd.T)
                single = .5*(retained@retained.T+complement@complement.T)
                ssl = .5*(dr[:, 128:]@dr[:, 128:].T+dd[:, 2048:]@dd[:, 2048:].T)
                np.testing.assert_allclose(dual-single, ssl, atol=1e-10, rtol=1e-10)

    def test_refit_is_not_old_head_with_ssl_columns_zeroed(self):
        rng = np.random.default_rng(761)
        x = rng.normal(size=(320, 198))
        x[:, 128:192] = .8*x[:, :64]+.2*x[:, 128:192]
        y = np.arange(len(x)) % 8
        transform = np.eye(198)
        heads = fit_source_heads(x, y, {'dual': transform, 'r50': single_transform(transform)})
        keep = np.r_[np.arange(128), np.arange(192, 198)]
        independent = fit_source_heads(x[:, keep], y, {'one': np.eye(len(keep))})['one']
        np.testing.assert_allclose(heads['r50']['coef'][keep], independent['coef'], atol=1e-10)
        np.testing.assert_array_equal(heads['r50']['coef'][128:192], 0)
        self.assertGreater(np.linalg.norm(heads['r50']['coef'][keep]-heads['dual']['coef'][keep]), .01)
        changed = x.copy()
        changed[:, 128:192] += rng.normal(size=(len(x), 64))*100
        changed_head = fit_source_heads(changed, y, {'r50': single_transform(transform)})['r50']
        np.testing.assert_allclose(changed_head['coef'][keep], independent['coef'], atol=1e-10)
        np.testing.assert_allclose(changed@changed_head['coef']+changed_head['intercept'],
                                   x@heads['r50']['coef']+heads['r50']['intercept'], atol=1e-10)

    def test_ridge8_dual_matches_independent_primal_with_complement(self):
        rng = np.random.default_rng(139)
        retained = rng.normal(size=(91, 7))
        complement = rng.normal(size=(91, 11))
        features = np.c_[retained/np.sqrt(2), complement/np.sqrt(2), np.ones(91)]
        support = rng.choice(91, 64, replace=False)
        y = rng.choice(8, 64, p=[.4, .1, .1, .1, .1, .08, .07, .05])
        prior = rng.normal(size=(91, 8))
        kernel = features@features[support].T
        actual, _ = score(kernel, prior, support, y)
        x, w = features[support], class_weights(y)
        coef = solve(x.T@(w[:, None]*x)+8*np.eye(x.shape[1]),
                     x.T@(w[:, None]*(2*np.eye(8)[y]-1-prior[support])), assume_a='pos')
        np.testing.assert_allclose(actual, prior+features@coef, atol=1e-10)
        truncated = retained@retained[support].T/2+1
        self.assertGreater(np.max(abs(actual-score(truncated, prior, support, y)[0])), .1)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        unittest.main()
