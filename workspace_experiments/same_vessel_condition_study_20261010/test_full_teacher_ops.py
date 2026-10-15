"""Read-only Santos parity and gradient checks; run with the original GPU Python.

    python -B -m unittest -v test_full_teacher_ops

Only small slices of existing raw/features/banks are read. No training,
geometry fitting, metrics, or historical artifact writes are performed.
"""

import json
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from full_teacher_ops import FullTeacherOps, residual_scores


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
REFERENCE = ROOT/'oof_reliability_check'
SANTOS = ROOT/'protected_drift_activation_20261008/ports/Santos'


class FullTeacherOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SANTOS/'geometry.pkl').exists():
            raise FileNotFoundError(f'The real Santos parity fixture is required: {SANTOS}')
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        limits = threadpool_limits(limits=2)
        cls.addClassCleanup(limits.restore_original_limits)
        sys.path.insert(0, str(REFERENCE))
        cls.addClassCleanup(sys.path.remove, str(REFERENCE))
        from unified_head_expert import ExpertContext, SOURCE, _conditions

        with (SANTOS/'geometry.pkl').open('rb') as stream:
            cls.geometry = pickle.load(stream)
        with (ROOT/'direct_incidence_comparison_20261009/ports/Santos/direct_heads.pkl').open('rb') as stream:
            cls.head = pickle.load(stream)['joint_expert']
        cls.context = ExpertContext.__new__(ExpertContext)
        cls.context.geometry = cls.geometry
        source_rows = np.load(SANTOS/'source_rows.npy')
        # Global source row IDs and local R18/bank positions are different indices.
        positions = np.linspace(0, len(source_rows)-1, 96, dtype=int)
        rows = source_rows[positions]
        cls.raw = np.array(np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[rows])
        cls.extra = np.array(np.load(
            ROOT/'evidence_aware_handoff_20261008/source_features/Santos/features.npy',
            mmap_mode='r')[positions])
        cls.bank = np.array(np.load(SANTOS/'source_bank.npy', mmap_mode='r')[positions])
        cls.u = _conditions()[rows].copy()
        # Explicit missing-incidence rows exercise the supported zero encoding.
        cls.u[[0, 65]] = 0.
        cls.op = FullTeacherOps(cls.geometry)
        # Imbalanced supports, including a missing class, exercise balanced ridge.
        cls.labels = np.array([0]*20+[1]*12+[2]*10+[3]*8+[4]*6+[6]*4+[7]*4)

    def inputs(self, dtype=torch.float64, device='cpu', fixed=True):
        values = [torch.tensor(x, dtype=dtype, device=device)
                  for x in (self.raw, self.extra, self.u)]
        bank = torch.tensor(self.bank, dtype=dtype, device=device) if fixed else None
        return (*values, bank)

    def numpy_reference(self, bank, raw=None, extra=None, u=None):
        raw = self.raw if raw is None else raw
        extra = self.extra if extra is None else extra
        u = self.u if u is None else u
        aug = np.column_stack((bank, u))
        prior = aug @ self.head['coef']+self.head['intercept']
        kernel = self.context._port_kernel(bank, raw, extra, u, np.arange(64))
        from projection_residual_expert import fit_kernel_correction
        scores = prior[64:]+kernel[64:] @ fit_kernel_correction(
            kernel[:64], self.labels, prior[:64])
        return aug, prior, kernel, scores

    def check_parity(self, dtype, device):
        raw, extra, u, fixed = self.inputs(dtype, device)
        op = FullTeacherOps(self.geometry).to(device=device)
        bank, phi = op(raw, extra, u, fixed_bank=fixed)
        coef = torch.tensor(self.head['coef'], dtype=dtype, device=device)
        intercept = torch.tensor(self.head['intercept'], dtype=dtype, device=device)
        prior = bank @ coef+intercept
        kernel = phi @ phi[:64].T
        scores = residual_scores(phi[:64], prior[:64],
                                 torch.tensor(self.labels, device=device),
                                 phi[64:], prior[64:])
        expected = self.numpy_reference(self.bank)
        actual = (bank, prior, kernel, scores)
        errors = {}
        for name, value, target in zip(('bank', 'prior', 'kernel', 'scores'), actual, expected):
            array = value.detach().cpu().numpy()
            errors[name] = float(np.max(np.abs(array-target)))
            # Cached PCA coordinates include float32 rounding even for a double replay.
            tolerance = 2e-5 if dtype == torch.float64 else 2e-4
            np.testing.assert_allclose(array, target, atol=tolerance, rtol=tolerance,
                                       err_msg=f'{device}/{dtype}/{name}')
        self.assertEqual(bank.dtype, dtype)
        self.assertEqual(phi.dtype, dtype)
        self.assertEqual(bank.shape, (96, 236))
        self.assertEqual(phi.shape, (96, 2781))
        self.assertEqual(list(op.parameters()), [])
        print('Santos parity '+json.dumps(dict(device=device, dtype=str(dtype), **errors)), flush=True)

    def test_cached_bank_parity_cpu_float64(self):
        self.check_parity(torch.float64, 'cpu')

    def test_cached_bank_parity_cpu_float32(self):
        self.check_parity(torch.float32, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cached_bank_parity_cuda(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.check_parity(dtype, 'cuda')

    def test_source_transform_matches_current_head_geometry(self):
        from direct_incidence_conditioning import extend_transforms
        from run_protected_drift_activation import make_transforms
        expected = extend_transforms(make_transforms(self.geometry))['joint_expert']
        np.testing.assert_allclose(self.op.source_transform.numpy(), expected,
                                   atol=1e-12, rtol=1e-12)

    def test_raw_projection_parity_and_cache_boundary(self):
        raw, extra, u, _ = self.inputs()
        bank, phi = self.op(raw, extra, u)
        g, model = self.geometry, self.geometry['model']
        xs = ((model['pca'].transform(self.raw.astype(float))-model['center'])
              / model['scale'])
        ssl = model['pca_ssl'].transform(self.extra.astype(float))
        x = (self.raw.astype(float)-g['feature_center'])/g['feature_scale']
        stable = (x @ g['qs']-g['stable_center'])/g['stable_scale']
        expected_bank = np.column_stack((xs, ssl, stable, x @ g['qd']))
        np.testing.assert_allclose(bank.numpy(), np.c_[expected_bank, self.u],
                                   atol=2e-11, rtol=2e-11)
        expected_kernel = self.numpy_reference(expected_bank)[2]
        np.testing.assert_allclose((phi @ phi[:64].T).numpy(), expected_kernel,
                                   atol=2e-7, rtol=2e-7)
        print('Santos raw-vs-historical xs max_abs='+str(
            np.max(np.abs(xs-self.bank[:, :128]))), flush=True)

    def test_fixed_bank_refreshes_extra_and_incidence(self):
        raw, extra, u, fixed = self.inputs()
        stale = torch.cat((fixed.clone(), u+100), dim=1)
        stale[:, 128:192] = -999
        bank, phi = self.op(raw, extra+.1, u, fixed_bank=stale)
        expected, expected_phi = self.op(raw, extra+.1, u, fixed_bank=fixed)
        torch.testing.assert_close(bank, expected, atol=0, rtol=0)
        torch.testing.assert_close(phi, expected_phi, atol=0, rtol=0)
        self.assertGreater(float((bank[:, 128:192]-fixed[:, 128:192]).abs().max()), .01)

    def test_gradients_through_prior_retained_complement_and_support_solve(self):
        from aligned_residual_ops import residual_scores as reference_scores
        raw, extra, u, fixed = self.inputs()
        extra.requires_grad_()
        bank, phi = self.op(raw, extra, u, fixed_bank=fixed)
        coef = torch.tensor(self.head['coef'])
        intercept = torch.tensor(self.head['intercept'])
        prior = bank @ coef+intercept
        labels = torch.tensor(self.labels)
        scores = residual_scores(phi[:64], prior[:64], labels, phi[64:], prior[64:])
        torch.testing.assert_close(scores, reference_scores(
            phi[:64], prior[:64], labels, phi[64:], prior[64:]), atol=1e-12, rtol=1e-12)
        for value in (prior, phi[:, :192], phi[:, 192:2752], scores):
            grad, = torch.autograd.grad(value.square().mean(), extra, retain_graph=True)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad[:64].norm()), 0.)
            self.assertGreater(float(grad[64:].norm()), 0.)

        generator = torch.Generator().manual_seed(31415)
        direction = torch.randn(extra.shape, generator=generator, dtype=extra.dtype)
        direction /= direction.norm()
        support_direction, query_direction = direction.clone(), direction.clone()
        support_direction[64:] = 0
        query_direction[:64] = 0

        def objective(offset):
            changed = extra.detach()+offset[0]*support_direction+offset[1]*query_direction
            b, p = self.op(raw, changed, u, fixed_bank=fixed)
            pri = b @ coef+intercept
            s = residual_scores(p[:64], pri[:64], labels, p[64:], pri[64:])
            return s.square().mean()

        offset = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(objective, (offset,), eps=1e-5,
                                                atol=1e-8, rtol=1e-4))
        derivative, = torch.autograd.grad(objective(offset), offset)
        print('Santos support/query directional gradients='+str(derivative.tolist()), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
