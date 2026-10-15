"""Current protected/stable/incidence expert, differentiable into R18."""

import math
import numpy as np
import torch
from torch import nn


class FullExpertGeometry(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        model, stats = geometry['model'], geometry['stats']
        values = dict(ssl_mean=model['pca_ssl'].mean_, ssl_components=model['pca_ssl'].components_,
            raw_mean=model['pca'].mean_, raw_components=model['pca'].components_,
            expert_center=model['expert_center'], expert_scale=model['expert_scale'],
            discarded_center=stats['discarded_center'], discarded_scale=stats['discarded_scale'],
            drift_to_xs=geometry['drift_to_xs'], drift_to_complement=geometry['drift_to_complement'])
        self.rank = geometry['stable_rank']
        for key, value in values.items():
            self.register_buffer(key, torch.tensor(np.asarray(value), dtype=torch.float64))

    def forward(self, raw, extra, fixed_bank, incidence):
        # Fixed bank retains the exact deployed R50/stable/drift coordinates.
        dtype = self.ssl_mean.dtype
        raw, extra = raw.to(dtype), extra.to(dtype)
        bank, u = fixed_bank.to(dtype), incidence.to(dtype)
        ec = extra-self.ssl_mean
        ssl = (ec@self.ssl_components.T).float().to(dtype)
        bank = torch.cat((bank[:, :128], ssl, bank[:, 192:]), dim=1)
        drift = bank[:, 192+self.rank:]
        core = bank[:, :128]-.5*(drift@self.drift_to_xs)
        retained = (torch.cat((core, ssl), 1)-self.expert_center)/self.expert_scale
        rc = raw-self.raw_mean
        rd = rc-(rc@self.raw_components.T)@self.raw_components
        ed = ec-(ec@self.ssl_components.T)@self.ssl_components
        complement = (torch.cat((rd, ed), 1)-self.discarded_center)/self.discarded_scale
        complement = torch.cat((complement[:, :2048]-.5*(drift@self.drift_to_complement),
                                complement[:, 2048:]), 1)
        stable = bank[:, 192:192+self.rank]/math.sqrt(self.rank)
        phi = torch.cat((retained/math.sqrt(384), complement/math.sqrt(5120),
                         .5*stable, u/math.sqrt(8), bank.new_ones((len(bank), 1))), 1)
        return torch.cat((bank, u), 1), phi


def numpy_phi(bank, raw, extra, incidence, geometry):
    from run_protected_drift_activation import make_views
    views = make_views(bank, raw, extra, geometry)
    common = views['common'][True]
    return np.column_stack((common['retained']/np.sqrt(2), common['complement']/np.sqrt(2),
                            .5*views['stable'], incidence/np.sqrt(8), np.ones(len(bank))))


def parity_check(context, incidence):
    from aligned_residual_ops import residual_scores
    from unified_head_core import score
    n = 83
    bank = np.asarray(context.bank[:n])
    extra = np.asarray(context.extra[:n])
    raw = np.asarray(context.raw[context.source_rows[:n]])
    u = incidence[context.source_rows[:n]]
    from full_teacher_data import bank_from_extra, fit_head
    updated = bank_from_extra(bank, extra, context.geometry)
    head = fit_head(np.c_[context.bank, incidence[context.source_rows]], context.source_labels, context.transforms)
    module = FullExpertGeometry(context.geometry)
    et = torch.tensor(extra, dtype=torch.float32, requires_grad=True)
    aug, phi = module(torch.tensor(raw), et, torch.tensor(bank), torch.tensor(u))
    expected_phi = numpy_phi(updated, raw, extra, u, context.geometry)
    prior = np.c_[updated, u]@head['coef']+head['intercept']
    sy = context.source_labels[:64]
    expected, _ = score(expected_phi@expected_phi[:64].T, prior, np.arange(64), sy)
    p = aug@torch.tensor(head['coef'])+torch.tensor(head['intercept'])
    actual = residual_scores(phi[:64], p[:64], torch.tensor(sy), phi[64:], p[64:])
    np.testing.assert_allclose(phi.detach().numpy(), expected_phi, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(actual.detach().numpy(), expected[64:], atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert torch.isfinite(et.grad).all() and float(et.grad.norm()) > 0
    return dict(phi_error=float(np.max(abs(phi.detach().numpy()-expected_phi))),
                score_error=float(np.max(abs(actual.detach().numpy()-expected[64:]))),
                extra_gradient_norm=float(et.grad.norm()))
