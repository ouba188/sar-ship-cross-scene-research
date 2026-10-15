"""Differentiable deployment geometry and support-only residual ridge."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def residual_scores(support_phi, support_prior, support_labels, query_phi,
                    query_prior, strength=8.):
    """Adapt query scores with the exact balanced dual ridge deployment head.

    Phi already includes its intercept column. For 64 support rows the solve
    is 64x64; all floating-point operations retain the input dtype.
    """
    _, inverse, counts = torch.unique(support_labels, return_inverse=True,
                                      return_counts=True)
    weights = len(support_labels)/(len(counts)*counts[inverse].to(support_phi.dtype))
    root = weights.sqrt().unsqueeze(1)
    desired = 2*F.one_hot(support_labels.long(), num_classes=support_prior.shape[1]).to(
        support_prior.dtype)-1
    weighted = support_phi*root
    gram = weighted@weighted.T
    dual = torch.linalg.solve(
        gram+strength*torch.eye(len(support_labels), dtype=gram.dtype, device=gram.device),
        root*(desired-support_prior))
    correction = weighted.T@dual
    return query_prior+query_phi@correction


class FixedExpertGeometry(nn.Module):
    """Frozen source projection with gradients into the neural extra features.

    xs is the already standardized R50 PCA128 representation. NumPy arrays
    are copied into buffers; forward uses the extra features' dtype/device.
    """

    def __init__(self, arrays):
        super().__init__()
        for name in ('ssl_mean', 'ssl_components', 'expert_center', 'expert_scale',
                     'expert_coef', 'expert_intercept'):
            self.register_buffer(name, torch.as_tensor(arrays[name]).clone())

    def forward(self, xs, extra):
        z = (extra-self.ssl_mean.to(extra))@self.ssl_components.to(extra).T
        joint = torch.cat((xs, z), dim=1)
        xe = (joint-self.expert_center.to(joint))/self.expert_scale.to(joint)
        prior = xe@self.expert_coef.to(xe).T+self.expert_intercept.to(xe)
        phi = torch.cat((xe/math.sqrt(xe.shape[1]), xe.new_ones((len(xe), 1))), dim=1)
        return prior, phi
