"""Differentiable retained-PCA and source-normalized complement geometry."""

import math

import torch
from torch import nn

from aligned_residual_ops import FixedExpertGeometry, residual_scores


class ComplementExpertGeometry(nn.Module):
    """Frozen source expert with equally weighted retained/complement kernels.

    xs is the standardized R50 PCA representation. base_discarded contains
    frozen, source-normalized R50 complement coordinates, without dimension
    scaling. Extra complement coordinates are recomputed to preserve gradients.
    """

    def __init__(self, arrays):
        super().__init__()
        self.retained = FixedExpertGeometry(arrays)
        for name in ('discarded_center', 'discarded_scale'):
            self.register_buffer(name, torch.as_tensor(arrays[name]).detach().clone())

    def forward(self, xs, extra, base_discarded):
        prior, retained_phi = self.retained(xs, extra)
        centered = extra-self.retained.ssl_mean.to(extra)
        components = self.retained.ssl_components.to(extra)
        residual = centered-(centered@components.T)@components
        offset = base_discarded.shape[1]
        normalized = ((residual-self.discarded_center[offset:].to(extra))/
                      self.discarded_scale[offset:].to(extra))
        complement = torch.cat((base_discarded, normalized), dim=1)
        phi = torch.cat((retained_phi[:, :-1]/math.sqrt(2.),
                         complement/math.sqrt(2.*complement.shape[1]),
                         retained_phi[:, -1:]), dim=1)
        return prior, phi
