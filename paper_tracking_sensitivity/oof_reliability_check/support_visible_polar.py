"""Put learned polar evidence in directions invisible to the source prior."""

import torch

from aligned_residual_ops import FixedExpertGeometry
from polar_residual_expert import PolarResidualExpert


class SupportVisibleGeometry(FixedExpertGeometry):
    def __init__(self, arrays):
        super().__init__(arrays)
        # SVD handles the dependent one-vs-rest class rows without ridge leakage.
        weight = self.expert_coef[:, 128:].double()
        projector = torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device)-torch.linalg.pinv(weight)@weight
        self.register_buffer('evidence_null', projector.to(self.expert_coef))

    def project_evidence(self, residual):
        components = self.ssl_components.to(residual)
        scale = self.expert_scale[128:].to(residual)
        standardized = (residual@components.T)/scale
        visible_to_support = standardized@self.evidence_null.to(residual)
        return (visible_to_support*scale)@components


class SupportVisiblePolarExpert(PolarResidualExpert):
    def __init__(self, backbone_state, mode='contrast'):
        super().__init__(backbone_state, mode)
        # Keep the original prior's VH representation as well as its classifier.
        self.layer4.requires_grad_(False)

    def fuse(self, h_map, v_map, xs, geometry, disable_contrast=False, return_details=False):
        unconstrained, details = super().fuse(h_map, v_map, xs, geometry,
            disable_contrast=disable_contrast, return_details=True)
        base = details['base_features']
        residual = geometry.project_evidence(unconstrained-base)
        features = base+residual
        if return_details:
            details['residual_norm'] = residual.norm(dim=1)
            return features, details
        return features
