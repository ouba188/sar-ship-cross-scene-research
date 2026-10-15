"""Directional embedding drift along fixed true-versus-rival head directions."""

import torch


def directional_rows(
    anchors: torch.Tensor,
    donors: torch.Tensor,
    head_weight: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return (N,) masked mean squared projections for (N, D) embeddings.

    ``head_weight`` is (C, D), ``labels`` is long (N,), and ``mask`` is
    bool (N, C). True classes are excluded without mutating the input mask.
    Empty rows and identical head-row directions contribute zero. Gradients
    flow to both embeddings, but never to the head through this helper.
    """
    weight = head_weight.detach()
    directions = weight[labels, None, :] - weight[None, :, :]
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    directions = directions / norms.clamp_min(torch.finfo(weight.dtype).tiny)
    active = mask & (
        torch.arange(weight.shape[0], device=labels.device)[None, :]
        != labels[:, None]
    )
    projections = torch.einsum('nd,ncd->nc', anchors - donors, directions)
    squared = projections.square().masked_fill(~active, 0)
    return squared.sum(dim=1) / active.sum(dim=1).clamp_min(1)
