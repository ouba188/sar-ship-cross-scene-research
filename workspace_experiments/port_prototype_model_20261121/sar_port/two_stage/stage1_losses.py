"""Four Stage1 responsibilities and the alternating discriminator update.

Donor selection, source-only target standardization, and legal-view masks are
provided by the runner. Natural images must not be silently treated as controlled
views for leakage confusion. No feature-swap/reconstruction objective is used.
"""
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F


def _sample_mean(values, valid):
    valid = valid.to(values.dtype)
    values = values.flatten(1)
    valid = valid.flatten(1)
    count = valid.sum(1)
    per_ship = (values * valid).sum(1) / count.clamp_min(1)
    usable = count > 0
    return per_ship[usable].mean() if usable.any() else values.sum() * 0


def classification_loss(logits, labels, valid_scales):
    """[B,L,C] or [B,V,L,C]; average views, then scales, then valid ships."""
    if logits.ndim == 3:
        logits, valid_scales = logits[:, None], valid_scales[:, None]
    b, views, levels, classes = logits.shape
    targets = labels.to(logits.device)[:, None, None].expand(b, views, levels)
    loss = F.cross_entropy(logits.reshape(-1, classes), targets.reshape(-1), reduction="none")
    loss = loss.reshape(b, views, levels)
    valid = valid_scales.to(device=logits.device, dtype=logits.dtype)
    counts = valid.sum(1)
    per_level = (loss * valid).sum(1) / counts.clamp_min(1)
    return _sample_mean(per_level, counts > 0)


def scene_regression_loss(model, output, background_weights, targets, reliable):
    """targets [B,2,2], reliable [B,2]; Q weights are pure-background coverage.

    Targets are median and q90-q10 AFTER the runner's source-only standardization.
    Both target values and qualification flags are external to the predictor.
    """
    prediction, usable = model.scene_predictions(output, background_weights)
    targets = targets.detach().to(prediction)
    valid = usable & reliable.to(device=prediction.device, dtype=torch.bool)[:, None]
    expected = targets[:, None].expand_as(prediction)
    # Unqualified target slots may be NaN; they must not poison other gradients.
    expected = torch.where(valid[..., None], expected, prediction.detach())
    loss = F.huber_loss(prediction, expected, reduction="none").mean(-1)
    return _sample_mean(loss, valid)


def content_invariance_loss(output, intervened, target_weights, paired=None):
    """Raw Fc Huber at corresponding locations, without projection/normalization.

    target_weights: three [B,2,1,H,W] (or [B,2,H,W]) geometry weights.
    paired: [B] legal intervention flags; omitted means all supplied pairs legal.
    """
    losses, eligible = [], []
    for level, (original, changed) in enumerate(zip(output["content"], intervened["content"])):
        weights = target_weights[level].to(original)
        if weights.ndim == 5:
            weights = weights.squeeze(2)
        weights = weights * (output["valid"][level][:, :, 0] > 0)
        weights = weights * (intervened["valid"][level][:, :, 0] > 0)
        area = weights.sum((-2, -1))
        local = F.huber_loss(original, changed, reduction="none").mean(2)
        losses.append((local * weights).sum((-2, -1)) / area.clamp_min(1e-12))
        eligible.append(area > 0)
    values, valid = torch.stack(losses, dim=1), torch.stack(eligible, dim=1)
    if paired is not None:
        valid = valid & paired.to(device=values.device, dtype=torch.bool)[:, None, None]
    return _sample_mean(values, valid)


def _controlled_loss(logits, values, legal_views):
    legal = legal_views.to(device=logits.device, dtype=torch.bool)
    return values[legal].mean() if legal.any() else logits.sum() * 0


def discriminator_loss(model, output, labels, legal_views):
    """D step: updates D only; caller must zero/step its dedicated optimizer."""
    logits = model.discriminator(tuple(x.detach() for x in output["residual"]), output["valid"])
    loss = F.cross_entropy(logits, labels.to(logits.device), reduction="none")
    return _controlled_loss(logits, loss, legal_views)


@contextmanager
def frozen_discriminator(module):
    """Hold D parameters/state fixed while preserving gradients to its inputs."""
    flags = [p.requires_grad for p in module.parameters()]
    modes = [(m, m.training) for m in module.modules()]
    module.requires_grad_(False)
    module.eval()
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)
        for m, mode in modes:
            m.training = mode


def leakage_loss(model, output, prior, legal_views):
    """Encoder step KL(fixed prior || D(Fs)); no D parameter gradients created.

    D gradients from its PREVIOUS step are not erased here: runner zero_grad must
    separate the two alternating updates. Prior is [C], fixed by source sampling.
    """
    with frozen_discriminator(model.discriminator):
        logits = model.discriminator(output["residual"], output["valid"])
        prior = prior.detach().to(logits)
        if prior.ndim != 1 or prior.numel() != logits.shape[1] or (prior < 0).any():
            raise ValueError("prior must be a nonnegative vector over all classes")
        if not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-6):
            raise ValueError("prior must sum to one")
        target = prior.expand_as(logits)
        loss = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="none").sum(1)
        return _controlled_loss(logits, loss, legal_views)
