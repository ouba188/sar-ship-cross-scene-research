"""Frozen-teacher targets: all candidate classes, signed and operation-specific."""

import torch
from torch.nn import functional as F

from .priors import OPERATIONS, REGIONS


def candidate_margins(logits):
    """m_c = z_c - max(z_j, j != c), preserving all leading dimensions."""
    if logits.shape[-1] < 2:
        raise ValueError("Candidate margins require at least two classes")
    top, index = logits.topk(2, dim=-1)
    classes = torch.arange(logits.shape[-1], device=logits.device)
    strongest_other = torch.where(classes == index[..., :1], top[..., 1:2], top[..., :1])
    return logits - strongest_other


@torch.no_grad()
def candidate_effects(clean, changed):
    """[B,C], [B,2,3,J,C] -> signed effects [B,2,C,3,J].

    Positive means the operation weakens that candidate's teacher margin.
    Targets include non-ground-truth candidates and use the strongest competitor
    of each image independently, including competitor switches.
    """
    if changed.ndim != 5 or changed.shape[1:3] != (2, 3):
        raise ValueError("changed must have shape [B,2,3,J,C]")
    if clean.shape != (changed.shape[0], changed.shape[-1]):
        raise ValueError("clean and changed batch/class dimensions must agree")
    difference = candidate_margins(clean)[:, None, None, None] - candidate_margins(changed)
    return difference.movedim(-1, 2)


@torch.no_grad()
def effects_from_legacy(clean, intervention_scores, conditions, eligible=None,
                        operation_names=OPERATIONS):
    """Map legacy [action,batch,class] full logits by named operation/region.

    conditions entries are (polarization, region, operation). Integer regions
    mean the legacy body/ends/side order. Missing combinations get zero mask;
    this never consumes old positive-clipped gains or best-action selections.
    Existing all-port caches remain development data, not clean LOPO targets.
    """
    if len(conditions) != len(intervention_scores):
        raise ValueError("Each intervention score needs its condition")
    batch, classes = clean.shape
    changed = clean[:, None, None, None].expand(batch, 2, 3, len(operation_names), classes).clone()
    mask = clean.new_zeros(batch, 2, 3, len(operation_names))
    seen = set()
    for action, (polarization, region, operation) in enumerate(conditions):
        p = ("VV", "VH").index(polarization) if isinstance(polarization, str) else int(polarization)
        r = REGIONS.index(region) if isinstance(region, str) else int(region)
        j = operation_names.index(operation)
        if p not in (0, 1) or r not in (0, 1, 2) or (p, r, j) in seen:
            raise ValueError("Invalid or duplicate intervention condition")
        seen.add((p, r, j))
        changed[:, p, r, j] = intervention_scores[action]
        mask[:, p, r, j] = 1 if eligible is None else eligible[action].to(mask)
    return {"effects": candidate_effects(clean, changed), "effect_mask": mask}


def _weighted_mean(value, weight):
    weight = weight.detach().to(value).expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1e-12)


def distillation_losses(pred, target_attention, target_effects, attention_mask, effect_mask,
                        effect_target="frequency"):
    """Independent weak-prior KL and explicitly typed signed-response Huber terms.

    attention_mask [B,2,C]; effect_mask [B,2,8] or [B,2,3,J] follows target type.
    Callers combine source availability and intervention validity into these
    masks; teacher correctness is not a universal effect-supervision filter.
    Empty masks yield graph-connected zero losses. Targets never receive grads.
    """
    attention = pred["attention"].float()
    target = target_attention.detach().float()
    target_mass = target.sum((-1, -2), keepdim=True)
    target = target / target_mass.clamp_min(1e-20)
    kl = (target * (target.clamp_min(1e-20).log() - attention.clamp_min(1e-20).log())).sum((-1, -2))
    active = (target_mass[..., 0, 0] > 0) & (attention.sum((-1, -2)) > 0)
    prior = _weighted_mean(kl, attention_mask * active)
    if effect_target not in ("frequency", "utility"):
        raise ValueError("effect_target must be frequency or utility")
    key = "utility" if effect_target == "utility" else "effects"
    if pred[key].shape != target_effects.shape:
        raise ValueError(f"{key} target shape differs from the student; frequency targets are not utility targets")
    expected_mask = target_effects.shape[:2] + target_effects.shape[3:]
    if effect_mask.shape != expected_mask:
        raise ValueError(f"{key} mask shape must be {tuple(expected_mask)}")
    huber = F.huber_loss(pred[key].float(), target_effects.detach().float(), reduction="none")
    effect = _weighted_mean(huber, effect_mask[:, :, None])
    return {"prior_kl": prior, "utility_huber" if key == "utility" else "effect_huber": effect}
