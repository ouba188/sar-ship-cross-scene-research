"""Explicit optimizer boundaries for Stage1 -> frozen encoder/G -> h0 -> H."""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .data import background_targets, intervene_background
from .stage1_losses import (classification_loss, scene_regression_loss,
                            content_invariance_loss, discriminator_loss, leakage_loss)


@dataclass
class LossConfig:
    scene: float = 1.
    invariance: float = 1.
    leakage: float = .1
    hard_gate_after: int = 200
    warmup_steps: int = 50
    ramp_steps: int = 150


def stage1_optimizers(model, lr=1e-4, weight_decay=.01):
    discriminator = set(model.discriminator.parameters())
    main = [p for p in model.parameters() if p not in discriminator]
    return (torch.optim.AdamW(main, lr=lr, weight_decay=weight_decay),
            torch.optim.AdamW(model.discriminator.parameters(), lr=lr, weight_decay=weight_decay))


def stage1_step(model, batch, labels, optimizers, prior, scene_mean, scene_std,
                controlled_views, config=None, step=0, generator=None):
    """controlled_views certifies class-independent nuisance/mask/position/size design.

    Natural Q replacement alone must NOT set this flag. The caller supplies an
    audited controlled-view manifest; absent flags disable the leakage update.
    """
    config = config or LossConfig()
    model.train()
    main, disc = optimizers
    changed, paired = intervene_background(batch, generator)
    hard = step >= config.hard_gate_after
    cpu_rng = torch.get_rng_state() if model.config.stochastic_depth else None
    cuda_rng = (torch.cuda.get_rng_state(batch['images'].device)
                if cpu_rng is not None and batch['images'].is_cuda else None)
    original = model(batch['images'], batch['valid'], hard_gate=hard)
    if cpu_rng is not None:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, batch['images'].device)
    intervened = model(changed, batch['valid'], hard_gate=hard)
    legal = paired & controlled_views.to(paired.device)
    ramp = min(1., max(0., (step-config.warmup_steps+1)/max(config.ramp_steps, 1)))
    # D sees detached residuals. Its update cannot change encoder or G.
    disc.zero_grad(set_to_none=True)
    d_loss = discriminator_loss(model, intervened, labels, legal)
    if legal.any():
        d_loss.backward()
        disc.step()
    disc.zero_grad(set_to_none=True)
    main.zero_grad(set_to_none=True)
    flags = model.class_valid(batch['geometries']).to(labels.device)
    logits = torch.stack([model.class_logits(original, batch['geometries']),
                          model.class_logits(intervened, batch['geometries'])], dim=1)
    valid_views = torch.stack([flags, flags & paired[:, None]], dim=1)
    cls = classification_loss(logits, labels, valid_views)
    qweights = tuple(F.avg_pool2d(batch['background'].flatten(0, 1), s, s)
                     .unflatten(0, (len(labels), 2)) for s in (4, 8, 16))
    # Only cells completely inside trusted Q supervise the scene predictor.
    qweights = tuple((w >= 1-1e-6).to(w.dtype) for w in qweights)
    scene_values = []
    reliable_count = 0
    for images, output, use in ((batch['images'], original, torch.ones_like(paired)),
                                (changed, intervened, paired)):
        targets, reliable = background_targets(images, batch['background'])
        reliable &= use[:, None]
        reliable &= torch.stack([w.flatten(2).sum(-1) > 0 for w in qweights]).any(0)
        targets = (targets-scene_mean.to(targets))/scene_std.to(targets).clamp_min(1e-6)
        if reliable.any():
            scene_values.append(scene_regression_loss(model, output, qweights, targets, reliable))
        reliable_count += int(reliable.sum())
    scene = sum(scene_values)/len(scene_values) if scene_values else cls*0
    target_weights = tuple(torch.stack([g.levels[l].slots[:, 0] for g in batch['geometries']])
                           for l in range(3))
    inv = content_invariance_loss(original, intervened, target_weights, paired)
    leak = leakage_loss(model, intervened, prior, legal)
    total = cls + config.scene*scene + ramp*(config.invariance*inv + config.leakage*leak)
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite Stage1 objective')
    total.backward()
    main.step()
    return {**{k: float(v.detach()) for k, v in dict(total=total, cls=cls, scene=scene,
             invariance=inv, leakage=leak, discriminator=d_loss).items()},
            'paired_samples': int(paired.sum()), 'controlled_samples': int(legal.sum()),
            'scene_targets': reliable_count, 'constraint_ramp': ramp, 'hard_gate': int(hard),
            'controlled_class_counts': torch.bincount(labels[legal], minlength=len(prior)).detach().cpu().tolist()}


def base_step(h, features, labels, optimizer):
    h.set_stage('base'); h.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(h.forward_base(features), labels)
    loss.backward(); optimizer.step()
    return float(loss.detach())


def matching_step(h, features, labels, references, reference_labels, reference_ports,
                  legal, optimizer):
    h.set_stage('match'); h.train()
    optimizer.zero_grad(set_to_none=True)
    output = h(features, references, reference_labels, reference_ports, legal)
    losses = h.loss(output, labels)
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('Nonfinite H objective')
    losses['total'].backward(); optimizer.step()
    return {k: float(v.detach()) for k, v in losses.items()}
