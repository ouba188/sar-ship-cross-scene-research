"""Source-only non-regression objectives for the existing M3 writer."""
import torch
from torch.nn import functional as F


def correct_kl(score, teacher, labels):
    per_sample = F.kl_div(score.log_softmax(1), teacher.detach().log_softmax(1),
                         log_target=True, reduction='none').sum(1)
    return (per_sample * teacher.argmax(1).eq(labels)).mean()


def focal_logits(score, teacher, labels):
    weight = 1 + 5 * teacher.argmax(1).eq(labels).to(score)
    return (.5 * (score - teacher.detach()).square().sum(1) * weight).mean()


def margin_keep(score, teacher, labels):
    old = teacher.detach()
    old_gaps = old.gather(1, labels[:, None]) - old
    new_gaps = score.gather(1, labels[:, None]) - score
    decrease = F.relu(old_gaps - new_gaps)
    correct = old.argmax(1).eq(labels)
    return (.5 * decrease.square().sum(1) * correct).mean()


def objective(score, teacher, labels, arm):
    ce = F.cross_entropy(score, labels)
    if arm == 'fd_lm':
        guard = focal_logits(score, teacher, labels)
    elif arm == 'margin':
        guard = .1 * correct_kl(score, teacher, labels) + margin_keep(score, teacher, labels)
    else:
        raise ValueError(arm)
    return ce, guard
