"""Source module fitting and intervention-selected local backbone adaptation."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import assert_source, batch, pair_mask, prepare_record
from .model import LIVE, forward_batch, reference_bank, save_model, training_policy


def margin_floor(student, original, labels, weights, useful, altered=None):
    def margins(z):
        return z.gather(1, labels[:, None])-z
    floor = margins(original.detach())
    if altered is not None:
        floor = torch.maximum(floor, margins(altered.detach()))
    error = (floor-margins(student)).relu().square().sum(1)/(student.shape[1]-1)
    return (weights*useful*error).sum()/weights.sum()


def adaptation_loss(score, labels, teacher, weights, useful, coefficient):
    ce = (weights*F.cross_entropy(score, labels, reduction='none')).sum()/weights.sum()
    correct = teacher.argmax(1).eq(labels)
    kl = F.kl_div(score.log_softmax(1), teacher.detach().log_softmax(1),
                  reduction='none', log_target=True).sum(1)
    keep = (weights*correct*kl).sum()/weights.sum()
    floor = margin_floor(score, teacher, labels, weights, useful)
    return ce+.1*keep+coefficient*floor, dict(ce=ce, keep=keep, floor=floor)


def fitting_loss(score, detail, labels, weights, reference_labels):
    ce = (F.cross_entropy(score, labels, reduction='none')*weights).mean()
    aux = torch.stack([(F.cross_entropy(x, labels, reduction='none')*weights).mean()
                       for x in detail['aux_logits']]).mean()
    relation, active = [], []
    for logits, valid in detail['relations']:
        same = labels[:, None].eq(reference_labels[None]) & valid
        exists = same.any(1) & valid.any(1)
        logp = logits.masked_fill(~valid, -1e4).log_softmax(1)
        relation.append(-(logp*same).sum(1)/same.sum(1).clamp_min(1)*exists)
        active.append(exists)
    rel = (torch.stack(relation).sum(0)/torch.stack(active).sum(0).clamp_min(1)*weights).mean()
    return ce+.1*aux+.1*rel, dict(ce=ce, auxiliary=aux, relation=rel)


def class_weights(records, stage):
    counts = Counter(r['label'] for r in records)
    if stage == 'adapt':
        return np.array([len(records)/(len(counts)*counts[r['label']]) for r in records], np.float32)
    vessels = defaultdict(set)
    observations = Counter(str(r['mmsi']) for r in records)
    for r in records:
        vessels[r['label']].add(str(r['mmsi']))
    return np.array([len(records)/(8*len(vessels[r['label']])*observations[str(r['mmsi'])])
                     for r in records], np.float32)


def train(model, records, references, output, device, *, stage='adapt', selection=None,
          seed=17, epochs=None, batch_size=None, coefficient=.3683066666126251,
          max_steps=None):
    assert_source(records)
    required = 'source_fit' if stage == 'fit' else 'source_adapt'
    if any(r['split'] != required for r in records):
        raise ValueError(f'{stage} requires the {required} manifest.')
    training_policy(model, stage)
    epochs = epochs or (8 if stage == 'fit' else 6)
    batch_size = batch_size or (24 if stage == 'fit' else 16)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bank = reference_bank(model, references, device)
    weights = class_weights(records, stage)
    targets = None
    if stage == 'adapt':
        if selection is None:
            raise ValueError('Adaptation requires source-only intervention selection.')
        lookup = {str(r['id']): r for r in selection}
        targets = [lookup[str(r['id'])] for r in records]
        if any(r['label'] != t['label'] for r, t in zip(records, targets)):
            raise ValueError('Selection labels do not match source records.')
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [dict(params=[p for n, p in parameters if n in LIVE], lr=3e-5)]
    if stage == 'fit':
        groups.append(dict(params=[p for n, p in parameters if n not in LIVE], lr=.001))
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    rng = np.random.default_rng(seed)
    ref_labels = torch.tensor([r['label'] for r in references], device=device)
    curve, steps = [], 0
    for epoch in range(1, epochs+1):
        terms_epoch = []
        for indices in np.array_split(rng.permutation(len(records)),
                                      range(batch_size, len(records), batch_size)):
            rows = [records[i] for i in indices]
            values = batch([prepare_record(r) for r in rows], device)
            legal = pair_mask(rows, references, device)
            labels = torch.tensor([r['label'] for r in rows], device=device)
            w = torch.tensor(weights[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            if stage == 'fit':
                score, detail = forward_batch(model, values, bank, legal, True)
                loss, terms = fitting_loss(score, detail, labels, w, ref_labels)
            else:
                score = forward_batch(model, values, bank, legal)
                teacher = torch.tensor([targets[i]['original'] for i in indices], device=device)
                useful = torch.tensor([targets[i]['useful'] for i in indices], device=device)
                if steps == 0 and float((score.detach()-teacher).abs().max()) > 1e-4:
                    raise ValueError('Selection was produced by a different teacher/reference/input pipeline.')
                loss, terms = adaptation_loss(score, labels, teacher, w, useful, coefficient)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            steps += 1
            terms_epoch.append({k: float(v.detach()) for k, v in terms.items()})
            if max_steps is not None and steps >= max_steps:
                break
        summary = dict(epoch=epoch, steps=steps,
            **{k: float(np.mean([x[k] for x in terms_epoch])) for k in terms_epoch[0]})
        curve.append(summary)
        print(json.dumps(summary), flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    receipt = dict(stage=stage, source_count=len(records), reference_count=len(references),
        selected_count=None if targets is None else sum(t['useful'] for t in targets),
        seed=seed, completed_epochs=len(curve), steps=steps, coefficient=coefficient if stage=='adapt' else None,
        trainable_parameters=sum(p.numel() for _, p in parameters), curve=curve,
        smoke_only=max_steps is not None, validation_selection=False)
    save_model(model, output/'model.pt', **receipt)
    (output/'training.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt
