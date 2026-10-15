"""Learn explicit content/residual evidence from observed same-vessel changes."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from paired_evidence_factors import acquisition_table, row_weights, class_signal, fit_probe, score_probe, support_operator, adapted_scores
from crossfit_evidence import class_contrasts


def acquisition_pairs(values, meta):
    """Each class, ship, and distinct-product pair gets equal nested weight."""
    x, table = acquisition_table(values, meta)
    pairs, records = [], []
    for ship, group in table.groupby('mmsi', sort=True):
        selected = [(a, b) for a, b in itertools.combinations(group.index, 2)
                    if table.loc[a, 'product'] != table.loc[b, 'product']]
        for a, b in selected:
            pairs.append((a, b))
            records.append(dict(mmsi=ship, class_id=int(group.class_id.iloc[0]),
                                cross_port=bool(table.loc[a, 'port'] != table.loc[b, 'port'])))
    info = pd.DataFrame(records)
    weights = np.zeros(len(info))
    for _, group in info.groupby('class_id', sort=True):
        for _, ship in group.groupby('mmsi', sort=True):
            weights[ship.index] = 1/(info.class_id.nunique()*group.mmsi.nunique()*len(ship))
    a, b = np.asarray(pairs).T
    return x[a], x[b], weights, info


class EvidenceDecomposition(nn.Module):
    """Orthogonal output components avoid encoder/decoder scale cancellation.

    A nonlinear coordinate residual permits imperfect reconstruction in exchange
    for task-useful separation. Both components remain available, never deleted.
    """
    def __init__(self, source=None, meta=None, rank=7):
        super().__init__()
        if source is None:
            d, q = 32, np.eye(32)
        else:
            between, _, _ = class_signal(source, meta)
            _, q = np.linalg.eigh(between)
            q = q[:, ::-1].copy()
            d = source.shape[1]
        self.rank = rank
        self.frame = nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.offset = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.head = nn.Linear(rank, 8)
        if source is not None:
            coef = fit_probe(source@q[:, :rank], meta.class_id.to_numpy(), row_weights(meta))
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor(coef[:-1].T, dtype=torch.float32))
                self.head.bias.copy_(torch.tensor(coef[-1], dtype=torch.float32))

    def forward(self, z):
        q, r = torch.linalg.qr(self.frame)
        q = q*torch.where(r.diagonal() < 0, -1., 1.)
        coordinates = z@q+self.offset(z)
        c, residual = coordinates[:, :self.rank], coordinates[:, self.rank:]
        content_component = c@q[:, :self.rank].T
        residual_component = residual@q[:, self.rank:].T
        return dict(content=c, residual=residual, content_component=content_component,
                    residual_component=residual_component, scores=self.head(c))


def decomposition_losses(model, x, labels, weights, pair_a, pair_b, pair_weights):
    out = model(x)
    target = 2*torch.eye(8)[labels]-1
    task = (weights*(out['scores']-target).square().mean(1)).sum()
    reconstruction = (weights*(out['content_component']+out['residual_component']-x).square().mean(1)).sum()
    a, b = model(pair_a), model(pair_b)
    delta = pair_a-pair_b
    denominator = (pair_weights*delta.square().mean(1)).sum().clamp_min(1e-8)
    residual_delta = a['residual_component']-b['residual_component']
    change = (pair_weights*(residual_delta-delta).square().mean(1)).sum()/denominator
    return task, reconstruction, change


def train_decomposition(z, meta, fitting, validation, seed, arm, steps=400, every=20):
    """Fit only source rows; choose content readability on new-ship validation."""
    fm = meta.iloc[fitting].reset_index(drop=True)
    if set(fm.mmsi)&set(meta.iloc[validation].mmsi):
        raise ValueError('Decomposition validation must contain unseen ships')
    weights = row_weights(fm)
    pa, pb, pw, coverage = acquisition_pairs(z[fitting], fm)
    torch.manual_seed(seed)
    model = EvidenceDecomposition(z[fitting], fm)
    x = torch.tensor(z[fitting], dtype=torch.float32)
    y = torch.tensor(fm.class_id.to_numpy())
    w = torch.tensor(weights, dtype=torch.float32)
    pair_args = [torch.tensor(t, dtype=torch.float32) for t in (pa, pb, pw)]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, best_score, selected, trace = None, -np.inf, 0, []
    labels = meta.class_id.to_numpy()

    def evaluate(step, losses=None):
        nonlocal best, best_score, selected
        ids = np.r_[fitting, validation]
        with torch.no_grad():
            c = model(torch.tensor(z[ids], dtype=torch.float32))['content'].numpy()
        coef = fit_probe(c[:len(fitting)], labels[fitting], weights)
        pred = score_probe(c[len(fitting):], coef).argmax(1)
        ba = 100*np.mean([np.mean(pred[labels[validation] == k] == k) for k in np.unique(labels[validation])])
        trace.append(dict(step=step, validation_BA=float(ba), **(losses or {})))
        if ba > best_score+1e-10:
            best, best_score, selected = copy.deepcopy(model.state_dict()), float(ba), step

    evaluate(0)
    for step in range(1, steps+1):
        task, reconstruction, change = decomposition_losses(model, x, y, w, *pair_args)
        loss = task+.25*reconstruction+(.25*change if arm == 'paired' else 0)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step%every == 0 or step == steps:
            evaluate(step, dict(task=float(task.detach()), reconstruction=float(reconstruction.detach()),
                                change=float(change.detach()), loss=float(loss.detach())))
    state = dict(selected=best, last=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        output = {key: value.numpy() for key, value in model(torch.tensor(z, dtype=torch.float32)).items()}
    return model, state, output, trace, dict(step=selected, validation_BA=best_score), coverage


def pair_matrix(classes=8):
    pairs = list(itertools.combinations(range(classes), 2))
    matrix = np.zeros((classes, len(pairs)), np.float32)
    for j, (a, b) in enumerate(pairs):
        matrix[a, j], matrix[b, j] = 1, -1
    return matrix


def relation_features(features, prior):
    """Only features and predicted scores; no query labels or query identities."""
    matrix = pair_matrix()
    class_evidence = np.asarray(features)@class_contrasts().T
    return np.tanh((class_evidence@matrix)/2)/(1+np.abs(np.asarray(prior)@matrix))


def relation_delta(features, prior, coefficient):
    # At most .5 absolute score change per class; every relation sums to zero.
    return (.5/7)*(relation_features(features, prior)*np.tanh(coefficient))@pair_matrix().T


def relation_episode_loss(coefficient, fs, fq, prior_s, prior_q, operator, present, ys, yq, objective='mse'):
    matrix = torch.tensor(pair_matrix())
    ds = (.5/7)*(fs*torch.tanh(coefficient))@matrix.T
    dq = (.5/7)*(fq*torch.tanh(coefficient))@matrix.T
    target_s = 2*torch.eye(8)[ys]-1
    baseline = prior_q+operator@((target_s-prior_s)*present)
    adapted = baseline+dq-operator@(ds*present)
    target = 2*torch.eye(8)[yq]-1
    count = torch.bincount(yq, minlength=8)
    w = len(yq)/((count > 0).sum()*count[yq].float())
    if objective == 'ce':
        task = (w*torch.nn.functional.cross_entropy(adapted, yq, reduction='none')).mean()
    else:
        task = (w*(adapted-target).square().mean(1)).mean()
    correct = baseline.argmax(1) == yq
    rival = adapted.masked_fill(torch.nn.functional.one_hot(yq, 8).bool(), -torch.inf).max(1).values
    margin = adapted[torch.arange(len(yq)), yq]-rival
    # Penalize crossing a small correct-decision margin, not all confidence change.
    protection = (w*correct*torch.relu(.05-margin).square()).mean()
    return task+.25*protection+.001*coefficient.square().mean(), adapted, baseline


def train_relation_correction(features, prior, labels, kernel, episodes, objective='mse'):
    design = relation_features(features, prior)
    coefficient = nn.Parameter(torch.zeros(28))
    optimizer = torch.optim.Adam([coefficient], lr=.02)
    trace = []
    for step, (s, q) in enumerate(episodes, 1):
        op, present = support_operator(kernel[s], kernel[q], labels[s])
        fs, fq, ps, pq, op, present = [torch.tensor(a, dtype=torch.float32) for a in
                                      (design[s], design[q], prior[s], prior[q], op, present)]
        ys, yq = torch.tensor(labels[s]), torch.tensor(labels[q])
        loss, _, _ = relation_episode_loss(coefficient, fs, fq, ps, pq, op, present, ys, yq, objective=objective)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step%20 == 0:
            trace.append(dict(step=step, loss=float(loss.detach())))
    return coefficient.detach().numpy(), trace
