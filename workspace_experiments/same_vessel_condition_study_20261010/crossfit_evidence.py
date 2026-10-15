"""Grouped out-of-fold evidence learning with fold-local upstream estimators."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from explicit_candidate_projection import CandidateProjector, ridge_probe, identity_episode, balanced_mse
from relative_content_condition import vessel_targets
from scene_training import sample_weights
from train_explicit_candidate import phi, metrics
from target_response_adapter import residual_head
from paired_evidence_factors import separate, row_weights, fit_probe, score_probe


def ship_folds(meta, seed=20261011, nfolds=3):
    """Fixed class-stratified group assignment; all observations of a ship stay together."""
    meta = meta.reset_index(drop=True)
    ships = meta.groupby('mmsi', sort=True).class_id.agg(['first', 'nunique'])
    if (ships['nunique'] != 1).any():
        raise ValueError('Conflicting vessel class labels')
    rng = np.random.default_rng(seed)
    assignment = {}
    for label, group in ships.groupby('first', sort=True):
        ids = rng.permutation(group.index.to_numpy())
        assignment.update({ship: int((j+label)%nfolds) for j, ship in enumerate(ids)})
    return np.array([assignment[v] for v in meta.mmsi])


def assert_excluded(meta, fitting, validation, excluded):
    excluded_ships = set(meta.iloc[excluded].mmsi)
    if excluded_ships & set(meta.iloc[np.r_[fitting, validation]].mmsi):
        raise ValueError('OOF identities entered fitting or checkpoint selection')


def class_contrasts(classes=8):
    """An orthonormal, fixed, class-indexed basis, not fold-dependent latent axes."""
    matrix = np.zeros((classes, classes-1))
    for j in range(classes-1):
        matrix[:j+1, j] = 1/np.sqrt((j+1)*(j+2))
        matrix[j+1, j] = -(j+1)/np.sqrt((j+1)*(j+2))
    return matrix


def semantic_features(all_factors, fit, labels, weights):
    """Convert arbitrary factor bases to class-score coordinates with source-fit RMS."""
    coefficient = fit_probe(all_factors[fit], labels[fit], weights)
    contrast = class_contrasts()
    evidence = score_probe(all_factors, coefficient)@contrast
    rms = float(np.sqrt(np.sum(weights*np.square(evidence[fit]).sum(1))))
    if rms < 1e-10:
        raise ValueError('No readable class evidence in candidate')
    return evidence/rms, dict(probe=coefficient, contrasts=contrast, rms=rms)


def correction_episodes(meta, assignment, seed, count=400):
    result = []
    for step in range(count):
        fold = step%3
        ids = np.flatnonzero(assignment == fold)
        local = meta.iloc[ids].reset_index(drop=True)
        s, q = identity_episode(local, np.random.default_rng(seed+70000+step))
        result.append((ids[s], ids[q]))
    return result


def assemble_training_view(bundles, fitting, assignment, method, mode):
    """Stitch fold outputs only for episodes that stay within a single fold."""
    if mode == 'full_in':
        b = bundles['full']
        return b['features'][method][fitting], b['prior'][fitting], b['kernel'][fitting]
    features = np.empty((len(fitting), 7), np.float64)
    prior, kernel = np.empty((len(fitting), 8), np.float64), np.empty((len(fitting), 32), np.float64)
    for fold in range(3):
        rows = np.flatnonzero(assignment == fold)
        owner = fold if mode == 'crossfit' else (fold+1)%3
        b = bundles[f'fold{owner}']
        ids = fitting[rows]
        if mode == 'crossfit':
            assert not np.intersect1d(ids, b['fitting']).size
        else:
            assert np.isin(ids, b['fitting']).all()
        features[rows] = b['features'][method][ids]
        prior[rows], kernel[rows] = b['prior'][ids], b['kernel'][ids]
    return features, prior, kernel


def train_stage(initial, x, labels, meta, fitting, validation, teacher, valid,
                seed, kernel=None, steps=400, every=20):
    """The previous ordinary objective, without unused graph/stability branches."""
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    fm = meta.iloc[fitting].reset_index(drop=True)
    probability = sample_weights(fm); probability /= probability.sum()
    best, best_score, selected = copy.deepcopy(model.state_dict()), -np.inf, 0
    trace = []

    def evaluate(step, losses=None):
        nonlocal best, best_score, selected
        # Selection uses ONLY original new-ship validation; new-time may share
        # an OOF identity and therefore cannot choose a fold's checkpoint.
        bas = []
        with torch.no_grad():
            val_ids = np.unique(np.concatenate([np.r_[s, q] for s, q in validation]))
            z, p = model(x[val_ids])
            positions = {int(v): j for j, v in enumerate(val_ids)}
            design = phi(z if kernel is None else kernel[val_ids])
            for support, query in validation:
                s, q = [np.array([positions[int(v)] for v in ids]) for ids in (support, query)]
                beta, _, _ = residual_head(design[s], p[s], labels[support], balanced=True)
                score = p[q]+design[q]@beta
                bas.append(metrics(score.argmax(1).numpy(), labels[query].numpy())['BA'])
        value = float(np.mean(bas))
        trace.append(dict(step=step, validation_BA=value, **(losses or {})))
        if value > best_score+1e-10:
            best, best_score, selected = copy.deepcopy(model.state_dict()), value, step

    evaluate(0)
    for step in range(1, steps+1):
        ids = np.random.default_rng(seed+step).choice(len(fitting), 128, replace=True, p=probability)
        s, q = identity_episode(fm, np.random.default_rng(seed+30000+step))
        inputs = x[fitting[np.r_[ids, s, q]]]
        z, score = model(inputs)
        zs, zq = z[128:192], z[192:]
        ps, pq = score[128:192], score[192:]
        ks, kq = (zs, zq) if kernel is None else (kernel[fitting[s]], kernel[fitting[q]])
        beta, _, _ = residual_head(phi(ks), ps, labels[fitting[s]], balanced=True)
        adapted = pq+phi(kq)@beta
        supervised = F.mse_loss(score[:128], 2*F.one_hot(labels[fitting[ids]], 8).float()-1)
        mask = valid[ids]
        if mask.any():
            student = score[:128][mask]; target = teacher[ids][mask]
            distillation = F.mse_loss(student-student.mean(1, keepdim=True), target-target.mean(1, keepdim=True))
        else:
            distillation = score.sum()*0
        episode = balanced_mse(adapted, labels[fitting[q]])
        loss = supervised+.5*distillation+.5*episode
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step%every == 0 or step == steps:
            evaluate(step, dict(loss=float(loss.detach()), supervised=float(supervised.detach()),
                                teacher_loss=float(distillation.detach()), episode_loss=float(episode.detach())))
    state = dict(selected=best, last=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best)
    return model.eval(), state, trace, dict(step=selected, validation_BA=best_score)


def fit_pipeline(raw, meta, fitting, validation, excluded, seed, steps=400, every=20):
    """Refit normalizer, teacher, case projection, score projection and head locally."""
    validation_ids = np.unique(np.concatenate([np.r_[s, q] for s, q in validation]))
    assert_excluded(meta, fitting, validation_ids, excluded)
    fm = meta.iloc[fitting].reset_index(drop=True)
    probability = sample_weights(fm); probability /= probability.sum()
    center = raw[fitting].mean(0)
    scale = np.maximum(raw[fitting].std(0), 1e-5)
    x = torch.tensor((raw-center)/scale, dtype=torch.float32)
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    teacher_coef = ridge_probe(x[fitting]/np.sqrt(x.shape[1]), labels[fitting], probability)
    teacher_prior = (phi(x[fitting]/np.sqrt(x.shape[1]))@teacher_coef).numpy()
    _, teacher, valid = vessel_targets(teacher_prior, fm)
    teacher, valid = torch.tensor(teacher), torch.tensor(valid)
    torch.manual_seed(seed)
    initial = CandidateProjector(x[fitting], labels[fitting], probability)
    case, case_state, case_trace, case_choice = train_stage(initial, x, labels, meta, fitting,
        validation, teacher, valid, seed, steps=steps, every=every)
    with torch.no_grad():
        kernel = case.encode(x).detach()
    prior, prior_state, prior_trace, prior_choice = train_stage(initial, x, labels, meta, fitting,
        validation, teacher, valid, seed, kernel=kernel, steps=steps, every=every)
    with torch.no_grad():
        z, score = prior(x)
    bundle = dict(z=z.numpy(), prior=score.numpy(), kernel=kernel.numpy(),
                  head_weight=prior.head.weight.detach().numpy(), raw_center=center, raw_scale=scale,
                  teacher_coef=teacher_coef.numpy(), fitting=fitting.copy(), excluded=excluded.copy(),
                  validation_ids=validation_ids, case_state=case_state, prior_state=prior_state,
                  case_trace=case_trace, prior_trace=prior_trace,
                  case_choice=case_choice, prior_choice=prior_choice)
    return bundle


def attach_factors(bundle, meta):
    fit = bundle['fitting']; fm = meta.iloc[fit].reset_index(drop=True)
    weights = row_weights(fm)
    factor = separate(bundle['z'][fit], fm, bundle['head_weight'])
    u = (bundle['z']-factor['center'])@factor['whitener']
    features, maps = {}, {}
    labels = meta.class_id.to_numpy()
    for method in ('paired', 'class_only'):
        q = factor['bases'][method]
        latent = u@q/np.sqrt(7)
        features[method], reading = semantic_features(latent, fit, labels, weights)
        maps[method] = dict(center=factor['center'], whitener=factor['whitener'], q=q, **reading)
    bundle['features'], bundle['factor_maps'] = features, maps
    bundle['pair_coverage'] = factor['coverage']
    return bundle
