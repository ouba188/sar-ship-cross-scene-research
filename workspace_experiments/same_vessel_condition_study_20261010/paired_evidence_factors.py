"""Source-only class-anchored factor separation using real cross-port repeats.

The paired covariance measures observational within-vessel changes. It does not
identify noise or a physical causal intervention. Complements retain information.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.linalg import eigh


def acquisition_table(values, meta):
    meta = meta.reset_index(drop=True)
    rows, means = [], []
    for (ship, product, port), group in meta.groupby(['mmsi', 'product', 'port'], sort=True):
        if group.class_id.nunique() != 1:
            raise ValueError('A vessel/acquisition has conflicting labels')
        rows.append(dict(mmsi=ship, product=product, port=port,
                         class_id=int(group.class_id.iloc[0]), chips=len(group)))
        means.append(np.asarray(values)[group.index].mean(0))
    return np.asarray(means, dtype=np.float64), pd.DataFrame(rows)


def row_weights(meta):
    """Equal classes, vessels within class, acquisitions within vessel, chips."""
    meta = meta.reset_index(drop=True)
    weights = np.zeros(len(meta))
    for _, cls in meta.groupby('class_id', sort=True):
        ships = list(cls.groupby('mmsi', sort=True))
        for _, ship in ships:
            cells = list(ship.groupby(['product', 'port'], sort=True))
            for _, cell in cells:
                weights[cell.index] = 1/(meta.class_id.nunique()*len(ships)*len(cells)*len(cell))
    return weights


def paired_covariance(values, meta, scope='cross_port'):
    """Equal covered classes then vessels then distinct acquisition pairs."""
    x, mm = acquisition_table(values, meta)
    d = x.shape[1]
    records, matrices = [], []
    for ship, group in mm.groupby('mmsi', sort=True):
        if group.class_id.nunique() != 1:
            raise ValueError('A vessel has conflicting class labels')
        pairs = [(i, j) for i, j in itertools.combinations(group.index, 2)
                 if mm.loc[i, 'product'] != mm.loc[j, 'product'] and
                 (scope == 'all' or (mm.loc[i, 'port'] != mm.loc[j, 'port']) == (scope == 'cross_port'))]
        if not pairs:
            continue
        a, b = np.asarray(pairs).T
        delta = x[a]-x[b]
        matrices.append(delta.T@delta/len(delta))
        records.append(dict(mmsi=ship, class_id=int(group.class_id.iloc[0]),
                            pairs=len(pairs), acquisitions=len(group)))
    coverage = pd.DataFrame(records)
    if not len(records):
        return np.full((d, d), np.nan), coverage
    matrices = np.asarray(matrices)
    result = np.mean([matrices[coverage.class_id.to_numpy() == c].mean(0)
                      for c in sorted(coverage.class_id.unique())], 0)
    return result, coverage


def class_signal(values, meta):
    x, mm = acquisition_table(values, meta)
    centroids, labels = [], []
    for label, cls in mm.groupby('class_id', sort=True):
        ship_means = [x[group.index].mean(0) for _, group in cls.groupby('mmsi', sort=True)]
        centroids.append(np.mean(ship_means, 0)); labels.append(int(label))
    centroids = np.asarray(centroids)
    between = centroids-centroids.mean(0)
    return between.T@between/len(centroids), centroids, np.asarray(labels)


def orthogonal_basis(vectors, rank):
    q, r = np.linalg.qr(vectors[:, :rank])
    q *= np.where(np.diag(r) < 0, -1., 1.)
    return q


def separate(values, meta, head_weight, rank=7):
    """Return matched subspaces in a common SOURCE-fit whitened coordinate system."""
    values = np.asarray(values, np.float64)
    w = row_weights(meta)
    center = w@values
    delta = values-center
    cov = delta.T@(w[:, None]*delta)
    eig, axes = np.linalg.eigh(cov)
    floor = max(float(eig.mean())*1e-8, 1e-12)
    safe = np.maximum(eig, floor)
    whitener = (axes/np.sqrt(safe))@axes.T
    inverse = (axes*np.sqrt(safe))@axes.T
    u = delta@whitener
    signal, centroids, labels = class_signal(u, meta)
    drift, coverage = paired_covariance(u, meta)
    if not len(coverage):
        raise ValueError('No source-fit cross-port repeated vessels')
    rank = min(rank, len(labels)-1)
    drift_scale = np.trace(drift)/len(drift)
    nuisance_metric = drift/drift_scale + .1*np.eye(len(drift))
    paired_eigen, paired = eigh(signal, nuisance_metric)
    signal_eigen, discriminative = np.linalg.eigh(signal)
    _, stable = np.linalg.eigh(drift)
    # A centered eight-way linear head has at most seven independent contrasts.
    contrasts = inverse@(head_weight-head_weight.mean(0)).T
    head_axes, head_singular, _ = np.linalg.svd(contrasts, full_matrices=False)
    bases = dict(paired=orthogonal_basis(paired[:, ::-1], rank),
                 class_only=orthogonal_basis(discriminative[:, ::-1], rank),
                 stable_only=orthogonal_basis(stable, rank),
                 pca=orthogonal_basis(axes[:, ::-1], rank),
                 head=orthogonal_basis(head_axes, rank))
    result = dict(center=center, whitener=whitener, inverse_whitener=inverse,
                  covariance=cov, signal=signal, drift=drift, drift_scale=drift_scale,
                  paired_eigenvalues=paired_eigen[::-1], signal_eigenvalues=signal_eigen[::-1],
                  head_singular_values=head_singular, class_centroids=centroids,
                  labels=labels, rank=rank, bases=bases, coverage=coverage)
    return result


def complement(q):
    return np.linalg.svd(q.T, full_matrices=True)[2][q.shape[1]:].T


def encode_raw(raw_features, mapping, part='candidate'):
    """Apply an exported mapping to frozen R18 512D features of new chips."""
    x = (np.asarray(raw_features)-mapping['old_raw_center'])/mapping['old_raw_scale']
    z = ((x-mapping['old_input_center'])@mapping['old_basis'])/mapping['old_scale']/np.sqrt(32)
    u = (z-mapping['center'])@mapping['whitener']
    q = mapping['q'] if part == 'candidate' else mapping['q_remainder']
    return u@q/np.sqrt(q.shape[1])


def factorized_scores(raw_features, mapping):
    """Exact additive score explanation; remaining information is NOT discarded."""
    candidate = encode_raw(raw_features, mapping, 'candidate')
    remaining = encode_raw(raw_features, mapping, 'remainder')
    transformed_head = mapping['inverse_whitener']@mapping['head_weight'].T
    candidate_score = candidate@(np.sqrt(candidate.shape[1])*mapping['q'].T@transformed_head)
    remaining_score = remaining@(np.sqrt(remaining.shape[1])*mapping['q_remainder'].T@transformed_head)
    intercept = mapping['center']@mapping['head_weight'].T+mapping['head_bias']
    return dict(candidate=candidate, remaining=remaining, candidate_score=candidate_score,
                remaining_score=remaining_score, intercept=intercept,
                scores=candidate_score+remaining_score+intercept)


def fit_probe(values, labels, weights, ridge=1.):
    x = np.c_[values, np.ones(len(values))]
    w = np.asarray(weights)/np.mean(weights)
    target = 2*np.eye(8)[labels]-1
    penalty = np.eye(x.shape[1])*ridge; penalty[-1, -1] = 0
    return np.linalg.solve(x.T@(w[:, None]*x)+penalty, x.T@(w[:, None]*target))


def score_probe(values, coef):
    return np.c_[values, np.ones(len(values))]@coef


def support_operator(kernel_support, kernel_query, labels, ridge=8.):
    """Original balanced residual operator including its unbalanced intercept."""
    ks = np.c_[kernel_support, np.ones(len(kernel_support))]
    kq = np.c_[kernel_query, np.ones(len(kernel_query))]
    counts = np.bincount(labels, minlength=8)
    present = counts > 0
    weights = len(labels)/(present.sum()*counts[labels])
    root = np.sqrt(weights)
    weighted = ks*root[:, None]
    inv = np.linalg.solve(weighted@weighted.T+ridge*np.eye(len(ks)), np.eye(len(ks)))
    operator = (kq@ks.T)*(root[None, :])
    operator = (operator@inv)*root[None, :]
    return operator, present


def adapted_scores(prior_s, prior_q, operator, present, labels):
    target = 2*np.eye(8)[labels]-1
    return prior_q + operator@((target-prior_s)*present[None, :])


def fit_episode_correction(factors, prior, labels, episodes, kernel, ridge=8.):
    """Exact least-squares score correction through a fixed support readout.

    There is one per-class operator because absent support classes have no
    residual correction in the existing implementation. No query target labels
    enter; episodes here are exclusively the old source-training episodes.
    """
    design = np.c_[factors, np.ones(len(factors))]
    d = design.shape[1]
    lhs, rhs = np.zeros((8, d, d)), np.zeros((8, d))
    for s, q in episodes:
        operator, present = support_operator(kernel[s], kernel[q], labels[s])
        baseline = adapted_scores(prior[s], prior[q], operator, present, labels[s])
        target = 2*np.eye(8)[labels[q]]-1
        counts = np.bincount(labels[q], minlength=8)
        weight = len(q)/((counts > 0).sum()*counts[labels[q]])
        projected = operator@design[s]
        for c in range(8):
            effect = design[q]-projected*present[c]
            lhs[c] += effect.T@(weight[:, None]*effect)
            rhs[c] += effect.T@(weight*(target[:, c]-baseline[:, c]))
    penalty = np.eye(d)*ridge
    coef = np.stack([np.linalg.solve(lhs[c]+penalty, rhs[c]) for c in range(8)], axis=1)
    return coef
