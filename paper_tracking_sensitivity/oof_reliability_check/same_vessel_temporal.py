"""Source-only same-vessel temporal diagnostics in reduced coordinates.

Temporal directions are candidate nuisance modes, not identified pure noise.
"""

import numpy as np
import pandas as pd


OBS_COLUMNS = ['port', 'mmsi', 'class_id', 'acq_time', 'row_count',
               'first_row', 'product']
PAIR_COLUMNS = ['port', 'mmsi', 'class_id', 'obs_a', 'obs_b', 'gap_days']


def collect_observations(frame, rows, z):
    """Aggregate eligible source rows, with z aligned to the supplied row IDs.

    Mixed-label MMSIs are excluded using all supplied rows, including rows with
    unparseable timestamps. No other frame rows or labels are inspected.
    First row/product means first in supplied order. Output is time-sorted.
    """
    rows = np.asarray(rows)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(rows) != len(z):
        raise ValueError('Expected z (N,r) aligned with rows (N,)')
    source = frame.loc[rows, ['port', 'mmsi', 'class_id', 'product']].copy()
    source['first_row'] = rows
    source = source.reset_index(drop=True)
    mixed = source.groupby('mmsi')['class_id'].transform('nunique') > 1
    valid_mmsi = (source['mmsi'].str.fullmatch(r'[0-9]{9}', na=False)
                  & source['mmsi'].ne('000000000'))
    timestamp = source['product'].str.extract(r'_([0-9]{8}T[0-9]{6})_',
                                               expand=False)
    source['acq_time'] = pd.to_datetime(timestamp, format='%Y%m%dT%H%M%S',
                                        errors='coerce')
    source = source.loc[valid_mmsi & ~mixed & source['acq_time'].notna()]
    records, coords = [], []
    for (port, mmsi, when), group in source.groupby(
            ['port', 'mmsi', 'acq_time'], sort=False):
        first = group.iloc[0]
        records.append((port, mmsi, first['class_id'], when, len(group),
                        first['first_row'], first['product']))
        coords.append(z[group.index.to_numpy()].mean(axis=0))
    obs = pd.DataFrame(records, columns=OBS_COLUMNS)
    obs['acq_time'] = pd.to_datetime(obs['acq_time'])
    if not records:
        return obs, np.empty((0, z.shape[1]), dtype=np.float64)
    obs = obs.sort_values(['acq_time', 'port', 'mmsi'], kind='stable')
    zobs = np.asarray(coords)[obs.index.to_numpy()]
    return obs.reset_index(drop=True), zobs


def make_pairs(obs, zobs):
    """Pair chronological adjacent acquisitions within each port and MMSI.

    Weights balance represented classes, ports per class, ships per class-port,
    and pairs per ship. Singleton ships contribute no pairs or weight.
    obs and zobs follow collect_observations' unique-acquisition contract.
    """
    zobs = np.asarray(zobs, dtype=np.float64)
    records = []
    ordered = obs.reset_index(drop=True).sort_values('acq_time', kind='stable')
    for (port, mmsi), group in ordered.groupby(['port', 'mmsi'], sort=True):
        indices = group.index.to_numpy()
        for a, b in zip(indices[:-1], indices[1:]):
            gap = (ordered.loc[b, 'acq_time'] - ordered.loc[a, 'acq_time'])
            records.append((port, mmsi, group['class_id'].iloc[0], a, b,
                            gap.total_seconds() / 86400.0))
    pairs = pd.DataFrame(records, columns=PAIR_COLUMNS)
    if not records:
        return (pairs, np.empty((0, zobs.shape[1]), dtype=np.float64),
                np.empty(0, dtype=np.float64))
    deltas = zobs[pairs['obs_b'].to_numpy()] - zobs[pairs['obs_a'].to_numpy()]
    nclasses = pairs['class_id'].nunique()
    nports = pairs.groupby('class_id')['port'].transform('nunique')
    nships = pairs.groupby(['class_id', 'port'])['mmsi'].transform('nunique')
    npairs = pairs.groupby(['class_id', 'port', 'mmsi'])['obs_a'].transform('size')
    weights = 1.0 / (nclasses * nports.to_numpy(dtype=float)
                     * nships.to_numpy(dtype=float) * npairs.to_numpy(dtype=float))
    return pairs, deltas, weights


def pair_covariance(deltas, weights):
    """Return half the weighted second moment, without mean centering."""
    deltas = np.asarray(deltas, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return 0.5 * (deltas.T @ (weights[:, None] * deltas))


def class_port_centers(obs, zobs):
    """Return sorted classes and equal-port class means without count cutoffs.

    Within a port-class cell, each aggregated observation has equal weight,
    regardless of row_count or the number of acquisitions belonging to a ship.
    """
    zobs = np.asarray(zobs, dtype=np.float64)
    classes, centers = [], []
    for label, group in obs.reset_index(drop=True).groupby('class_id', sort=True):
        means = [zobs[port.index.to_numpy()].mean(axis=0)
                 for _, port in group.groupby('port', sort=True)]
        classes.append(label)
        centers.append(np.mean(means, axis=0))
    return (np.asarray(classes, dtype=obs['class_id'].dtype),
            np.asarray(centers, dtype=np.float64).reshape(-1, zobs.shape[1]))


def fit_temporal_directions(cov, centers, rank=8):
    """Fit up to eight positive modes, optionally protecting only class span.

    Inputs live in the reduced r-dimensional space. SVD operates only on the
    centered class-by-r matrix; covariance eigendecompositions are r-by-r.
    Eigenvalues are descending. Scale-relative tolerances suppress numerical
    rank from centering/projection roundoff, including fully removed covariance.
    """
    cov = np.asarray(cov, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError('Expected square covariance (r,r)')
    if centers.ndim != 2 or centers.shape[1] != cov.shape[0]:
        raise ValueError('Expected class centers (C,r)')
    if not isinstance(rank, (int, np.integer)) or rank < 0:
        raise ValueError('rank must be a nonnegative integer')
    rank = min(rank, 8)
    cov = (cov + cov.T) / 2.0
    r = cov.shape[0]
    qclass = np.empty((r, 0), dtype=np.float64)
    if len(centers) > 1:
        centered = centers - centers.mean(axis=0, keepdims=True)
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        scale = max(singular[0], np.linalg.norm(centers))
        tolerance = 10 * np.finfo(float).eps * max(centers.shape) * scale
        class_rank = min(int(np.count_nonzero(singular > tolerance)),
                         len(centers) - 1)
        qclass = vh[:class_rank].T
    values, vectors = np.linalg.eigh(cov)
    tolerance = 10 * np.finfo(float).eps * r * np.max(np.abs(values))

    def top_positive(eigenvalues, eigenvectors):
        selected = np.flatnonzero(eigenvalues > tolerance)[::-1][:rank]
        return eigenvectors[:, selected], eigenvalues[selected]

    temporal, temporal_values = top_positive(values, vectors)
    complement = np.eye(r) - qclass @ qclass.T
    protected_cov = complement @ cov @ complement
    values, vectors = np.linalg.eigh((protected_cov + protected_cov.T) / 2.0)
    protected, protected_values = top_positive(values, vectors)
    return dict(temporal=temporal, class_protected=protected,
                class_basis=qclass, temporal_eigenvalues=temporal_values,
                class_protected_eigenvalues=protected_values,
                class_rank=qclass.shape[1])
