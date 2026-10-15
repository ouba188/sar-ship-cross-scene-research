"""Paired condition-response probes; observational associations, not causal IDs."""

import numpy as np


def split_observations(obs, seed=20261009):
    identities = np.random.default_rng(seed).permutation(np.sort(obs.mmsi.unique()))
    held_identity = obs.mmsi.isin(identities[::3]).to_numpy()
    held_month = obs.acq_time.dt.month.mod(3).eq(0).to_numpy()
    return {
        'fit': ~held_identity & ~held_month,
        'new_ship': held_identity & ~held_month,
        'new_month': ~held_identity & held_month,
        'new_ship_month': held_identity & held_month,
    }


def condition_differences(obs, conditions, pairs):
    """Incidence polynomial and local displacement; no elapsed-time scaling."""
    a = pairs.obs_a.to_numpy(dtype=int)
    b = pairs.obs_b.to_numpy(dtype=int)
    theta = (conditions[:, 0] - 38.) / 5.
    angle = np.column_stack((theta, theta**2))
    lat, lon = conditions[:, 1], conditions[:, 2]
    dlon = (lon[b] - lon[a] + 180.) % 360. - 180.
    east = 111.195 * np.cos(np.deg2rad((lat[a] + lat[b]) / 2.)) * dlon / 10.
    north = 111.195 * (lat[b] - lat[a]) / 10.
    return np.column_stack((angle[b] - angle[a], east, north))


def fit_response(x, y, weights, ridge=.01):
    """Zero-intercept paired ridge with fixed, physically scaled inputs."""
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    return np.linalg.solve(x.T @ (weights[:, None] * x) + ridge*np.eye(x.shape[1]),
                           x.T @ (weights[:, None] * y))


def response_metrics(actual, predicted, weights):
    baseline = float(np.sum(weights[:, None] * actual**2))
    error = float(np.sum(weights[:, None] * (actual-predicted)**2))
    predicted_energy = float(np.sum(weights[:, None] * predicted**2))
    cross = float(np.sum(weights[:, None] * actual*predicted))
    return dict(skill=1-error/baseline if baseline > 0 else np.nan,
                baseline_energy=baseline, error_energy=error,
                predicted_energy=predicted_energy, cross_energy=cross,
                best_scalar=cross/predicted_energy if predicted_energy > 0 else np.nan)


def incidence_shift(angles, coefficients):
    """Modelled change relative to 38 degrees; missing angles give no shift."""
    angles = np.asarray(angles, dtype=float)
    shift = np.zeros((len(angles), coefficients.shape[1]))
    valid = np.isfinite(angles)
    t = (angles[valid]-38.)/5.
    shift[valid] = np.column_stack((t, t**2)) @ coefficients[:2]
    return shift
