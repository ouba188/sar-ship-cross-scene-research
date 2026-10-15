"""Query-specific evidence; no query labels enter these operators."""

import numpy as np

NAMES = (
    'case_margin_update', 'case_log_absolute_mass', 'case_signed_agreement',
    'case_max_vessel_share', 'case_effective_vessel_fraction',
    'case_positive_vessel_fraction', 'case_initial_role_share', 'case_raw_role_share',
    'drop_initial_retained', 'drop_raw_retained', 'drop_both_retained',
    'drop_margin_update_mean', 'drop_margin_update_sd', 'drop_margin_update_min',
)


def contributions(kr, ki, ar, ai, r, i, sy, groups):
    terms = ki * (ai[:, i] - ai[:, r]).T - kr * (ar[:, i] - ar[:, r]).T
    unique = np.unique(groups)
    vessel = np.column_stack([terms[:, groups == g].sum(1) for g in unique])
    mass = np.abs(vessel).sum(1)
    safe = np.maximum(mass, 1e-12)
    share = np.abs(vessel) / safe[:, None]
    effective = np.divide(1., np.square(share).sum(1),
                          out=np.zeros(len(terms)), where=mass > 1e-12) / len(unique)
    return np.column_stack([
        vessel.sum(1), np.log1p(mass), vessel.sum(1)/safe, share.max(1),
        effective, (vessel > 0).mean(1),
        (terms * (sy[None, :] == i[:, None])).sum(1)/safe,
        (terms * (sy[None, :] == r[:, None])).sum(1)/safe,
    ]), terms


def deletion_features(raw, initial, r, i):
    # Arrays are query x deleted-vessel x class. r/i stay fixed to full support.
    rows = np.arange(len(r))[:, None]
    drops = np.arange(raw.shape[1])[None, :]
    rm = raw.argmax(2) == r[:, None]
    im = initial.argmax(2) == i[:, None]
    delta = (initial[rows, drops, i[:, None]] - initial[rows, drops, r[:, None]]
             - raw[rows, drops, i[:, None]] + raw[rows, drops, r[:, None]])
    return np.column_stack([im.mean(1), rm.mean(1), (rm & im).mean(1),
                            delta.mean(1), delta.std(1), delta.min(1)])
