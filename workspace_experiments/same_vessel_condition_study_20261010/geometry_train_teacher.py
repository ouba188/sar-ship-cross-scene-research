"""Training-only incidence knowledge changes other-acquisition teachers, not inputs."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from relative_content_condition import vessel_targets

ARMS = ('B', 'far', 'near', 'sham', 'sham_near')
ANGLE_SCALE = 5.0


def make_teachers(scores, meta, permutation_seed=51011):
    """Equal product mass, leave focal product out; unknown-angle mass unchanged.

    Known donor weights range from one to two before normalization within known
    donors. This prevents missingness alone from changing their total mass.
    Shuffling is within vessel among known product means, never across vessels.
    """
    meta = meta.reset_index(drop=True)
    scores = np.asarray(scores, np.float32)
    _, baseline, valid = vessel_targets(scores, meta)
    teachers = {arm: baseline.copy() for arm in ARMS}
    edges, rows = [], []
    rng = np.random.default_rng(permutation_seed)
    for ship, group in meta.groupby('mmsi', sort=True):
        cells = []
        for product, cell in group.groupby('product', sort=True):
            a = pd.to_numeric(cell.incidence_deg, errors='coerce').to_numpy(float)
            a = a[np.isfinite(a)]
            cells.append(dict(ids=cell.index.to_numpy(), product=str(product),
                              angle=float(a.mean()) if len(a) else np.nan,
                              score=scores[cell.index].mean(0)))
        if len(cells) < 2:
            continue
        angle = np.array([c['angle'] for c in cells])
        known = np.flatnonzero(np.isfinite(angle))
        shuffled = angle.copy()
        shuffled[known] = rng.permutation(angle[known])
        for focal, cell in enumerate(cells):
            donor = np.array([j for j in range(len(cells)) if j != focal])
            means = np.stack([cells[j]['score'] for j in donor])
            for arm in ARMS:
                a = shuffled if arm.startswith('sham') else angle
                eligible = np.isfinite(a[donor]) & np.isfinite(a[focal])
                weights = np.ones(len(donor), dtype=float)
                changed = False
                if arm != 'B' and eligible.sum() >= 2:
                    distance = np.minimum(np.abs(a[donor[eligible]]-a[focal])/ANGLE_SCALE, 1.)
                    local = 2-distance if arm in ('near', 'sham_near') else 1+distance
                    weights[eligible] = local/local.mean()
                    changed = bool(np.ptp(local) > 1e-12)
                weights /= weights.sum()
                # Keep untouched rows bit-identical to the historical baseline.
                if changed:
                    teachers[arm][cell['ids']] = weights@means
                for j, weight in zip(donor, weights):
                    edges.append(dict(mmsi=int(ship), focal_product=cell['product'],
                        donor_product=cells[j]['product'], arm=arm, weight=float(weight),
                        incidence_focal=float(angle[focal]), incidence_donor=float(angle[j]),
                        used_focal=float(a[focal]), used_donor=float(a[j])))
                for row in cell['ids']:
                    delta = teachers[arm][row]-baseline[row]
                    rows.append(dict(local_id=int(row), mmsi=int(ship), arm=arm,
                        class_id=int(meta.iloc[row].class_id), port=str(meta.iloc[row].port),
                        donors=len(donor), known_donors=int(eligible.sum()), weights_changed=changed,
                        teacher_delta_norm=float(np.linalg.norm(delta)),
                        centered_delta_norm=float(np.linalg.norm(delta-delta.mean()))))
    return teachers, valid, pd.DataFrame(rows), pd.DataFrame(edges)
