"""Independent, descriptive source embedding geometry without file I/O."""

import numpy as np


CONTROL_SAMPLES_PER_VESSEL = 64
CONTROL_SEED = 41033


def _mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


def embedding_diagnostics(values, meta) -> dict:
    """Describe aligned N x D embeddings and source mmsi/product/class_id metadata.

    Rows align by position; metadata index labels are ignored. Coordinates retain
    their input dimension and are never centered per chip or projected. Global
    centering is used only for covariance/variance. Effective rank is the entropy
    exponential of positive population-covariance eigenvalues, or 0 at collapse.

    Within-vessel distances average all chip pairs from DIFFERENT products, then
    average vessels equally. Drift is mean squared distance per coordinate;
    direction distance is 1-cosine of the original embeddings. A zero vector in
    a temporal pair makes that vessel's direction distance (and its mean) NaN.

    Controls draw 64 pairs per covered anchor vessel with replacement: uniformly
    choose its chip, a same-class other identity, and that identity's chip. The
    donor need not have repeated products. Only anchors with temporal pairs and
    a same-class donor enter controls AND matched_same_vessel_drift. Labels must
    be source labels supplied by the caller; no target labels or files are read.

    normalized_drift divides all-paired-vessel drift by global variance / D.
    class_separation divides equal-class centroid scatter around the equal-class
    centroid mean by equal-class within scatter (chips equal within each class).
    Empty means and ratios with zero denominators are NaN; empty rank is 0.
    These are descriptive stability/separation diagnostics, not causal evidence.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(meta) or values.shape[1] == 0:
        raise ValueError('values must be N x D with D > 0, aligned with meta')
    if not np.isfinite(values).all():
        raise ValueError('values must be finite')
    meta = meta.reset_index(drop=True)
    n, dimension = values.shape
    variance, effective_rank = float('nan'), 0.
    if n:
        # Reference subtraction also makes exactly constant float inputs collapse
        # exactly, avoiding mean-roundoff masquerading as nonzero covariance.
        centered = values-values[0]
        centered -= centered.mean(axis=0)
        variance = float(np.square(centered).sum()/n)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered/n)
        positive = eigenvalues[eigenvalues > 0]
        if len(positive):
            probabilities = positive/positive.sum()
            effective_rank = float(np.exp(-np.sum(probabilities*np.log(probabilities))))

    vessels, drifts, directions = [], {}, []
    pairs = 0
    for ship, group in meta.groupby('mmsi', sort=True):
        ids = group.index.to_numpy()
        vessels.append((ship, group.class_id.iloc[0], ids))
        reference = values[ids[0]]
        count, total, energy = 0, np.zeros(dimension), 0.
        unit_total = np.zeros(dimension)
        distance, cosine_sum, ship_pairs = 0., 0., 0
        has_zero = False
        for _, acquisition in group.groupby('product', sort=True):
            raw = values[acquisition.index.to_numpy()]
            x = raw-reference
            size, sx, ex = len(x), x.sum(axis=0), float(np.square(x).sum())
            # Incrementally combine products, counting every cross-product pair
            # once without building a quadratic chip-pair matrix.
            distance += count*ex+size*energy-2*float(total @ sx)
            ship_pairs += count*size
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            has_zero |= bool(np.any(norms == 0))
            units = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)
            unit_sum = units.sum(axis=0)
            cosine_sum += float(unit_total @ unit_sum)
            unit_total += unit_sum
            count, total, energy = count+size, total+sx, energy+ex
        if ship_pairs:
            drifts[ship] = max(0., distance/(dimension*ship_pairs))
            directions.append(float('nan') if has_zero else
                              float(np.clip(1-cosine_sum/ship_pairs, 0., 2.)))
            pairs += ship_pairs

    classes = {}
    for ship, label, ids in vessels:
        classes.setdefault(label, []).append((ship, ids))
    rng = np.random.default_rng(CONTROL_SEED)
    controls, matched = [], []
    for ships in classes.values():
        if len(ships) < 2:
            continue
        for k, (ship, ids) in enumerate(ships):
            if ship not in drifts:
                continue
            anchors = rng.choice(ids, CONTROL_SAMPLES_PER_VESSEL)
            choices = rng.integers(len(ships)-1, size=CONTROL_SAMPLES_PER_VESSEL)
            choices += choices >= k
            donors = np.array([rng.choice(ships[j][1]) for j in choices])
            controls.append(float(np.square(values[anchors]-values[donors]).mean()))
            matched.append(drifts[ship])
    control, matched_drift = _mean(controls), _mean(matched)
    drift = _mean(list(drifts.values()))

    centroids, within = [], []
    for _, group in meta.groupby('class_id', sort=True):
        x = values[group.index.to_numpy()]-values[0]
        centroid = x.mean(axis=0)
        centroids.append(centroid)
        within.append(float(np.square(x-centroid).sum(axis=1).mean()))
    within_scatter = _mean(within)
    between_scatter = float('nan')
    if centroids:
        centroids = np.asarray(centroids)
        between_scatter = float(np.square(centroids-centroids.mean(axis=0)).sum(axis=1).mean())

    return dict(
        dimension=dimension, total_centered_variance=variance,
        effective_rank=effective_rank, same_vessel_drift=drift,
        same_vessel_direction_distance=_mean(directions),
        same_class_other_distance=control,
        matched_same_vessel_drift=matched_drift,
        drift_ratio=matched_drift/control if control > 0 else float('nan'),
        normalized_drift=drift/(variance/dimension) if variance > 0 else float('nan'),
        paired_vessels=len(drifts), same_vessel_pairs=pairs,
        control_vessels=len(controls),
        control_pairs=len(controls)*CONTROL_SAMPLES_PER_VESSEL,
        class_separation=between_scatter/within_scatter if within_scatter > 0 else float('nan'))
