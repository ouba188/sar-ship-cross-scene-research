"""Source-only descriptive score diagnostics; no data loading or causal claims."""

import numpy as np


CONTROL_SAMPLES_PER_VESSEL = 64
CONTROL_SEED = 41033


def _mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


def _geometry(centered, meta):
    vessels = []
    drifts, pairs = {}, 0
    for ship, group in meta.groupby('mmsi', sort=True):
        ids = group.index.to_numpy()
        vessels.append((ship, int(group.class_id.iloc[0]), ids))
        # Sufficient statistics give the exact cross-product chip-pair MSE
        # without allocating quadratic pairs. Subtract a reference for stability.
        reference = centered[ids[0]]
        count, total, energy = 0, np.zeros(8), 0.
        distance, ship_pairs = 0., 0
        for _, acquisition in group.groupby('product', sort=True):
            x = centered[acquisition.index.to_numpy()]-reference
            n, sx, ex = len(x), x.sum(0), float(np.square(x).sum())
            distance += count*ex+n*energy-2*float(total@sx)
            ship_pairs += count*n
            count, total, energy = count+n, total+sx, energy+ex
        if ship_pairs:
            drifts[ship] = max(0., distance/(8*ship_pairs))
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
            # Fixed, bounded sampling: 64 directed draws per eligible vessel,
            # uniform anchor chip, other vessel, then donor chip (replacement).
            # No all-vessel quadratic matrix; products are NOT matched here.
            anchors = rng.choice(ids, CONTROL_SAMPLES_PER_VESSEL)
            choices = rng.integers(len(ships)-1, size=CONTROL_SAMPLES_PER_VESSEL)
            choices += choices >= k
            donors = np.array([rng.choice(ships[j][1]) for j in choices])
            controls.append(float(np.square(centered[anchors]-centered[donors]).mean()))
            matched.append(drifts[ship])
    control, matched_drift = _mean(controls), _mean(matched)
    return dict(
        same_vessel_pairs=pairs, paired_vessels=len(drifts),
        centered_score_drift=_mean(list(drifts.values())),
        same_class_other_vessel_distance=control,
        control_vessels=len(controls),
        control_pairs=len(controls)*CONTROL_SAMPLES_PER_VESSEL,
        control_matched_centered_score_drift=matched_drift,
        drift_ratio=matched_drift/control if control > 0 else float('nan'))


def diagnose_scores(scores, baseline, teacher, meta, valid) -> dict:
    """Diagnose aligned N x 8 source scores without reading any data files.

    All arrays and metadata correspond by ROW POSITION; pandas index labels are
    ignored. A caller must slice/reorder them together: unseen array permutations
    cannot be detected without observation IDs. Labels are source diagnostics only.
    OA/BA are percentages, BA averages classes present in this call. Rescue/harm
    are counts relative to baseline. Rates are fractions among baseline-wrong /
    baseline-correct rows, respectively, and are NaN for zero denominators.

    Teacher groups and teacher MSE use valid rows (other-acquisition teacher
    availability, as returned by vessel_targets). Invalid teacher rows may be NaN.
    Mean correct-class margin uses all rows: true score minus best rival score.
    teacher_mse and baseline_teacher_mse average 8 centered score coordinates,
    then valid chips. They describe teacher fit, not teacher correctness.

    Drift averages cross-product chip-pair centered MSE within each vessel, then
    vessels equally, matching pair_metrics weighting. Same-product chips are not
    temporal pairs. Controls use the same distance and equal anchor-vessel weights;
    only paired vessels with a same-class other vessel contribute to controls.
    drift_ratio uses the within-vessel drift on that SAME covered vessel subset.
    Geometry ignores valid: it depends on available local products, whereas an
    externally supplied teacher may cover a different source observation subset.
    Empty populations produce NaN means and explicit zero counts. All findings
    are descriptive source associations, not causal or target-domain evidence.
    """
    scores, baseline, teacher = [np.asarray(x, dtype=np.float64)
                                 for x in (scores, baseline, teacher)]
    valid = np.asarray(valid, dtype=bool)
    n = len(meta)
    if any(x.shape != (n, 8) for x in (scores, baseline, teacher)) or valid.shape != (n,):
        raise ValueError('scores/baseline/teacher must be N x 8 and valid must be N, aligned with meta')
    meta = meta.reset_index(drop=True)
    y = meta.class_id.to_numpy()
    pred, prior = scores.argmax(1), baseline.argmax(1)
    correct, base_correct = pred == y, prior == y
    centered, base_centered = [x-x.mean(1, keepdims=True) for x in (scores, baseline)]
    teacher_valid = teacher[valid]
    teacher_centered = teacher_valid-teacher_valid.mean(1, keepdims=True)
    teacher_correct = np.zeros(n, dtype=bool)
    teacher_correct[valid] = teacher_valid.argmax(1) == y[valid]
    result = dict(n=n, OA=100*_mean(correct),
                  BA=100*_mean([_mean(correct[y == c]) for c in np.unique(y)]),
                  baseline_OA=100*_mean(base_correct),
                  baseline_BA=100*_mean([_mean(base_correct[y == c]) for c in np.unique(y)]))
    for prefix, mask in [('', np.ones(n, dtype=bool)), ('repeated_', valid),
                         ('teacher_correct_', valid & teacher_correct),
                         ('teacher_wrong_', valid & ~teacher_correct)]:
        wrong = int((mask & ~base_correct).sum())
        right = int((mask & base_correct).sum())
        rescue = int((mask & ~base_correct & correct).sum())
        harm = int((mask & base_correct & ~correct).sum())
        result.update({prefix+k: v for k, v in dict(
            count=int(mask.sum()), baseline_wrong=wrong, baseline_correct=right,
            rescue=rescue, harm=harm,
            rescue_rate=rescue/wrong if wrong else float('nan'),
            harm_rate=harm/right if right else float('nan')).items()})
    for prefix, values in [('', scores), ('baseline_', baseline)]:
        rival = values.copy()
        rival[np.arange(n), y] = -np.inf
        result[prefix+'correct_margin'] = _mean(values[np.arange(n), y]-rival.max(1))
    result.update(teacher_mse=_mean(np.square(centered[valid]-teacher_centered).mean(1)),
                  baseline_teacher_mse=_mean(np.square(base_centered[valid]-teacher_centered).mean(1)))
    result.update(_geometry(centered, meta))
    result.update({'baseline_'+k: v for k, v in _geometry(base_centered, meta).items()})
    return result
