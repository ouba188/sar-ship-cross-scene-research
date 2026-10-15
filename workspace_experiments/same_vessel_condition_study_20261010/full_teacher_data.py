"""Source-only teachers for a bounded continuation of the deployed expert."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
HELPERS = ROOT/'oof_reliability_check'
sys.path.insert(0, str(HELPERS))
from sar_joint_completion import SOURCE, PREFIX
from protected_drift_activation import fit_source_heads
from check_translation_teacher_feasibility import make_cells, motion_features, pairs_for, margins
from incidence_transport import pair_weights
from scene_training import sample_weights
from check_teacher_weight_strength import local_weights

OUT = BASE/'full_teacher_integration_20261012'
PORTS = ('Los Angeles', 'Port Klang', 'Santos')
ARMS = ('continued', 'equal_teacher', 'knowledge_teacher')


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def bank_from_extra(bank, extra, geometry):
    answer = np.array(bank, dtype=float, copy=True)
    pca = geometry['model']['pca_ssl']
    answer[:, 128:192] = ((np.asarray(extra, dtype=float)-pca.mean_)@np.asarray(pca.components_, dtype=float).T).astype(np.float32)
    return answer


def fit_head(bank, labels, transforms):
    return fit_source_heads(bank, labels, transforms)['joint_expert']


def teaching_metadata(allowed):
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        ids, fold = z['fit_obs_ids'], z['assignment']
    use = np.isin(meta.iloc[ids].manifest_row.to_numpy(), allowed)
    ids, fold = ids[use], fold[use]
    mm = meta.iloc[ids].reset_index(drop=True)
    motion = pd.read_csv(BASE/'motion_knowledge_20261011/motion_knowledge.csv').set_index('obs_id').loc[ids]
    grid = pd.read_csv(BASE/'interaction_feasibility_20261012/geometry_join.csv').set_index('obs_id').loc[ids]
    np.testing.assert_array_equal(motion.mmsi, mm.mmsi)
    np.testing.assert_array_equal(motion['product'], mm['product'])
    sog = motion.sog_knots.to_numpy(float)
    cog = motion.cog_deg.to_numpy(float).copy()
    cog[~motion.cog_deg_valid.to_numpy(bool)] = np.nan
    az = grid.range_azimuth.to_numpy(float).copy()
    az[grid.grid_status.to_numpy() != 'ok'] = np.nan
    valid = motion.sog_knots_valid.to_numpy(bool) & motion.match_status.eq('matched').to_numpy()
    valid &= np.isfinite(sog) & (sog >= 0) & (sog < 102.2)
    return mm, fold, motion_features(sog, cog, az, valid)


def aggregate_teachers(cells, scores, members, quality, n):
    equal = np.zeros((n, 8)); weighted = equal.copy()
    multi = np.zeros(n, bool); active = multi.copy()
    for _, group in cells.groupby('mmsi', sort=True):
        ix = group.index.to_numpy()
        for focal in ix:
            donor = ix[ix != focal]
            if not len(donor):
                continue
            assert not np.any(cells.loc[donor, 'product'].to_numpy() == cells.loc[focal, 'product'])
            rows = members[focal]
            equal[rows] = scores[donor].mean(0)
            weighted[rows] = equal[rows]
            multi[rows] = True
            known = cells.loc[donor, 'known'].to_numpy(bool)
            if known.sum() < 2:
                continue
            w = np.ones(len(donor))
            local = local_weights(quality[donor[known]], 'minmax')
            w[known] = local/local.mean()
            np.testing.assert_allclose(w.sum(), len(donor), atol=1e-12)
            weighted[rows] = w@scores[donor]/len(donor)
            active[rows] = True
    return equal, weighted, multi, active


def prepare_teachers(context, bank, manifest, dest):
    """Refit source readers in THIS expert's coordinates; no compact-reader reuse."""
    rows = context.source_rows
    mm, folds, motion = teaching_metadata(rows)
    locations = pd.Index(rows).get_indexer(mm.manifest_row.to_numpy())
    assert (locations >= 0).all()
    source_ships = manifest.iloc[rows].mmsi.to_numpy()
    labels = manifest.iloc[rows].class_id.to_numpy()
    scores = np.zeros((len(mm), 8)); heads = []
    audit = []
    for k in range(3):
        val = folds == k
        excluded = mm.loc[val].mmsi.unique()
        train = ~np.isin(source_ships, excluded)
        head = fit_head(bank[train], labels[train], context.transforms)
        heads.append(head)
        scores[val] = bank[locations[val]]@head['coef']+head['intercept']
        assert not set(source_ships[train]) & set(excluded)
        audit.append(dict(fold=k, fit_rows=int(train.sum()), excluded_ships=len(excluded),
                          teacher_rows=int(val.sum())))
    cells, x, cell_scores, members = make_cells(mm, scores, motion, folds)
    x = x[:, 1:5]
    quality = np.full(len(cells), np.nan)
    betas = []
    target_margin = margins(cell_scores, cells.class_id.to_numpy())
    for k in range(3):
        pairs = pairs_for(cells, cells.fold.to_numpy() != k)
        a, b = pairs.T
        dx = x[b]-x[a]; dy = target_margin[b]-target_margin[a]
        w = pair_weights(cells, pairs)
        beta = np.linalg.solve(dx.T@(w[:, None]*dx)+.1*np.eye(4), dx.T@(w*dy))
        val = cells.fold.to_numpy() == k
        quality[val] = x[val]@beta
        betas.append(beta)
    equal, weighted, multi, active = aggregate_teachers(cells, cell_scores, members, quality, len(mm))
    y = mm.class_id.to_numpy()
    def metrics(s):
        pred = s[multi].argmax(1); truth = y[multi]
        return dict(oa=100*float(np.mean(pred == truth)),
                    ba=100*float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)])))
    changed = multi & (equal.argmax(1) != weighted.argmax(1))
    rescue = changed & (weighted.argmax(1) == y)
    harm = changed & (equal.argmax(1) == y)
    delta = weighted-equal
    record = dict(rows=len(mm), ships=int(mm.mmsi.nunique()), multi_rows=int(multi.sum()),
        active_rows=int(active.sum()), active_ships=int(mm.loc[active].mmsi.nunique()),
        equal=metrics(equal), knowledge=metrics(weighted), rescues=int(rescue.sum()), harms=int(harm.sum()),
        target_rms_difference=float(np.sqrt(np.mean(delta[multi]**2))),
        head_audit=audit, quality_reader_oof=True, upstream_reader_nested=False,
        representation_vessel_held=False, target_metadata_used=False)
    dest.mkdir(parents=True, exist_ok=True)
    mm.to_csv(dest/'teacher_rows.csv', index=False)
    cells.to_csv(dest/'teacher_cells.csv', index=False)
    np.savez_compressed(dest/'teachers.npz', rows=mm.manifest_row.to_numpy(), fold=folds,
        equal=equal, knowledge=weighted, multi=multi, active=active, scores=scores,
        weights=sample_weights(mm), beta=np.stack(betas), quality=quality,
        coef=np.stack([h['coef'] for h in heads]), intercept=np.stack([h['intercept'] for h in heads]))
    write(dest/'TEACHER_DIAGNOSTIC.json', record)
    print('TEACHER', context.task, json.dumps(record), flush=True)
    return dict(rows=mm.manifest_row.to_numpy(), fold=folds, equal=equal, knowledge=weighted,
                multi=multi, active=active, weights=sample_weights(mm), heads=heads, diagnostics=record)


def self_test():
    cells = pd.DataFrame(dict(mmsi=[1, 1, 1, 1], product=['a', 'b', 'c', 'd'], known=[True, True, True, False]))
    scores = np.zeros((4, 8)); scores[:, 0] = [100., 2., 8., 11.]
    members = [np.array([0, 1]), np.array([2]), np.array([3]), np.array([4])]
    q = np.array([9., 0., 1., np.nan])
    eq, weighted, mask, active = aggregate_teachers(cells, scores, members, q, 5)
    np.testing.assert_allclose(eq[:2, 0], 7.)
    np.testing.assert_allclose(weighted[:2, 0], ((2/3)*2+(4/3)*8+11)/3)
    altered = scores.copy(); altered[0] = -999.
    eq2, wt2, _, _ = aggregate_teachers(cells, altered, members, q, 5)
    np.testing.assert_array_equal(eq[:2], eq2[:2])
    np.testing.assert_array_equal(weighted[:2], wt2[:2])
    assert mask.all() and active.all()


if __name__ == '__main__':
    self_test()
    print('teacher aggregation tests PASS')
