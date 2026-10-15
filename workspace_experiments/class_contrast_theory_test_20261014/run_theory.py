"""Test the class-graph identity and its proposed SAR shift mechanism."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'class_contrast_core_controls_20261014'))
import run_controls as core
run, ext = core.run, core.run.old.ext
PAIRS = core.probe.PAIRS


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def same_ship_pairs(meta, rows, name):
    frame = meta.loc[rows].copy()
    frame = frame[frame.identity.ne('')]
    if 'known_ship' in frame:
        frame = frame[frame.known_ship.astype(bool)]
    consistent = frame.groupby('identity').class_id.nunique()
    frame = frame[frame.identity.isin(consistent[consistent == 1].index)]
    pairs = []
    for ship, g in frame.groupby('identity', sort=True):
        if name == 'oss2':
            products = [(p, h) for p, h in g.groupby('product_type') if h.acquisition.nunique() >= 2]
            if not products:
                continue
            products.sort(key=lambda item: (-item[1].acquisition.nunique(), item[0] != 'GRDH'))
            g = products[0][1]
        g = g.sort_index().drop_duplicates('acquisition').sort_values('acquisition')
        if len(g) >= 2:
            pairs.append(dict(dataset=name, ship=str(ship), class_id=int(g.class_id.iloc[0]),
                              row_a=int(g.index[0]), row_b=int(g.index[-1]),
                              acq_a=str(g.acquisition.iloc[0]), acq_b=str(g.acquisition.iloc[-1])))
    return pd.DataFrame(pairs)


def distance(delta, metric):
    return np.einsum('ij,jk,ik->i', delta, metric, delta, optimize=True)


def values(z, specs, models):
    maps = {'original': z}
    for model in models.values():
        if model['spec'] is not None and model['spec'] not in maps:
            maps[model['spec']] = core.project(z, specs[model['spec']])
    return maps


def measure(dataset, f, meta, pairs, specs, models, centers, scene_tasks, basis):
    rows = f['rows']
    assert np.all(np.isin(pairs.row_a, rows)) and np.all(np.isin(pairs.row_b, rows))
    ia, ib = np.searchsorted(rows, pairs.row_a), np.searchsorted(rows, pairs.row_b)
    maps = values(f['z'], specs, models)
    maps_center = values(centers, specs, models)
    raw_delta = f['z'][ia] - f['z'][ib]
    raw_energy = np.einsum('ij,ij->i', raw_delta, raw_delta)
    projection = np.sum((raw_delta @ basis) ** 2, axis=1)
    pairs = pairs.copy()
    pairs['raw_change'] = raw_energy
    pairs['class_span_fraction'] = projection / raw_energy
    records, means = [], []
    for arm, model in models.items():
        key = model['spec'] if model['spec'] is not None else 'original'
        metric = model['metric'] if model['metric'] is not None else np.eye(maps[key].shape[1])
        delta = maps[key][ia] - maps[key][ib]
        shift = distance(delta, metric)
        class_delta = maps_center[key][PAIRS[:, 0]] - maps_center[key][PAIRS[:, 1]]
        source_between = float(distance(class_delta, metric).mean())
        assert source_between > 0
        chunk = pairs[['dataset', 'ship', 'class_id', 'row_a', 'row_b', 'raw_change', 'class_span_fraction']].copy()
        chunk['arm'] = arm
        chunk['shift_energy'] = shift
        chunk['source_between'] = source_between
        chunk['shift_over_source_between'] = shift / source_between
        records.append(chunk)
        between_scenes = []
        for scene in scene_tasks:
            q = np.asarray(scene['query'])
            g = meta.loc[q].copy()
            g['row'] = q
            reference = []
            for label, group in g.groupby('class_id'):
                if group.identity.nunique() >= 2:
                    reference.append((int(label), maps[key][np.searchsorted(rows, group.row)].mean(axis=0)))
            if len(reference) >= 2:
                cb = np.stack([r[1] for r in reference])
                ci, cj = np.triu_indices(len(cb), 1)
                between_scenes.append(dict(scene=scene['episode'], classes=len(reference),
                                           energy=float(distance(cb[ci] - cb[cj], metric).mean())))
        means.append(dict(dataset=dataset, arm=arm, ships=len(pairs), source_between=source_between,
                          same_ship_change=float(shift.mean()), shift_over_source_between=float(shift.mean() / source_between),
                          scene_between=float(np.mean([e['energy'] for e in between_scenes])) if between_scenes else np.nan,
                          scene_between_count=len(between_scenes),
                          scene_between_over_ship_change=float(np.mean([e['energy'] for e in between_scenes]) / shift.mean()) if between_scenes else np.nan))
    return pd.concat(records, ignore_index=True), pd.DataFrame(means)


def main():
    save('PROTOCOL.json', dict(status='Frozen before this analysis', question='Do learned class-pair weights reduce observed same-ship cross-acquisition variation relative to type separation, and does that explain full-system gains?',
        models='Reused class_utility seeds3, fixed class pairs, free class metric seeds3, nonlinear class mean, PCA7 and original visual',
        source='Existing validation_fit geometry/8825 source-ship class bank; no new fitting or target-label model selection',
        pair='One pair between lexicographically first/last acquisition keys per consistent-label known ship; not assumed to be chronological. OSS2 retains product type; one observation per ship/acquisition',
        between='28 source class-center pairs; secondary target within-acquisition class means from at least two distinct query ships per class, scene equal-weighted; target labels diagnostic only',
        ratio='Mean same-ship squared distance divided by mean source-center between-class squared distance; lower is conditionally favorable, not a BA guarantee',
        outcome='Existing cross-acquisition B64 ship BA contributions for F/OSS2 and B32 for ports; compare learned vs fixed pairs, with no new predictions',
        attribution='Same-ship difference includes viewpoint/state/crop, not isolated causal acquisition drift; support/query and source-fit exposure retain previous bounds',
        main_model_changed=False))
    with (ROOT / 'class_contrast_core_controls_20261014/specs.pkl').open('rb') as h:
        specs = pickle.load(h)
    with (ROOT / 'class_contrast_core_controls_20261014/models.pkl').open('rb') as h:
        all_models = pickle.load(h)
    arms = ['original', 'class_mean', 'class_fixed', 'pca_fixed'] + [f'class_utility_{i}' for i in range(3)] + [f'class_metric_{i}' for i in range(3)]
    models = {arm: all_models[arm] for arm in arms}
    bank = specs['class_edge']['bank']
    centers = bank['centers']
    contrast = centers[PAIRS[:, 0]] - centers[PAIRS[:, 1]]
    basis_full = np.linalg.svd(centers - centers.mean(0), full_matrices=False)[2].T
    rank = int(np.linalg.matrix_rank(centers - centers.mean(0)))
    basis = basis_full[:, :rank]
    assert rank <= 7
    if rank == 7:
        np.testing.assert_allclose(contrast @ basis @ basis.T, contrast, atol=1e-8)
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    with (run.PREV / 'validation_memory.pkl').open('rb') as h:
        memory = pickle.load(h)
    ev = run.old.operator()
    source_meta = run.old.source_metadata()
    source_rows = np.unique(np.concatenate([np.load(run.PREV / 'validation_fit_rows.npy'), memory['source_rows']]))
    source = run.old.encoded(ev, run.old.source_arrays(), source_rows, geometry)
    bank_z = source['z'][np.searchsorted(source_rows, memory['source_rows'])]
    mean_z = bank_z.mean(axis=0)
    checks = []
    for i in range(3):
        w = np.diag(models[f'class_utility_{i}']['metric'])
        b = (2 * bank['scale'] / bank['tau2']) * (np.sqrt(w) / bank['pair_scale'])[:, None] * contrast
        analytic = (source['z'][:128] - mean_z) @ b.T
        mapped = core.project(source['z'][:128], specs['class_edge']) * np.sqrt(w)
        error = float(np.max(abs(analytic - mapped)))
        np.testing.assert_allclose(analytic, mapped, atol=3e-6, rtol=1e-7)
        singular = np.linalg.svd(b, compute_uv=False)
        checks.append(dict(seed=i, max_map_error=error, rank=int(np.sum(singular > singular[0] * 1e-9)),
                           min_weight=float(w.min()), max_weight=float(w.max())))
    assert all(r['rank'] <= 7 and r['min_weight'] > 0 for r in checks)
    plans = json.loads((ROOT / 'class_contrast_scope_validation_20261014/ENROLLMENT.json').read_text(encoding='utf-8'))
    source_pairs = same_ship_pairs(source_meta, source_rows, 'source')
    pieces, summaries = [], []
    source_records, source_summary = measure('source', source, source_meta, source_pairs, specs, models, centers, [], basis)
    pieces.append(source_records); summaries.append(source_summary)
    del source
    for name in ('f0921', 'oss2', 'ports'):
        scene_tasks = [t for t in plans if t['dataset'] == name and t['budget'] == 64 and t['draw'] == 0 and t['cell'] == 'other_acquisition']
        rows = np.unique(np.concatenate([t['query'] for t in scene_tasks]))
        meta = source_meta if name == 'ports' else pd.read_csv(run.old.OLD / f'{name}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
        arrays = run.old.source_arrays() if name == 'ports' else ext.arrays_for(name, meta)
        f = run.old.encoded(ev, arrays, rows, geometry)
        pairs = same_ship_pairs(meta, rows, name)
        print('PAIRS', name, len(pairs), 'query', len(rows), flush=True)
        rec, summ = measure(name, f, meta, pairs, specs, models, centers, scene_tasks, basis)
        pieces.append(rec); summaries.append(summ)
        del f, arrays
    record = pd.concat(pieces, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    record.to_csv(OUT / 'pair_energies.csv', index=False)
    summary.to_csv(OUT / 'representation_summary.csv', index=False)
    checks.append(dict(kind='known_class_centers', rank=rank, source_bank_ships=len(memory['source_rows']),
                       source_pairs=len(source_pairs)))
    save('ALGEBRA_CHECKS.json', checks)
    # The existing event ledger stores zero only for changed decisions; absent ships have zero net effect.
    contributions = pd.read_csv(ROOT / 'class_contrast_scope_validation_20261014/ship_contributions.csv', dtype={'ship': str})
    rows = []
    for name, budget in [('f0921', 64), ('oss2', 64), ('ports', 32)]:
        pairs = record[(record.dataset == name) & record.arm.eq('class_fixed')][['ship', 'class_id', 'shift_over_source_between']].rename(columns={'shift_over_source_between': 'fixed_shift_ratio'})
        # A ship appears once in this dataset's pair register for each model.
        learned = record[(record.dataset == name) & record.arm.str.startswith('class_utility')].groupby('ship', as_index=False).shift_over_source_between.mean().rename(columns={'shift_over_source_between': 'learned_shift_ratio'})
        joined = pairs.merge(learned, on='ship', validate='one_to_one')
        outcome = contributions[(contributions.dataset == name) & (contributions.cohort == 'all') &
            (contributions.budget == budget) & (contributions.cell == 'other_acquisition') & (contributions.comparator == 'class_fixed')]
        joined = joined.merge(outcome[['ship', 'ba_contribution', 'oa_contribution']], on='ship', how='left', validate='one_to_one').fillna({'ba_contribution': 0., 'oa_contribution': 0.})
        joined['relative_shift'] = joined.learned_shift_ratio / joined.fixed_shift_ratio
        joined['dataset'] = name
        joined['budget'] = budget
        joined['quartile'] = pd.qcut(joined.fixed_shift_ratio.rank(method='first'), 4, labels=False) + 1
        rows.append(joined)
    linked = pd.concat(rows, ignore_index=True)
    linked.to_csv(OUT / 'ship_outcome_link.csv', index=False)
    groups = linked.groupby(['dataset', 'budget', 'quartile']).agg(ships=('ship', 'size'),
        relative_shift=('relative_shift', 'median'), ba_contribution=('ba_contribution', 'sum'),
        rescued_ships=('ba_contribution', lambda x: int((x > 1e-12).sum())),
        harmed_ships=('ba_contribution', lambda x: int((x < -1e-12).sum()))).reset_index()
    groups.to_csv(OUT / 'ship_shift_quartiles.csv', index=False)
    associations = []
    for name, g in linked.groupby('dataset'):
        correlation = spearmanr(g.relative_shift, g.ba_contribution)
        associations.append(dict(dataset=name, budget=int(g.budget.iloc[0]), ships=len(g),
                                 relative_shift_median=float(g.relative_shift.median()),
                                 improved_shift_fraction=float((g.relative_shift < 1).mean()),
                                 linked_ba_contribution=float(g.ba_contribution.sum()),
                                 spearman=float(correlation.statistic), p_exploratory=float(correlation.pvalue)))
    pd.DataFrame(associations).to_csv(OUT / 'outcome_associations.csv', index=False)
    save('COMPLETE.json', dict(source_pairs=len(source_pairs), target_pairs={n:int((linked.dataset == n).sum()) for n in ('f0921','oss2','ports')},
                               new_model_training=0, new_predictions=0, main_model_changed=False))
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(associations).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
