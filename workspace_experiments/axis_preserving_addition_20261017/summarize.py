"""Independent paired accounting, source curves, and native-chip identities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'axis_access_margin_20261017'
KEYS = ['fold', 'seed', 'origin', 'draw']
PKEYS = KEYS+['global_row']
VALUES = ['oa', 'ba', 'brier', 'macro_brier', 'nll']


def recompute(frame):
    rows = []
    for keys, group in frame.groupby(['arm']+KEYS):
        p = group[[f'p{k}' for k in range(8)]].to_numpy()
        y = group.truth.to_numpy(int)
        correct = p.argmax(1) == y
        np.testing.assert_array_equal(p.argmax(1), group.prediction)
        brier = np.sum((p-np.eye(8)[y])**2, axis=1)
        row = dict(zip(['arm']+KEYS, keys))
        row.update(oa=100*correct.mean(), ba=100*np.mean([correct[y==c].mean() for c in np.unique(y)]),
                   brier=brier.mean(), macro_brier=np.mean([brier[y==c].mean() for c in np.unique(y)]),
                   nll=-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    m = pd.concat([pd.read_csv(HERE/'metrics.csv'), pd.read_csv(HERE/'frozen_metrics.csv')], ignore_index=True)
    p = pd.concat([pd.read_csv(HERE/'predictions.csv'), pd.read_csv(HERE/'frozen_predictions.csv')], ignore_index=True)
    assert not p.duplicated(['arm']+PKEYS).any()
    calculated = recompute(p).set_index(['arm']+KEYS)
    provided = m.set_index(['arm']+KEYS).loc[calculated.index]
    errors = abs(calculated[VALUES]-provided[VALUES]).max()
    assert float(errors.max()) < 1e-10
    calculated.reset_index().to_csv(HERE/'independent_metrics.csv', index=False)
    extras = ['support_map_r', 'descriptor_energy', 'cached_solve_seconds']
    for name, keys in [('summary', ['origin','arm']), ('seed_summary', ['origin','arm','seed']),
                       ('fold_summary', ['origin','arm','fold']), ('draw_summary', ['origin','arm','draw'])]:
        m.groupby(keys)[VALUES+extras].mean().to_csv(HERE/(name+'.csv'))
    reference_metrics = m[m.arm.eq('margin')].set_index(KEYS)
    effects = []
    for arm in m.arm.unique():
        current = m[m.arm.eq(arm)].set_index(KEYS)
        delta = current[VALUES+extras]-reference_metrics.loc[current.index, VALUES+extras]
        delta = delta.rename(columns={c: 'delta_'+c for c in delta.columns})
        delta['arm'] = arm
        effects.append(delta.reset_index())
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(HERE/'paired_effects.csv', index=False)
    measures = ['delta_'+c for c in VALUES+extras]
    for name, keys in [('effect_summary', ['origin','arm']), ('effect_seeds', ['origin','arm','seed']),
                      ('effect_folds', ['origin','arm','fold']), ('effect_draws', ['origin','arm','draw'])]:
        effects.groupby(keys)[measures].mean().to_csv(HERE/(name+'.csv'))
    effects.groupby(['origin','arm','seed'])[measures].mean().groupby(['origin','arm']).agg(['mean','std']).to_csv(HERE/'seed_uncertainty.csv')

    reference = p[p.arm.eq('margin')].set_index(PKEYS)
    matched = reference.loc[pd.MultiIndex.from_frame(p[PKEYS])]
    p['reference_prediction'] = matched.prediction.to_numpy()
    p['change'] = p.prediction.eq(p.truth).astype(int)-p.reference_prediction.eq(p.truth).astype(int)
    p['max_probability_change'] = abs(p[[f'p{k}' for k in range(8)]].to_numpy()-matched[[f'p{k}' for k in range(8)]].to_numpy()).max(1)
    p['true_probability_change'] = (p[[f'p{k}' for k in range(8)]].to_numpy()-matched[[f'p{k}' for k in range(8)]].to_numpy())[np.arange(len(p)), p.truth]
    cohort = reference.reset_index().drop_duplicates(['fold', 'global_row'])
    class_counts = cohort.groupby(['fold','truth']).size().rename('class_queries')
    fold_counts = cohort.groupby('fold').agg(fold_queries=('global_row','size'),fold_classes=('truth','nunique'))
    p = p.join(class_counts, on=['fold','truth']).join(fold_counts, on='fold')
    context_counts = reference_metrics.reset_index().groupby('origin').size()
    p['oa_contribution_pp'] = 100*p.change/(p.origin.map(context_counts)*p.fold_queries)
    p['ba_contribution_pp'] = 100*p.change/(p.origin.map(context_counts)*p.fold_classes*p.class_queries)
    p.to_csv(HERE/'paired_predictions.csv', index=False)
    for name, keys in [('subgroup_contributions', ['origin','arm','axis_valid']),
                       ('class_contributions', ['origin','arm','truth']),
                       ('query_contributions', ['origin','arm','global_row','truth'])]:
        p.groupby(keys).agg(rescues=('change',lambda s: int((s>0).sum())),
            harms=('change',lambda s: int((s<0).sum())),
            delta_oa=('oa_contribution_pp','sum'), delta_ba=('ba_contribution_pp','sum'),
            query_contexts=('global_row','size'), unique_queries=('global_row','nunique'),
            mean_max_probability_change=('max_probability_change','mean'),
            mean_true_probability_change=('true_probability_change','mean')).to_csv(HERE/(name+'.csv'))
    sums = p.groupby(['origin','arm'])[['oa_contribution_pp','ba_contribution_pp']].sum()
    expected = effects.groupby(['origin','arm'])[['delta_oa','delta_ba']].mean().loc[sums.index]
    np.testing.assert_allclose(sums.to_numpy(), expected.to_numpy(), atol=1e-10)
    geometry = pd.read_csv(HERE.parent/'oss1_joint_geometry_20261017'/'geometry.csv', dtype={'mmsi':str})
    changed = p[p.change.ne(0)].merge(geometry, on='global_row', how='left', validate='many_to_one', suffixes=('','_metadata'))
    changed.to_csv(HERE/'changed_native_cases.csv', index=False)
    history = []
    for fold in m.fold.unique():
        for arm in ['margin','joint','additive']:
            for seed in range(3):
                root = HERE if arm == 'additive' else PREVIOUS
                frame = pd.read_csv(root/fold/f'{arm}_{seed}'/'learning.csv')
                frame['fold'], frame['arm'], frame['seed'] = fold, arm, seed
                history.append(frame)
    history = pd.concat(history, ignore_index=True)
    history.to_csv(HERE/'learning_curves.csv', index=False)
    history.groupby(['arm','evaluation','step'])[['brier','retrieval_ce','margin_loss','oa','ba']].mean().to_csv(HERE/'learning_summary.csv')

    replay = pd.read_csv(HERE/'baseline_replay.csv')
    verification = dict(metric_rows=len(m), prediction_rows=len(p), unique_queries=int(cohort.global_row.nunique()),
        old_axis_invalid_queries=int((~cohort.axis_valid).sum()), independent_metric_max_error=float(errors.max()),
        baseline_replay_max_error=float(replay.maximum_probability_error.max()),
        new_training_fits=int(pd.read_csv(HERE/'timings.csv').trained.sum()))
    (HERE/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(m.groupby(['origin','arm'])[VALUES].mean().to_string())
    print(effects[effects.arm.eq('additive')].groupby(['origin','seed'])[['delta_oa','delta_ba']].mean().to_string())
    print(changed[changed.arm.eq('additive')&changed.origin.eq('cross')][['fold','seed','draw','global_row','truth','reference_prediction','prediction','ba_contribution_pp']].to_string(index=False))
    print(verification)


if __name__ == '__main__':
    main()
