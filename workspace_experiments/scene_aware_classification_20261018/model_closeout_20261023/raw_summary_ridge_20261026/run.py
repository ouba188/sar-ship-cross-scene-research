"""Complete the raw/summary x ridge/MLP comparison with source-only selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'readout_bottleneck_20261026'))
import probe_compression as prior
base = prior.base
NEW = ROOT/'raw_region_bypass_20261026'


def fit_source(name):
    root, cross, table, target, raw, eraw, local, elocal = base.dataset(name)
    assert 'class_id' not in target.columns
    fit = np.flatnonzero(table.role.eq('fit_train'))
    val = np.flatnonzero(table.role.eq('fit_validation'))
    for field in ('mmsi', 'pass_group'):
        assert not set(table.iloc[fit][field]) & set(table.iloc[val][field])
        assert not set(table.loc[table.role.ne('evaluation'), field]) & set(target[field])
    x = base.representations(raw, local)['regional']
    xe = base.representations(eraw, elocal)['regional']
    aux, eaux = prior.summary_inputs(root, cross, table, target)
    out = HERE/name
    out.mkdir(parents=True, exist_ok=True)
    sc, sa = base.standardize(x, fit), base.standardize(aux, fit)
    for key in sc:
        np.testing.assert_array_equal(sc[key], np.load(NEW/name/'raw_scaler.npz')[key])
        np.testing.assert_array_equal(sa[key], np.load(NEW/name/'summary_scaler.npz')[key])
    np.savez_compressed(out/'raw_scaler.npz', **sc)
    np.savez_compressed(out/'summary_scaler.npz', **sa)
    x, xe = base.transform(x, sc), base.transform(xe, sc)
    aux, eaux = base.transform(aux, sa), base.transform(eaux, sa)
    src, dst, receipts, checks = {}, {}, [], []
    for arm, xx, ee in [('raw', x, xe), ('raw_summary', np.c_[x,aux], np.c_[xe,eaux])]:
        dest = out/arm
        dest.mkdir(exist_ok=True)
        s, e, receipt = base.train_ridge(xx, ee, table, fit, val, dest)
        src[arm+'_ridge'], dst[arm+'_ridge'] = s, e
        receipts.append(dict(arm=arm, **receipt))
    for side, data, tab in [('source', src, table), ('cross', dst, target)]:
        old = base.tr.load_npz(prior.PRIOR/f'classical_fit_only/{name}/{side}_predictions.npz')
        added = base.tr.load_npz(NEW/name/f'{side}_predictions.npz')
        np.testing.assert_array_equal(old['obs_id'], tab.obs_id)
        np.testing.assert_array_equal(added['obs_id'], tab.obs_id)
        np.testing.assert_allclose(data['raw_ridge'], old['regional_ridge'], rtol=1e-5, atol=2e-5)
        np.testing.assert_array_equal(data['raw_ridge'].argmax(1), old['regional_ridge'].argmax(1))
        checks.append(dict(side=side, raw_replay_error=float(abs(data['raw_ridge']-old['regional_ridge']).max())))
        for seed in base.SEEDS:
            data[f'raw_mlp_seed{seed}'] = old[f'regional_mlp_seed{seed}']
            data[f'raw_summary_mlp_seed{seed}'] = added[f'raw_summary_seed{seed}']
        np.savez_compressed(out/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(), **data)
    pd.DataFrame(receipts).to_csv(out/'training.csv', index=False)
    base.write_json(out/'checks.json', checks)
    base.write_json(out/'protocol.json', dict(source=name, fit=len(fit), validation=len(val),
        raw_dim=x.shape[1], aux_dim=aux.shape[1], source_only=True,
        fit_roles=['fit_train'], selection_roles=['fit_validation'], alphas=base.ALPHAS,
        new_fits=2, mlp='cached three seeds, no ensembling', full_M123=False))
    print('FROZEN', name, receipts, checks, flush=True)


def evaluate():
    rows = []
    for name in ('oss1', 'oss2'):
        other = 'oss2' if name == 'oss1' else 'oss1'
        cross = base.TRANSFER/f'{name}-to-{other}'
        source = pd.read_csv(ROOT/f'{name}_m123_replication_20261026/stack_table.csv', dtype={'mmsi':str})
        target = base.tr.read_table(cross/'target_table.csv', labels=True)
        for side, table in [('source', source), ('cross', target)]:
            z = base.tr.load_npz(HERE/name/f'{side}_predictions.npz')
            np.testing.assert_array_equal(z['obs_id'], table.obs_id)
            groups = table.groupby('role').indices if side == 'source' else {'cross':np.arange(len(table))}
            for key, score in z.items():
                if key == 'obs_id':
                    continue
                for cohort, ids in groups.items():
                    rows.append(dict(source=name, cohort=cohort, method=key,
                        **base.classification_metrics(table.iloc[ids].class_id.to_numpy(int), score[ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
    frame = pd.DataFrame(rows)
    frame['arm'] = frame.method.str.replace(r'_seed\d+$', '', regex=True)
    summary = frame.groupby(['source','cohort','arm'])[['OA','BA']].agg(['mean','std'])
    summary.to_csv(HERE/'summary.csv')
    base.write_json(HERE/'complete.json', dict(training_finished=True, new_fits=4,
        target_selection=False, raw_review_pending=True, all_predictions_frozen_before_target_evaluation=True))
    print(frame.query("cohort=='cross'").groupby(['source','arm'])[['OA','BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    for source in ('oss1', 'oss2'):
        fit_source(source)
    evaluate()
