"""Read-only norm/direction accounting of existing eight-class spatial std."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
EIGHT = PARENT.parent


def main():
    started = time.perf_counter()
    t = pd.read_csv(EIGHT/'cohort.csv', dtype={'mmsi': str})
    z = np.load(PARENT/'std_features.npz')
    assert np.array_equal(z['instance_ids'], t.instance_id)
    delta, valid = z['delta_pol'], z['valid']
    assert delta.shape == (3, len(t), 2, 8)
    assert np.isfinite(delta).all() and not delta[:, ~valid].any()
    y = t.class_id.to_numpy()
    names = t[['class_id', 'class_name']].drop_duplicates().set_index('class_id').class_name.to_dict()
    norm_rows, stats, correlations, pairs, checks = [], [], [], [], []
    norms = []
    for fold in range(3):
        model = np.load(EIGHT/'baseline/models'/f'fold{fold}_outer.npz')
        inputs = np.load(EIGHT/'baseline'/f'fold{fold}_inputs.npz')
        assert np.array_equal(model['fit'], inputs['train'])
        assert np.array_equal(inputs['instance_ids'], t.instance_id)
        w = model['W'].reshape(8, 2, 1024).transpose(1, 0, 2)
        effective = w-w.mean(1, keepdims=True)
        raw_norm = np.linalg.norm(w, axis=-1)
        norm = np.linalg.norm(effective, axis=-1)
        assert (norm > 0).all()
        norms.append(norm)
        unit = delta[fold]/norm[None]
        for p, pol in enumerate(['vv', 'vh']):
            for c in range(8):
                norm_rows.append(dict(fold=fold, pol=pol, direction=c, class_name=names[c],
                    raw_W_norm=raw_norm[p, c], effective_W_norm=norm[p, c],
                    relative_norm_difference=abs(norm[p, c]-raw_norm[p, c])/raw_norm[p, c],
                    Wmean_over_Wrms=np.linalg.norm(w[p].mean(0))/np.sqrt(np.mean(raw_norm[p]**2))))
        checks.append(dict(fold=fold, cache_factorization_error=float(abs(unit*norm[None]-delta[fold]).max()),
            pol_sum_error=float(abs(delta[fold].sum(1)-z['delta'][fold]).max())))
        for role in ['train', 'test']:
            indices = inputs[role]
            groups = [('all', indices)]+[(str(c), indices[y[indices] == c]) for c in range(8)]
            for truth, ix in groups:
                if not len(ix):
                    continue
                # Missing polarizations remain zero in integrated means, matching the real score.
                mean_raw = delta[fold, ix].mean(0)
                mean_unit = unit[ix].mean(0)
                for p, pol in enumerate(['vv', 'vh']):
                    observed = ix[valid[ix, p]]
                    mr = delta[fold, observed, p].mean(0)
                    mu = unit[observed, p].mean(0)
                    for c in range(8):
                        stats.append(dict(fold=fold, role=role, truth=truth, n=len(ix), pol=pol,
                            valid_n=len(observed), valid_fraction=len(observed)/len(ix), direction=c,
                            class_name=names[c], W_norm=norm[p, c], raw_mean_valid=mr[c], unit_mean_valid=mu[c],
                            raw_mean_all=mean_raw[p, c], unit_mean_all=mean_unit[p, c]))
                    correlations.append(dict(fold=fold, role=role, truth=truth, pol=pol,
                        norm_vs_raw_pearson=float(np.corrcoef(norm[p], mr)[0, 1]),
                        norm_vs_raw_spearman=float(spearmanr(norm[p], mr).statistic),
                        norm_vs_unit_pearson=float(np.corrcoef(norm[p], mu)[0, 1]),
                        raw_order='>'.join(map(str, np.argsort(-mr))),
                        norm_order='>'.join(map(str, np.argsort(-norm[p]))),
                        unit_order='>'.join(map(str, np.argsort(-mu)))))
                for c in [5, 7]:
                    for b in range(8):
                        if c == b:
                            continue
                        nc, nb = norm[:, c], norm[:, b]
                        uc, ub = mean_unit[:, c], mean_unit[:, b]
                        # Symmetric two-factor accounting; additive identity, not causal attribution.
                        norm_term = (nc-nb)*(uc+ub)/2
                        direction_term = (uc-ub)*(nc+nb)/2
                        gap = mean_raw[:, c]-mean_raw[:, b]
                        assert np.max(abs(gap-norm_term-direction_term)) < 1e-12
                        for p, pol in [(0, 'vv'), (1, 'vh'), (None, 'sum')]:
                            take = lambda a: float(a.sum() if p is None else a[p])
                            ng, dg, gg = take(norm_term), take(direction_term), take(gap)
                            pairs.append(dict(fold=fold, role=role, truth=truth, n=len(ix), pol=pol,
                                direction=c, reference=b, direction_name=names[c], reference_name=names[b],
                                raw_gap=gg, norm_component=ng, unit_direction_component=dg,
                                norm_abs_fraction=abs(ng)/(abs(ng)+abs(dg)) if abs(ng)+abs(dg) else np.nan,
                                signed_norm_fraction=ng/gg if abs(gg)>1e-12 else np.nan))
    norms = np.stack(norms)
    maps = np.load(PARENT/'validation_maps.npz')
    replay, accum = [], {}
    for key in maps.files:
        if not key.endswith('_r'):
            continue
        prefix = key[:-2]
        row_token, pol, rotation = prefix.split('_')
        row = int(row_token[3:]); p = ['vv', 'vh'].index(pol)
        r = maps[key].astype(float)
        weight = maps[prefix+'_weight'].astype(float)
        weight /= weight.sum()
        unit_r = r/norms[:, p, :, None, None]
        center = (unit_r*weight).sum((-2, -1), keepdims=True)
        unit_std = np.sqrt(((unit_r-center)**2*weight).sum((-2, -1)))
        rebuilt = unit_std*norms[:, p]
        accum.setdefault((row, p), np.zeros((3, 8)))
        accum[(row, p)] += rebuilt/4
    for (row, p), value in accum.items():
        err = float(abs(value-delta[:, row, p]).max())
        assert err < 1e-12
        replay.append(dict(row_index=row, instance_id=t.iloc[row].instance_id, pol=['vv', 'vh'][p],
            C4_mean_reconstruction_error=err))
    tables = {'weight_norms': norm_rows, 'source_target_direction_stats': stats,
              'direction_correlations': correlations, 'pair_factorization': pairs,
              'factorization_checks': checks, 'cached_map_replay': replay}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
    pp = pd.DataFrame(pairs)
    pooled = []
    for keys, g in pp[pp.pol.eq('sum')].groupby(['role', 'truth', 'direction', 'reference']):
        role, truth, c, b = keys
        avg = lambda col: float(np.average(g[col], weights=g.n))
        ng, dg, gap = [avg(k) for k in ['norm_component', 'unit_direction_component', 'raw_gap']]
        pooled.append(dict(role=role, truth=truth, direction=c, reference=b, direction_name=names[c],
            reference_name=names[b], fold_sample_exposures=int(g.n.sum()), raw_gap=gap,
            norm_component=ng, unit_direction_component=dg,
            norm_abs_fraction=abs(ng)/(abs(ng)+abs(dg)),
            signed_norm_fraction=ng/gap if abs(gap)>1e-12 else np.nan))
    pd.DataFrame(pooled).to_csv(HERE/'pooled_pair_accounting.csv', index=False)
    summary = dict(status='complete',seconds=time.perf_counter()-started,reader_fits=0,new_arms=0,
        selections=0,GPU_extractions=0,rows=len(t),train_exposures=13229,test_unique=4744,
        exact_weight='W_c - class_mean(W), separately per polarization',
        unit_statistic='C4 average of footprint-weighted population std in unit effective-W direction',
        max_raw_effective_norm_relative_difference=max(r['relative_norm_difference'] for r in norm_rows),
        max_Wmean_over_Wrms=max(r['Wmean_over_Wrms'] for r in norm_rows),
        max_cached_map_reconstruction_error=max(r['C4_mean_reconstruction_error'] for r in replay),
        source_summary='source train only; all and each true class; unweighted descriptive means',
        pooling_scope='train exposures can repeat across folds; test IDs are disjoint; no independent replication claim',
        interpretation='symmetric factor accounting is not causal and does not recommend normalized scoring')
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    p = pd.DataFrame(pooled)
    print(p[p.truth.eq('all') & p.reference.isin([0, 3])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
