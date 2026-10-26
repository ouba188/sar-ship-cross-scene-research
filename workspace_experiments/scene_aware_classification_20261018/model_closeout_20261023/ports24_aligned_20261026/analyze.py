"""Evaluate frozen predictions, paired coverage, and traceable native-chip cases."""
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), '.mplcache')
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_bundle import HERE, ROOT, OLD, UP, EXT, LOCAL, PACKAGE, load, write
from run import SEEDS
from sar_m123 import train_ops as op


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(y, score):
    value = op.metrics(y, score)
    for key in ('OA', 'BA', 'J'):
        value[key] *= 100
    return value


def margins(score, y):
    other = np.where(np.arange(8)[None] == y[:, None], -np.inf, score).max(1)
    return score[np.arange(len(y)), y]-other


def all_scores(source, target):
    out = {}
    n = len(source)
    for seed in SEEDS:
        scores = {}
        for name in ('both', 'frozen_metric', 'cosine_fixed'):
            z = load(HERE/'results'/f'seed{seed}'/f'{name}.npz')
            for end in ('bare', 'B_fixed', 'spatial', 'frequency'):
                scores[name+'_'+end] = z[end]
        oldsrc = source.source_index.to_numpy(int)
        dev = np.flatnonzero(source.role.eq('evaluation'))
        scores['old_s12'] = np.concatenate((load(UP/f'strategies_seed{seed}.npz')['logits'][oldsrc, 2],
                                            load(EXT/f'strategies_seed{seed}.npz')['logits'][:, 2]))
        for name, path, key, indexkey in (
            ('old_frequency', ROOT/f'evidence_preserving_redesign/anchored_polar_adjustment_20261025/results/frequency_{seed}.npz', 'score', 'up_index'),
            ('old_both', LOCAL/f'pair_conditioned_joint_metric/results/pair_joint_{seed}.npz', 'score', 'ids')):
            zz = load(path)
            ix = pd.Index(zz[indexkey]).get_indexer(oldsrc[dev])
            assert (ix >= 0).all()
            s = np.full((n+len(target), 8), np.nan, np.float32)
            s[dev] = zz[key][ix]
            old = load(OLD/'results'/f'seed{seed}.npz')
            np.testing.assert_array_equal(old['instance_ids'].astype(str), target.instance_id)
            s[n:] = old['frequency' if name == 'old_frequency' else 'old_both']
            scores[name] = s
        out[seed] = scores
    # Baseline script writes prediction-only arrays before labels are loaded here.
    if (HERE/'baselines/complete.json').exists():
        manifest = json.loads((HERE/'baselines/complete.json').read_text(encoding='utf-8'))
        for mode in ('fit_only', 'pooled'):
            src = load(HERE/'baselines'/mode/'source_predictions.npz')
            dst = load(HERE/'baselines'/mode/'external_predictions.npz')
            np.testing.assert_array_equal(src['obs_id'], source.obs_id)
            np.testing.assert_array_equal(dst['obs_id'], target.obs_id)
            for seed in SEEDS:
                for key in src:
                    if key in ('obs_id', 'class_id'):
                        continue
                    if 'seed' in key and not key.endswith(f'seed{seed}'):
                        continue
                    name = mode+'_'+key.replace(f'_seed{seed}', '')
                    out[seed][name] = np.concatenate((src[key], dst[key]))
    return out


def main():
    assert (HERE/'prediction_complete.json').exists(), 'Freeze predictions first.'
    source = pd.read_csv(HERE/'source_truth.csv', dtype={'mmsi': str})
    target = pd.read_csv(HERE/'target_truth.csv', dtype={'mmsi': str})
    table = pd.concat((source, target), ignore_index=True)
    y = table.class_id.to_numpy(int)
    cohorts = dict(development485=np.flatnonzero(source.role.eq('evaluation')),
                   coverage3075=np.arange(len(source), len(table)))
    scores = all_scores(source, target)
    rows, paired, byclass, byport, stratified, intervals, predrows = [], [], [], [], [], [], []
    comparisons = [('both_frequency', 'old_frequency'), ('both_frequency', 'old_both'),
                   ('both_frequency', 'both_bare'), ('both_frequency', 'frozen_metric_frequency'),
                   ('both_frequency', 'cosine_fixed_frequency'), ('both_frequency', 'both_B_fixed'),
                   ('both_frequency', 'both_spatial'), ('both_bare', 'old_s12'),
                   ('old_frequency', 'old_s12'), ('both_bare', 'fit_only_raw_mlp')]
    for baseline in ('fit_only_raw_mlp', 'pooled_raw_mlp', 'pooled_raw_ridge',
                     'pooled_raw_summary_mlp', 'pooled_raw_summary_ridge'):
        if baseline in scores[17]:
            comparisons.append(('both_frequency', baseline))
    srcfeatures = load(HERE/'bundle/features.npz')
    extfeatures = load(HERE/'bundle/external.npz')
    valid = np.concatenate((srcfeatures['mass'], extfeatures['mass'])).sum(-1) > 0
    for cohort, ix in cohorts.items():
        f = table.iloc[ix].reset_index(drop=True)
        yy = y[ix]
        for seed in SEEDS:
            for method, s in scores[seed].items():
                assert np.isfinite(s[ix]).all(), (seed, method, cohort)
                rows.append(dict(cohort=cohort, seed=seed, method=method, **metrics(yy, s[ix])))
                if method in ('both_frequency', 'both_bare', 'old_frequency', 'cosine_fixed_frequency'):
                    p = f[['obs_id', 'mmsi', 'pass_group', 'port', 'class_id', 'class_name']].copy()
                    p['cohort'], p['seed'], p['method'] = cohort, seed, method
                    p['prediction'], p['margin'] = s[ix].argmax(1), margins(s[ix], yy)
                    predrows.append(p)
                    for paired_pol in (True, False):
                        mask = valid[ix].all(1) == paired_pol
                        if mask.any():
                            stratified.append(dict(cohort=cohort, seed=seed, method=method,
                                paired_polarization=paired_pol, **metrics(yy[mask], s[ix][mask])))
            for method, baseline in comparisons:
                b = scores[seed][baseline][ix].argmax(1) == yy
                a = scores[seed][method][ix].argmax(1) == yy
                d = a.astype(float)-b
                paired.append(dict(cohort=cohort, seed=seed, method=method, baseline=baseline,
                    rescued=int((a & ~b).sum()), harmed=int((b & ~a).sum()),
                    OA_pp=100*d.mean(), BA_pp=100*np.mean([d[yy == c].mean() for c in range(8)])))
                for c in range(8):
                    mask = yy == c
                    byclass.append(dict(cohort=cohort, seed=seed, method=method, baseline=baseline,
                        class_id=c, n=int(mask.sum()), recall_pp=100*d[mask].mean(),
                        rescued=int((a[mask] & ~b[mask]).sum()), harmed=int((b[mask] & ~a[mask]).sum())))
                for field in ('port', 'pass_group'):
                    for group, ids in f.groupby(field).indices.items():
                        byport.append(dict(cohort=cohort, seed=seed, method=method, baseline=baseline,
                            field=field, group=group, n=len(ids), OA_pp=100*d[ids].mean()))
        groups = list(f.groupby('pass_group').indices.values())
        counts = np.array([np.bincount(yy[g], minlength=8) for g in groups])
        draws = np.random.default_rng(1026).integers(len(groups), size=(2000, len(groups)))
        ct = counts[draws].sum(1)
        complete = (ct > 0).all(1)
        for method, baseline in comparisons:
            d = np.mean([(scores[seed][method][ix].argmax(1) == yy).astype(float)-
                         (scores[seed][baseline][ix].argmax(1) == yy) for seed in SEEDS], 0)
            totals = np.array([np.bincount(yy[g], weights=d[g], minlength=8) for g in groups])
            sums = totals[draws].sum(1)
            oa = 100*sums.sum(1)/ct.sum(1)
            ba = 100*(sums[complete]/ct[complete]).mean(1)
            intervals.append(dict(cohort=cohort, method=method, baseline=baseline,
                OA_low=np.quantile(oa, .025), OA_high=np.quantile(oa, .975),
                BA_low=np.quantile(ba, .025), BA_high=np.quantile(ba, .975), complete_draws=int(complete.sum())))
    for name, data in [('metrics', rows), ('paired', paired), ('by_class', byclass), ('by_group', byport),
                       ('polar_availability', stratified), ('uncertainty', intervals)]:
        pd.DataFrame(data).to_csv(HERE/f'{name}.csv', index=False)
    pd.concat(predrows).to_csv(HERE/'predictions.csv.gz', compression='gzip', index=False)
    pd.DataFrame(rows).groupby(['cohort', 'method'])[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(HERE/'summary.csv')
    cases = panels(table, scores, cohorts['coverage3075'])
    activity = []
    for seed in SEEDS:
        z = load(HERE/'results'/f'seed{seed}/both.npz')
        for cohort, ix in cohorts.items():
            for endpoint in ('frequency', 'spatial'):
                gap = z['weight_'+endpoint][ix]-z['weight_B_fixed'][ix]
                activity.append(dict(cohort=cohort, seed=seed, endpoint=endpoint,
                    nonzero_weights=int((abs(gap).max(1) > 1e-7).sum()), max_weight_change=float(abs(gap).max()),
                    predictions_changed=int((z[endpoint][ix].argmax(1) != z['B_fixed'][ix].argmax(1)).sum())))
    pd.DataFrame(activity).to_csv(HERE/'gate_activity.csv', index=False)
    selections = []
    for seed in SEEDS:
        folder = HERE/'training'/f'seed{seed}'
        record = json.loads((folder/'selection.json').read_text(encoding='utf-8'))
        record['base_step'] = int(op.load_checkpoint(folder/'raw_summary.pt')['step'])
        record['spatial_step'] = int(op.load_checkpoint(folder/'spatial.pt')['selected_step'])
        selections.append(record)
    pd.DataFrame(selections).to_csv(HERE/'training_selection.csv', index=False)
    write(HERE/'analysis_complete.json', dict(cases=cases, new_training_in_analysis=False,
        uncertainty='paired by pass; average seed effects first; 2000 draws',
        native_visual_review='panels generated; human/model inspection recorded separately'))
    print(pd.DataFrame(rows).groupby(['cohort', 'method'])[['OA', 'BA']].mean().to_string(), flush=True)


def panels(table, scores, ids):
    y = table.class_id.to_numpy(int)
    plans = [('both_frequency', 'old_frequency', ('rescue', 'harm', 'CC', 'WW')),
             ('both_frequency', 'both_bare', ('rescue', 'harm')),
             ('both_frequency', 'both_B_fixed', ('rescue', 'harm')),
             ('both_frequency', 'frozen_metric_frequency', ('rescue', 'harm')),
             ('both_frequency', 'cosine_fixed_frequency', ('rescue', 'harm')),
             ('both_spatial', 'both_B_fixed', ('rescue', 'harm'))]
    selected = []
    for method, baseline, statuses in plans:
        for status in statuses:
            eligible = []
            for seed in SEEDS:
                a, b = scores[seed][method], scores[seed][baseline]
                ac, bc = a[ids].argmax(1) == y[ids], b[ids].argmax(1) == y[ids]
                mask = dict(rescue=ac & ~bc, harm=bc & ~ac, CC=ac & bc, WW=~ac & ~bc)[status]
                delta = abs(margins(a[ids], y[ids])-margins(b[ids], y[ids]))
                eligible.extend((float(delta[k]), seed, int(ids[k])) for k in np.flatnonzero(mask))
            if eligible:
                # Select a median-sized evidence change, not only an extreme outlier.
                eligible.sort()
                change, seed, j = eligible[len(eligible)//2]
                selected.append(dict(method=method, baseline=baseline, status=status, seed=seed, index=j))
    ex = module('ports24_native_review', LOCAL/'frozen_joint_coverage/extract.py')
    snapshots = {int(v['obs_id']): v for v in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    details = []
    for case in selected:
        j, seed = case['index'], case['seed']
        z = load(HERE/'results'/f'seed{seed}/both.npz')
        a, b = scores[seed][case['method']][j], scores[seed][case['baseline']][j]
        true = int(y[j])
        rival = int(a.argmax()) if a.argmax() != true else int(np.where(np.arange(8) == true, -np.inf, b).argmax())
        refs = z['refs'][j, [true, rival]].ravel()
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        provenance = []
        for p, pol in enumerate(('VV', 'VH')):
            for k, index in enumerate([j, *refs]):
                pix, footprint, g, placement = ex.joint.geo.native(table, snapshots, int(index), p)
                axes[p, k].imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if g is not None:
                    mask = ex.joint.geo.soft_regions(g)[0]*footprint
                    if mask.max() > .5:
                        axes[p, k].contour(mask, levels=[.5], colors=['#20a780'], linewidths=.7)
                row = table.iloc[index]
                label = 'QUERY' if k == 0 else ('TRUE REF' if k < 3 else 'RIVAL REF')
                axes[p, k].set_title(f'{pol} {label} {int(row.obs_id)}\n{row.class_name}', fontsize=8)
                axes[p, k].set(xticks=[], yticks=[])
                provenance.append(dict(obs_id=int(row.obs_id), polarization=pol, path=row['image_'+pol.lower()],
                    geometry=None if g is None else np.asarray(g).tolist(), placement=placement))
        w = z['weight_'+case['method'].removeprefix('both_')][j]
        branch = [float(w[p]*(z[key][j, true]-z[key][j, rival])) for p, key in enumerate(('delta_vv', 'delta_vh'))]
        base_margin = float(z['bare'][j, true]-z['bare'][j, rival])
        assert abs(base_margin+sum(branch)-(a[true]-a[rival])) < 1e-4
        fig.suptitle(f"{case['method']} vs {case['baseline']}: {case['status']} seed{seed} true={true} pred {b.argmax()}->{a.argmax()}\n"
            f"pair margin {b[true]-b[rival]:+.3f}->{a[true]-a[rival]:+.3f}; current base={base_margin:+.3f}, VV={branch[0]:+.3f}, VH={branch[1]:+.3f}")
        fig.text(.02, .02, 'Native crop/pad, no resize. Actual retrieved references. Score decomposition is not a physical-causal attribution.', fontsize=9)
        fig.subplots_adjust(top=.82, bottom=.08, wspace=.15, hspace=.2)
        name = f"{case['baseline']}_{case['status']}_{int(table.iloc[j].obs_id)}_{seed}.png"
        fig.savefig(out/name, dpi=110)
        plt.close(fig)
        # Placement contains numpy scalar values in some legacy extractors.
        provenance = json.loads(json.dumps(provenance, default=lambda x: np.asarray(x).tolist()))
        details.append(dict(**case, obs_id=int(table.iloc[j].obs_id), true=true, rival=rival,
            before_prediction=int(b.argmax()), after_prediction=int(a.argmax()),
            before_margin=float(b[true]-b[rival]), after_margin=float(a[true]-a[rival]),
            base_margin=base_margin, branch_contributions=branch, weights=w.tolist(),
            reference_obs=table.iloc[refs].obs_id.astype(int).tolist(), image=name, provenance=provenance))
    write(out/'cases.json', details)
    return len(details)


if __name__ == '__main__':
    main()
