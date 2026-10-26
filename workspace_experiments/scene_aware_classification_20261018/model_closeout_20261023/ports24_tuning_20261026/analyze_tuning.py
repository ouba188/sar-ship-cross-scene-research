"""Post-freeze paired evaluation and native query/reference evidence panels."""
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), '.mplcache')
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tune import HERE, PREV, SEEDS, load, write
import analyze as previous


def main():
    assert (HERE/'prediction_complete.json').exists()
    source = pd.read_csv(PREV/'source_truth.csv', dtype={'mmsi': str})
    target = pd.read_csv(PREV/'target_truth.csv', dtype={'mmsi': str})
    table = pd.concat((source, target), ignore_index=True)
    yy = table.class_id.to_numpy(int)
    prior = previous.all_scores(source, target)
    scores, payloads = {}, {}
    for seed in SEEDS:
        scores[seed] = {k: prior[seed][v] for k, v in (
            ('untuned', 'both_frequency'), ('bare', 'both_bare'), ('old_full', 'old_frequency'),
            ('pooled_raw_mlp', 'pooled_raw_mlp'), ('pooled_raw_summary_mlp', 'pooled_raw_summary_mlp'))}
        for arm in ('primary', 'runner_up', 'lambda_only'):
            z = load(HERE/'results'/f'seed{seed}'/f'{arm}.npz')
            np.testing.assert_array_equal(z['obs_id'].astype(str), table.obs_id.astype(str))
            scores[seed][arm] = z['score']
            payloads[seed, arm] = z
    cohorts = dict(development485=np.flatnonzero(source.role.eq('evaluation')),
                   coverage3075=np.arange(len(source), len(table)))
    comparisons = [('primary', k) for k in ('untuned', 'old_full', 'bare', 'lambda_only',
                                           'pooled_raw_mlp', 'pooled_raw_summary_mlp')]
    comparisons += [('runner_up', 'untuned'), ('lambda_only', 'untuned')]
    rows, paired, classes, grouped, intervals, pp = [], [], [], [], [], []
    for cohort, ix in cohorts.items():
        f, y = table.iloc[ix].reset_index(drop=True), yy[ix]
        for seed in SEEDS:
            for arm, s in scores[seed].items():
                rows.append(dict(cohort=cohort, seed=seed, method=arm, **previous.metrics(y, s[ix])))
                p = f[['obs_id', 'mmsi', 'pass_group', 'port', 'class_id', 'class_name']].copy()
                p['cohort'], p['seed'], p['method'] = cohort, seed, arm
                p['prediction'] = s[ix].argmax(1)
                p['margin'] = previous.margins(s[ix], y)
                pp.append(p)
            for arm, baseline in comparisons:
                a, b = (scores[seed][name][ix].argmax(1) == y for name in (arm, baseline))
                d = a.astype(float)-b
                paired.append(dict(cohort=cohort, seed=seed, method=arm, baseline=baseline,
                    rescued=int((a & ~b).sum()), harmed=int((b & ~a).sum()),
                    OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y == c].mean() for c in range(8)])))
                for c in range(8):
                    mask = y == c
                    classes.append(dict(cohort=cohort, seed=seed, method=arm, baseline=baseline,
                        class_id=c, n=int(mask.sum()), recall_pp=100*d[mask].mean(),
                        rescued=int((a[mask] & ~b[mask]).sum()), harmed=int((b[mask] & ~a[mask]).sum())))
                for field in ('pass_group', 'port'):
                    for group, gi in f.groupby(field).indices.items():
                        grouped.append(dict(cohort=cohort, seed=seed, method=arm, baseline=baseline,
                            field=field, group=group, n=len(gi), OA_pp=100*d[gi].mean()))
        groups = list(f.groupby('pass_group').indices.values())
        counts = np.array([np.bincount(y[g], minlength=8) for g in groups])
        draws = np.random.default_rng(1026).integers(len(groups), size=(2000, len(groups)))
        ct = counts[draws].sum(1)
        complete = (ct > 0).all(1)
        for arm, baseline in comparisons:
            d = np.mean([(scores[s][arm][ix].argmax(1) == y).astype(float)-
                         (scores[s][baseline][ix].argmax(1) == y) for s in SEEDS], 0)
            sums = np.array([np.bincount(y[g], weights=d[g], minlength=8) for g in groups])[draws].sum(1)
            oa, ba = 100*sums.sum(1)/ct.sum(1), 100*(sums[complete]/ct[complete]).mean(1)
            intervals.append(dict(cohort=cohort, method=arm, baseline=baseline,
                OA_low=np.quantile(oa, .025), OA_high=np.quantile(oa, .975),
                BA_low=np.quantile(ba, .025), BA_high=np.quantile(ba, .975)))
    for name, values in (('metrics', rows), ('paired', paired), ('by_class', classes),
                         ('by_group', grouped), ('uncertainty', intervals)):
        pd.DataFrame(values).to_csv(HERE/f'{name}.csv', index=False)
    pd.concat(pp).to_csv(HERE/'predictions.csv.gz', index=False, compression='gzip')
    summary = pd.DataFrame(rows).groupby(['cohort', 'method'])[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std'])
    summary.to_csv(HERE/'summary.csv')
    cases = panels(table, scores, payloads, cohorts['coverage3075'])
    gate_diagnostic()
    write(HERE/'analysis_complete.json', dict(cases=cases, target_based_selection=False,
        native_review='generated; inspection recorded in FINDINGS_ZH.md'))
    print(summary.to_string(), flush=True)


def gate_diagnostic():
    selected = json.loads((HERE/'selected.json').read_text(encoding='utf-8'))['primary']
    source = pd.read_csv(PREV/'source_truth.csv')
    target = pd.read_csv(PREV/'target_truth.csv')
    y = target.class_id.to_numpy(int)
    n, rows = len(source), []
    for seed in SEEDS:
        z = load(HERE/'results'/f'seed{seed}/primary.npz')
        receipt = json.loads((HERE/'training'/selected['config']/f'seed{seed}/complete.json').read_text(encoding='utf-8'))
        if selected['config'] == 'default':
            receipt = json.loads((PREV/'training'/f'seed{seed}/selection.json').read_text(encoding='utf-8'))
        alpha, lam = receipt['alpha'], selected['correction_scale']
        available = load(PREV/'results'/f'seed{seed}/both.npz')['available']
        w = lam*available*np.array([alpha, 1-alpha])
        fixed = z['bare']+w[:, :1]*z['delta_vv']+w[:, 1:]*z['delta_vh']
        a, b = z['score'][n:].argmax(1) == y, fixed[n:].argmax(1) == y
        delta = a.astype(float)-b
        rows.append(dict(seed=seed, alpha=alpha, frequency_step=receipt['frequency_step'],
            changed_weights=int((abs(z['weight_frequency'][n:]-w[n:]).max(1)>1e-7).sum()),
            changed_predictions=int((z['score'][n:].argmax(1)!=fixed[n:].argmax(1)).sum()),
            rescued=int((a & ~b).sum()), harmed=int((b & ~a).sum()),
            OA_gain_pp=100*delta.mean(), BA_gain_pp=100*np.mean([delta[y==c].mean() for c in range(8)])))
    pd.DataFrame(rows).to_csv(HERE/'gate_activity.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def panels(table, scores, payloads, ids):
    y = table.class_id.to_numpy(int)
    plans = [('untuned', ('rescue', 'harm', 'CC', 'WW')), ('old_full', ('rescue', 'harm'))]
    chosen = []
    for baseline, statuses in plans:
        for status in statuses:
            eligible = []
            for seed in SEEDS:
                a, b = scores[seed]['primary'][ids], scores[seed][baseline][ids]
                ac, bc = a.argmax(1) == y[ids], b.argmax(1) == y[ids]
                mask = dict(rescue=ac & ~bc, harm=bc & ~ac, CC=ac & bc, WW=~ac & ~bc)[status]
                delta = abs(previous.margins(a, y[ids])-previous.margins(b, y[ids]))
                eligible.extend((float(delta[k]), seed, int(ids[k])) for k in np.flatnonzero(mask))
            if eligible:
                eligible.sort()
                _, seed, ix = eligible[len(eligible)//2]
                chosen.append(dict(baseline=baseline, status=status, seed=seed, index=ix))
    for obs in (32936, 12862):
        ix = int(np.flatnonzero(table.obs_id.eq(obs))[0])
        chosen.append(dict(baseline='untuned', status='predeclared', seed=43, index=ix))
    ex = previous.module('tuning_native_review', previous.LOCAL/'frozen_joint_coverage/extract.py')
    snapshots = {int(v['obs_id']): v for v in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    details = []
    for case in chosen:
        j, seed = case['index'], case['seed']
        z = payloads[seed, 'primary']
        a, b = scores[seed]['primary'][j], scores[seed][case['baseline']][j]
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
        w = z['weight_frequency'][j]
        parts = [float(w[p]*(z[key][j, true]-z[key][j, rival])) for p, key in enumerate(('delta_vv', 'delta_vh'))]
        bm = float(z['bare'][j, true]-z['bare'][j, rival])
        assert abs(bm+sum(parts)-(a[true]-a[rival])) < 1e-4
        fig.suptitle(f"Tuned vs {case['baseline']}: {case['status']} seed{seed} true={true} pred {b.argmax()}->{a.argmax()}\n"
            f"pair margin {b[true]-b[rival]:+.3f}->{a[true]-a[rival]:+.3f}; base={bm:+.3f}, VV={parts[0]:+.3f}, VH={parts[1]:+.3f}")
        fig.text(.02, .02, 'Native crop/pad, no resize. Actual references. Source-selected parameters; no physical-causal claim.', fontsize=9)
        fig.subplots_adjust(top=.82, bottom=.08, wspace=.15, hspace=.2)
        name = f"{case['baseline']}_{case['status']}_{int(table.iloc[j].obs_id)}_{seed}.png"
        fig.savefig(out/name, dpi=110)
        plt.close(fig)
        provenance = json.loads(json.dumps(provenance, default=lambda x: np.asarray(x).tolist()))
        details.append(dict(**case, obs_id=int(table.iloc[j].obs_id), true=true, rival=rival,
            before_prediction=int(b.argmax()), after_prediction=int(a.argmax()),
            before_margin=float(b[true]-b[rival]), after_margin=float(a[true]-a[rival]),
            base_margin=bm, branch_contributions=parts, effective_weights=w.tolist(),
            reference_obs=table.iloc[refs].obs_id.astype(int).tolist(), image=name, provenance=provenance))
    write(out/'cases.json', details)
    return len(details)


if __name__ == '__main__':
    main()
