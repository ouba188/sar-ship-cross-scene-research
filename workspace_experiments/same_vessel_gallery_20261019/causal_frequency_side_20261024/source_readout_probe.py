"""Trace frozen perturbation effects to source data and refit source-only heads."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import RidgeClassifier
import public_oss1_probe as p
import polarization_attribution as a

OUT = p.ROOT/'round25_source_readout'


def fit(raw, labels, ids, old, scaler='fixed', branch='joint'):
    mean = old['mean'] if scaler == 'fixed' else raw[ids].mean(0)
    std = old['std'] if scaler == 'fixed' else np.maximum(raw[ids].std(0), 1e-6)
    x = (raw-mean)/std/np.sqrt(raw.shape[1])
    if branch == 'vv':
        x[:, 128:256] = 0
    elif branch == 'vh':
        x[:, :128] = 0
    y = labels[ids]
    counts = np.bincount(y, minlength=8)
    assert (counts > 0).all()
    weights = len(ids)/(8*counts[y])
    model = RidgeClassifier(alpha=10., solver='cholesky').fit(x[ids], y, sample_weight=weights)
    coef = model.coef_.copy()
    if branch == 'vv':
        coef[:, 128:256] = 0
    elif branch == 'vh':
        coef[:, :128] = 0
    return dict(mean=mean, std=std, coef=coef, intercept=model.intercept_), x, weights


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(p.OUT/'baseline.csv', dtype={'mmsi': str}).set_index('obs_id', drop=False)
    splits = pd.read_csv(p.OUT/'splits.csv')
    raw = np.load(p.OUT/'native_features.npy')
    labels = frame.class_id.to_numpy()
    coords = pd.read_csv(a.OUT/'coordinate_contributions.csv')
    focus = pd.read_csv(p.OUT/'case_records.csv').drop_duplicates('obs_id').set_index('obs_id')
    old_effects = pd.read_csv(a.OUT/'interventions.csv')
    old_effects = old_effects[old_effects['mode'].eq('common') & old_effects.dose.eq(1.)]
    interventions = []
    for (oid, region, band), g in coords.groupby(['obs_id', 'region', 'band']):
        g = g.sort_values('coordinate')
        assert np.array_equal(g.coordinate, np.arange(128))
        for k, pol in enumerate(('vv', 'vh')):
            df = np.zeros(raw.shape[1])
            df[k*128:(k+1)*128] = g[pol+'_df']
            interventions.append(dict(obs_id=int(oid), region=region, band=band, pol=pol, df=df,
                focus=bool(region == focus.loc[oid, 'region'] and band == focus.loc[oid, 'band'])))
    records, contributions, coverage, skips, channel_stats = [], [], [], [], []
    errors = dict(head_replay=0., effect_replay=0., source_sum=0.)
    fit_count = 0

    def evaluate(head, items, fold, arm, removed='', scaler='fixed', n=0):
        for item in items:
            oid, df = item['obs_id'], item['df']
            row = frame.loc[oid]
            y, rival = int(row.class_id), int(row.rival)
            scores = p.q.d.score(raw[oid], head)
            altered = p.q.d.score(raw[oid]+df, head)
            effect = float((head['coef'][y]-head['coef'][rival])@(df/head['std']/np.sqrt(len(df))))
            assert abs((altered[y]-altered[rival])-(scores[y]-scores[rival])-effect) < 1e-12
            records.append(dict(**{k:v for k,v in item.items() if k != 'df'}, fold=fold, arm=arm,
                removed=removed, scaler=scaler, source_n=n, class_id=y, rival=rival,
                base_pred=int(scores.argmax()), altered_pred=int(altered.argmax()),
                margin=float(scores[y]-scores[rival]), delta=effect,
                rescued=bool(scores.argmax()!=y and altered.argmax()==y),
                harmed=bool(scores.argmax()==y and altered.argmax()!=y),
                **{f'base_score_{c}':float(v) for c,v in enumerate(scores)},
                **{f'altered_score_{c}':float(v) for c,v in enumerate(altered)}))

    for fold in range(3):
        ids = splits[splits.fold.eq(fold) & splits.partition.eq('train')].obs_id.to_numpy()
        source = frame.loc[ids]
        query = frame[frame.fold.eq(fold)]
        assert not set(source.overpass) & set(query.overpass)
        assert not set(source.mmsi) & set(query.mmsi)
        items = [v for v in interventions if frame.loc[v['obs_id'], 'fold'] == fold]
        old = dict(np.load(p.OUT/f'head_f{fold}.npz'))
        head, x, weights = fit(raw, labels, ids, old)
        fit_count += 1
        error = max(float(abs(head[k]-old[k]).max()) for k in head)
        errors['head_replay'] = max(errors['head_replay'], error)
        assert error < 1e-10
        evaluate(head, items, fold, 'baseline', n=len(ids))
        for item in items:
            previous = old_effects[(old_effects.obs_id==item['obs_id']) & old_effects.pol.eq(item['pol'])
                & old_effects.region.eq(item['region']) & old_effects.band.eq(item['band'])].iloc[0]
            latest = next(r for r in reversed(records) if r['obs_id']==item['obs_id']
                and all(r[k]==item[k] for k in ('region','band','pol')))
            errors['effect_replay'] = max(errors['effect_replay'], abs(latest['delta']-previous.delta_fixed))
        for (pas, cid), g in source.groupby(['overpass', 'class_id']):
            coverage.append(dict(fold=fold, overpass=pas, class_id=int(cid), n=len(g), ships=g.mmsi.nunique()))

        # This exact linear contribution includes all sources in the covariance inverse.
        xc = x[ids]-np.average(x[ids], axis=0, weights=weights)
        covariance = xc.T@(weights[:, None]*xc)+10*np.eye(raw.shape[1])
        factor = cho_factor(covariance, lower=True)
        for item in items:
            oid = item['obs_id']
            y, rival = int(frame.loc[oid, 'class_id']), int(frame.loc[oid, 'rival'])
            direction = item['df']/head['std']/np.sqrt(raw.shape[1])
            target = 2*((labels[ids]==y).astype(float)-(labels[ids]==rival).astype(float))
            target -= np.average(target, weights=weights)
            c = weights*(xc@cho_solve(factor, direction))*target
            expected = (head['coef'][y]-head['coef'][rival])@direction
            errors['source_sum'] = max(errors['source_sum'], abs(c.sum()-expected))
            if item['focus']:
                for sid, value in zip(ids, c):
                    sr = frame.loc[sid]
                    contributions.append(dict(obs_id=oid, pol=item['pol'], region=item['region'], band=item['band'],
                        fold=fold, source_id=int(sid), source_mmsi=sr.mmsi, source_pass=sr.overpass,
                        source_class=int(sr.class_id), contribution=float(value)))
        # Report the previously highlighted maximum channel without selecting a new channel.
        for pol, coord in [('vv', 75), ('vh', 203)]:
            for (pas, cid), g in source.groupby(['overpass','class_id']):
                channel_stats.append(dict(fold=fold, pol=pol, coordinate=coord, source_pass=pas,
                    class_id=int(cid), n=len(g), mean=float(raw[g.obs_id,coord].mean()),
                    sd=float(raw[g.obs_id,coord].std())))
        for pol in ('vv', 'vh'):
            branch_head, _, _ = fit(raw, labels, ids, old, branch=pol)
            fit_count += 1
            evaluate(branch_head, [v for v in items if v['pol']==pol], fold, 'single_'+pol, n=len(ids))
        for pas in sorted(source.overpass.unique()):
            kept = ids[source.overpass.to_numpy()!=pas]
            missing = sorted(set(range(8))-set(labels[kept]))
            if missing:
                skips.append(dict(fold=fold, removed=pas, missing_classes=','.join(map(str,missing)), source_n=len(kept)))
                continue
            for scaler in ('fixed', 'refit'):
                new_head, _, _ = fit(raw, labels, kept, old, scaler=scaler)
                fit_count += 1
                evaluate(new_head, items, fold, 'leave_pass_out', removed=pas, scaler=scaler, n=len(kept))
        print('fold', fold, 'source', len(ids), 'passes', source.overpass.nunique(), 'fits', fit_count, flush=True)

    assert errors['effect_replay'] < 1e-12 and errors['source_sum'] < 1e-12
    results = pd.DataFrame(records)
    contrib = pd.DataFrame(contributions)
    results.to_csv(OUT/'refit_effects.csv', index=False)
    contrib.to_csv(OUT/'source_contributions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(coverage).to_csv(OUT/'source_coverage.csv', index=False)
    pd.DataFrame(skips, columns=['fold','removed','missing_classes','source_n']).to_csv(OUT/'skipped_refits.csv', index=False)
    pd.DataFrame(channel_stats).to_csv(OUT/'channel11_source_statistics.csv', index=False)
    contrib.groupby(['obs_id','pol','source_pass']).contribution.agg(['sum','size']).reset_index().to_csv(OUT/'pass_contributions.csv',index=False)
    contrib.groupby(['obs_id','pol','source_class']).contribution.agg(['sum','size']).reset_index().to_csv(OUT/'class_contributions.csv',index=False)
    summaries = []
    for (oid, region, band, pol, scaler), g in results[results.arm.eq('leave_pass_out')].groupby(['obs_id','region','band','pol','scaler']):
        b = results[results.arm.eq('baseline') & results.obs_id.eq(oid) & results.region.eq(region)
            & results.band.eq(band) & results.pol.eq(pol)].iloc[0]
        summaries.append(dict(obs_id=oid, region=region, band=band, pol=pol, scaler=scaler, focus=bool(b.focus),
            baseline=b.delta, n=len(g), min=float(g.delta.min()), max=float(g.delta.max()),
            median=float(g.delta.median()), same_sign=int((g.delta*b.delta>0).sum()),
            positive=int(g.delta.gt(0).sum()), rescued=int(g.rescued.sum()), harmed=int(g.harmed.sum())))
    pd.DataFrame(summaries).to_csv(OUT/'stability_summary.csv', index=False)
    results[results.focus].to_csv(OUT/'focus_effects.csv', index=False)
    selected = panels(frame, contrib, results)
    selected.to_csv(OUT/'source_case_selection.csv', index=False)
    run = dict(status='complete', fits=fit_count, skipped_passes=len(skips), errors=errors,
        seconds=time.perf_counter()-started, cpu_threads=2, cases=sorted(focus.index.tolist()),
        source_contribution_rows=len(contrib), effects=len(results))
    (OUT/'run.json').write_text(json.dumps(run,indent=2), encoding='utf-8')
    print(json.dumps(run,indent=2),flush=True)


def panels(frame, contrib, results):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, dets = p.load()
    selection = []
    for oid in (10, 81):
        chosen = []
        for pol in ('vv','vh'):
            group = contrib[contrib.obs_id.eq(oid) & contrib.pol.eq(pol)]
            for sign in ('positive','negative'):
                row = group.loc[group.contribution.idxmax() if sign=='positive' else group.contribution.idxmin()]
                chosen.append((int(row.source_id), f'{pol} {sign}: {row.contribution:+.6f}'))
                selection.append(dict(query_id=oid, sign=sign, **row.to_dict()))
        entries = [(oid,'query')]+chosen
        fig, axes = plt.subplots(len(entries), 2, figsize=(9, 16), constrained_layout=True)
        for r,(sid,label) in enumerate(entries):
            sr = frame.loc[sid]
            for k,pol in enumerate(('vv','vh')):
                im = p.image(sr,pol)
                axes[r,k].imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                det = dets[sid,pol]
                if det['axis_valid']:
                    corners = np.array(p.make_view(det)['priors']['detections'][0]['corners'])
                    corners = np.vstack([corners,corners[0]])
                    axes[r,k].plot(corners[:,0],corners[:,1],color='lime',lw=.6)
                axes[r,k].set_title(f'{sid} {sr.class_name} {pol} {im.shape}\n{label}',fontsize=8)
                axes[r,k].set_xlabel(f'{sr.overpass}; MMSI {sr.mmsi}',fontsize=7)
        fig.suptitle(f'Query {oid}: exact signed TRAINING contributions, not visual similarity\nNative pixel images; each subplot auto-fitted, no inference rescaling')
        fig.savefig(OUT/f'source_cases_{oid}.png',dpi=125)
        plt.close(fig)
    return pd.DataFrame(selection)


if __name__ == '__main__':
    main()
