"""Frozen round21 interventions over all previously enumerated candidate pairs."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
import time
import numpy as np
import pandas as pd
import classification_joint_probe as q

ROOT = q.ROOT
OUT = ROOT/'round22_classification_coverage'
KEYS = ['pol', 'region', 'band', 'dose']


def overpass(product):
    fields = product.split('_')
    return '_'.join([fields[0], fields[6], fields[7]])


def infer():
    started = time.perf_counter()
    q.d.torch.set_num_threads(2)
    frame, views, disabled = q.d.ip.load_data()
    pairs = pd.read_csv(ROOT/'round21_classification_joint/candidate_pairs.csv', dtype={'mmsi': str})
    pilot = pd.read_csv(ROOT/'round21_classification_joint/selected_pairs.csv', dtype={'mmsi': str})
    pairs['previous_pilot_ship'] = pairs.mmsi.isin(pilot.mmsi)
    pairs.to_csv(OUT/'pairs.csv', index=False)
    keep = sorted(set(pairs.history_id) | set(pairs.query_id))
    cache = np.load(q.d.CONTROL/'features.npz')['full_global']
    heads = [np.load(q.d.CONTROL/f'head_f{k}_full_global.npz') for k in range(3)]
    old = pd.read_csv(q.d.CONTROL/'predictions.csv')
    old = old[old.arm.eq('full_global')].set_index('obs_id')
    model = q.d.resnet50(weights=None)
    model.load_state_dict(q.d.torch.load(q.d.ip.WEIGHTS, map_location='cpu', weights_only=True))
    stem = q.d.torch.nn.Sequential(model.conv1, model.bn1, model.relu).eval()
    del model
    records, baseline, skipped = [], [], []
    checks = dict(feature_replay=0., margin_replay=0.)
    with q.d.torch.inference_mode():
        for count, oid in enumerate(keep):
            i = frame.index[frame.obs_id.eq(oid)][0]
            row = frame.loc[i]
            head = heads[int(row.fold)]
            base = q.d.score(cache[i], head)
            y = int(row.class_id)
            wrong = base.copy(); wrong[y] = -np.inf
            rival = int(wrong.argmax()); bp = int(base.argmax())
            margin = float(base[y]-base[rival])
            checks['margin_replay'] = max(checks['margin_replay'], abs(margin-old.loc[oid, 'margin']))
            info = dict(obs_id=oid, mmsi=str(row.mmsi), class_id=y, fold=int(row.fold),
                        product=row['product'], overpass=overpass(row['product']), port=row.port,
                        previous_pilot_ship=str(row.mmsi) in set(pilot.mmsi),
                        base_pred=bp, baseline_margin=margin, rival=rival)
            baseline.append(dict(**info, **{f'score_{k}': float(v) for k, v in enumerate(base)}))
            for p, pol in enumerate(('vv', 'vh')):
                if p == 1 and disabled[i]:
                    skipped.append(dict(obs_id=oid, pol=pol, reason='existing_crossview_disabled'))
                    continue
                view = views[i]['images'][pol]
                transform = np.array(view['priors']['scale']['transform'])[[1, 2, 4, 5]].reshape(2, 2)
                assert np.allclose(transform.T@transform, np.eye(2)*100, atol=1e-5)
                im = np.asarray(q.d.ip.Image.open(row['image_'+pol]).convert('L'), dtype=float)/255
                items = q.perturb(im, view)
                if not items:
                    skipped.append(dict(obs_id=oid, pol=pol, reason='zero_matched_change'))
                    continue
                images = [im]+[x for a, level, meta in items for x in (a, level)]
                desc = q.d.ip.describe(stem, *q.d.prepare_batch(images, view), inputs=('full',))['full_global'].numpy()
                sl = slice(p*128, (p+1)*128)
                err = float(abs(desc[0]-cache[i, sl]).max())
                checks['feature_replay'] = max(checks['feature_replay'], err)
                assert err < 2e-5, (oid, pol, err)
                raw = np.tile(cache[i], (len(desc)-1, 1)); raw[:, sl] = desc[1:]
                scores = q.d.score(raw, head)
                for n, (_, _, meta) in enumerate(items):
                    s, level = scores[2*n:2*n+2]
                    pred = int(s.argmax())
                    delta = float(s[y]-s[rival]-margin)
                    ld = float(level[y]-level[rival]-margin)
                    records.append(dict(**info, pol=pol, image_path=row['image_'+pol], **meta,
                        pred=pred, delta_fixed=delta, level_delta=ld, residual=delta-ld,
                        true_score_change=float(s[y]-base[y]), rival_score_change=float(s[rival]-base[rival]),
                        rescued=bp != y and pred == y, harmed=bp == y and pred != y,
                        **{f'score_{k}': float(v) for k, v in enumerate(s)}))
            if count % 25 == 0 or count == len(keep)-1:
                print('processed', count+1, '/', len(keep), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    assert checks['margin_replay'] < 1e-5
    table = pd.DataFrame(records)
    baseline = pd.DataFrame(baseline)
    table.to_csv(OUT/'interventions.csv', index=False)
    baseline.to_csv(OUT/'baseline.csv', index=False)
    pd.DataFrame(skipped, columns=['obs_id', 'pol', 'reason']).to_csv(OUT/'skipped.csv', index=False)
    # Same endpoint can occur in two pairs; compute each pair's rival here, not via an endpoint dictionary.
    bases = baseline.set_index('obs_id')
    changes = []
    for pair in pairs.itertuples():
        a, b = bases.loc[pair.history_id], bases.loc[pair.query_id]
        assert a.fold == b.fold and a.class_id == b.class_id
        scores = np.array([a[f'score_{k}']+b[f'score_{k}'] for k in range(8)])
        scores[int(a.class_id)] = -np.inf
        rival = int(scores.argmax()); y = int(a.class_id)
        gap = (b[f'score_{y}']-b[f'score_{rival}'])-(a[f'score_{y}']-a[f'score_{rival}'])
        ta = table[table.obs_id.eq(pair.history_id)].set_index(KEYS)
        tb = table[table.obs_id.eq(pair.query_id)].set_index(KEYS)
        for key in ta.index.intersection(tb.index):
            x, z = ta.loc[key], tb.loc[key]
            altered = (z[f'score_{y}']-z[f'score_{rival}'])-(x[f'score_{y}']-x[f'score_{rival}'])
            changes.append(dict(history_id=pair.history_id, query_id=pair.query_id, mmsi=pair.mmsi,
                class_id=y, state=pair.state, previous_pilot_ship=pair.previous_pilot_ship,
                pass_pair=a.overpass+'__'+b.overpass, pair_rival=rival, **dict(zip(KEYS, key)),
                original_margin_change=gap, altered_margin_change=altered,
                intervention_gap_change=altered-gap, energy_change=z.band_energy_before-x.band_energy_before,
                history_effect=x.delta_fixed, query_effect=z.delta_fixed,
                history_residual=x.residual, query_residual=z.residual,
                same_rival=a.rival == b.rival,
                history_l2=x.l2, query_l2=z.l2))
    pd.DataFrame(changes).to_csv(OUT/'same_ship_changes.csv', index=False)
    prior = pd.read_csv(ROOT/'round21_classification_joint/interventions.csv')
    matched = prior.merge(table, on=['obs_id']+KEYS, suffixes=('_old', '_new'), validate='one_to_one')
    checks['pilot_rows'] = len(matched)
    assert len(matched) == len(prior)
    checks['pilot_effect_replay'] = float(abs(matched.delta_fixed_old-matched.delta_fixed_new).max())
    assert checks['pilot_effect_replay'] < 1e-7
    manifest = dict(pairs=len(pairs), ships=pairs.mmsi.nunique(), observations=len(keep),
                    products=baseline['product'].nunique(), overpasses=baseline.overpass.nunique(),
                    interventions=len(table), checks=checks, seconds=time.perf_counter()-started,
                    model='same frozen full_global diagnostic, not current complete model',
                    new_training=False, support_adaptation=False, physical_causality=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest), flush=True)


def estimate(g, col, rng):
    ship = g.groupby('mmsi')[col].mean().to_numpy()
    boot = ship[rng.integers(len(ship), size=(2000, len(ship)))].mean(1)
    return dict(ships=len(ship), ship_mean=float(ship.mean()), ship_ci_low=float(np.quantile(boot, .025)),
                ship_ci_high=float(np.quantile(boot, .975)), ship_median=float(np.median(ship)))


def analyze():
    t = pd.read_csv(OUT/'interventions.csv', dtype={'mmsi': str})
    b = pd.read_csv(OUT/'baseline.csv', dtype={'mmsi': str})
    pair = pd.read_csv(OUT/'same_ship_changes.csv', dtype={'mmsi': str})
    rng = np.random.default_rng(20261024)
    summaries, classes, passes, contrasts, repeated = [], [], [], [], []
    for scope, use in [('all', t), ('expansion_only', t[~t.previous_pilot_ship])]:
        for key, g in use.groupby(KEYS):
            for effect in ('delta_fixed', 'residual'):
                summaries.append(dict(scope=scope, **dict(zip(KEYS, key)), effect=effect,
                    observations=len(g), **estimate(g, effect, rng),
                    rescued=int(g.rescued.sum()), harmed=int(g.harmed.sum()),
                    mean_abs=float(abs(g[effect]).mean()), positive_fraction=float((g[effect]>0).mean())))
            for c, h in g.groupby('class_id'):
                classes.append(dict(scope=scope, **dict(zip(KEYS, key)), class_id=c, observations=len(h),
                    **estimate(h, 'delta_fixed', rng), residual_mean=h.groupby('mmsi').residual.mean().mean(),
                    rescued=int(h.rescued.sum()), harmed=int(h.harmed.sum())))
            for p, h in g.groupby('overpass'):
                passes.append(dict(scope=scope, **dict(zip(KEYS, key)), overpass=p,
                    observations=len(h), ships=h.mmsi.nunique(), mean_delta=h.delta_fixed.mean(),
                    mean_residual=h.residual.mean(), rescued=int(h.rescued.sum()), harmed=int(h.harmed.sum())))
        for effect in ('delta_fixed', 'residual'):
            wide = use.pivot(index=['obs_id', 'mmsi', 'pol', 'dose', 'class_id'],
                             columns=['region', 'band'], values=effect)
            for region in ('body', 'end', 'side'):
                z = (wide[region, 'fine']-wide[region, 'middle']).rename('effect').reset_index()
                for key, h in z.groupby(['pol', 'dose']):
                    contrasts.append(dict(scope=scope, effect=effect, pol=key[0], dose=key[1],
                        contrast=region+'_fine_minus_middle', **estimate(h, 'effect', rng)))
            for neighbor in ('end', 'side'):
                z = ((wide['body', 'fine']-wide['body', 'middle'])-
                     (wide[neighbor, 'fine']-wide[neighbor, 'middle'])).rename('effect').reset_index()
                for key, h in z.groupby(['pol', 'dose']):
                    contrasts.append(dict(scope=scope, effect=effect, pol=key[0], dose=key[1],
                        contrast='interaction_body_vs_'+neighbor, **estimate(h, 'effect', rng)))
        p = pair if scope == 'all' else pair[~pair.previous_pilot_ship]
        for key, g in p.groupby(KEYS):
            for effect in ('effect', 'residual'):
                a, z = g['history_'+effect], g['query_'+effect]
                informative = (abs(a)>1e-5) & (abs(z)>1e-5)
                stable = (a*z>0).astype(float)
                h = g.assign(stable=stable)
                repeated.append(dict(scope=scope, **dict(zip(KEYS, key)), effect=effect,
                    pairs=len(g), ships=g.mmsi.nunique(),
                    ship_weighted_same_sign=h.groupby('mmsi').stable.mean().mean(),
                    informative_pairs=int(informative.sum()),
                    informative_same_sign=float(stable[informative].mean()) if informative.any() else np.nan,
                    same_rival_pairs=int(g.same_rival.sum()),
                    same_rival_same_sign=float(stable[g.same_rival].mean()) if g.same_rival.any() else np.nan))
    for name, rows in [('summary', summaries), ('by_class', classes), ('by_overpass', passes),
                       ('contrasts', contrasts), ('same_ship_repetition', repeated)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False)
    pd.DataFrame(classes).groupby(['scope']+KEYS).agg(
        classes=('class_id', 'size'), equal_class_mean=('ship_mean', 'mean'),
        equal_class_residual=('residual_mean', 'mean'),
        positive_classes=('ship_mean', lambda x: int((x>0).sum()))).to_csv(OUT/'equal_class_summary.csv')
    pair.groupby(['previous_pilot_ship', 'state']+KEYS).agg(
        pairs=('mmsi', 'size'), ships=('mmsi', 'nunique'),
        mean_history_effect=('history_effect', 'mean'), mean_query_effect=('query_effect', 'mean'),
        natural_gap=('original_margin_change', 'mean'),
        change_in_gap=('intervention_gap_change', 'mean')).to_csv(OUT/'by_pair_state.csv')
    coverage = b.groupby(['previous_pilot_ship', 'class_id']).agg(observations=('obs_id','size'),
        ships=('mmsi','nunique'), products=('product','nunique'), overpasses=('overpass','nunique'),
        correct=('base_pred', lambda x: int((x == b.loc[x.index,'class_id']).sum())))
    coverage.to_csv(OUT/'coverage_by_class.csv')
    t[t.rescued | t.harmed].to_csv(OUT/'flips.csv', index=False)
    doses = t.pivot(index=['obs_id','pol','region','band'], columns='dose', values='delta_fixed')
    metrics = dict(dose_same_sign=float((doses[.5]*doses[1]>0).mean()),
                   dose_monotonic=float((abs(doses[1])>=abs(doses[.5])).mean()),
                   local_energy_decreased=float((t.band_energy_after<t.band_energy_before).mean()),
                   unique_rescued=int(t.loc[t.rescued,'obs_id'].nunique()),
                   unique_harmed=int(t.loc[t.harmed,'obs_id'].nunique()),
                   expansion_ships=int(b.loc[~b.previous_pilot_ship,'mmsi'].nunique()),
                   expansion_observations=int((~b.previous_pilot_ship).sum()))
    (OUT/'analysis_summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    print(json.dumps(metrics), flush=True)


def panels():
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = pd.read_csv(OUT/'interventions.csv')
    pairs = pd.read_csv(OUT/'pairs.csv')
    frame, views, _ = q.d.ip.load_data()
    choices = []
    for state, g in pairs[~pairs.previous_pilot_ship].groupby('state'):
        pair = g.sort_values(['class_id','mmsi','history_id']).iloc[0]
        h = t[t.obs_id.eq(pair.query_id) & t.pol.eq('vv') & t.region.eq('body') & t.band.eq('fine') & t.dose.eq(1)]
        if not h.empty:
            choices.append(dict(reason='natural_'+state, **h.iloc[0].to_dict()))
    for flag in ('rescued', 'harmed'):
        h = t[t[flag] & ~t.previous_pilot_ship & t.dose.eq(1)].sort_values('obs_id').drop_duplicates('obs_id')
        if not h.empty:
            r = h.loc[(abs(h.delta_fixed)-abs(h.delta_fixed).median()).abs().idxmin()]
            choices.append(dict(reason=flag, **r.to_dict()))
    selected = pd.DataFrame(choices)
    selected.to_csv(OUT/'case_selection.csv', index=False)
    for chosen in selected.itertuples():
        pair = pairs[pairs.history_id.eq(chosen.obs_id) | pairs.query_id.eq(chosen.obs_id)].iloc[0]
        fig, axes = plt.subplots(4,4,figsize=(12,11),constrained_layout=True)
        for endpoint, oid in enumerate((pair.history_id,pair.query_id)):
            i=frame.index[frame.obs_id.eq(oid)][0]
            for p,pol in enumerate(('vv','vh')):
                view=views[i]['images'][pol]
                im=np.asarray(q.d.ip.Image.open(frame.loc[i,'image_'+pol]).convert('L'),dtype=float)/255
                altered,level,_=next(v for v in q.perturb(im,view)
                    if v[2]['region']==chosen.region and v[2]['band']==chosen.band and v[2]['dose']==1)
                component=q.bands(im)[0][chosen.band]
                limit=max(abs(component).max(),1e-8)
                axrow=axes[endpoint*2+p]
                for col,(label,image) in enumerate((('native',im),('component',component),('attenuated',altered),('mean control',level))):
                    ax=axrow[col]
                    ax.imshow(image,cmap='RdBu_r' if col==1 else 'gray',vmin=-limit if col==1 else 0,
                              vmax=limit if col==1 else 1,interpolation='nearest')
                    ax.contour(q.regions(im,view)[chosen.region],levels=[.5],colors='lime',linewidths=.6)
                    ax.set_title(f'{oid} {pol.upper()} {label}',fontsize=8)
                r=t[t.obs_id.eq(oid)&t.pol.eq(pol)&t.region.eq(chosen.region)&t.band.eq(chosen.band)&t.dose.eq(1)]
                if not r.empty:
                    z=r.iloc[0]
                    axrow[2].set_xlabel(f'margin {z.delta_fixed:+.5f}; {z.base_pred}->{z.pred}',fontsize=9)
                    axrow[3].set_xlabel(f'margin {z.level_delta:+.5f}',fontsize=9)
        fig.suptitle(f'{chosen.reason}: {pair.history_id}->{pair.query_id} natural {pair.state}\n'
                     f'{chosen.region}/{chosen.band}, each polarization intervened separately')
        fig.savefig(OUT/f'case_{chosen.reason}_{chosen.obs_id}.png',dpi=115)
        plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not args.analyze_only:
        infer()
    analyze()
    panels()
