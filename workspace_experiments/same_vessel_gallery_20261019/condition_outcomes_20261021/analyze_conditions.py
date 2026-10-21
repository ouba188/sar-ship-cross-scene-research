"""Frozen-prediction diagnostics; no ship classifier is trained or changed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / 'paired_relation_probe_20261020'
ARMS = ('baseline', 'marginal', 'joint', 'joint_condition')
GEO = ('log_length', 'log_width', 'cos2', 'sin2', 'incidence', 'other_boxes')
PIX = ('target_mean', 'target_std', 'target_clip', 'rest_mean', 'rest_std', 'contrast')


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def polygon_mask(size, corners):
    mask = Image.new('1', size)
    ImageDraw.Draw(mask).polygon([tuple(x) for x in corners], fill=1)
    return np.asarray(mask, dtype=bool)


def pixel_statistics(raw, target, others):
    rest = ~(target | others)
    t, r = raw[target].astype(float) / 255, raw[rest].astype(float) / 255
    return dict(target_mean=t.mean() if len(t) else np.nan,
                target_std=t.std() if len(t) else np.nan,
                target_clip=np.mean(t == 1) if len(t) else np.nan,
                rest_mean=r.mean() if len(r) else np.nan,
                rest_std=r.std() if len(r) else np.nan,
                contrast=t.mean()-r.mean() if len(t) and len(r) else np.nan,
                target_pixels=len(t), rest_pixels=len(r))


def scores(cohort):
    records = []
    for arm in ARMS:
        for seed in range(3):
            for fold in range(3):
                with np.load(OLD / f'scores_f{fold}_{arm}_s{seed}.npz') as z:
                    rows, s = z['rows'], z['scores']
                c = cohort.iloc[rows]
                yy = c.class_id.to_numpy()
                wrong = s.copy()
                wrong[np.arange(len(s)), yy] = -np.inf
                for j, row in enumerate(c.itertuples()):
                    pred = int(s[j].argmax())
                    records.append(dict(obs_id=row.obs_id, mmsi=str(row.mmsi), fold=fold,
                        seed=seed, arm=arm, class_id=row.class_id, class_name=row.class_name,
                        port=row.port, product=row.product, correct=int(pred == row.class_id),
                        prediction=pred, margin=float(s[j, yy[j]]-wrong[j].max())))
    result = pd.DataFrame(records)
    assert not result.duplicated(['obs_id', 'arm', 'seed']).any()
    assert result.groupby(['arm', 'seed']).size().eq(4744).all()
    return result


def extract_pixels(views, snapshot):
    cache = OUT / 'image_statistics.csv'
    if cache.exists():
        return pd.read_csv(cache)
    result = []
    for k, row in enumerate(views.itertuples()):
        view = snapshot[row.obs_id]['images'][row.pol]
        p = view['priors']
        raw = np.asarray(Image.open(row.image_path).convert('L'))
        assert raw.shape == (row.height, row.width)
        target = polygon_mask((row.width, row.height), p['detections'][p['target_index']]['corners'])
        other = np.zeros_like(target)
        for j, det in enumerate(p['detections']):
            if j != p['target_index'] and det.get('shape_usable') and det['confidence'] >= .25:
                other |= polygon_mask((row.width, row.height), det['corners'])
        result.append(dict(obs_id=row.obs_id, pol=row.pol, **pixel_statistics(raw, target, other)))
        if k % 500 == 0:
            print('pixels', k, '/', len(views), flush=True)
    result = pd.DataFrame(result)
    assert len(result) == 3640 and not result.duplicated(['obs_id', 'pol']).any()
    result.to_csv(cache, index=False)
    return result


def associations(pairs, features):
    result = []
    rng = np.random.default_rng(20261021)
    for feature in features:
        x, y = pairs[feature].to_numpy(), pairs.delta_margin.to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        xx, yy = x[ok], y[ok]
        boot = [corr(xx[ix], yy[ix]) for ix in rng.integers(0, len(xx), size=(300, len(xx)))]
        row = dict(feature=feature, n=int(ok.sum()), rho=corr(x, y),
                   low=np.nanquantile(boot, .025), high=np.nanquantile(boot, .975),
                   same_port_rho=corr(pairs.loc[pairs.same_port.eq(1), feature],
                                      pairs.loc[pairs.same_port.eq(1), 'delta_margin']))
        for fold in range(3):
            part = pairs[pairs.fold.eq(fold)]
            row[f'fold{fold}_rho'] = corr(part[feature], part.delta_margin)
        result.append(row)
    pd.DataFrame(result).to_csv(OUT / 'associations.csv', index=False)
    rows = []
    for name, group in pairs.groupby('class_name'):
        for feature in features:
            rows.append(dict(class_name=name, feature=feature, ships=len(group),
                             rho=corr(group[feature], group.delta_margin)))
    pd.DataFrame(rows).to_csv(OUT / 'associations_by_class.csv', index=False)


def diagnostic(pairs, groups):
    metrics, predictions, coefficients = [], [], []
    y = pairs.delta_margin.to_numpy()
    for name, cols in groups.items():
        pred = np.zeros(len(pairs)); null = np.zeros(len(pairs))
        for fold in range(3):
            test = pairs.fold.eq(fold).to_numpy(); train = ~test
            assert set(pairs.loc[test, 'mmsi']).isdisjoint(pairs.loc[train, 'mmsi'])
            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                  StandardScaler(), Ridge(alpha=10))
            model.fit(pairs.loc[train, cols], y[train])
            pred[test] = model.predict(pairs.loc[test, cols]); null[test] = y[train].mean()
            metrics.append(dict(inputs=name, fold=str(fold), ships=int(test.sum()),
                mse=float(np.mean((pred[test]-y[test])**2)),
                constant_mse=float(np.mean((null[test]-y[test])**2)),
                rho=corr(pred[test], y[test])))
            cn = model[0].get_feature_names_out(cols)
            coefficients.extend(dict(inputs=name, fold=fold, feature=feature, weight=float(value))
                                for feature, value in zip(cn, model[-1].coef_))
        mse, nmse = np.mean((pred-y)**2), np.mean((null-y)**2)
        metrics.append(dict(inputs=name, fold='pooled', ships=len(pairs), mse=mse,
            constant_mse=nmse, rho=corr(pred, y), error_reduction=1-mse/nmse))
        predictions.extend(dict(pair_id=p.pair_id, inputs=name, fold=p.fold,
                                actual=y[i], predicted=pred[i], constant=null[i])
                           for i, p in enumerate(pairs.itertuples()))
    pd.DataFrame(metrics).to_csv(OUT / 'diagnostic_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'diagnostic_predictions.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'diagnostic_coefficients.csv', index=False)


def grouped_outcomes(pairs, original, all_pred):
    results = []
    pair_pred = pd.read_csv(OLD / 'pair_predictions.csv')
    for field in ('same_port', 'angle_bin', 'incidence_bin', 'class_name', 'platform_changed', 'days_bin'):
        for group, p in pairs.groupby(field, observed=True):
            for arm in ARMS:
                z = pair_pred[pair_pred.arm.eq(arm) & pair_pred.pair_id.isin(p.pair_id)]
                assert len(z) == 3*len(p)
                results.append(dict(field=field, group=str(group), arm=arm, ships=len(p),
                    both_correct=z.outcome.eq('CC').mean(), both_wrong=z.outcome.eq('WW').mean(),
                    one_correct=z.outcome.isin(['CW','WC']).mean(),
                    delta_oa_pp=100*z.gained_correct.mean()/2,
                    correct_to_wrong=z.outcome.eq('CW').mean(), wrong_to_correct=z.outcome.eq('WC').mean()))
    pd.DataFrame(results).to_csv(OUT / 'pair_groups.csv', index=False)
    base = all_pred[all_pred.arm.eq('baseline')][['obs_id','seed','correct']].rename(columns={'correct':'base_correct'})
    z = all_pred.merge(base, on=['obs_id','seed'], validate='many_to_one')
    z['net'] = z.correct-z.base_correct
    z['rescued'] = z.correct.eq(1)&z.base_correct.eq(0)
    z['harmed'] = z.correct.eq(0)&z.base_correct.eq(1)
    records = []
    for field in ('class_name', 'port', 'fold'):
        for (group, arm), g in z.groupby([field, 'arm']):
            per = g.groupby('seed').net.mean()*100
            records.append(dict(field=field, group=str(group), arm=arm, observations=g.obs_id.nunique(),
                ships=g.mmsi.nunique(), products=g['product'].nunique(), oa=100*g.correct.mean(),
                delta_oa_pp=100*g.net.mean(), rescued_mean=g.rescued.sum()/3,
                harmed_mean=g.harmed.sum()/3, seed0=per.get(0),seed1=per.get(1),seed2=per.get(2)))
    pd.DataFrame(records).to_csv(OUT / 'full_cohort_groups.csv', index=False)
    return pd.DataFrame(results)


def cases(pairs, views, original):
    base = original[original.arm.eq('baseline')]
    # One pair per outcome, selected by largest absolute seed-mean margin change.
    outcomes = base.pivot(index='pair_id', columns='seed', values='outcome')
    chosen = []
    for outcome in ('CW','WC','CC','WW'):
        ids = outcomes.index[outcomes.eq(outcome).all(axis=1)]
        g = pairs[pairs.pair_id.isin(ids)]
        chosen.append((outcome, g.loc[g.delta_margin.abs().idxmax()]))
    # Geometry contrast examples are chosen without knowing correctness.
    ordered = pairs.sort_values(['vv_relative_axis_gap_deg','pair_id'])
    chosen.extend([('small_angle',ordered.iloc[0]), ('large_angle', ordered.iloc[-1])])
    view_lookup = views.set_index(['obs_id','pol'])
    records = []
    for label, p in chosen:
        fig, axs = plt.subplots(2,2,figsize=(10,8))
        fig.suptitle(f'{label} | {p.pair_id} | {p.class_name} | same-port={p.same_port}\n'
                     f'Baseline mean margin change={p.delta_margin:+.3f}; chronological A -> B')
        for i, side in enumerate(('a','b')):
            oid = int(p['obs_id_'+side])
            for j, pol in enumerate(('vv','vh')):
                v = view_lookup.loc[oid,pol]
                with Image.open(v.image_path) as im:
                    raw = np.asarray(im.convert('L'))
                axs[i,j].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axs[i,j].set_title(f'{side.upper()} {pol.upper()} obs={oid}, {p["port_"+side]}\n'
                    f'native {v.width}x{v.height}, L={v.length_px:.1f}px, rel={v.relative_axis_deg:.1f}, inc={v.incidence:.1f}',fontsize=9)
                axs[i,j].set_xlabel('Native pixels; original uint8 display')
        fig.tight_layout(rect=(0,0,1,.93))
        fig.savefig(OUT/f'case_{label}_{p.pair_id}.png',dpi=120); plt.close(fig)
        records.append(dict(selection=label, **p.to_dict()))
    pd.DataFrame(records).to_csv(OUT / 'selected_cases.csv',index=False)


def plots(groups):
    assoc=pd.read_csv(OUT/'associations.csv').sort_values('rho')
    fig,ax=plt.subplots(figsize=(9,9))
    ax.errorbar(assoc.rho, np.arange(len(assoc)),
                xerr=[np.maximum(0,assoc.rho-assoc.low),np.maximum(0,assoc.high-assoc.rho)],fmt='o',markersize=3)
    ax.set_yticks(np.arange(len(assoc)),assoc.feature,fontsize=8)
    ax.axvline(0,color='gray',linewidth=.8)
    ax.set_xlabel('Spearman rho with same-vessel margin change; exploratory ship-pair CI')
    fig.tight_layout();fig.savefig(OUT/'factor_associations.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,field in zip(axs,('angle_bin','incidence_bin')):
        z=groups[groups.field.eq(field)&groups.arm.eq('baseline')]
        x=np.arange(len(z))
        ax.bar(x,z.both_correct,label='Both correct',color='#287d8e')
        ax.bar(x,z.one_correct,bottom=z.both_correct,label='One correct',color='#e4af37')
        ax.bar(x,z.both_wrong,bottom=z.both_correct+z.one_correct,label='Both wrong',color='#a95063')
        ax.set_xticks(x,[f'{g}\nn={n}' for g,n in zip(z.group,z.ships)])
        ax.set_title(field+' (degrees)');ax.set_ylim(0,1)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'condition_outcomes.png',dpi=140);plt.close(fig)


def main():
    cohort=pd.read_csv(ROOT/'conditional_evidence_pilot/cohort.csv',dtype={'mmsi':str})
    pairs=pd.read_csv(OLD/'data_review/main_pairs.csv',dtype={'mmsi':str})
    assert len(pairs)==910 and pairs.mmsi.is_unique
    views=pd.read_csv(OLD/'data_review/main_pair_views.csv')
    snap={x['obs_id']:x for x in json.loads((ROOT/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
          if x['obs_id'] in set(views.obs_id)}
    pixels=extract_pixels(views,snap)
    v=views.merge(pixels,on=['obs_id','pol'],validate='one_to_one')
    v['log_length']=np.log(v.length_px);v['log_width']=np.log(v.width_px)
    v['cos2']=np.cos(np.deg2rad(2*v.relative_axis_deg));v['sin2']=np.sin(np.deg2rad(2*v.relative_axis_deg))
    features=[]
    for pol in ('vv','vh'):
        q=v[v.pol.eq(pol)].set_index('obs_id')
        for name in GEO+PIX:
            key=pol+'_'+name
            pairs[key]=pairs.obs_id_b.map(q[name])-pairs.obs_id_a.map(q[name]);features.append(key)
    conditions=pd.read_csv(OLD/'data_review/main_pair_conditions.csv')
    pairs=pairs.merge(conditions[['pair_id','vv_relative_axis_gap_deg','vv_incidence_delta_b_minus_a']],on='pair_id',validate='one_to_one')
    original=pd.read_csv(OLD/'pair_predictions.csv')
    base=original[original.arm.eq('baseline')].copy()
    base['dm']=base.margin_b-base.margin_a
    pairs['delta_margin']=pairs.pair_id.map(base.groupby('pair_id').dm.mean())
    check=pd.read_csv(OLD/'data_review/main_pair_predictions.csv').groupby('pair_id').margin_delta_b_minus_a.mean()
    np.testing.assert_allclose(pairs.delta_margin,pairs.pair_id.map(check),atol=1e-6)
    pairs['angle_bin']=pd.cut(pairs.vv_relative_axis_gap_deg,[-.001,15,45,90.001],labels=['0-15','15-45','45-90'])
    pairs['incidence_bin']=pd.cut(pairs.vv_incidence_delta_b_minus_a.abs(),[-.001,2,5,100],labels=['0-2','2-5','>5'])
    pairs['platform_changed']=pairs.platform_a.ne(pairs.platform_b).astype(int)
    pairs['days_bin']=pd.cut(pairs.days_apart,[0,14,30,np.inf],labels=['0-14','14-30','>30'])
    pairs.to_csv(OUT/'pair_factors.csv',index=False)
    pred=scores(cohort)
    pred.to_csv(OUT/'frozen_predictions.csv.gz',index=False,compression='gzip')
    associations(pairs,features)
    diagnostic(pairs,{'metadata':[p+'_'+n for p in ('vv','vh') for n in GEO],
                      'image':[p+'_'+n for p in ('vv','vh') for n in PIX], 'both':features})
    groups=grouped_outcomes(pairs,original,pred)
    cases(pairs,views,original);plots(groups)
    summary=dict(observations=4744,ships_full=int(pred.mmsi.nunique()),pairs=910,raw_views=len(pixels),
        pair_classes=pairs.class_name.value_counts().to_dict(),same_port=int(pairs.same_port.sum()),
        new_classifier_fits=0,diagnostic_ridge_fits=9,source='24-port historical experiment cohort; not F_0921')
    (OUT/'RECEIPT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'diagnostic_metrics.csv').to_string(index=False),flush=True)
    print(pd.read_csv(OUT/'associations.csv').sort_values('rho').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
