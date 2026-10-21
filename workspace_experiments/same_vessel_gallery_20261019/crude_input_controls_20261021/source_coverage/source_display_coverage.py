"""Native PNG display coverage, weighted by the existing source training loss."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze_source_coverage import OUT, ROOT, LOCAL, PILOT, wquantile, save

CLASSES = (0, 3, 4, 5)
METRICS = ('target_mean', 'rest_mean', 'contrast', 'target_clip', 'rest_clip')
TRACE = ROOT / 'crude_input_controls_20261021/radiometry_trace'


def get_masks_for():
    tree = ast.parse((LOCAL/'prepare.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'masks_for')
    namespace = dict(np=np, Image=Image, ImageDraw=ImageDraw)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'existing_masks_for', 'exec'), namespace)
    return namespace['masks_for']


def main():
    frame = pd.read_csv(PILOT/'cohort.csv', dtype={'mmsi': str}).set_index('obs_id')
    members = pd.read_csv(OUT/'split_membership.csv', dtype={'mmsi': str})
    # Detect a changed matched-input cohort before reusing the source membership.
    current = pd.read_csv(OUT.parent/'partitions.csv')
    for (fold, part), g in members.groupby(['split_fold','partition']):
        np.testing.assert_array_equal(g.obs_id, current[current.fold.eq(fold) & current.partition.eq(part)].obs_id)
    source = members[members.partition.eq('train') & members.class_id.isin(CLASSES)].copy()
    endpoint = pd.read_csv(OUT/'crude_endpoints.csv', dtype={'mmsi': str})
    assert len(endpoint) == 200 and endpoint.obs_id.is_unique
    source_ids, endpoint_ids = set(source.obs_id), set(endpoint.obs_id)
    assert not source_ids & endpoint_ids
    requested = source_ids | endpoint_ids
    views = {v['obs_id']:v for v in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8')) if v['obs_id'] in requested}
    alignment = pd.read_csv(LOCAL/'crossview_alignment.csv').set_index('obs_id')
    cached = pd.read_csv(TRACE/'png_statistics.csv').set_index(['obs_id','pol','region'])
    provenance = pd.read_csv(TRACE/'view_trace.csv', usecols=['obs_id','pol','image_path']).set_index(['obs_id','pol'])
    old = pd.read_csv(ROOT/'condition_outcomes_20261021/image_statistics.csv').set_index(['obs_id','pol'])
    mask_fn = get_masks_for()
    stats = []; read_views = 0; reused_views = 0
    for step, oid in enumerate(sorted(requested)):
        row = frame.loc[oid]
        for pol in ('vv','vh'):
            view = views[oid]['images'][pol]
            masks = mask_fn(view).astype(bool)
            target, other, rest = masks
            assert np.all(masks.sum(0) == 1)
            disabled = pol == 'vh' and bool(alignment.loc[oid,'disable_new_vh'])
            good = bool(target.any() and rest.any() and not disabled)
            record = dict(obs_id=oid, pol=pol, role='endpoint' if oid in endpoint_ids else 'source',
                mmsi=row.mmsi, class_id=row.class_id, class_name=row.class_name, product=row['product'],
                port=row.port, platform=row.platform, image_path=row['image_'+pol],
                target_pixels=int(target.sum()), rest_pixels=int(rest.sum()), other_pixels=int(other.sum()),
                image_width=view['w'], image_height=view['h'], disabled_alignment=disabled,
                display_usable=good, origin='unavailable', **{m:np.nan for m in METRICS})
            if oid in endpoint_ids:
                assert good
                assert provenance.loc[(oid,pol),'image_path'] == row['image_'+pol]
                t, r = cached.loc[(oid,pol,'target')], cached.loc[(oid,pol,'rest')]
                assert t['n'] == target.sum() and r['n'] == rest.sum()
                assert t['dtype'] == r['dtype'] == 'uint8'
                record.update(target_mean=t['mean']/255, rest_mean=r['mean']/255,
                    contrast=(t['mean']-r['mean'])/255, target_clip=t.at255_fraction,
                    rest_clip=r.at255_fraction, origin='reused_verified_png_statistics_400views')
                np.testing.assert_allclose([record[m] for m in METRICS[:4]],
                    old.loc[(oid,pol),list(METRICS[:4])].to_numpy(float), atol=1e-12, rtol=0)
                reused_views += 1
            elif good:
                with Image.open(row['image_'+pol]) as im:
                    raw = np.asarray(im)
                    assert im.mode == 'L' and raw.dtype == np.uint8
                    assert raw.shape == (view['h'],view['w'])
                    t, r = raw[target], raw[rest]
                    record.update(target_mean=float(t.mean()/255), rest_mean=float(r.mean()/255),
                        contrast=float((t.mean()-r.mean())/255), target_clip=float(np.mean(t==255)),
                        rest_clip=float(np.mean(r==255)), origin='new_native_png_read')
                read_views += 1
            stats.append(record)
        if step % 500 == 0:
            print('PNG metadata/reads',step,len(requested),'new',read_views,'reused',reused_views,flush=True)
    stats = save('display_raw_statistics',stats)
    weighted = source[['split_fold','obs_id','source_weight']].merge(stats[stats.role.eq('source')],on='obs_id',validate='many_to_many')
    quantiles = []; coverage = []
    for keys, g in weighted.groupby(['split_fold','class_id','class_name','pol']):
        good = g[g.display_usable]
        identifiers = dict(zip(['split_fold','class_id','class_name','pol'],keys))
        coverage.append(dict(**identifiers, observations=len(g), ships=g.mmsi.nunique(),
            valid_observations=len(good), valid_ships=good.mmsi.nunique(), valid_products=good['product'].nunique(),
            missing_target=int(g.target_pixels.eq(0).sum()), disabled=int(g.disabled_alignment.sum()),
            empty_rest=int(g.rest_pixels.eq(0).sum()), weight_total=g.source_weight.sum(),
            weight_available=good.source_weight.sum(), available_weight_fraction=good.source_weight.sum()/g.source_weight.sum()))
        for metric in METRICS:
            x, w = good[metric].to_numpy(), good.source_weight.to_numpy()
            assert len(x) and np.isfinite(x).all() and (w>0).all()
            quantiles.append(dict(**identifiers, metric=metric, source_observations=len(good),
                minimum=float(x.min()), maximum=float(x.max()), weighted_mean=float(np.average(x,weights=w)),
                **{name:wquantile(x,w,q) for name,q in [('q025',.025),('q05',.05),('q50',.5),('q95',.95),('q975',.975)]}))
    save('display_source_availability',coverage)
    qtable = save('display_source_quantiles',quantiles)
    endpoints = endpoint[['obs_id','pair_id','side','split_fold']].merge(stats[stats.role.eq('endpoint')],on='obs_id',validate='one_to_many')
    relative = []
    for keys, g in endpoints.groupby(['split_fold','pol']):
        fold, pol = keys
        src = weighted[weighted.split_fold.eq(fold) & weighted.pol.eq(pol) & weighted.class_id.eq(5) & weighted.display_usable]
        ref = qtable[qtable.split_fold.eq(fold) & qtable.pol.eq(pol) & qtable.class_id.eq(5)].set_index('metric')
        for metric in METRICS:
            x, w = src[metric].to_numpy(), src.source_weight.to_numpy()
            q = ref.loc[metric]
            for row in g.itertuples():
                value = getattr(row,metric)
                lo, hi = float(w[x<value].sum()/w.sum()), float(w[x<=value].sum()/w.sum())
                relative.append(dict(obs_id=row.obs_id,pair_id=row.pair_id,side=row.side,split_fold=fold,
                    mmsi=row.mmsi,product=row.product,port=row.port,pol=pol,metric=metric,value=value,
                    percentile_lower=100*lo,percentile_upper=100*hi,percentile_mid=50*(lo+hi),
                    source_min=q.minimum,source_max=q.maximum,source_q025=q.q025,source_q975=q.q975,
                    outside_extrema=bool(value<q.minimum or value>q.maximum),
                    below_min=bool(value<q.minimum),above_max=bool(value>q.maximum),
                    tail='low' if value<q.q025 else ('high' if value>q.q975 else 'central')))
    rel = save('display_endpoint_percentiles',relative)
    counts = []
    for fold in ('pooled',0,1,2):
        sub = rel if fold=='pooled' else rel[rel.split_fold.eq(fold)]
        for (pol,metric),g in sub.groupby(['pol','metric']):
            counts.append(dict(split_fold=fold,pol=pol,metric=metric,endpoints=len(g),
                below_min=int(g.below_min.sum()),above_max=int(g.above_max.sum()),
                outside_extrema=int(g.outside_extrema.sum()),low_tail=int(g['tail'].eq('low').sum()),
                high_tail=int(g['tail'].eq('high').sum()),outside_95=int(g['tail'].ne('central').sum()),
                median_percentile=g.percentile_mid.median()))
    save('display_endpoint_tail_counts',counts)
    # Existing original paired_relation baseline, not the new five matched heads.
    outcomes = pd.read_csv(OUT/'endpoint_baseline_outcomes.csv')
    assert len(outcomes)==600 and outcomes.groupby('obs_id').size().eq(3).all()
    mean = outcomes.groupby('obs_id').agg(correct=('correct','mean'),margin=('margin','mean'),
        all_seeds_wrong=('correct',lambda s:bool(s.eq(0).all())))
    joined = rel.merge(mean,on='obs_id',validate='many_to_one')
    associations = []
    for fold in ('pooled',0,1,2):
        sub = joined if fold=='pooled' else joined[joined.split_fold.eq(fold)]
        for (pol,metric),g in sub.groupby(['pol','metric']):
            total_error = (1-g.correct).sum()
            for label,chosen in [('low',g['tail'].eq('low')),('central',g['tail'].eq('central')),
                                 ('high',g['tail'].eq('high')),('either_tail',g['tail'].ne('central'))]:
                h=g[chosen]; error=(1-h.correct).sum(); population=len(h)/len(g)
                share=error/total_error if total_error else np.nan
                associations.append(dict(split_fold=fold,pol=pol,metric=metric,group=label,
                    endpoints=len(h),ships=h.mmsi.nunique(),products=h['product'].nunique(),
                    population_fraction=population,error_equivalent_endpoints=error,
                    error_rate=1-h.correct.mean(),share_of_all_errors=share,
                    error_share_over_population=share/population if population else np.nan,
                    mean_margin=h.margin.mean(),all_seeds_wrong=int(h.all_seeds_wrong.sum())))
    save('display_tail_error_associations',associations)
    seedwise=rel.merge(outcomes,on='obs_id',validate='many_to_many')
    save('display_tail_error_by_seed',seedwise.groupby(['pol','metric','tail','seed']).agg(
        endpoints=('obs_id','size'),correct_rate=('correct','mean'),mean_margin=('margin','mean')).reset_index())
    # One union per polarization avoids counting the same endpoint once per metric.
    unions = rel.groupby(['obs_id','pol']).agg(any_display_tail=('tail',lambda s:s.ne('central').any()),
        any_mean_tail=('metric',lambda s:False)).reset_index().drop(columns='any_mean_tail')
    mean_flags=rel[rel.metric.isin(['target_mean','rest_mean'])].groupby(['obs_id','pol'])['tail'].agg(lambda s:s.ne('central').any())
    unions=unions.set_index(['obs_id','pol']);unions['any_mean_tail']=mean_flags
    unions=unions.reset_index().merge(mean,on='obs_id',validate='many_to_one')
    save('display_endpoint_tail_unions',unions)
    union_summary=[]
    for pol,g in unions.groupby('pol'):
        for flag in ('any_mean_tail','any_display_tail'):
            for state,h in g.groupby(flag):
                union_summary.append(dict(pol=pol,flag=flag,tail=state,endpoints=len(h),
                    error_rate=1-h.correct.mean(),share_of_all_errors=(1-h.correct).sum()/(1-g.correct).sum()))
    save('display_tail_union_summary',union_summary)
    assert reused_views==400 and len(rel)==200*2*len(METRICS)
    assert not rel.duplicated(['obs_id','pol','metric']).any()
    np.testing.assert_allclose(stats.loc[stats.display_usable,'contrast'],
        stats.loc[stats.display_usable,'target_mean']-stats.loc[stats.display_usable,'rest_mean'],atol=1e-14)
    receipt=dict(status='complete',source_classes=list(CLASSES),source_unique_observations=len(source_ids),
        endpoint_observations=200,endpoint_views_reused=reused_views,new_source_png_reads=read_views,
        unavailable_source_views=int((~stats[stats.role.eq('source')].display_usable).sum()),
        native_no_crop_pad_resize=True,physical_values=False,classifier_fits=0,feature_arrays_read=0,
        thresholds='Fixed weighted source q025/q975, strict outside; no target correctness selection',
        percentiles='100 * weighted (P(X<value) + 0.5 P(X=value)); lower/upper CDF also exported',
        clip='Fraction of original uint8 region pixels exactly equal to 255; not proof of physical saturation',
        masks='Existing masks_for: valid shape and confidence >=0.25; T target, R complement of target and valid other boxes',
        weights='Original per-fold N/(8*n_obs_ship*n_ships_class); filter unavailable views, renormalize remaining original weights only',
        original_baseline='source_coverage/endpoint_baseline_outcomes.csv: paired_relation baseline, 3 seeds',
        verified=['Matched-input partitions equal previous source memberships','400 endpoint image paths and T/R counts match verified raw cache',
                  '400 endpoint means/contrast/target_clip match old image_statistics','Native source PNG uint8 L shape checks',
                  'All finite source distributions; 2000 endpoint metric records; contrast identity'],
        perturb_status='User confirmed baseline replay <1e-5 passed before this supplement; no rerun')
    (OUT/'DISPLAY_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(pd.DataFrame(counts).query('split_fold == "pooled"').to_string(index=False))


if __name__=='__main__':
    main()
