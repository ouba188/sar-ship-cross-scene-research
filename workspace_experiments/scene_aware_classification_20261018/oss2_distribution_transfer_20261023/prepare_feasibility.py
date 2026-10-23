"""Fixed acquisition/MMSI split and native-image feasibility; no model fitting."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parent
MANIFEST = Path('E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv')
META = WORK / 'ports_root_cause_20261013/dataset_audit/public_metadata.csv'
SEED = 20261023


def acquisition(product):
    s = product.split('_')
    return '_'.join([s[0], s[6], s[7]])


def summarize(t, ix, fold, role, inner=-1):
    g = t.iloc[ix]
    present = sorted(g.class_id.unique().tolist())
    rows = []
    for c in range(8):
        q = g[g.class_id.eq(c)]
        rows.append(dict(fold=fold, inner=inner, role=role, class_id=c,
                         n=len(q), mmsi=q.mmsi.nunique(), passes=q.pass_group.nunique(),
                         total_n=len(g), classes=len(present),
                         all8=len(present) == 8,
                         missing_classes='|'.join(map(str, sorted(set(range(8))-set(present))))))
    return rows


def splits(t):
    passes = np.array(sorted(t.pass_group.unique()))
    shuffled = np.random.default_rng(SEED).permutation(passes)
    groups = np.array_split(shuffled, 6)
    lookup = {p: k for k, group in enumerate(groups) for p in group}
    t['pass_bag'] = t.pass_group.map(lookup).astype(int)
    pass_rows = []
    for order, p in enumerate(shuffled):
        q = t[t.pass_group.eq(p)]
        pass_rows.append(dict(pass_group=p, random_order=order, bag=lookup[p], n=len(q),
                              mmsi=q.mmsi.nunique(), products=q.scene_id.nunique(),
                              **{f'class{c}': int(q.class_id.eq(c).sum()) for c in range(8)}))
    pd.DataFrame(pass_rows).to_csv(HERE / 'pass_assignment.csv', index=False)
    coverage, members, innermembers, checks = [], [], [], []
    indices = dict(sample_ids=t.sample_id.to_numpy(str), row_index=t.row_index.to_numpy())
    for fold in range(6):
        te = np.flatnonzero(t.pass_bag.eq(fold))
        vc = np.flatnonzero(t.pass_bag.eq((fold+1) % 6))
        tc = np.flatnonzero(~t.pass_bag.isin([fold, (fold+1) % 6]))
        test_ids = set(t.iloc[te].mmsi)
        va = vc[~t.iloc[vc].mmsi.isin(test_ids).to_numpy()]
        blocked = test_ids | set(t.iloc[vc].mmsi)
        tr = tc[~t.iloc[tc].mmsi.isin(blocked).to_numpy()]
        assert not (set(t.iloc[tr].mmsi) & (set(t.iloc[va].mmsi) | test_ids))
        assert not (set(t.iloc[va].mmsi) & test_ids)
        role = np.full(len(t), '', dtype=object)
        role[tc], role[vc], role[te] = 'excluded_train_mmsi', 'excluded_val_test_mmsi', 'test'
        role[tr], role[va] = 'train', 'validation'
        g = t[['row_index','manifest_position','sample_id','class_id','mmsi','pass_group','pass_bag']].copy()
        g['fold'], g['role'] = fold, role
        g['shares_test_mmsi'] = t.mmsi.isin(test_ids)
        g['shares_validation_candidate_mmsi'] = t.mmsi.isin(set(t.iloc[vc].mmsi))
        members.append(g)
        for name, ix in [('train',tr),('validation',va),('test',te),('train_candidate',tc),('validation_candidate',vc)]:
            indices[f'fold{fold}_{name}'] = ix
            coverage.extend(summarize(t, ix, fold, name))
        count = np.zeros(len(t), int)
        for inner, bag in enumerate(sorted(set(range(6)) - {fold, (fold+1) % 6})):
            held = tr[t.iloc[tr].pass_bag.eq(bag).to_numpy()]
            candidate = tr[~t.iloc[tr].pass_bag.eq(bag).to_numpy()]
            fit = candidate[~t.iloc[candidate].mmsi.isin(set(t.iloc[held].mmsi)).to_numpy()]
            excluded = np.setdiff1d(candidate, fit)
            assert not (set(t.iloc[fit].mmsi) & set(t.iloc[held].mmsi))
            assert not (set(t.iloc[fit].pass_group) & set(t.iloc[held].pass_group))
            count[held] += 1
            for name, ix in [('fit',fit),('held',held),('excluded_mmsi',excluded)]:
                indices[f'fold{fold}_inner{inner}_{name}'] = ix
                coverage.extend(summarize(t, ix, fold, 'inner_'+name, inner))
                g = t.iloc[ix][['row_index','sample_id','class_id','mmsi','pass_group','pass_bag']].copy()
                g['fold'], g['inner'], g['role'], g['held_bag'] = fold, inner, name, bag
                innermembers.append(g)
        assert (count[tr] == 1).all() and count.sum() == len(tr)
        checks.append(dict(fold=fold, train=len(tr), validation=len(va), test=len(te),
                           excluded_train=len(tc)-len(tr), excluded_val=len(vc)-len(va),
                           train_val_test_mmsi_overlap=0, inner_held_coverage_exact=True))
    pd.DataFrame(coverage).to_csv(HERE / 'split_coverage.csv', index=False)
    pd.concat(members, ignore_index=True).to_csv(HERE / 'membership.csv', index=False)
    pd.concat(innermembers, ignore_index=True).to_csv(HERE / 'inner_membership.csv', index=False)
    pd.DataFrame(checks).to_csv(HERE / 'split_checks.csv', index=False)
    np.savez_compressed(HERE / 'indices.npz', **indices)
    failures = pd.DataFrame(coverage).query("role in ['train','inner_fit'] and all8 == False")
    failures = failures[['fold','inner','role','total_n','missing_classes']].drop_duplicates()
    failures.to_csv(HERE / 'fit_failures.csv', index=False)
    return dict(outer_folds=6, inner_folds=4, seed=SEED, fit_failures=failures.to_dict('records'))


def images(t):
    meta = pd.read_csv(META, dtype={'mmsi_or_ship_id':str, 'xml_mmsi':str})
    meta = meta[meta.dataset.eq('oss2')].set_index('sample_id').loc[t.sample_id]
    assert np.array_equal(meta['product'], t.scene_id)
    assert np.array_equal(meta.xml_mmsi, t.mmsi)
    rows = []
    for i, r in t.iterrows():
        m = meta.iloc[i]
        for pol in ['vv', 'vh']:
            path = Path(r['chip_path_'+pol])
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
            assert w == r.chip_width_px and h == r.chip_height_px
            # This origin is an estimate until checked against XML for the eight examples.
            x0, y0 = m.center_x-(w-1)/2, m.center_y-(h-1)/2
            hx, hy, tx, ty = m.head_x-x0, m.head_y-y0, m.tail_x-x0, m.tail_y-y0
            left, top = max((w-160)//2, 0), max((h-160)//2, 0)
            right, bottom = left+min(w,160), top+min(h,160)
            finite = bool(np.isfinite([hx,hy,tx,ty]).all())
            inside = finite and min(hx,tx)>=left and max(hx,tx)<=right-1 and min(hy,ty)>=top and max(hy,ty)<=bottom-1
            rows.append(dict(row_index=i,sample_id=r.sample_id,class_id=r.class_id,pol=pol,path=str(path),
                             width=w,height=h,mode=mode,center_x=m.center_x,center_y=m.center_y,
                             head_x=hx,head_y=hy,tail_x=tx,tail_y=ty,
                             estimated_origin_x=x0,estimated_origin_y=y0,origin_source='center_minus_half_extent_estimate',
                             crop160_removes_native=w>160 or h>160,
                             estimated_head_tail_inside160=inside,head_tail_finite=finite,
                             max_dimension=max(w,h),degenerate_dimension=min(w,h)<=1,
                             retained_native_fraction160=min(w,160)*min(h,160)/(w*h)))
    g = pd.DataFrame(rows)
    g.to_csv(HERE / 'native_geometry.csv', index=False)
    g.groupby(['class_id','pol']).agg(n=('sample_id','size'), min_w=('width','min'),max_w=('width','max'),
        max_h=('height','max'), cropped160=('crop160_removes_native','sum'),
        endpoints_not_confirmed=('estimated_head_tail_inside160',lambda s:int((~s).sum())),
        degenerate=('degenerate_dimension','sum')).to_csv(HERE / 'native_by_class.csv')
    return g


def overlap(t):
    references = {
        'ports24_manifest': WORK / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',
        'active_eight15251': ROOT / 'eight_class_evidence_20261023/cohort.csv',
    }
    rows, identities = [], []
    for name, path in references.items():
        q = pd.read_csv(path, dtype={'mmsi':str}, keep_default_na=False)
        q['pass_group'] = q['product'].map(acquisition)
        q['year'] = q['product'].str.extract(r'_(\d{4})\d{4}T', expand=False)
        common = set(t.mmsi) & set(q.mmsi)
        rows.append(dict(reference=name,path=str(path),rows=len(q),ports=q.port.nunique(),
                         years='|'.join(sorted(q.year.unique())),
                         oss2_years='|'.join(sorted(t.year.unique())),
                         shared_years='|'.join(sorted(set(t.year)&set(q.year))),
                         shared_products=len(set(t.scene_id)&set(q['product'])),
                         shared_passes=len(set(t.pass_group)&set(q.pass_group)),
                         shared_mmsi=len(common),oss2_rows_shared_mmsi=int(t.mmsi.isin(common).sum()),
                         reference_rows_shared_mmsi=int(q.mmsi.isin(common).sum())))
        for m in sorted(common):
            identities.append(dict(reference=name,mmsi=m,oss2_n=int(t.mmsi.eq(m).sum()),
                                   reference_n=int(q.mmsi.eq(m).sum())))
    pd.DataFrame(rows).to_csv(HERE / 'overlap_summary.csv', index=False)
    pd.DataFrame(identities).to_csv(HERE / 'shared_mmsi.csv', index=False)
    return rows


def main():
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    source = pd.read_csv(MANIFEST, dtype={'mmsi_or_ship_id':str})
    source['manifest_position'] = np.arange(len(source))
    t = source[source.scene_id.str.contains('_GRDH_')].copy().reset_index(drop=True)
    t.insert(0,'row_index',np.arange(len(t)))
    t['mmsi'] = t.mmsi_or_ship_id
    assert t.mmsi.str.fullmatch(r'[0-9]{9}').all() and not t.mmsi.eq('000000000').any()
    t['pass_group'] = t.scene_id.map(acquisition)
    t['year'] = t.scene_id.str.extract(r'_(\d{4})\d{4}T', expand=False)
    assert len(t)==2746 and t.pass_group.nunique()==27
    assert t.groupby('class_id').size().tolist()==[590,352,231,611,770,118,48,26]
    assert t.sample_id.is_unique
    stats = splits(t)
    t.to_csv(HERE / 'cohort.csv', index=False)
    g = images(t)
    overlaps = overlap(t)
    stats.update(n=len(t),mmsi=t.mmsi.nunique(),passes=t.pass_group.nunique(),products=t.scene_id.nunique(),
                 classes=t.groupby('class_id').agg(n=('sample_id','size'),mmsi=('mmsi','nunique'),
                                                  passes=('pass_group','nunique')).reset_index().to_dict('records'),
                 year_counts=t.year.value_counts().to_dict(), max_native=int(g.max_dimension.max()),
                 full_native_fixed_side=int(np.ceil(g.max_dimension.max()/32)*32),
                 native_cropped160=int(g[g.pol.eq('vv')].crop160_removes_native.sum()),
                 endpoint_not_confirmed160=int((~g[g.pol.eq('vv')].estimated_head_tail_inside160).sum()),
                 degenerate_rows=int(g[g.pol.eq('vv')].degenerate_dimension.sum()),
                 overlaps=overlaps,training_runs=0,downloads=0)
    (HERE / 'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
