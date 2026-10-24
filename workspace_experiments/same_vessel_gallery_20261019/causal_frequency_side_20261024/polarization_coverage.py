"""Expand matched region-scale interventions with joint/single image readouts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import numpy as np
import pandas as pd
import polarization_attribution as a
import source_readout_probe as s

p, q = a.p, a.q
OUT = p.ROOT/'round26_polarization_coverage'
KEYS = ['pol','region','band','dose']
OLD_IDS = [0,4,6,10,18,81]
EPS = 1e-6


def prepare():
    OUT.mkdir(exist_ok=True)
    frame = pd.read_csv(p.OUT/'baseline.csv',dtype={'mmsi':str})
    _, dets = p.load()
    coverage = []
    for row in frame.itertuples():
        reason, counts = '', {}
        if not row.intervention_eligible:
            reason = 'previous_geometry_ineligible'
        else:
            images = [p.image(frame.loc[row.Index],pol) for pol in a.POLS]
            if images[0].shape != images[1].shape:
                reason = 'different_native_shapes'
            elif max(images[0].shape) > 160:
                reason = 'native_exceeds_common_canvas'
            else:
                own = [q.regions(im,p.make_view(dets[row.obs_id,pol])) for im,pol in zip(images,a.POLS)]
                counts = {r+'_pixels':int((own[0][r]&own[1][r]).sum()) for r in a.REGIONS}
                if min(counts.values()) < 8:
                    reason = 'insufficient_shared_region'
        coverage.append(dict(obs_id=row.obs_id,reason=reason,**counts))
    cov = pd.DataFrame(coverage)
    cov.to_csv(OUT/'coverage.csv',index=False)
    frame['eligible'] = cov.reason.eq('').to_numpy()
    frame['old_case'] = frame.obs_id.isin(OLD_IDS)
    frame.to_csv(OUT/'cohort.csv',index=False)
    print(frame.groupby(['class_name','eligible']).size().to_string(),flush=True)
    print('eligible',int(frame.eligible.sum()),'ships',frame[frame.eligible].mmsi.nunique(),
          'passes',frame[frame.eligible].overpass.nunique(),flush=True)


def infer():
    started = time.perf_counter()
    frame = pd.read_csv(OUT/'cohort.csv',dtype={'mmsi':str})
    split = pd.read_csv(p.OUT/'splits.csv')
    raw = np.load(p.OUT/'native_features.npy')
    _, dets = p.load()
    stem = p.stem_model()
    heads, baseline = {}, []
    for fold in range(3):
        joint = dict(np.load(p.OUT/f'head_f{fold}.npz'))
        heads[fold,'joint'] = joint
        train = split[split.fold.eq(fold)&split.partition.eq('train')].obs_id.to_numpy()
        query = frame[frame.fold.eq(fold)]
        assert not set(frame.loc[train,'mmsi'])&set(query.mmsi)
        assert not set(frame.loc[train,'overpass'])&set(query.overpass)
        for pol in a.POLS:
            head,_,_ = s.fit(raw,frame.class_id.to_numpy(),train,joint,branch=pol)
            heads[fold,pol] = head
            np.savez(OUT/f'head_{fold}_{pol}.npz',**head)
        for arm in ('joint','vv','vh'):
            scores = q.d.score(raw[query.obs_id],heads[fold,arm])
            for (_,row),sc in zip(query.iterrows(),scores):
                wrong = sc.copy()
                wrong[int(row.class_id)] = -np.inf
                baseline.append(dict(obs_id=int(row.obs_id),arm=arm,pred=int(sc.argmax()),
                    own_rival=int(wrong.argmax()),own_margin=float(sc[int(row.class_id)]-wrong.max()),
                    **{f'score_{c}':float(v) for c,v in enumerate(sc)}))
    base = pd.DataFrame(baseline)
    base.to_csv(OUT/'baseline_predictions.csv',index=False)
    stored = base[base.arm.eq('joint')].sort_values('obs_id')[[f'score_{c}' for c in range(8)]].to_numpy()
    error_score = float(abs(stored-frame[[f'score_{c}' for c in range(8)]].to_numpy()).max())
    assert error_score < 1e-12
    cache, index = [], []
    error_feature = 0.
    with q.d.torch.inference_mode():
        selected = frame[frame.eligible]
        for n,(_,row) in enumerate(selected.iterrows()):
            oid = int(row.obs_id)
            images = [p.image(row,pol) for pol in a.POLS]
            views = [p.make_view(dets[oid,pol]) for pol in a.POLS]
            changes,_,_ = a.make_changes(images,views)
            for k,pol in enumerate(a.POLS):
                entries = [v for v in changes if v[0]==k and v[3]['mode']=='common']
                inputs = [images[k]]+[im for _,alt,level,meta in entries for im in (alt,level)]
                desc = q.d.ip.describe(stem,*q.d.prepare_batch(inputs,views[k]),inputs=('full',))['full_global'].numpy().astype(float)
                sl = slice(k*128,(k+1)*128)
                err = float(abs(desc[0]-raw[oid,sl]).max())
                error_feature = max(error_feature,err)
                assert err < 2e-5,(oid,pol,err)
                for j,(_,alt,level,meta) in enumerate(entries):
                    cache.append(np.stack([desc[1+2*j]-raw[oid,sl],desc[2+2*j]-raw[oid,sl]]))
                    index.append(dict(obs_id=oid,**meta))
            if (n+1)%100==0 or n+1==len(selected):
                print('inferred',n+1,'/',len(selected),'seconds',round(time.perf_counter()-started,1),flush=True)
    np.save(OUT/'private_feature_changes.npy',np.stack(cache))
    pd.DataFrame(index).to_csv(OUT/'intervention_index.csv',index=False)
    (OUT/'inference.json').write_text(json.dumps(dict(status='complete',observations=len(selected),fits=6,
        interventions=len(index),controls=len(index),feature_replay=error_feature,score_replay=error_score,
        seconds=time.perf_counter()-started,cpu_threads=2),indent=2),encoding='utf-8')


def evaluate():
    frame = pd.read_csv(OUT/'cohort.csv',dtype={'mmsi':str}).set_index('obs_id',drop=False)
    raw = np.load(p.OUT/'native_features.npy')
    base = pd.read_csv(OUT/'baseline_predictions.csv').set_index(['obs_id','arm'])
    index = pd.read_csv(OUT/'intervention_index.csv')
    changes = np.load(OUT/'private_feature_changes.npy',mmap_mode='r')
    heads = {(f,'joint'):dict(np.load(p.OUT/f'head_f{f}.npz')) for f in range(3)}
    heads.update({(f,pol):dict(np.load(OUT/f'head_{f}_{pol}.npz')) for f in range(3) for pol in a.POLS})
    records = []
    for j,item in index.iterrows():
        oid,pol = int(item.obs_id),item.pol
        row = frame.loc[oid]
        k = a.POLS.index(pol)
        sl = slice(k*128,(k+1)*128)
        modified = np.tile(raw[oid],(2,1))
        modified[:,sl] += changes[j]
        y,rival = int(row.class_id),int(row.rival)
        for arm in ('joint',pol):
            head = heads[int(row.fold),arm]
            b = base.loc[oid,arm]
            bc = b[[f'score_{c}' for c in range(8)]].to_numpy(float)
            sc,lc = q.d.score(modified,head)
            wrong = sc.copy()
            wrong[y] = -np.inf
            info = {key:row[key] for key in ('mmsi','overpass','fold','class_id','class_name','old_case')}
            records.append(dict(**item.to_dict(),**info,arm='joint' if arm=='joint' else 'single',
                rival=rival,own_rival=int(b.own_rival),base_pred=int(b.pred),pred=int(sc.argmax()),level_pred=int(lc.argmax()),
                delta=float(sc[y]-sc[rival]-bc[y]+bc[rival]),
                level_delta=float(lc[y]-lc[rival]-bc[y]+bc[rival]),
                own_delta=float(sc[y]-sc[int(b.own_rival)]-b.own_margin),
                decision_margin_delta=float(sc[y]-wrong.max()-b.own_margin),
                rescued=bool(b.pred!=y and sc.argmax()==y),harmed=bool(b.pred==y and sc.argmax()!=y),
                **{f'score_{c}':float(v) for c,v in enumerate(sc)},
                **{f'level_score_{c}':float(v) for c,v in enumerate(lc)}))
    table = pd.DataFrame(records)
    table['residual'] = table.delta-table.level_delta
    table.to_csv(OUT/'effects.csv.gz',index=False,compression='gzip')
    old = pd.read_csv(a.OUT/'interventions.csv')
    old = old[old['mode'].eq('common')]
    replay = table[table.arm.eq('joint')].merge(old,on=['obs_id']+KEYS,suffixes=('','_old'),validate='one_to_one')
    err = float(abs(replay.delta-replay.delta_fixed).max())
    assert len(replay)==144 and err < 1e-12,(len(replay),err)
    (OUT/'evaluation.json').write_text(json.dumps(dict(rows=len(table),old_case_replay=err),indent=2),encoding='utf-8')
    print('evaluated',len(table),'old case replay',err,flush=True)


def ci(values,rng):
    values = np.asarray(values,float)
    boots = values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return float(values.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def analyze():
    frame = pd.read_csv(OUT/'cohort.csv',dtype={'mmsi':str})
    base = pd.read_csv(OUT/'baseline_predictions.csv').merge(frame,on='obs_id',validate='many_to_one')
    metrics = []
    for scope,g in [('all',base),('eligible',base[base.eligible]),('new_coverage',base[base.eligible&~base.old_case])]:
        for arm,h in g.groupby('arm'):
            correct = h.pred.eq(h.class_id)
            metrics.append(dict(scope=scope,arm=arm,n=len(h),ships=h.mmsi.nunique(),
                oa=float(correct.mean()),ba=float(correct.groupby(h.class_id).mean().mean())))
    pd.DataFrame(metrics).to_csv(OUT/'baseline_metrics.csv',index=False)
    class_metrics = []
    for scope,g in [('all',base),('eligible',base[base.eligible])]:
        for (arm,cid,cname),h in g.groupby(['arm','class_id','class_name']):
            class_metrics.append(dict(scope=scope,arm=arm,class_id=int(cid),class_name=cname,n=len(h),
                ships=h.mmsi.nunique(),accuracy=float(h.pred.eq(h.class_id).mean())))
    pd.DataFrame(class_metrics).to_csv(OUT/'baseline_classes.csv',index=False)
    table = pd.read_csv(OUT/'effects.csv.gz',dtype={'mmsi':str})
    index = ['obs_id','mmsi','overpass','class_id','class_name','old_case']+KEYS
    cols = ['delta','level_delta','residual','own_delta','decision_margin_delta','base_pred','pred','rescued','harmed']
    pair = table[table.arm.eq('joint')][index+cols].merge(table[table.arm.eq('single')][index+cols],on=index,
        suffixes=('_joint','_single'),validate='one_to_one')
    pair['readout_difference'] = pair.delta_joint-pair.delta_single
    pair['residual_difference'] = pair.residual_joint-pair.residual_single
    pair['opposite'] = (pair.delta_joint*pair.delta_single<0)&(pair.delta_joint.abs()>EPS)&(pair.delta_single.abs()>EPS)
    pair['joint_correct'] = pair.base_pred_joint.eq(pair.class_id)
    pair['single_correct'] = pair.base_pred_single.eq(pair.class_id)
    pair['state'] = np.select([pair.joint_correct&pair.single_correct,pair.joint_correct&~pair.single_correct,
        ~pair.joint_correct&pair.single_correct],['both_correct','joint_only_correct','single_only_correct'],default='both_wrong')
    pair.to_csv(OUT/'paired_effects.csv.gz',index=False,compression='gzip')
    high = pair[pair.dose.eq(1.)]
    rng = np.random.default_rng(20261024)
    summary = []
    for scope,g in [('all_eligible',high),('new_coverage',high[~high.old_case]),('old_six',high[high.old_case])]:
        for key,h in g.groupby(KEYS):
            record = dict(scope=scope,**dict(zip(KEYS,key)),n=len(h),ships=h.mmsi.nunique(),passes=h.overpass.nunique(),
                          opposite=int(h.opposite.sum()))
            for field in ('delta_joint','delta_single','readout_difference','residual_difference'):
                for unit in ('mmsi','overpass'):
                    vals = h.groupby(unit)[field].mean()
                    avg,lo,hi = ci(vals,rng)
                    record.update({field+'_'+unit+'_mean':avg,field+'_'+unit+'_lo':lo,field+'_'+unit+'_hi':hi,
                                   field+'_'+unit+'_positive':int(vals.gt(EPS).sum())})
            for arm in ('joint','single'):
                correct = h['pred_'+arm].eq(h.class_id)
                original = h['base_pred_'+arm].eq(h.class_id)
                record.update({arm+'_rescued':int(h['rescued_'+arm].sum()),arm+'_harmed':int(h['harmed_'+arm].sum()),
                    arm+'_oa_delta':float(correct.mean()-original.mean()),
                    arm+'_ba_delta':float(correct.groupby(h.class_id).mean().mean()-original.groupby(h.class_id).mean().mean())})
            summary.append(record)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    for groups,name in [(['class_id','class_name'],'by_class'),(['overpass'],'by_pass'),(['state'],'by_state')]:
        rows = []
        for key,h in high.groupby(KEYS+groups):
            record = dict(zip(KEYS+groups,key),n=len(h),ships=h.mmsi.nunique(),opposite=int(h.opposite.sum()))
            for field in ('delta_joint','delta_single','readout_difference','residual_difference'):
                record[field] = float(h.groupby('mmsi')[field].mean().mean())
            for arm in ('joint','single'):
                record[arm+'_rescued'] = int(h['rescued_'+arm].sum())
                record[arm+'_harmed'] = int(h['harmed_'+arm].sum())
            rows.append(record)
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    details,pass_details = [],[]
    for scope,subset in [('all_eligible',high),('new_coverage',high[~high.old_case])]:
        for key,h in subset.groupby(KEYS+['class_id','class_name']):
            record = dict(scope=scope,**dict(zip(KEYS+['class_id','class_name'],key)),
                n=len(h),ships=h.mmsi.nunique(),passes=h.overpass.nunique(),opposite=int(h.opposite.sum()))
            for field in ('delta_joint','delta_single','own_delta_joint','own_delta_single',
                          'decision_margin_delta_joint','decision_margin_delta_single',
                          'residual_joint','residual_single','readout_difference','residual_difference'):
                record[field] = float(h.groupby('mmsi')[field].mean().mean())
            for field in ('readout_difference','residual_difference'):
                avg,lo,hi = ci(h.groupby('mmsi')[field].mean(),rng)
                record[field+'_ship_lo'],record[field+'_ship_hi'] = lo,hi
            for pas,g in h.groupby('overpass'):
                pass_details.append(dict(scope=scope,**dict(zip(KEYS+['class_id','class_name'],key)),
                    overpass=pas,n=len(g),ships=g.mmsi.nunique(),
                    **{c:float(g[c].mean()) for c in ('delta_joint','delta_single','readout_difference','residual_difference')}))
            record['pass_difference_positive'] = int(h.groupby('overpass').readout_difference.mean().gt(EPS).sum())
            details.append(record)
    pd.DataFrame(details).to_csv(OUT/'class_detail.csv',index=False)
    pd.DataFrame(pass_details).to_csv(OUT/'class_pass_detail.csv',index=False)
    dose = pair.pivot(index=['obs_id','pol','region','band'],columns='dose',values='readout_difference')
    (OUT/'analysis.json').write_text(json.dumps(dict(n=int(frame.eligible.sum()),ships=int(frame[frame.eligible].mmsi.nunique()),
        passes=int(frame[frame.eligible].overpass.nunique()),classes=int(frame[frame.eligible].class_id.nunique()),
        dose_same_sign=float((dose[.5]*dose[1.]>0).mean()),pair_rows=len(pair)),indent=2),encoding='utf-8')
    panels(frame,high)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


def panels(frame,high):
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _,dets = p.load()
    new = high[~high.old_case].sort_values(['obs_id','pol','region','band'])
    selected,seen = [],set()
    for kind in ('rescued','harmed','both_correct','both_wrong','opposite','side_rescued','side_harmed'):
        if kind.startswith('side_'):
            candidates = new[new.pol.eq('vh')&new.region.eq('side')&new.band.eq('middle')
                &new[kind.removeprefix('side_')+'_joint']]
        elif kind in ('rescued','harmed'):
            candidates = new[new[kind+'_joint']]
        elif kind == 'opposite':
            candidates = new[new.opposite]
        else:
            candidates = new[new.state.eq(kind)&~new.rescued_joint&~new.harmed_joint]
        candidates = candidates[~candidates.obs_id.isin(seen)]
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        seen.add(row.obs_id)
        selected.append(dict(kind=kind,**row.to_dict()))
        original = frame[frame.obs_id.eq(row.obs_id)].iloc[0]
        images = [p.image(original,pol) for pol in a.POLS]
        views = [p.make_view(dets[int(row.obs_id),pol]) for pol in a.POLS]
        changes,_,common = a.make_changes(images,views)
        fig,axes = plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
        entries = [next(v for v in changes if v[3]['mode']=='common' and v[3]['pol']==pol
            and v[3]['region']==row.region and v[3]['band']==row.band and v[3]['dose']==1.) for pol in a.POLS]
        limit = max(abs(v[1]-images[k]).max() for k,v in enumerate(entries))
        for k,pol in enumerate(a.POLS):
            entry = entries[k]
            effect = high[(high.obs_id==row.obs_id)&high.pol.eq(pol)&high.region.eq(row.region)&high.band.eq(row.band)].iloc[0]
            for col,(title,im) in enumerate([('native',images[k]),('pixel change',entry[1]-images[k]),('altered',entry[1])]):
                axes[k,col].imshow(im,cmap='RdBu_r' if col==1 else 'gray',vmin=-limit if col==1 else 0,
                    vmax=limit if col==1 else 1,interpolation='nearest')
                axes[k,col].contour(common[row.region],levels=[.5],colors='lime',linewidths=.6)
                axes[k,col].set_title(f'{pol} {title}',fontsize=8)
            axes[k,2].set_xlabel(f'joint {effect.delta_joint:+.6f}: {effect.base_pred_joint}->{effect.pred_joint}\n'
                f'single {effect.delta_single:+.6f}: {effect.base_pred_single}->{effect.pred_single}',fontsize=8)
        fig.suptitle(f'{kind}: OSS1 {row.obs_id} {row.class_name}; {row.region}/{row.band}\nOne polarization altered at a time; shared region and L2')
        fig.savefig(OUT/f'case_{kind}_{int(row.obs_id)}.png',dpi=135)
        plt.close(fig)
    pd.DataFrame(selected).to_csv(OUT/'case_selection.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','infer','evaluate','analyze'])
    args = parser.parse_args()
    globals()[args.stage]()
