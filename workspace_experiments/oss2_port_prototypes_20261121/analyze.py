"""OSS2 source-only vessel/port phenotype study; no fitting or held-out inspection."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OLD = WORK/'oss2_grd_port_class_analysis_20261116'
META = WORK/'all_factor_probe_20261017/metadata/oss2_factor_metadata.csv'
MANIFEST = WORK/'m1_coverage_20261101/manifest.json'
REFS = WORK/'scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/manifests/references.json'
SEED = 20261121

def csv(name, frame):
    frame.to_csv(ROOT/name, index=False, encoding='utf-8-sig')

def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x:x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')

def vessels(df, port='port_proxy'):
    return df.groupby(['class_name',port,'mmsi'],as_index=False).agg(
        length=('ais_length_m','median'), beam=('ais_width_m','median'),
        observations=('sample_id','size'), products=('product','nunique'))

def pair_values(df,a,b):
    x,y=df[df.port_proxy==a],df[df.port_proxy==b]
    shared=set(x.mmsi)&set(y.mmsi)
    return x[~x.mmsi.isin(shared)],y[~y.mmsi.isin(shared)],len(shared)

def bootdiff(x,y):
    rng=np.random.default_rng(SEED)
    values=np.median(rng.choice(y,(2000,len(y))),axis=1)-np.median(rng.choice(x,(2000,len(x))),axis=1)
    return np.quantile(values,[.025,.975]).tolist()

def main():
    rows=json.loads(MANIFEST.read_text(encoding='utf-8'))
    meta=pd.read_csv(META,dtype={'mmsi':str,'ais_mmsi_raw':str})
    loc=pd.read_csv(OLD/'experiment_manifest_join.csv',dtype={'mmsi':str})
    # Existing geographic lookup is reused unchanged. Join by exact sample identity.
    d=loc[['sample_id','port_proxy','nearest_km','distance_gap_km','conservative_assignment']].merge(meta,on='sample_id',validate='one_to_one')
    mr=pd.DataFrame([dict(sample_id=r['id'],index=i,role=r['role'],label=r['label'],manifest_mmsi=r['mmsi'],manifest_scene=r['scene_id']) for i,r in enumerate(rows)])
    d=mr.merge(d,on='sample_id',validate='one_to_one')
    assert (d.mmsi==d.manifest_mmsi).all() and (d.scene_id==d.manifest_scene).all()
    assert (d.product_type=='GRDH').all()
    s=d[d.role.isin(['fit_train','utility_train'])].copy()
    held=set(d.loc[d.role.isin(['evaluation','fit_validation']),'mmsi'])
    # Detect source label/static dimension disagreements instead of inventing a prototype.
    conflicts=s.groupby('mmsi').agg(nclass=('class_name','nunique'),nlength=('ais_length_m','nunique'),nbeam=('ais_width_m','nunique'))
    bad=set(conflicts[(conflicts.nclass>1)|(conflicts.nlength>1)|(conflicts.nbeam>1)].index)
    s['valid']=s.identity_match_valid & s.ais_length_valid & s.ais_width_valid & ~s.mmsi.isin(bad) & s.mmsi.str.fullmatch(r'\d{9}')
    s['plausible']=s.ais_length_m.between(10,500)&s.ais_width_m.between(2,100)&(s.ais_length_m/s.ais_width_m).between(1.5,15)
    s['quarter']=pd.to_datetime(s.sar_time_utc).dt.quarter
    s['month']=pd.to_datetime(s.sar_time_utc).dt.strftime('%Y-%m')
    s['heldout_identity_overlap']=s.mmsi.isin(held)
    csv('source_observations_local.csv',s)
    valid=s[s.valid & ~s.heldout_identity_overlap & (s.port_proxy!='outside_150km')]
    v=vessels(valid)
    csv('source_vessels_local.csv',v)
    stats=v.groupby(['class_name','port_proxy'],as_index=False).agg(vessels=('mmsi','size'),observations=('observations','sum'),length_q10=('length',lambda x:x.quantile(.1)),length_q25=('length',lambda x:x.quantile(.25)),length_median=('length','median'),length_q75=('length',lambda x:x.quantile(.75)),length_q90=('length',lambda x:x.quantile(.9)),beam_median=('beam','median'))
    csv('port_class_sizes.csv',stats)
    results=[]
    # 15 ships per side is declared for this smaller source queue; no inferential p-value filtering.
    for cls,g in v.groupby('class_name'):
        for a,b in itertools.combinations(sorted(g.port_proxy.unique()),2):
            x,y,shared=pair_values(g,a,b)
            if min(len(x),len(y))<15:continue
            if x.length.median()>y.length.median():a,b=b,a;x,y=y,x
            ci=bootdiff(x.length.to_numpy(),y.length.to_numpy())
            results.append(dict(class_name=cls,port_low=a,port_high=b,n_low=len(x),n_high=len(y),shared_removed=shared,median_low=x.length.median(),median_high=y.length.median(),gap_m=y.length.median()-x.length.median(),wasserstein_m=wasserstein_distance(x.length,y.length),bootstrap_low=ci[0],bootstrap_high=ci[1]))
    pairs=pd.DataFrame(results).sort_values(['class_name','gap_m'],ascending=[True,False])
    csv('port_pair_size_differences.csv',pairs)
    selected=pairs.groupby('class_name',sort=True).head(1)
    csv('selected_pairs.csv',selected)
    sens=[]
    for r in selected.itertuples():
        for scope,q in [('primary',valid),('fit_train_only',valid[valid.role=='fit_train']),('conservative_geo',valid[valid.conservative_assignment]),('plausible_sizes',valid[valid.plausible]),('ascending',valid[valid.product_pass=='ASCENDING']),('descending',valid[valid.product_pass=='DESCENDING'])]+[(f'quarter_{i}',valid[valid.quarter==i]) for i in [1,2,3,4]]:
            g=vessels(q[q.class_name==r.class_name]);x,y,shared=pair_values(g,r.port_low,r.port_high)
            sens.append(dict(class_name=r.class_name,port_low=r.port_low,port_high=r.port_high,scope=scope,n_low=len(x),n_high=len(y),median_low=x.length.median(),median_high=y.length.median(),gap_m=y.length.median()-x.length.median(),shared_removed=shared))
        # Exact product strata separate image acquisition from geographic membership where supported.
        for prod,q in valid[valid.class_name==r.class_name].groupby('product'):
            g=vessels(q);x,y,_=pair_values(g,r.port_low,r.port_high)
            if min(len(x),len(y))>=5:
                sens.append(dict(class_name=r.class_name,port_low=r.port_low,port_high=r.port_high,scope='same_product_'+prod,n_low=len(x),n_high=len(y),median_low=x.length.median(),median_high=y.length.median(),gap_m=y.length.median()-x.length.median(),shared_removed=0))
    csv('selected_pair_sensitivity.csv',pd.DataFrame(sens))
    merged=valid.copy();merged['port_proxy']=merged.port_proxy.replace({'Hong_Kong':'Pearl_River','Shenzhen':'Pearl_River','Shanghai':'Yangtze_Delta','Ningbo':'Yangtze_Delta'})
    mv=vessels(merged)
    csv('coarse_area_sizes.csv',mv.groupby(['class_name','port_proxy'],as_index=False).agg(vessels=('mmsi','size'),median_length=('length','median'),q10=('length',lambda x:x.quantile(.1)),q90=('length',lambda x:x.quantile(.9))))
    # Fine labels and length within subtype; one vessel per area, no re-labelling.
    fine=valid.sort_values('sample_id').drop_duplicates(['class_name','port_proxy','mmsi'])
    csv('fine_label_sizes.csv',fine.groupby(['class_name','port_proxy','marine_elaborated_type'],as_index=False).agg(vessels=('mmsi','size'),median_length=('ais_length_m','median')))
    comp=valid.groupby(['class_name','port_proxy'],as_index=False).agg(products=('product','nunique'),orbits=('product_relative_orbit','nunique'),months=('month','nunique'),incidence_min=('incidence_deg','min'),incidence_max=('incidence_deg','max'),ascending_fraction=('product_pass',lambda x:(x=='ASCENDING').mean()))
    csv('acquisition_composition.csv',comp)
    refs=json.loads(REFS.read_text(encoding='utf-8'))
    refids={r['id'] for r in refs};rd=meta[meta.sample_id.isin(refids)].copy()
    assert len(rd)==len(refs)
    rd=rd.merge(loc[['sample_id','port_proxy']],on='sample_id',how='left',validate='one_to_one')
    csv('references_local.csv',rd[['sample_id','mmsi','class_name','port_proxy','ais_length_m','ais_width_m']])
    cov=[]
    for cls,q in valid.groupby('class_name'):
        x=q.sort_values('sample_id').drop_duplicates('mmsi');rr=rd[rd.class_name==cls];rr=rr[rr.ais_length_m.gt(0)&rr.ais_width_m.gt(0)]
        vals=[]
        for r in x.itertuples():
            ref=rr[rr.mmsi!=r.mmsi]
            if ref.empty:continue
            length=np.maximum(r.ais_length_m/ref.ais_length_m.to_numpy(),ref.ais_length_m.to_numpy()/r.ais_length_m)
            beam=np.maximum(r.ais_width_m/ref.ais_width_m.to_numpy(),ref.ais_width_m.to_numpy()/r.ais_width_m)
            vals.append((float(length.min()),float(np.maximum(length,beam).min())))
        arr=np.array(vals)
        cov.append(dict(class_name=cls,query_ships=len(arr),references=len(rr),reference_min_length=rr.ais_length_m.min(),reference_max_length=rr.ais_length_m.max(),nearest_length_over_1p25_pct=100*(arr[:,0]>1.25).mean(),nearest_length_over_1p5_pct=100*(arr[:,0]>1.5).mean(),nearest_joint_over_1p25_pct=100*(arr[:,1]>1.25).mean()))
    csv('reference_size_coverage.csv',pd.DataFrame(cov))
    audit=dict(seed=SEED,manifest_observations=len(d),source_observations=len(s),source_ships=s.mmsi.nunique(),source_products=s['product'].nunique(),source_class_counts=s.class_name.value_counts().to_dict(),static_or_class_conflicted_ships=len(bad),valid_observations=int(s.valid.sum()),heldout_overlap_source_observations=int(s.heldout_identity_overlap.sum()),outside_observations=int((s.port_proxy=='outside_150km').sum()),primary_observations=len(valid),primary_ships=valid.mmsi.nunique(),primary_vessel_port_cells=len(v),primary_area_groups=valid.port_proxy.nunique(),eligible_pairs=len(pairs),reference_count=len(refs),scope='Existing fit_train + utility_train only. No heldout images/predictions used. Four classes; local GRDH source subset, not full OSS2.',geography='Reused 20261116 nearest-anchor water-area proxies, <=150km; conservative <=75km and second-nearest gap>=10km. Not official port labels.',bootstrap='2000 within-group vessel resamples; exploratory selected extrema; CI does not account for product clustering or selection.',prediction_cache='Existing m1_coverage initial_source, source-side descriptive evidence, not unseen-port performance')
    write('coverage.json',audit)
    plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,r in zip(axs.flat,selected.itertuples()):
        g=v[v.class_name==r.class_name];x,y,_=pair_values(g,r.port_low,r.port_high)
        for q,p,c in [(x,r.port_low,'#4C8C9B'),(y,r.port_high,'#D99A6C')]:
            xx=np.sort(q.length);ax.step(xx,np.arange(1,len(xx)+1)/len(xx),where='post',label=f'{p} (n={len(xx)})',color=c,lw=2)
        ax.set_title(r.class_name);ax.set_xlabel('AIS length (m)');ax.set_ylabel('Vessel-weighted ECDF');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('OSS2 source observations: within-class size varies across water-area groups\nExploratory pairs; common MMSIs removed; port labels are geographic proxies',fontsize=12)
    for ext in ['png','svg']:fig.savefig(ROOT/f'port_size_ecdf.{ext}',dpi=170)
    plt.close(fig)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(selected.to_string(index=False));print(pd.DataFrame(cov).to_string(index=False))

if __name__=='__main__':main()
