"""Exploratory regional utility strata, with source-only numerical cutpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'conditional_evidence_pilot'


def main():
    f=pd.read_csv(PREV/'cohort.csv',dtype={'mmsi':str})
    p=pd.read_csv(ROOT/'predictions.csv')
    partitions=pd.read_csv(PREV/'partitions.csv')
    coverage=pd.read_csv(ROOT/'view_coverage.csv').set_index(['obs_id','pol'])
    alignment=pd.read_csv(ROOT/'crossview_alignment.csv').set_index('obs_id').loc[f.obs_id]
    vv=coverage.xs('vv',level='pol').loc[f.obs_id]
    vh=coverage.xs('vh',level='pol').loc[f.obs_id]
    area=(vv.target_pixels/(vv.target_pixels+vv.other_pixels+vv.rest_pixels)).to_numpy()
    has_other=(vv.other_pixels.to_numpy()>0)|((vh.other_pixels.to_numpy()>0)&~alignment.disable_new_vh.to_numpy(bool))
    records=[]; thresholds=[]
    for fold in range(3):
        base=np.load(ROOT/f'baseline_fold{fold}.npy',mmap_mode='r')
        sorted_scores=np.sort(base,axis=1); gap=sorted_scores[:,-1]-sorted_scores[:,-2]
        valid_ids=partitions[(partitions.fold==fold)&partitions.partition.eq('val')].obs_id
        val=f.obs_id.isin(valid_ids).to_numpy()&f.obs_id.ne(31237).to_numpy()
        gap_cut=np.quantile(gap[val],[.25,.5,.75]); area_cut=np.quantile(area[val],[.25,.5,.75])
        thresholds.append(dict(fold=fold,validation_n=int(val.sum()),top_two_gap_cuts=gap_cut.tolist(),vv_target_fraction_cuts=area_cut.tolist()))
        meta=f[['obs_id','class_id','mmsi','port']].copy()
        meta['gap_group']=np.searchsorted(gap_cut,gap,side='right')+1
        meta['target_area_group']=np.searchsorted(area_cut,area,side='right')+1
        meta['other_present']=has_other.astype(int)
        meta['alignment']=alignment.status.to_numpy()
        meta=meta.set_index('obs_id')
        for seed in (20261019,20261020,20261021):
            full=p[(p.fold==fold)&(p.seed==seed)&p.arm.eq('pre')&p.selection.eq('source_selected')].set_index('obs_id')
            for region in ('target','other','rest'):
                data=np.load(ROOT/f'intervention_f{fold}_s{seed}_drop_{region}.npz')
                ids=data['obs_id']; g=meta.loc[ids].copy()
                assert np.array_equal(full.loc[ids].class_id.to_numpy(),g.class_id.to_numpy())
                drop_correct=data['scores'].argmax(1)==g.class_id.to_numpy()
                actual=full.loc[ids].correct.to_numpy(bool)
                g['delta']=actual.astype(int)-drop_correct.astype(int)
                g['rescue']=actual&~drop_correct; g['harm']=~actual&drop_correct
                g['region']=region; g['fold']=fold; g['seed']=seed
                records.append(g.reset_index())
    allrows=pd.concat(records,ignore_index=True)
    rows=[]
    for field in ('gap_group','target_area_group','other_present','alignment'):
        for (region,seed,key),g in allrows.groupby(['region','seed',field]):
            rows.append(dict(condition=field,group=str(key),region=region,seed=seed,n=len(g),ships=g.mmsi.nunique(),oa_net_pp=100*g.delta.mean(),ba_net_pp=100*g.groupby('class_id').delta.mean().mean(),rescues=int(g.rescue.sum()),harms=int(g.harm.sum())))
    by_seed=pd.DataFrame(rows); by_seed.to_csv(ROOT/'response_conditions_by_seed.csv',index=False)
    summary=by_seed.groupby(['condition','group','region']).agg(n=('n','first'),ships=('ships','first'),oa_net_pp=('oa_net_pp','mean'),oa_min_pp=('oa_net_pp','min'),oa_max_pp=('oa_net_pp','max'),ba_net_pp=('ba_net_pp','mean'),positive_seeds=('oa_net_pp',lambda x:int((x>0).sum())),rescues_mean=('rescues','mean'),harms_mean=('harms','mean')).reset_index()
    summary.to_csv(ROOT/'response_conditions_summary.csv',index=False)
    (ROOT/'response_conditions_thresholds.json').write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__': main()
