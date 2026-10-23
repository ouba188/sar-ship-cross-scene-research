"""Fixed source-defined score matching and competitor-switch accounting."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
CONTROLS = HERE.parent
ALIGN = CONTROLS.parent
OLD = ALIGN.parent/'old_partition_replication'
CALIPER = .25


def read_predictions(partition, role):
    name = 'predictions.csv' if role=='test' else 'validation_predictions.csv'
    if partition=='new':
        sc = pd.read_csv(ALIGN/name,dtype={'mmsi':str})
        sc = sc[sc.cell.eq('jointCE__J')]
        ca = pd.read_csv(CONTROLS/name,dtype={'mmsi':str})
        ca = ca[ca.arm.eq('score_only') & ca.selector.eq('J')]
    else:
        df = pd.read_csv(OLD/name,dtype={'mmsi':str})
        sc = df[df.arm.eq('signed_only') & df.selector.eq('J')]
        ca = df[df.arm.eq('score_only') & df.selector.eq('J')]
    return sc,ca


def paired(sc, ca, seed):
    left = sc[sc.seed.eq(seed)].set_index('instance_id').sort_index()
    right = ca[ca.seed.eq(seed)].set_index('instance_id').loc[left.index]
    np.testing.assert_allclose(left[['b0','b1','b2']],right[['b0','b1','b2']],atol=1e-12,rtol=0)
    g = left[['class_id','fold','port','pass_group','mmsi']].copy().reset_index()
    b, a, s = [frame[cols].to_numpy(float) for frame,cols in [(left,['b0','b1','b2']),
        (right,['s0','s1','s2']),(left,['s0','s1','s2'])]]
    y = g.class_id.to_numpy()
    for name,score in [('base',b),('calibration',a),('signed',s)]:
        pred = score.argmax(1)
        wrong = score.copy()
        wrong[np.arange(len(y)),y] = -np.inf
        g[f'{name}_pred'] = pred
        g[f'{name}_correct'] = pred==y
        g[f'{name}_competitor'] = wrong.argmax(1)
        g[f'{name}_margin'] = score[np.arange(len(y)),y]-wrong.max(1)
    g['signed_only_correct'] = g.signed_correct & ~g.calibration_correct
    g['calibration_only_correct'] = ~g.signed_correct & g.calibration_correct
    g['both_wrong'] = ~g.signed_correct & ~g.calibration_correct
    raw = np.c_[b-b.mean(1,keepdims=True),a-a.mean(1,keepdims=True)]
    return g,raw


def matching(g, z):
    rows=[]
    for _, group in g.groupby(['fold','class_id','base_pred','calibration_pred']):
        positive=group.index[group.signed_only_correct].to_numpy()
        negative=group.index[group.both_wrong].to_numpy()
        if not len(positive) or not len(negative):
            continue
        distances=cdist(z[positive],z[negative])/np.sqrt(z.shape[1])
        order=np.argsort(distances,axis=None,kind='stable')
        used_pos,used_neg=set(),set()
        for flat in order:
            i,j=np.unravel_index(flat,distances.shape)
            if distances[i,j]>CALIPER:
                break
            if i in used_pos or j in used_neg:
                continue
            used_pos.add(i);used_neg.add(j)
            pp,nn=g.loc[positive[i]],g.loc[negative[j]]
            rows.append(dict(fold=int(pp.fold),class_id=int(pp.class_id),distance=float(distances[i,j]),
                positive_id=pp.instance_id,negative_id=nn.instance_id,
                positive_port=pp.port,negative_port=nn.port,
                positive_pass=pp.pass_group,negative_pass=nn.pass_group,
                base_pred=int(pp.base_pred),calibration_pred=int(pp.calibration_pred),
                positive_base_margin=pp.base_margin,negative_base_margin=nn.base_margin,
                positive_calibration_margin=pp.calibration_margin,negative_calibration_margin=nn.calibration_margin,
                positive_signed_margin=pp.signed_margin,negative_signed_margin=nn.signed_margin))
    return rows


def main():
    counts, matches, groups, competition, summaries, scales, raw_rows=[] ,[],[],[],[],[],[]
    for partition in ['new','old']:
        loaded={role:read_predictions(partition,role) for role in ['validation','test']}
        for seed in [17,29,43]:
            vg,vraw=paired(*loaded['validation'],seed)
            scale=vraw.std(0).clip(1e-8)
            scales.append(dict(partition=partition,seed=seed,source_n=len(vg),
                **{f'scale_{i}':float(v) for i,v in enumerate(scale)}))
            for role in ['validation','test']:
                g,raw=(vg,vraw) if role=='validation' else paired(*loaded[role],seed)
                meta=dict(partition=partition,seed=seed,role=role)
                mm=matching(g,raw/scale)
                matches.extend(dict(**meta,**row) for row in mm)
                summaries.append(dict(**meta,n=len(g),signed_only_correct=int(g.signed_only_correct.sum()),
                    calibration_only_correct=int(g.calibration_only_correct.sum()),both_wrong=int(g.both_wrong.sum()),
                    matched=len(mm),coverage=len(mm)/max(1,int(g.signed_only_correct.sum())),
                    distance_median=float(np.median([r['distance'] for r in mm])) if mm else np.nan,
                    distance_max=max((r['distance'] for r in mm),default=np.nan)))
                for scope,group in [('all',g),('signed_only_correct',g[g.signed_only_correct]),
                                     ('calibration_only_correct',g[g.calibration_only_correct]),('both_wrong',g[g.both_wrong])]:
                    if group.empty:
                        continue
                    competition.append(dict(**meta,scope=scope,n=len(group),
                        calibration_competitor_changed=int((group.base_competitor!=group.calibration_competitor).sum()),
                        calibration_competitor_changed_rate=float((group.base_competitor!=group.calibration_competitor).mean()),
                        signed_competitor_changed=int((group.base_competitor!=group.signed_competitor).sum()),
                        signed_competitor_changed_rate=float((group.base_competitor!=group.signed_competitor).mean()),
                        calibration_top1_changed_rate=float((group.base_pred!=group.calibration_pred).mean()),
                        signed_top1_changed_rate=float((group.base_pred!=group.signed_pred).mean())))
                pos_ids={r['positive_id'] for r in mm}
                neg_ids={r['negative_id'] for r in mm}
                g['matched_positive']=g.instance_id.isin(pos_ids)
                g['matched_negative']=g.instance_id.isin(neg_ids)
                for (c,port),h in g.groupby(['class_id','port']):
                    groups.append(dict(**meta,class_id=c,port=port,n=len(h),
                        signed_only_correct=int(h.signed_only_correct.sum()),calibration_only_correct=int(h.calibration_only_correct.sum()),
                        both_wrong=int(h.both_wrong.sum()),matched_positive=int(h.matched_positive.sum()),matched_negative=int(h.matched_negative.sum())))
                for key,value in meta.items():
                    g[key]=value
                raw_rows.append(g)
    all_rows=pd.concat(raw_rows,ignore_index=True)
    recurrent=[]
    for (partition,role,c,port),g in all_rows.groupby(['partition','role','class_id','port']):
        success=g[g.signed_only_correct]
        seed_counts=success.groupby('instance_id').seed.nunique()
        stable_ids=seed_counts.index[seed_counts>=2]
        stable=success[success.instance_id.isin(stable_ids)].drop_duplicates('instance_id')
        matched=g[g.matched_positive].drop_duplicates('instance_id')
        recurrent.append(dict(partition=partition,role=role,class_id=c,port=port,
            n_unique=g.instance_id.nunique(),signed_only_unique=success.instance_id.nunique(),
            stable_2seed_ids=len(stable),stable_2seed_passes=stable.pass_group.nunique(),
            matched_positive_unique=len(matched),matched_positive_passes=matched.pass_group.nunique(),
            stable_source_condition=len(stable)>=3 and stable.pass_group.nunique()>=2))
    recurrence=pd.DataFrame(recurrent)
    source=recurrence[recurrence.role.eq('validation')]
    cross=source.pivot(index=['class_id','port'],columns='partition',values=['stable_2seed_ids','stable_2seed_passes','matched_positive_unique','stable_source_condition'])
    cross.columns=['_'.join(c) for c in cross.columns]
    cross=cross.reset_index()
    cross['source_repeats_both_partitions']=cross.stable_source_condition_new.fillna(False)&cross.stable_source_condition_old.fillna(False)
    cross['source_repeats_and_score_matches']=cross.source_repeats_both_partitions&(cross.matched_positive_unique_new.fillna(0)>0)&(cross.matched_positive_unique_old.fillna(0)>0)
    stable_rows=all_rows[all_rows.signed_only_correct].groupby(['partition','role','instance_id']).seed.nunique().reset_index(name='supporting_seeds')
    stable_rows=stable_rows[stable_rows.supporting_seeds>=2]
    case_meta=all_rows.drop_duplicates(['partition','role','instance_id'])[['partition','role','instance_id','class_id','port','pass_group']]
    stable_rows.merge(case_meta,on=['partition','role','instance_id']).to_csv(HERE/'stable_signed_only_cases.csv',index=False)
    pd.DataFrame(scales).to_csv(HERE/'source_score_scales.csv',index=False)
    pd.DataFrame(summaries).to_csv(HERE/'matching_summary.csv',index=False)
    pd.DataFrame(matches).to_csv(HERE/'matched_pairs.csv',index=False)
    pd.DataFrame(groups).to_csv(HERE/'class_port_groups.csv',index=False)
    pd.DataFrame(competition).to_csv(HERE/'competitor_changes.csv',index=False)
    recurrence.to_csv(HERE/'recurrence.csv',index=False)
    cross.to_csv(HERE/'source_cross_partition.csv',index=False)
    receipt=dict(fits=0,caliper=CALIPER,matching='one-to-one within fold/truth/base-top1/calibrated-top1',
        scales='source validation only',test_used_for_design=False,matched_pairs=len(matches))
    (HERE/'summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(cross[cross.source_repeats_both_partitions].to_string(index=False),flush=True)
    print(pd.DataFrame(competition).query('scope == "all"').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
