"""Fixed-task accounting; no target-dependent model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'geometry_relation_slots_20261016'
MET=['oa','ba','macro_nll','macro_brier']
KEY=['port','fold','budget','origin','draw','sample_id','seed']
P=[f'p{i}' for i in range(8)]


def markdown(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'---|'*len(frame.columns)]
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return lines


def main(stage):
    root=HERE/stage
    met=pd.concat([pd.read_csv(p) for p in (root/'evaluation').glob('*/*_metrics.csv')],ignore_index=True)
    pred=pd.concat([pd.read_csv(p,dtype={'mmsi':str}) for p in (root/'evaluation').glob('*/*_predictions.csv')],ignore_index=True)
    assert not pred.duplicated(KEY+['arm']).any()
    counts=met.loc[met.scope.eq('full')].groupby(['arm','seed']).size()
    expected={'B','spatial','conditional','geometry'} if stage=='bounded' else {'B','geometry_single','geometry_pair','geometry_rank','conditional_rank'}
    assert set(counts.index)=={(arm,seed) for arm in expected for seed in range(3)}
    assert counts.nunique()==1 and counts.iloc[0]==190
    b=pred.loc[pred.arm.eq('B')]
    historical=pd.read_csv(PREV/'predictions.csv',dtype={'mmsi':str})
    historical=historical.loc[historical.arm.eq('B')]
    check=b.merge(historical[KEY+P],on=KEY,suffixes=('','_old'),validate='one_to_one')
    assert len(check)==len(b)==len(historical)
    error=float(np.max(np.abs(check[P].to_numpy()-check[[p+'_old' for p in P]].to_numpy())))
    assert error<1e-10
    # Equal acquisition weighting inside each port, then equal port weighting.
    acq=met.groupby(['scope','origin','arm','seed','port','acquisition'])[MET].mean()
    port=acq.groupby(['scope','origin','arm','seed','port']).mean().reset_index()
    means=port.groupby(['scope','origin','arm','seed'])[MET].mean().reset_index()
    met.to_csv(root/'metrics.csv',index=False); pred.to_csv(root/'predictions.csv',index=False)
    port.to_csv(root/'per_port.csv',index=False); means.to_csv(root/'seed_means.csv',index=False)
    summary=means.groupby(['scope','origin','arm'])[MET].agg(['mean','std'])
    summary.to_csv(root/'summary.csv')
    refs=[(a,'B') for a in pred.arm.unique() if a!='B']
    refs+=([('geometry','conditional'),('conditional','spatial')] if stage=='bounded' else
           [('geometry_pair','geometry_single'),('geometry_rank','geometry_pair'),('geometry_rank','conditional_rank')])
    contrast=[]
    for arm,ref in refs:
        m=means.loc[means.arm.eq(arm)].merge(means.loc[means.arm.eq(ref)],on=['scope','origin','seed'],suffixes=('','_ref'),validate='one_to_one')
        m['reference']=ref
        for k in MET:
            m['delta_'+k]=m[k]-m[k+'_ref']
        contrast.append(m[['scope','origin','arm','reference','seed']+['delta_'+k for k in MET]])
    contrast=pd.concat(contrast,ignore_index=True); contrast.to_csv(root/'paired_contrasts.csv',index=False)
    events=[]
    for arm in pred.arm.unique():
        if arm=='B':
            continue
        m=pred.loc[pred.arm.eq(arm)].merge(b[KEY+['pred','correct','ap',*P]],on=KEY,suffixes=('','_B'),validate='one_to_one')
        np.testing.assert_array_equal(m.loc[~m.enabled,P],m.loc[~m.enabled,[p+'_B' for p in P]])
        m['rescue']=m.correct&~m.correct_B; m['harm']=~m.correct&m.correct_B
        m['delta_correct']=m.correct.astype(int)-m.correct_B.astype(int)
        events.append(m)
    events=pd.concat(events,ignore_index=True)
    events.to_csv(root/'paired_events.csv',index=False)
    events.loc[events.rescue|events.harm].to_csv(root/'changed_cases.csv',index=False)
    event_summary=events.groupby(['arm','seed','port','origin'])[['rescue','harm']].sum().reset_index()
    event_summary.to_csv(root/'event_counts.csv',index=False)
    contributions=[]
    unit=['arm','seed','port','origin','acquisition','fold','draw']
    for key,g in events.groupby(unit,sort=False):
        for cls,c in g.groupby('truth'):
            contributions.append(dict(zip(unit,key),truth=int(cls),delta_ba=100*c.delta_correct.mean()/g.truth.nunique(),
                                      delta_oa=100*c.delta_correct.sum()/len(g),rescues=int(c.rescue.sum()),harms=int(c.harm.sum())))
    contribution=pd.DataFrame(contributions)
    contribution.to_csv(root/'class_contributions.csv',index=False)
    classsum=contribution.groupby(unit)[['delta_ba','delta_oa']].sum().groupby(['arm','seed','port','origin','acquisition']).mean().groupby(['arm','seed','port','origin']).mean()
    for r in port.loc[port.scope.eq('full')&port.arm.ne('B')].itertuples():
        ref=port.loc[port.scope.eq('full')&port.arm.eq('B')&port.seed.eq(r.seed)&port.port.eq(r.port)&port.origin.eq(r.origin)].iloc[0]
        actual=classsum.loc[(r.arm,r.seed,r.port,r.origin)]
        assert abs(actual.delta_ba-(r.ba-ref.ba))<1e-9
        assert abs(actual.delta_oa-(r.oa-ref.oa))<1e-9
    retrieval=[]
    for scope,f in [('full',pred),('geometry',pred.loc[pred.enabled])]:
        x=f.groupby(['arm','seed','port','origin','acquisition','fold','draw','truth'])[['ap']].mean()
        for grouping in [['arm','seed','port','origin','acquisition','fold','draw'],['arm','seed','port','origin','acquisition'],['arm','seed','port','origin'],['arm','seed','origin']]:
            x=x.groupby(grouping).mean()
        x=x.reset_index(); x['scope']=scope; retrieval.append(x)
    pd.concat(retrieval).to_csv(root/'retrieval.csv',index=False)
    draws=met.groupby(['scope','origin','arm','seed','draw','port','acquisition'])[MET].mean().groupby(['scope','origin','arm','seed','draw','port']).mean().groupby(['scope','origin','arm','seed','draw']).mean().reset_index()
    draws.to_csv(root/'support_draw_means.csv',index=False)
    training=[]
    for file in (root/'trained').glob('*/*_seed*.pt'):
        saved=torch.load(file,map_location='cpu',weights_only=False)
        training.append(dict(pool=file.parent.name,arm=file.stem.split('_seed')[0],seed=int(file.stem.split('_seed')[1]),
                             unavailable=saved.get('unavailable',False),seconds=saved.get('seconds'),first_gradient=saved.get('first_gradient')))
    training=pd.DataFrame(training); training.to_csv(root/'training_summary.csv',index=False)
    curves=[]
    for p in (root/'trained').glob('*/*_curve.csv'):
        c=pd.read_csv(p)
        first=c.loc[c.step.eq(0)].iloc[0]; last=c.loc[c.step.eq(200)].iloc[0]
        r=dict(pool=p.parent.name,arm=p.name.split('_seed')[0],seed=int(p.name.split('_seed')[1].split('_')[0]))
        for col in c.columns:
            if col.startswith(('fit_','selection_')) or col=='raw_centered_energy':
                r[col+'_initial']=first[col]; r[col+'_final']=last[col]; r[col+'_delta']=last[col]-first[col]
        r['logged_max_feature_norm2']=c.max_feature_norm2.max()
        assert r['logged_max_feature_norm2']<=1.000001
        curves.append(r)
    pd.DataFrame(curves).to_csv(root/'source_learning.csv',index=False)
    sampling=0
    for p in (root/'trained').glob('*/*_sampling.json'):
        if stage=='bounded':
            old=PREV/'trained'/p.parent.name/p.name
            assert json.loads(p.read_text())==json.loads(old.read_text())
        else:
            arm=p.name.split('_seed')[0]
            if arm in ('geometry_rank','conditional_rank'):
                ref=p.with_name(p.name.replace(arm,'geometry_pair'))
                assert json.loads(p.read_text())==json.loads(ref.read_text())
        sampling+=1
    check=dict(stage=stage,trained=int((~training.unavailable).sum()),fallback=int(training.unavailable.sum()),
               tasks_per_arm_seed=int(counts.iloc[0]),probability_rows=len(pred),baseline_max_error=error,
               fallback_exact=True,class_accounting=True,sampling_traces=sampling)
    (root/'RESULT_CHECK.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    table=means.loc[means.scope.eq('full')&means.origin.eq('cross')].groupby('arm')[MET].mean().reset_index()
    lines=[f'# {stage}: fixed training result','', 'S32; three training seeds; five paired support draws; development cohort.', '']+markdown(table)
    lines+=['','## Paired Differences','']+markdown(contrast.loc[contrast.origin.eq('cross')].groupby(['scope','arm','reference'])[['delta_'+k for k in MET]].mean().reset_index())
    lines+=['','## Checks','',json.dumps(check)]
    (root/'RESULT_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(table.to_string(index=False)); print(json.dumps(check))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['bounded','recovered'])
    main(parser.parse_args().stage)
