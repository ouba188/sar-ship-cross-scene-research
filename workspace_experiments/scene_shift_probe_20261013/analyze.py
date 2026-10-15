"""Frozen-model scene diagnosis and matched same/other-acquisition support probe."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SAR = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = OUT.parent/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916'
CODE = SAR/'oof_reliability_check'
sys.path.insert(0, str(CODE))
SEED = 20261013
MODES = ('base', 'r50_late', 'dual_refit_late')


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metadata():
    d = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id','port','class_id','class_name','mmsi','product'],
                    dtype={'mmsi':str}).fillna({'mmsi':''})
    tokens = d['product'].str.split('_')
    assert tokens.str.len().eq(9).all()
    d['acquisition'] = tokens.map(lambda x: '_'.join((x[0],x[6],x[7])))
    d['start'] = tokens.str[4]
    d['known_ship'] = d.mmsi.str.fullmatch('[1-9][0-9]{8}')
    d['row'] = np.arange(len(d))
    return d


def tasks():
    f = json.loads((OUT.parent/'paper_core_freeze_20261013/FROZEN_REFERENCE.json').read_text(encoding='utf-8'))
    return [x for x in f['tasks'] if x['task'].startswith('self24ports/')]


def metrics(y, p):
    labels = np.unique(y)
    return dict(n=len(y), classes=len(labels), oa=100*float(np.mean(y==p)),
                ba=100*float(np.mean([np.mean(p[y==c]==c) for c in labels])))


def inventory():
    write('PROTOCOL.json', dict(seed=SEED, primary='Within-port acquisition support transfer, frozen full model',
        grouping='platform + absolute orbit + datatake; not tile/product filename',
        archival='Original B64 predictions: same-acquisition support exposure, with support-vessel exclusion; observational only',
        matched='Fixed query vessels; same-acquisition vs other-acquisition same-port supports, exact same class counts and B',
        budgets=[16,64], draws=5, max_acquisitions_per_port_budget=3,
        scene_selection='Eligible scenes sorted by start; select at most3 evenly spaced, no prediction-based selection',
        query='Seeded 30 percent known vessels per scene, minimum4; one random observation per vessel; fixed across arms and budgets',
        support='One observation per known vessel, exclude all query vessels, exact matched class quota by shared pool capacity',
        eligibility='At least2 support classes and2 query classes, >=5 query rows, sufficient matched capacity',
        label_usage='Diagnostic label-matched support sampling; not a label-blind deployment acquisition policy',
        ridge=8., fusion=.5, model_changes=False, neural_training=False, source_fits=False,
        scope='Source models remain port-held legacy models, not acquisition-held or identity-held at all training stages',
        uncertainty='Average5draws then scenes within port then ports; port bootstrap descriptive development CI',
        inference='Other-scene minus same-scene; cannot identify pure imaging cause or decompose original cross-port gap'))
    d = metadata()
    scene = d.groupby(['port','acquisition']).agg(n=('row','size'),ships=('mmsi','nunique'),
        classes=('class_id','nunique'),products=('product','nunique'),start=('start','min')).reset_index()
    scene.to_csv(OUT/'scene_inventory.csv',index=False)
    acq_ports = d.groupby('acquisition').port.nunique()
    enroll = []
    archival = []
    switch = []
    coverage = []
    for ti, task in enumerate(tasks()):
        port=task['task'].split('/',1)[1]
        with np.load(task['predictions']) as z:
            pred=z['predictions']; names=list(z['modes']); rows=z['rows']; support=z['support']; masks=z['eval_masks']
            frame=d.iloc[rows].reset_index(drop=True)
            np.testing.assert_array_equal(z['ids'],frame.instance_id)
        assert frame.port.eq(port).all()
        y=frame.class_id.to_numpy(); a=frame.acquisition.to_numpy(); ships=frame.mmsi.to_numpy()
        same=np.array([np.isin(a,a[s]) for s in support])
        clean=np.array([m & frame.known_ship.to_numpy() & ~np.isin(ships,ships[s]) for m,s in zip(masks,support)])
        for draw in range(5):
            for exposed in (False,True):
                q=clean[draw] & (same[draw]==exposed)
                if not q.any(): continue
                for mode in MODES:
                    p=pred[draw,names.index(mode)]
                    archival.append(dict(port=port,draw=draw,same_acquisition_support=exposed,mode=mode,**metrics(y[q],p[q])))
        # Identical observations compared only across existing draws where exposure switches.
        for mode in MODES:
            ok=pred[:,names.index(mode)]==y[None,:]
            for i in np.flatnonzero((clean & same).any(0) & (clean & ~same).any(0)):
                yes=clean[:,i]&same[:,i]; no=clean[:,i]&~same[:,i]
                switch.append(dict(port=port,row=int(rows[i]),class_id=int(y[i]),acquisition=a[i],mode=mode,
                    same_correct=float(ok[yes,i].mean()),other_correct=float(ok[no,i].mean()),
                    delta=100*float(ok[no,i].mean()-ok[yes,i].mean()),same_draws=int(yes.sum()),other_draws=int(no.sum())))
        coverage.append(dict(port=port,n=len(frame),acquisitions=frame.acquisition.nunique(),
            products=frame['product'].nunique(),known_ship_n=int(frame.known_ship.sum()),
            acquisitions_shared_other_ports=int(frame.acquisition.map(acq_ports).gt(1).groupby(frame.acquisition).any().sum()),
            same_scene_query_events=int((clean&same).sum()),other_scene_query_events=int((clean&~same).sum()),
            switcher_observations=int(((clean&same).any(0)&(clean&~same).any(0)).sum())))
        eligible={16:[],64:[]}
        for si, (acq, group) in enumerate(frame.groupby('acquisition',sort=True)):
            valid=group[group.known_ship]
            unique=valid.mmsi.unique()
            if len(unique)<8: continue
            rng=np.random.default_rng(SEED+ti*1000+si)
            query_ships=rng.choice(unique,max(4,int(np.ceil(.3*len(unique)))),replace=False)
            q=np.array([rng.choice(valid.index[valid.mmsi==ship]) for ship in query_ships])
            if len(q)<5 or np.unique(y[q]).size<2: continue
            allowed=frame[frame.known_ship & ~frame.mmsi.isin(query_ships)]
            # Inconsistent vessel labels are excluded from support selection only.
            conflicting=allowed.groupby('mmsi').class_id.nunique()
            allowed=allowed[~allowed.mmsi.isin(conflicting[conflicting>1].index)]
            inside=allowed[allowed.acquisition==acq]
            outside=allowed[allowed.acquisition!=acq]
            capacity=np.minimum(inside.groupby('class_id').mmsi.nunique().reindex(range(8),fill_value=0),
                                outside.groupby('class_id').mmsi.nunique().reindex(range(8),fill_value=0)).to_numpy()
            if (capacity>0).sum()<2: continue
            for budget in (16,64):
                if capacity.sum()<budget: continue
                # Allocate the same quota proportionally to shared available class capacity.
                quota=np.zeros(8,dtype=int)
                quota[capacity>0]=1
                while quota.sum()<budget:
                    score=capacity/capacity.sum()*budget-quota
                    score[quota>=capacity]=-np.inf
                    quota[int(np.argmax(score))]+=1
                pairs=[]
                for draw in range(5):
                    rr=np.random.default_rng(SEED+ti*10000+si*100+budget+draw)
                    arms={}
                    for arm,pool in [('same',inside),('other',outside)]:
                        picked=[]
                        for c in range(8):
                            if not quota[c]: continue
                            subset=pool[pool.class_id==c]
                            selected=rr.choice(subset.mmsi.unique(),quota[c],replace=False)
                            picked.extend(int(rr.choice(subset.index[subset.mmsi==ship])) for ship in selected)
                        ix=np.array(picked)
                        assert len(ix)==budget and len(set(ships[ix]))==budget
                        assert not set(ships[ix])&set(ships[q])
                        arms[arm]=ix.tolist()
                    pairs.append(dict(draw=draw,**arms))
                eligible[budget].append(dict(port=port,acquisition=acq,start=group.start.min(),budget=budget,
                    query=q.tolist(),query_classes=np.unique(y[q]).tolist(),quota=quota.tolist(),pairs=pairs,
                    same_pool=inside.mmsi.nunique(),other_pool=outside.mmsi.nunique(),known_query_ships=len(set(ships[q]))))
        for budget,items in eligible.items():
            items=sorted(items,key=lambda x:(x['start'],x['acquisition']))
            if items:
                pick=np.unique(np.rint(np.linspace(0,len(items)-1,min(3,len(items)))).astype(int))
                enroll.extend(items[k] for k in pick)
            coverage[-1][f'eligible_B{budget}']=len(items)
            coverage[-1][f'selected_B{budget}']=min(3,len(items))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(archival).to_csv(OUT/'archival_strata.csv',index=False)
    sw=pd.DataFrame(switch); sw.to_csv(OUT/'archival_switchers.csv',index=False)
    ss=sw.groupby(['port','mode','class_id']).delta.mean().groupby(['port','mode']).mean().rename('delta_ba').reset_index()
    ss=ss.merge(sw.groupby(['port','mode']).agg(delta_oa=('delta','mean'),n=('row','size')).reset_index())
    ss.to_csv(OUT/'archival_switcher_summary.csv',index=False)
    write('ENROLLMENT.json',enroll)
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    print('SWITCHERS',ss.groupby('mode')[['delta_oa','delta_ba']].mean().to_string(),flush=True)
    print('ENROLLED',pd.DataFrame(enroll).groupby('budget').agg(scenes=('port','size'),ports=('port','nunique')),flush=True)


def paired():
    from threadpoolctl import threadpool_limits
    from unified_head_expert import ExpertContext, _conditions
    from unified_head_run import base_kernel
    from unified_head_core import score, late
    enrollment=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    d=metadata()
    records=[]; class_records=[]; checks=[]; timing=[]
    started=time.perf_counter()
    for task in tasks():
        port=task['task'].split('/',1)[1]
        episodes=[e for e in enrollment if e['port']==port]
        if not episodes: continue
        print('START',port,flush=True)
        ts=time.perf_counter()
        folder=Path(task['prepared']).parent
        info=json.loads(Path(task['prepared']).read_text(encoding='utf-8'))
        with np.load(task['input_cache']) as z: data={k:z[k] for k in z.files}
        with np.load(task['frequency_embedding']) as z:
            embedding=z['embedding'].astype(float)
            np.testing.assert_array_equal(z['rows'],data['rows'])
        frame=d.iloc[data['rows']].reset_index(drop=True)
        np.testing.assert_array_equal(frame.instance_id,data['ids'])
        labels=frame.class_id.to_numpy(); scene=frame.acquisition.to_numpy(); ships=frame.mmsi.to_numpy()
        reference=np.unique(np.concatenate([data['support'].ravel()]+[np.array(p[a]) for e in episodes for p in e['pairs'] for a in ['same','other']]))
        ctx=ExpertContext(folder)
        bank=np.load(ctx.parent/'target_bank.npy',mmap_mode='r')
        extra=np.load(SAR/'module_completion_20261008/ports_projection'/port/'matched_joint/joint_extra.npy',mmap_mode='r')
        u=_conditions()[data['rows']]
        with np.load(task['coefficients']) as z:
            expert_prior=np.c_[bank,u]@z['dual_coef']+z['dual_intercept']
        with threadpool_limits(limits=2):
            expert_kernel=ctx._port_kernel(bank,ctx.raw[data['rows']],extra,u,reference)
            with np.load(task['predictions']) as z:
                oldpred=z['predictions']; oldmodes=list(z['modes'])
            for draw,support in enumerate(data['support']):
                bk=base_kernel(data,embedding,support,info['eta'])
                ek=expert_kernel[:,np.searchsorted(reference,support)]
                b,_=score(bk,data['prior'],support,labels[support])
                e,_=score(ek,expert_prior,support,labels[support])
                for name,value in [('base',b),('dual_refit_expert',e),('dual_refit_late',late(b,e))]:
                    differences=int(np.sum(value.argmax(1)!=oldpred[draw,oldmodes.index(name)]))
                    checks.append(dict(port=port,draw=draw,mode=name,differences=differences))
                    assert differences==0,(port,draw,name,differences)
            for ei,episode in enumerate(episodes):
                q=np.array(episode['query']); budget=episode['budget']
                assert frame.iloc[q].acquisition.eq(episode['acquisition']).all()
                assert len(set(ships[q]))==len(q)
                probabilities=np.empty((5,2,2,len(q),8),dtype=float)
                for pair in episode['pairs']:
                    draw=pair['draw']
                    for ai,arm in enumerate(['same','other']):
                        support=np.array(pair[arm]); take=np.r_[support,q]
                        assert len(set(support)&set(q))==0
                        assert not set(ships[support])&set(ships[q])
                        assert len(set(ships[support]))==budget
                        assert np.array_equal(np.bincount(labels[support],minlength=8),episode['quota'])
                        assert bool(np.all(scene[support]==episode['acquisition']))==(arm=='same')
                        if arm=='other': assert np.all(scene[support]!=episode['acquisition'])
                        local={k:data[k][take] for k in ['phi','mean','conditions','prior']}
                        s=np.arange(budget)
                        kb=base_kernel(local,embedding[take],s,info['eta'])
                        ke=expert_kernel[np.ix_(take,np.searchsorted(reference,support))]
                        base,_=score(kb,local['prior'],s,labels[support])
                        expert,_=score(ke,expert_prior[take],s,labels[support])
                        from scipy.special import softmax
                        for mi,(mode,prob) in enumerate([('base',softmax(base,axis=1)),('full',late(base,expert))]):
                            p=prob[budget:]; probabilities[draw,ai,mi]=p
                            result=dict(port=port,acquisition=episode['acquisition'],budget=budget,draw=draw,arm=arm,mode=mode,
                                        **metrics(labels[q],p.argmax(1)))
                            records.append(result)
                            for c in np.unique(labels[q]):
                                mask=labels[q]==c
                                class_records.append(dict(port=port,acquisition=episode['acquisition'],budget=budget,draw=draw,
                                    arm=arm,mode=mode,class_id=int(c),n=int(mask.sum()),support_n=int(episode['quota'][c]),
                                    recall=100*float(np.mean(p.argmax(1)[mask]==c))))
                dest=OUT/'predictions'/port
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/f"{episode['acquisition']}_B{budget}.npz",probabilities=probabilities,
                    query_local=q,query_rows=data['rows'][q],labels=labels[q],mmsi=ships[q].astype(str),
                    supports=np.array([[p[a] for a in ['same','other']] for p in episode['pairs']]),
                    modes=np.array(['base','full']),arms=np.array(['same','other']))
        timing.append(dict(port=port,seconds=time.perf_counter()-ts,n=len(frame),reference_n=len(reference),episodes=len(episodes)))
        print('DONE',port,round(timing[-1]['seconds'],2),'seconds',flush=True)
        pd.DataFrame(records).to_csv(OUT/'paired_metrics.csv',index=False)
        pd.DataFrame(class_records).to_csv(OUT/'paired_classes.csv',index=False)
        pd.DataFrame(checks).to_csv(OUT/'baseline_replay.csv',index=False)
        pd.DataFrame(timing).to_csv(OUT/'timing.csv',index=False)
        del ctx,expert_kernel,expert_prior,bank,extra,data,embedding
    df=pd.DataFrame(records)
    ix=['port','acquisition','budget','draw','mode']
    same=df[df.arm=='same'].set_index(ix)
    other=df[df.arm=='other'].set_index(ix)
    effect=other[['oa','ba']]-same[['oa','ba']]
    effect=effect.rename(columns={'oa':'delta_oa','ba':'delta_ba'}).reset_index()
    effect.to_csv(OUT/'paired_effects.csv',index=False)
    per_scene=effect.groupby(['budget','port','acquisition','mode'])[['delta_oa','delta_ba']].mean().reset_index()
    per_port=per_scene.groupby(['budget','port','mode'])[['delta_oa','delta_ba']].mean().reset_index()
    per_port.to_csv(OUT/'port_effects.csv',index=False)
    absolute=df.groupby(['budget','port','acquisition','mode','arm'])[['oa','ba']].mean().groupby(['budget','port','mode','arm']).mean().groupby(['budget','mode','arm']).mean()
    absolute.to_csv(OUT/'absolute_summary.csv')
    rng=np.random.default_rng(SEED)
    summaries=[]
    for (budget,mode),group in per_port.groupby(['budget','mode']):
        values=group[['delta_oa','delta_ba']].to_numpy()
        bootstrap=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
        ci=np.quantile(bootstrap,[.025,.975],axis=0)
        summaries.append(dict(budget=int(budget),mode=mode,ports=len(group),
            scenes=int(len(per_scene[(per_scene.budget==budget)&(per_scene['mode']==mode)])),
            delta_oa=float(values[:,0].mean()),delta_ba=float(values[:,1].mean()),
            oa_ci_low=float(ci[0,0]),oa_ci_high=float(ci[1,0]),ba_ci_low=float(ci[0,1]),ba_ci_high=float(ci[1,1]),
            ports_ba_other_worse=int((values[:,1]<0).sum()),ports_oa_other_worse=int((values[:,0]<0).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'paired_summary.csv',index=False)
    write('COMPLETE.json',dict(status='complete',scene_budget_contexts=len(enrollment),support_pairs=len(enrollment)*5,
        readout_solves=len(enrollment)*5*2*2,neural_training=0,source_fits=0,baseline_replays=len(checks),
        replay_argmax_differences=int(sum(c['differences'] for c in checks)),seconds=time.perf_counter()-started,
        main_model_changed=False,summary=summaries))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['inventory','paired'],default='inventory')
    args=parser.parse_args()
    if args.stage=='inventory': inventory()
    else: paired()
