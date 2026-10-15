"""Apply source-frozen utility rules to historical ports and external tasks."""

import run_source as r
from run_source import np,pd,c,OUT,BUDGET,fit_gate,support_oof,probabilities,metrics
from strong_external import CachedStrongFeatures
from pathlib import Path
import argparse
import time
import gc

CACHE=Path('C:/Users/25791/AppData/Local/Temp/SAR_classwise_utility_20261012')


def folders():
    return [('self24ports',p.parent) for p in sorted((BUDGET/'tasks/self24ports').glob('*/PREDICTIONS_FROZEN.json'))]+[
        (d,BUDGET/'strong_external'/d/d) for d in ('oss2','f0921')]


def identity(frame,ids):
    key=next((k for k in ('mmsi','mmsi_or_ship_id') if k in frame),None)
    if key is None:
        return np.array(['row:'+str(i) for i in ids])
    values=frame.iloc[ids][key].astype('string').fillna('').str.strip().str.replace(r'\.0$','',regex=True)
    known=values.str.fullmatch(r'[0-9]{9}')&values.ne('000000000')
    return np.where(known,'ship:'+values,'row:'+pd.Series(ids,index=values.index).astype(str)).astype(str)


def freeze():
    start=time.perf_counter()
    choice=c.read_json(OUT/'FROZEN_CHOICE.json')['taus']
    logs,gates=[],[]
    for ds,directory in folders():
        task=directory.name
        folder=c.TASKS/'self24ports'/task if ds=='self24ports' else c.TASKS/ds
        feat=c.Features(folder) if ds=='self24ports' else CachedStrongFeatures(ds)
        index=pd.Index(feat.target_rows)
        table=pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'],dtype=str,keep_default_na=False)
        units=[(draw,b,c.archive(directory/f'{draw}_{b}.npz')) for draw in range(5) for b in c.BUDGETS]
        union=np.unique(np.concatenate([z['support_ids'] for _,_,z in units]))
        local=index.get_indexer(union)
        assert (local>=0).all()
        sx,sp=feat.target(local)
        si=pd.Index(union)
        with np.load(folder/'target.npz') as data:
            np.testing.assert_array_equal(data['rows'],feat.target_rows)
            base_parts={k:data[k][local] for k in ('phi','mean','conditions')}
            base_prior=data['prior'][local]
        with np.load(folder/'selective_embedding.npz') as z:
            np.testing.assert_array_equal(z['rows'],feat.target_rows)
            emb=z['embedding'][local].astype(float)
        eta=c.read_json(folder/'PREPARED.json')['eta']
        for draw,budget,old in units:
            key=dict(dataset=ds,task=task,draw=draw,budget=budget)
            s=si.get_indexer(old['support_ids'])
            y=old['support_labels']
            np.testing.assert_array_equal(table.iloc[old['support_ids']].class_id.to_numpy(int),y)
            groups=identity(table,old['support_ids'])
            ke=sx[s]@sx[s].T
            kb=c.base_kernel({k:v[s] for k,v in base_parts.items()},emb[s],np.arange(budget),eta)
            oof,oy,folds,notes=support_oof(ke,sp[s],kb,base_prior[s],y,groups,c.coefficients)
            selected={}
            for mode,tau in choice.items():
                gate,optim=fit_gate(*oof,oy,tau,mode)
                selected[mode]=gate
                risk0=r.loss_grad(np.ones(8),*oof,oy,0.)[0]
                risk1=r.loss_grad(gate,*oof,oy,0.)[0]
                for cls in range(8):
                    gates.append(dict(**key,mode=mode,tau=tau,class_id=cls,support_n=int((y==cls).sum()),
                        gamma=gate[cls],oof_risk_gain=risk0-risk1))
            qlocal=index.get_indexer(old['query_ids'])
            assert (qlocal>=0).all() and not np.intersect1d(old['query_ids'],old['support_ids']).size
            # Validate the full-support solver, then reuse all existing query logits.
            probe=qlocal[:min(32,len(qlocal))]
            qx,qp=feat.target(probe)
            expected=qp+(qx@sx[s].T)@c.coefficients(ke,sp[s],y,budget/8.)
            np.testing.assert_allclose(expected,old['expert_raw'][:len(probe)],atol=1e-8,rtol=1e-8)
            if ds=='self24ports':
                prior=feat.target(qlocal)[1]
            else:
                prior=np.asarray(feat.prior[qlocal])
            residual=old['expert_raw']-prior
            scores={mode:prior+gate*residual for mode,gate in selected.items()}
            dest=CACHE/ds/task
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{draw}_{budget}.npz',**scores,prior=prior,
                query_ids=old['query_ids'],support_ids=old['support_ids'])
            small=OUT/'target_oof'/ds/task
            small.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(small/f'{draw}_{budget}.npz',prior=oof[0],residual=oof[1],
                base_probability=oof[2],labels=oy,**{'gate_'+m:g for m,g in selected.items()})
            c.write(small/f'{draw}_{budget}_folds.json',dict(parts=folds,warnings=notes,groups=groups.tolist()))
            logs.append(dict(**key,query_n=len(qlocal),group_n=len(np.unique(groups)),
                replay_max_error=float(abs(expected-old['expert_raw'][:len(probe)]).max())))
        print('TARGET FROZEN',ds,task,flush=True)
        del feat,units,sx,base_parts
        gc.collect()
    pd.DataFrame(logs).to_csv(OUT/'target_replays.csv',index=False)
    pd.DataFrame(gates).to_csv(OUT/'target_gates.csv',index=False)
    c.write(OUT/'TARGET_PREDICTIONS_FROZEN.json',dict(units=len(logs),tasks=len(folders()),budgets=c.BUDGETS,
        taus=choice,cache=str(CACHE),seconds=time.perf_counter()-start,
        query_labels_in_gate_fit=False,query_selection=False,new_neural_training=0))


def evaluate():
    assert c.read_json(OUT/'TARGET_PREDICTIONS_FROZEN.json')['units']==375
    records,classes=[] ,[]
    for ds,directory in folders():
        task=directory.name
        table=pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'],usecols=['class_id'])
        for draw in range(5):
            for budget in c.BUDGETS:
                old=c.archive(directory/f'{draw}_{budget}.npz')
                new=c.archive(CACHE/ds/task/f'{draw}_{budget}.npz')
                for k in ('query_ids','support_ids'):
                    np.testing.assert_array_equal(new[k],old[k])
                y=table.iloc[old['query_ids']].class_id.to_numpy(int)
                if ds=='self24ports':
                    count=c.archive(OUT.parent/'prior_protection_ports_20261012/predictions'/task/f'{draw}_{budget}.npz')['count']
                else:
                    count=c.archive(Path('C:/Users/25791/AppData/Local/Temp/SAR_prior_protection_20261012')/ds/f'{draw}_{budget}.npz')['count']
                scores={a:old['expert_'+a] for a in ('raw','simple','initial')}
                scores.update(count=count,**{a:new[a] for a in r.MODES})
                key=dict(dataset=ds,task=task,draw=draw,budget=budget)
                for layer in ('expert','fusion'):
                    baseline=r.softmax(scores['raw'],axis=1) if layer=='expert' else probabilities(old['base'],scores['raw'])
                    bp=baseline.argmax(1)
                    ref=metrics(y,baseline)
                    for arm,score in scores.items():
                        p=r.softmax(score,axis=1) if layer=='expert' else probabilities(old['base'],score)
                        mm=metrics(y,p)
                        records.append(dict(**key,layer=layer,arm=arm,**mm,**{'delta_'+k:mm[k]-ref[k] for k in mm}))
                        for cls in np.unique(y):
                            mask=y==cls
                            pred=p[mask].argmax(1)
                            resc=100*np.mean((bp[mask]!=cls)&(pred==cls))/len(np.unique(y))
                            harm=100*np.mean((bp[mask]==cls)&(pred!=cls))/len(np.unique(y))
                            classes.append(dict(**key,layer=layer,arm=arm,class_id=int(cls),n=int(mask.sum()),
                                support_n=int((old['support_labels']==cls).sum()),recall=100*np.mean(pred==cls),
                                rescue_ba=resc,harm_ba=harm,delta_ba=resc-harm))
        print('TARGET EVALUATED',ds,task,flush=True)
    f=pd.DataFrame(records)
    f.to_csv(OUT/'target_metrics.csv',index=False)
    cp=pd.DataFrame(classes)
    cp.to_csv(OUT/'target_classes.csv',index=False)
    keys=['dataset','task','draw','budget','layer','arm']
    np.testing.assert_allclose(cp.groupby(keys).delta_ba.sum().sort_index(),f.set_index(keys).delta_ba.sort_index(),atol=1e-10)
    values=['oa','ba','macro_brier','brier','delta_oa','delta_ba','delta_macro_brier','delta_brier']
    perport=f.groupby(['dataset','task','budget','layer','arm'])[values].mean().reset_index()
    perport.to_csv(OUT/'target_port_summary.csv',index=False)
    summary=perport.groupby(['dataset','budget','layer','arm'])[values].mean().reset_index()
    summary.to_csv(OUT/'target_summary.csv',index=False)
    print(summary[(summary.layer=='fusion')].to_string(index=False),flush=True)
    c.write(OUT/'TARGET_EVALUATION_COMPLETE.json',dict(units=375,tasks=25,main_model_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','evaluate','all'])
    args=parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('freeze','all'):
            freeze()
        if args.stage in ('evaluate','all'):
            evaluate()
