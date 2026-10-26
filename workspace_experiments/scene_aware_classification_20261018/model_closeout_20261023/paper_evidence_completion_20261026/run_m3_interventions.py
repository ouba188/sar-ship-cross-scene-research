"""Final-model evidence interventions, not pixel interventions or physical causality."""
import argparse
import numpy as np
import pandas as pd
import torch
from run_classical import HERE,ROOT,TRANSFER,SEEDS,tr,write_json


def scores_from_local(local, base, seedroot, original_gate, available, neutral_frequency=False):
    scalers=tr.load_checkpoint(seedroot/'scalers.pt')
    raw=tr.m3.local_raw(local,base)
    x=torch.tensor(tr.m3.transform(raw,scalers['polar_scaler']))
    delta=[]
    for pol,name in enumerate(('vv','vh_centered')):
        ck=tr.load_checkpoint(seedroot/f'{name}.pt')
        model=tr.frozen_model(tr.m3.PolarHead(pol,centered=bool(pol),null=ck['state'].get('null')),ck)
        with torch.inference_mode():
            delta.append(model(x).numpy())
    dv,dh=delta
    z=base-base.mean(1,keepdims=True)
    raw_gate=original_gate.copy()
    cols=[z,dv,dh]
    for score in (base,base+dv,base+dh):
        other=np.stack([np.delete(score,c,1).max(1) for c in range(4)],1)
        cols.append(score-other)
    cols.extend(tr.m3.softmax(score,axis=1) for score in (base,base+dv,base+dh))
    cols.extend(np.broadcast_to(available[:,p,None],base.shape) for p in range(2))
    raw_gate[...,:11]=np.stack(cols,-1)
    ck=tr.load_checkpoint(seedroot/'frequency.pt')
    xx=tr.m3.transform(raw_gate,ck['scaler'])
    if neutral_frequency:
        xx[...,41:]=0
    gate=tr.frozen_model(tr.m3.Gate(185,ck['alpha']),ck)
    with torch.inference_mode():
        weight=gate(torch.tensor(xx),torch.tensor(available)).numpy()
    return base+weight[:,:1]*dv+weight[:,1:]*dh


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=('oss1','oss2'),required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    name=args.source
    direction=f'{name}-to-'+('oss2' if name=='oss1' else 'oss1')
    root=ROOT/f'{name}_m123_replication_20261026'
    mroot=root/('m3_results_grid01' if name=='oss1' else 'm3_results')
    repair=TRANSFER/'normalization_repair_20261026'
    table=pd.read_csv(root/'stack_table.csv',dtype={'mmsi':str})
    target=tr.read_table(TRANSFER/direction/'target_table.csv')
    test=np.flatnonzero(table.role.eq('evaluation'))
    out=HERE/'m3_interventions'/name
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',dict(frozen_weights=True,target_selection=False,
       interventions=['remove VV/VH matching evidence', 'remove body/side/end matching evidence',
                      'neutralize extra gate frequency inputs at source training mean'],
       scope='model evidence dependence, not physical SAR factors; no new learned models',
       recompute='polar heads and score-dependent gate inputs; geometry/content descriptors remain original',
       frequency_control='same trained gate, standardized dimensions41:185=0; not matched retraining'))
    checks=[]
    for seed in SEEDS:
        src=tr.load_npz(repair/f'source_{name}/seed{seed}/predictions.npz')
        dst=tr.load_npz(repair/direction/f'seed{seed}/predictions.npz')
        local=np.r_[src['local'][test],dst['local']]
        base=np.r_[src['s12'][test],dst['s12']].astype(np.float32)
        gr=np.r_[src['gate_raw'][test],dst['gate_raw']]
        av=np.r_[src['available'][test],dst['available']]
        actual=np.r_[src['frequency'][test],dst['frequency']]
        replay=scores_from_local(local,base,mroot/f'seed{seed}',gr,av)
        error=float(np.max(np.abs(replay-actual)))
        np.testing.assert_allclose(replay,actual,atol=2e-5,rtol=0)
        np.testing.assert_array_equal(replay.argmax(1),actual.argmax(1))
        checks.append(dict(seed=seed,replay_max_error=error))
        results={'original':actual,'gate_frequency_neutral':scores_from_local(local,base,mroot/f'seed{seed}',gr,av,True)}
        for arm,axis,index in [('no_vv',2,0),('no_vh',2,1),('no_body',3,0),('no_side',3,1),('no_end',3,2)]:
            changed=local.copy()
            sl=[slice(None)]*changed.ndim
            sl[axis]=index
            changed[tuple(sl)]=0
            results[arm]=scores_from_local(changed,base,mroot/f'seed{seed}',gr,av)
        np.savez_compressed(out/f'seed{seed}.npz',**results,within_obs_id=table.iloc[test].obs_id.to_numpy(),cross_obs_id=target.obs_id.to_numpy())
    target=tr.read_table(TRANSFER/direction/'target_table.csv',labels=True)
    rows,groups=[],[]
    for seed in SEEDS:
        scores=tr.load_npz(out/f'seed{seed}.npz')
        for cohort,sl,tab in [('within',slice(0,len(test)),table.iloc[test].reset_index(drop=True)),
                             ('cross',slice(len(test),None),target)]:
            y=tab.class_id.to_numpy(int)
            old=scores['original'][sl]
            for arm,ss in scores.items():
                if arm.endswith('obs_id'):continue
                s=ss[sl]
                a,b=s.argmax(1)==y,old.argmax(1)==y
                d=a.astype(float)-b
                rows.append(dict(source=name,cohort=cohort,seed=seed,arm=arm,
                    rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),changed=int((s.argmax(1)!=old.argmax(1)).sum()),
                    max_score_change=float(np.max(abs(s-old))),OA_delta_pp=float(d.mean()*100),
                    BA_delta_pp=float(100*np.mean([d[y==c].mean() for c in range(4)])),**tr.m3.metrics(y,s)))
                for c in range(4):
                    groups.append(dict(source=name,cohort=cohort,seed=seed,arm=arm,class_id=c,n=int((y==c).sum()),
                                       net=int(d[y==c].sum())))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'class_changes.csv',index=False)
    write_json(out/'replay.json',checks)
    write_json(out/'complete.json',dict(seeds=SEEDS,interventions=6,trained=False))
    print(pd.DataFrame(rows).groupby(['cohort','arm'])[['OA_delta_pp','BA_delta_pp']].mean().to_string(),flush=True)


if __name__=='__main__':main()
