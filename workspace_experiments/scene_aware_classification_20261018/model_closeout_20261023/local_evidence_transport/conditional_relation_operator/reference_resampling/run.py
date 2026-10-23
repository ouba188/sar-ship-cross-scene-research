"""Frozen operator reference intervention; no new fitting or evaluation selection."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_relation_reader',HERE.parent/'learned/run.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
ARMS = ('independent','transfer','excess')
BANK_SEEDS = (101,307,509,711,907)


def banks_from_source(table,mass,legal,old):
    fit = np.load(exp.edge.support.UP/'indices.npz')['fit_train']
    eligible = fit[mass[fit,:,:8].sum((1,2))>0]
    pool = table.iloc[eligible].drop_duplicates('mmsi').index.to_numpy()
    allqueries = table.role.isin(['utility_train','selection','evaluation'])
    for key in ('mmsi','pass_group'):
        assert not table.iloc[pool][key].isin(table.loc[allqueries,key]).any(),key
    patterns = legal.sum(-1)>=2
    old_mmsi = set(table.iloc[old.ravel()].mmsi)
    banks = {'original':old.copy()}
    rows = []
    for seed in BANK_SEEDS:
        rng = np.random.default_rng(seed)
        bank = np.empty_like(old)
        for c in range(8):
            used = set()
            for signature in np.unique(patterns[old[c]],axis=0):
                slots = np.flatnonzero((patterns[old[c]]==signature).all(-1))
                options = pool[(table.iloc[pool].class_id.to_numpy()==c)&(patterns[pool]==signature).all(-1)]
                fresh = [int(i) for i in options if table.iloc[i].mmsi not in old_mmsi]
                repeat = [int(i) for i in options if table.iloc[i].mmsi in old_mmsi]
                choice = list(rng.permutation(fresh)) + list(rng.permutation(repeat))
                choice = [i for i in choice if table.iloc[i].mmsi not in used][:len(slots)]
                assert len(choice)==len(slots),(c,signature,len(choice),len(slots))
                bank[c,slots] = choice
                used.update(table.iloc[choice].mmsi)
        assert table.iloc[bank.ravel()].mmsi.nunique()==64
        np.testing.assert_array_equal(patterns[bank],patterns[old])
        banks[str(seed)] = bank
    for name,bank in banks.items():
        for c in range(8):
            for j,i in enumerate(bank[c]):
                rows.append(dict(bank=name,class_id=c,slot=j,index=int(i),obs_id=int(table.iloc[i].obs_id),
                    mmsi=table.iloc[i].mmsi,pass_group=table.iloc[i].pass_group,
                    from_original=table.iloc[i].mmsi in old_mmsi,
                    VV_side=bool(patterns[i,0]),VV_end=bool(patterns[i,1]),
                    VH_side=bool(patterns[i,2]),VH_end=bool(patterns[i,3])))
    return banks,pd.DataFrame(rows)


def relation(reader,ids,mb,mn):
    z,raw = [],[]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            a,b,_ = reader.scores(ids[start:start+128],mb,mn)
            z.append(a.cpu().numpy()); raw.append(b.cpu().numpy())
    return np.concatenate(z),np.concatenate(raw)


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    table,x,mass,old,idx,tr,va,te = exp.source_data()
    y = table.class_id.to_numpy(int)
    _,_,legal,_ = exp.edge.edges(x,mass)
    banks,inventory = banks_from_source(table,mass,legal,old)
    inventory.to_csv(HERE/'banks.csv',index=False)
    scales = pd.read_csv(exp.HERE/'scales.csv')
    weights = np.load(exp.HERE/'weights.npz')
    receipts = pd.read_csv(exp.HERE/'training_receipts.csv')
    full = {s:np.load(exp.HERE/f'predictions_{s}.npz')['full'] for s in exp.SEEDS}
    metrics,paired,classes,changes,gaps = [],[],[],[],[]
    saved = {'idx':idx,'y':y[idx]}
    replay = 0.
    reference_delta = {}
    for bankname,bank in banks.items():
        for arm in ARMS:
            reader = exp.Reader(x,mass,bank,arm)
            divisor = scales[scales.arm.eq(arm)].sort_values('feature').divisor.to_numpy()
            reader.divisor = torch.tensor(divisor,device='cuda',dtype=torch.float32)
            I = torch.eye(32,device='cuda')
            z0,raw0 = relation(reader,idx,I,I)
            saved[f'{bankname}_{arm}_identity'] = z0
            states = [(s,'selected') for s in exp.SEEDS]+[(43,'terminal200')]
            for seed,state in states:
                take = np.arange(len(idx)) if state=='selected' else np.r_[tr,va]
                if state=='selected':
                    mb,mn,a = [weights[f'{arm}_{seed}_{k}'] for k in ('Mb','Mn','a')]
                    step = int(receipts[(receipts.seed==seed)&(receipts.arm==arm)].selected_step.iloc[0])
                else:
                    cp = np.load(exp.HERE/f'checkpoints_{arm}_{seed}.npz')
                    mb,mn,a = [cp[f'{k}_200'] for k in ('Mb','Mn','a')]
                    step = 200
                z,raw = relation(reader,idx[take],torch.tensor(mb,device='cuda'),torch.tensor(mn,device='cuda'))
                dz = z-z0[take]
                delta = (dz*a).sum(-1)
                delta -= delta.mean(-1,keepdims=True)
                pred = full[seed][idx[take]]+delta
                inverse = np.full(len(idx),-1,dtype=int)
                inverse[take] = np.arange(len(take))
                if state=='selected':
                    saved[f'{bankname}_{arm}_{seed}'] = pred
                    saved[f'{bankname}_{arm}_{seed}_parts'] = dz*a
                key = (arm,seed,state)
                if bankname=='original':
                    reference_delta[key] = delta.copy()
                    if state=='selected':
                        expected = np.load(exp.HERE/f'predictions_{seed}.npz')[arm][idx]
                        replay = max(replay,float(np.max(abs(pred-expected))))
                        assert replay<3e-5,replay
                for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
                    if state!='selected' and role=='evaluation':
                        continue
                    pos = inverse[loc]
                    assert np.array_equal(take[pos],loc)
                    labels = y[idx[loc]]
                    before = full[seed][idx[loc]].argmax(-1)==labels
                    after = pred[pos].argmax(-1)==labels
                    meta = dict(bank=bankname,arm=arm,seed=seed,state=state,step=step,role=role)
                    metrics.append(dict(**meta,**exp.dep.metrics(labels,pred[pos])))
                    paired.append(dict(**meta,rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
                    for c in range(8):
                        sel = labels==c
                        classes.append(dict(**meta,class_id=c,n=int(sel.sum()),correct=int(after[sel].sum()),full_correct=int(before[sel].sum())))
                    d,prev = delta[pos],reference_delta[key][pos]
                    corr = float(np.corrcoef(d.ravel(),prev.ravel())[0,1]) if d.std()>1e-10 and prev.std()>1e-10 else np.nan
                    shifts = np.sqrt(np.mean((d-prev)**2))
                    original_pred = full[seed][idx[loc]]+prev
                    changes.append(dict(**meta,delta_correlation=corr,delta_rms=float(np.sqrt(np.mean(d*d))),change_rms=float(shifts),
                        changed_prediction=int((pred[pos].argmax(-1)!=original_pred.argmax(-1)).sum())))
                    if seed==43 and state=='selected':
                        for g,group in enumerate(exp.edge.GROUPS):
                            for name,zz in [('identity',z0[loc,:,g]),('learned',z[pos,:,g])]:
                                values = zz[np.arange(len(loc)),labels]-(zz.sum(-1)-zz[np.arange(len(loc)),labels])/7
                                gap = np.mean([values[labels==c].mean() for c in np.unique(labels)])
                                gaps.append(dict(**meta,group=group,representation=name,macro_gap=float(gap)))
            print(bankname,arm,'done',round(time.perf_counter()-tick,1),flush=True)
    for seed in exp.SEEDS:
        saved[f'full_{seed}'] = full[seed][idx]
        for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
            metrics.append(dict(bank='none',arm='full',seed=seed,state='selected',step=0,role=role,**exp.dep.metrics(y[idx[loc]],full[seed][idx[loc]])))
    for name,rows in [('metrics',metrics),('paired',paired),('classes',classes),('changes',changes),('gaps',gaps)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    np.savez_compressed(HERE/'predictions.npz',**saved)
    df = pd.DataFrame(metrics)
    summary = df[df.state.eq('selected')].groupby(['bank','arm','role'])[['OA','BA','J','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt = dict(no_training=True,original_replay_max_error=replay,bank_seeds=list(BANK_SEEDS),
        roles={'utility':644,'selection':612,'evaluation':485},reference_per_class=8,reference_source='fit_train1205 only',
        matching='exact 4-group availability pattern, maximal original-MMSI exclusion',
        fixed_original_scales=True,terminal_evaluation=False,seconds=time.perf_counter()-tick)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
