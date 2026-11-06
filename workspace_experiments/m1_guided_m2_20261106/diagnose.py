"""Outcome provenance and one live-forward equivalence check; no fitting."""
import study as s
from study import ROOT,prev,task,torch,np,pd


def main():
    rows,groups=prev.records();ids=groups['evaluation'];y=np.array([rows[i]['label'] for i in ids])
    cache={p.stem:np.load(p,mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')}
    frozen=cache['score'][ids];original=frozen.argmax(1)==y
    events=[];totals=[]
    for seed in (17,29,43):
        z={arm:np.load(ROOT/f'{arm}_s{seed}/evaluation.npz')['score'] for arm in ('mean','content','class')}
        good=z['class'].argmax(1)==y
        off=np.load(ROOT/f'class_s{seed}/evaluation_off.npz')['score'].argmax(1)==y
        for control in ('mean','content'):
            old=z[control].argmax(1)==y
            for pos in np.flatnonzero(good!=old):
                events.append(dict(seed=seed,control=control,id=rows[ids[pos]]['id'],mmsi=rows[ids[pos]]['mmsi'],
                    orbit=task.orbit(rows[ids[pos]]),label=int(y[pos]),rescue=bool(good[pos]),
                    frozen_correct=bool(original[pos]),own_off_correct=bool(off[pos])))
            rescued=good&~old
            totals.append(dict(seed=seed,control=control,rescues=int(rescued.sum()),
                already_correct_frozen=int((rescued&original).sum()),new_vs_frozen=int((rescued&~original).sum()),
                requires_own_added_path=int((rescued&~off).sum())))
    pd.DataFrame(events).to_csv(ROOT/'event_provenance.csv',index=False)
    unique=[]
    for control in ('mean','content'):
        part=[r for r in events if r['control']==control and r['rescue']]
        unique.append(dict(control=control,events=len(part),observations=len({r['id'] for r in part}),
            ships=len({r['mmsi'] for r in part}),orbits=len({r['orbit'] for r in part})))
    s.write('outcome_provenance.json',dict(frozen_metrics=task.metrics(y,frozen),totals=totals,unique_rescues=unique))
    # Compare the cached readout implementation with adding its residual at live M2 outputs.
    probe=np.array([0,719,1142,1241,1279])
    new=s.RegionalGuidance('class').to(task.DEVICE)
    new.load_state_dict(torch.load(ROOT/'class_s17/final.pt',map_location=task.DEVICE,weights_only=True));new.eval()
    b={k:torch.as_tensor(np.array(v[probe],copy=True),device=task.DEVICE) for k,v in cache.items() if k!='score'}
    with torch.no_grad():expected,detail=new(b,details=True)
    old=prev.Network('class');old.restore(torch.load(s.CHECKPOINT,map_location='cpu',weights_only=True));old.eval()
    old.base.head.load_state_dict(new.head.state_dict());handles=[]
    def hook(p):
        def apply(module,args,out):
            out=dict(out);out['pooled']=out['pooled']+detail[p]['delta'];return out
        return apply
    for p in range(2):handles.append(old.base.m2[p].register_forward_hook(hook(p)))
    a,c=prev.arrays()
    with torch.no_grad():actual=old(a,c,probe)[0]
    error=float((actual-expected).abs().max());assert error<1e-4,error
    for h in handles:h.remove()
    mask_range=[float(cache['mask'].min()),float(cache['mask'].max())]
    assert 0<=mask_range[0]<=mask_range[1]<=1
    s.write('live_integration.json',dict(ids=probe.tolist(),max_score_error=error,mask_range=mask_range,
        explanation='Shallow live M2 pooled outputs augmented before original deep summary addition/head. Upstream prefix is frozen and cached as in training.'))
    print(dict(error=error,unique=unique,totals=totals),flush=True)


if __name__=='__main__':
    task.current.setup(task.DEVICE)
    main()
