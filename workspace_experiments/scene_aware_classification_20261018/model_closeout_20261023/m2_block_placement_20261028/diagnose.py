"""Same-trained-model pathway removal; not a refit or a physical intervention."""
import gc
import numpy as np
import torch
from run import ROOT, RELEASE, PREVIOUS, ARMS, SEEDS, Cache, construct, setup, read_manifest, pair_mask, write


def metric(z,y):
    z=z.astype(np.float64)
    l=z-z.max(-1,keepdims=True)
    l-=np.log(np.exp(l).sum(-1,keepdims=True))
    p=np.exp(l); pred=p.argmax(-1)
    return dict(OA=float((pred==y).mean()*100),BA=float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)])*100),
                NLL=float(-l[np.arange(len(y)),y].mean()),Brier=float(((p-np.eye(z.shape[1])[y])**2).sum(-1).mean()))


def margin(z,y):
    return z[np.arange(len(y)),y]-np.where(np.eye(z.shape[-1])[y].astype(bool),-np.inf,z).max(-1)


@torch.no_grad()
def main():
    setup('cuda')
    data=np.load(ROOT/'predictions.npz'); y=data['labels'];scores=data['scores']
    records=read_manifest(RELEASE/'manifests/development.json')
    refs=read_manifest(RELEASE/'manifests/references.json')
    cache=Cache('dev');bank=Cache('refs').get(np.arange(len(refs)))
    state=torch.load(RELEASE/'weights/teacher.pt',weights_only=True,map_location='cpu')['state']
    output=[];case_records=[];saved={}
    for arm in ('pre3x3','post3x3'):
        ai=list(ARMS).index(arm)
        for si,seed in enumerate(SEEDS):
            m=construct(state,ARMS[arm],seed)
            delta=torch.load(ROOT/'checkpoints'/f'{arm}_{seed}.pt',weights_only=True,map_location='cpu')['state']
            m.load_state_dict(delta,strict=False)
            for route in ('backbone_only','readout_only'):
                chunks=[]
                for first in range(0,len(y),16):
                    ix=np.arange(first,min(first+16,len(y)))
                    chunks.append(m(cache.get(ix),bank,pair_mask(records[first:first+len(ix)],refs,'cuda'),route=route).cpu().numpy())
                z=np.concatenate(chunks);saved[f'{arm}_{seed}_{route}']=z
                output.append(dict(arm=arm,seed=seed,route=route,**metric(z,y)))
            output.append(dict(arm=arm,seed=seed,route='full',**metric(scores[ai,si],y)))
            output.append(dict(arm=arm,seed=seed,route='off',**metric(data['bypasses'][ai,si],y)))
            if seed==17:
                base=scores[0,0].argmax(-1);pred=scores[ai,0].argmax(-1)
                masks=dict(rescued=(base!=y)&(pred==y),harmed=(base==y)&(pred!=y),CC=(base==y)&(pred==y),WW=(base!=y)&(pred!=y))
                for kind,mask in masks.items():
                    candidates=np.flatnonzero(mask)
                    if not len(candidates):continue
                    differences=margin(scores[ai,0],y)-margin(scores[0,0],y)
                    i=int(candidates[np.argmax(abs(differences[candidates]))])
                    q=cache.get([i]);legal=pair_mask([records[i]],refs,'cuda')
                    z,d=m(q,bank,legal,details=True)
                    error=float(abs(z.cpu().numpy()[0]-scores[ai,0,i]).max())
                    assert error<1e-4,(arm,i,error)
                    rival=int(np.where(np.arange(8)==y[i],-np.inf,scores[0,0,i]).argmax())
                    def entry(value):
                        v=value.cpu().numpy()[0]
                        return dict(pred=int(v.argmax()),top1_margin=float(margin(v[None],y[i:i+1])[0]),fixed_margin=float(v[y[i]]-v[rival]))
                    paths={'full':entry(z)}
                    for p,pol in enumerate(('VV','VH')):
                        m.early[f'2_{p}'].enabled=False
                        paths[f'off_{pol}']=entry(m(q,bank,legal))
                        m.early[f'2_{p}'].enabled=True
                    for r,region in enumerate(('body','end','side')):
                        hooks=[mod.transforms[r].register_forward_hook(lambda module,args,value: torch.zeros_like(value)) for mod in m.early.values()]
                        try:paths[f'off_{region}']=entry(m(q,bank,legal))
                        finally:
                            for hook in hooks:hook.remove()
                    for mod in m.early.values():mod.enabled=False
                    paths['off_all']=entry(m(q,bank,legal))
                    for mod in m.early.values():mod.enabled=True
                    updates=[]
                    for p,pol in enumerate(('VV','VH')):
                        detail=d['trace']['early_2'][p]
                        updates.append(dict(pol=pol,available=bool(q['available'][0,p]),
                            l2=float(detail['update'][0].norm()),mean_region_mass=detail['mass'][0].cpu().tolist(),
                            band_gain=detail['gain'].cpu().tolist()))
                    case_records.append(dict(arm=arm,seed=seed,kind=kind,id=records[i]['id'],label=int(y[i]),
                        rival=rival,baseline_pred=int(base[i]),baseline_top1_margin=float(margin(scores[0,0,i:i+1],y[i:i+1])[0]),
                        paths=paths,updates=updates,replay_error=error))
            del m
            gc.collect();torch.cuda.empty_cache()
    np.savez(ROOT/'path_predictions.npz',**saved)
    write('path_diagnosis.json',dict(metrics=output,cases=case_records,
          scope='Fixed trained weights; feedback removed for BOTH query and references. No refit, no pixel intervention, no physical causal attribution. Outcome-selected examples.'))
    print('PATH DIAGNOSIS COMPLETE',flush=True)


if __name__=='__main__':main()
