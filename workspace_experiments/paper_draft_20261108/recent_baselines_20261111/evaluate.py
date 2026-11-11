"""Post-selection evaluation and paired statistics; never called by training."""
import csv
import json
import time
import numpy as np
import torch

from train import ROOT, Inputs, build, metrics, predict, write


def csv_save(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as stream:
        out=csv.DictWriter(stream,fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)


def main():
    selection=json.loads((ROOT/'selected.json').read_text(encoding='utf-8'))
    lr=selection['lr']
    inputs=Inputs();ids=inputs.groups['evaluation'];y=inputs.y[ids]
    # Test inputs are loaded only after all three training receipts exist.
    for seed in (17,29,43):
        receipt=json.loads((ROOT/f'cpinet_s{seed}_lr{lr:g}/complete.json').read_text(encoding='utf-8'))
        assert receipt['epochs']==300
    x=torch.tensor(np.array(inputs.pixels[ids]),device='cuda',dtype=torch.float32)/255
    mask=torch.tensor(np.array(inputs.footprint[ids]),device='cuda',dtype=torch.float32)
    count=mask.sum((-2,-1),keepdim=True).clamp_min(1)
    mean=(x*mask).sum((-2,-1),keepdim=True)/count
    std=(((x-mean).square()*mask).sum((-2,-1),keepdim=True)/count).sqrt().clamp_min(1e-5)
    offset=len(inputs.x)
    inputs.x=torch.cat((inputs.x,(x-mean)/std*mask))
    inputs.av=torch.cat((inputs.av,torch.tensor(inputs.available[ids],device='cuda')))
    inputs.lookup.update({int(i):offset+j for j,i in enumerate(ids)})
    del x,mask
    scores={};rows=[];classes=[];paired=[];scenes=[];cost=[];replays=[]
    available=inputs.available[ids]
    subsets={'all':np.ones(len(ids),bool),'both':available.sum(1)==2,'single':available.sum(1)==1}
    oldroot=ROOT.parent/'staged_ablation_20261109'
    oldselect=json.loads((oldroot/'selected.json').read_text(encoding='utf-8'))['scales']
    for seed in (17,29,43):
        folder=ROOT/f'cpinet_s{seed}_lr{lr:g}'
        net=build(seed)
        net.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True))
        validation=predict(net,inputs,inputs.groups['validation'])
        saved_validation=np.load(folder/'validation.npz')
        error=float(abs(validation-saved_validation['score']).max())
        changed=int((validation.argmax(1)!=saved_validation['score'].argmax(1)).sum())
        # Independent validation-only diagnosis found sub-1e-3 numerical differences,
        # but identical predictions, for both TF32 and full-float32 replay.
        assert error<1e-3 and changed==0,(seed,error,changed)
        replays.append(dict(seed=seed,validation_replay_max_error=error,changed_predictions=changed))
        score=predict(net,inputs,ids)
        np.savez_compressed(folder/'evaluation.npz',ids=ids,score=score,y=y)
        scores['CPINet',seed]=score
        for arm in ('full','plain','se'):
            data=np.load(oldroot/f'{arm}_s{seed}_lr{oldselect[arm]:g}'/'evaluation.npz')
            assert np.array_equal(data['ids'],ids)
            scores[arm,seed]=data['score']
        # GPU forward only, includes both complete backbones and all four heads.
        net.eval();xx,av,_=inputs.batch(ids[:1])
        with torch.no_grad():
            for _ in range(15):net(xx,av)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(50):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();net(xx,av);end.record();end.synchronize();times.append(start.elapsed_time(end))
        cost.append(dict(seed=seed,batch=1,median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),
                         p90_ms=float(np.quantile(times,.9)),parameters=sum(p.numel() for p in net.parameters())))
        del net
    for (arm,seed),score in scores.items():
        for subset,mask in subsets.items():
            m=metrics(y[mask],score[mask])
            rows.append(dict(arm=arm,seed=seed,subset=subset,n=int(mask.sum()),**{k:v for k,v in m.items() if k!='confusion'}))
            cm=np.array(m['confusion'])
            for c in range(4):
                classes.append(dict(arm=arm,seed=seed,subset=subset,label=c,n=int(cm[c].sum()),
                    recall=float(100*cm[c,c]/max(cm[c].sum(),1)),predicted=int(cm[:,c].sum())))
    scene_ids=np.array([inputs.rows[i]['pass_group'] for i in ids])
    for seed in (17,29,43):
        before=scores['CPINet',seed].argmax(1)==y;after=scores['full',seed].argmax(1)==y
        difference=after.astype(float)-before
        for subset,mask in subsets.items():
            paired.append(dict(seed=seed,subset=subset,n=int(mask.sum()),oa_delta=100*difference[mask].mean(),
                ba_delta=100*np.mean([difference[mask&(y==c)].mean() for c in range(4)]),
                rescued=int((after&~before&mask).sum()),harmed=int((~after&before&mask).sum())))
        for scene in np.unique(scene_ids):
            mask=scene_ids==scene
            scenes.append(dict(seed=seed,scene=scene,n=int(mask.sum()),
                cpinet_oa=100*before[mask].mean(),full_oa=100*after[mask].mean(),oa_delta=100*difference[mask].mean()))
    csv_save('metrics.csv',rows);csv_save('per_class.csv',classes);csv_save('paired_by_seed.csv',paired)
    csv_save('per_scene.csv',scenes);csv_save('cost.csv',cost)
    write(ROOT/'checkpoint_replays.json',replays)
    summary=[]
    for arm in ('CPINet','full','plain','se'):
        for subset in subsets:
            group=[r for r in rows if r['arm']==arm and r['subset']==subset]
            summary.append(dict(arm=arm,subset=subset,n=group[0]['n'],
                **{f'{metric}_{stat}':float(getattr(np,stat)([r[metric] for r in group],**({'ddof':1} if stat=='std' else {})))
                   for metric in ('oa','ba','macro_f1','nll','brier') for stat in ('mean','std')}))
    csv_save('summary.csv',summary)
    # Resample ships, retaining all observations and the paired three-seed differences.
    ship=np.array([inputs.rows[i]['mmsi'] for i in ids]);unique=np.unique(ship)
    ship_ids=[np.flatnonzero(ship==m) for m in unique]
    diff=np.stack([(scores['full',s].argmax(1)==y).astype(float)-(scores['CPINet',s].argmax(1)==y)
                   for s in (17,29,43)]).mean(0)
    rng=np.random.default_rng(20261111);sample=[]
    for _ in range(2000):
        ix=np.concatenate([ship_ids[i] for i in rng.integers(len(unique),size=len(unique))])
        sample.append([100*diff[ix].mean(),100*np.mean([diff[ix][y[ix]==c].mean() for c in range(4)])])
    write(ROOT/'paired_intervals.json',dict(unit='MMSI cluster, averaged paired seed effect',replicates=2000,
        oa=np.quantile(np.array(sample)[:,0],[.025,.975]).tolist(),
        ba=np.quantile(np.array(sample)[:,1],[.025,.975]).tolist()))
    write(ROOT/'status.json',dict(status='evaluated',selected_lr=lr,seeds=[17,29,43],test_read=True))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
