"""One bounded four-arm source-hardness experiment; no test-time label routes."""
import argparse
import time
import pandas as pd
from torch.nn import functional as F
from model_task import *


def episode_refs(rows,train,rng):
    selected=[]
    for cls in range(4):
        ships={}
        for i in train:
            if rows[i]['label']==cls:ships.setdefault(rows[i]['mmsi'],[]).append(int(i))
        for ship in rng.choice(list(ships),8,replace=False):selected.append(int(rng.choice(ships[ship])))
    return np.array(selected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('none','original','aggregate','competitive'),required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();arm=args.arm;seed=args.seed;current.setup(DEVICE)
    rows,train,hold,rid=records();a=arrays();difficulty=json.loads((ROOT/'source_difficulty.json').read_text(encoding='utf-8'))
    y=np.array([r['label'] for r in rows]);hard=set(difficulty['hard'])
    dest=(args.output_root or ROOT)/(arm+'_'+str(seed));dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').exists():print('ALREADY_DONE',arm,flush=True);return
    model=Model(arm,seed=seed)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith('base.backbone.') for n in params)
    assert all(n.startswith(('base.head.','matcher.')) for n in params)
    rates={}
    for name,p in params.items():
        lr=1e-5 if name.startswith('base.head.') else (1e-4 if '.output.' in name else 1e-3)
        rates.setdefault(lr,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=v,lr=k) for k,v in rates.items()],weight_decay=.01)
    frame=pd.DataFrame([rows[i] for i in train]);weight=np.zeros(len(rows),np.float32)
    weight[train]=1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    weight[train]*=np.array([2. if i in hard else 1. for i in train]);weight[train]/=weight[train].mean()
    weights=torch.tensor(weight,device=DEVICE)
    initial=predict(model,a,rows,hold,rid);np.savez_compressed(dest/'initial_validation.npz',**initial)
    metric=metrics(y[hold],initial['score']);best_key=(metric['ba'],metric['oa'])
    best_epoch=0;best=model.live_state();curve=[dict(epoch=0,**metric)];episodes=[];grads=[];rng=np.random.default_rng(seed)
    tick=time.perf_counter()
    for epoch in range(1,5):
        order=rng.permutation(train);losses=[]
        for start in range(0,len(order),24):
            ids=order[start:start+24];erid=episode_refs(rows,train,rng)
            optimizer.zero_grad(set_to_none=True)
            score,ratio=model(a,rows,ids,erid)
            loss=(F.cross_entropy(score,torch.tensor(y[ids],device=DEVICE),reduction='none')*weights[ids]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            if epoch==1 and start in (0,24):
                g={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
                assert g['base.head.weight']>0 and all(np.isfinite(list(g.values())))
                if arm!='none':assert any(v>0 for n,v in g.items() if n.startswith('matcher.'))
                grads.append(dict(step=start//24,loss=float(loss.detach()),gradients=g))
            torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
            optimizer.step();losses.append(float(loss.detach()))
            episodes.append(dict(epoch=epoch,ids=ids.tolist(),references=erid.tolist()))
        p=predict(model,a,rows,hold,rid);metric=metrics(y[hold],p['score'])
        curve.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),**metric))
        key=(metric['ba'],metric['oa'])
        if key>best_key:best_key=key;best_epoch=epoch;best=model.live_state()
        pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False)
        print('EPOCH',arm,epoch,metric,round(time.perf_counter()-tick,1),flush=True)
    torch.save(dict(arm=arm,trainable_state=model.live_state()),dest/'last.pt')
    modes=[('last_validation',None)]
    if arm!='none':modes.append(('last_off','off'))
    if arm=='competitive':modes.append(('last_aggregate','aggregate'))
    for name,mode in modes:
        p=predict(model,a,rows,hold,rid,override=mode);np.savez_compressed(dest/(name+'.npz'),**p)
    model.restore(best);torch.save(dict(arm=arm,trainable_state=best),dest/'selected.pt')
    results={}
    for cohort,ids in [('train',train),('validation',hold)]:
        p=predict(model,a,rows,ids,rid);np.savez_compressed(dest/f'{cohort}.npz',**p)
        results[cohort]=metrics(y[ids],p['score'])
    if arm!='none':
        p=predict(model,a,rows,hold,rid,override='off');np.savez_compressed(dest/'same_weights_off.npz',**p)
    if arm=='competitive':
        p=predict(model,a,rows,hold,rid,override='aggregate');np.savez_compressed(dest/'same_weights_aggregate.npz',**p)
    score=predict(model,a,rows,hold[:3],rid)['score'];del model
    model=Model(arm,seed=seed);model.restore(torch.load(dest/'selected.pt',map_location='cpu',weights_only=True)['trainable_state'])
    error=float(np.abs(score-predict(model,a,rows,hold[:3],rid)['score']).max());assert error==0
    write(dest/'episodes.json',episodes)
    write(dest/'receipt.json',dict(arm=arm,seed=seed,source_train=len(train),hard=len(hard),validation=len(hold),
        classes=4,frequency=False,backbone_frozen=True,selected_epoch=best_epoch,selection='BA then OA; earliest tied epoch; zero eligible',
        initial=metrics(y[hold],initial['score']),results=results,steps=len(episodes),
        trainable={n:p.numel() for n,p in params.items()},first_gradients=grads,reload_error=error,seconds=time.perf_counter()-tick))
    print('DONE',arm,results,'epoch',best_epoch,flush=True)


if __name__=='__main__':main()
