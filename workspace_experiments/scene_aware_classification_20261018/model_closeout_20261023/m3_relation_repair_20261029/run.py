"""M3-only paired repair; fixed endpoint, source fit, unchanged rest of model."""
from probe import HERE, ROOT, refit
import argparse
import json
import time
import numpy as np
import torch
from sar_m123.current.model import setup, forward_batch
from sar_m123.current.data import prepare_record, batch, pair_mask
from sar_m123.current.training import fitting_loss, class_weights
from model import fresh, save, load, configure

ARMS = ('control','fast_compare','metric')
SEEDS = (17,29,43)
FULL = ROOT/'spatial_m12_full_20261029'


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def metrics(z,y):
    pred=z.argmax(1)
    t=torch.from_numpy(z).double()
    p=t.softmax(1).numpy()
    return dict(oa=100*float(np.mean(pred==y)),
        ba=100*float(np.mean([np.mean(pred[y==c]==c) for c in np.unique(y)])),
        nll=float(torch.nn.functional.cross_entropy(t,torch.tensor(y))),
        brier=float(np.square(p-np.eye(z.shape[1])[y]).sum(1).mean()))


@torch.no_grad()
def evaluate(net, arrays, ids, rows, refs, bank, name):
    z=refit.predict(net,arrays,ids,rows,refs,bank)
    folder=HERE/'predictions'
    folder.mkdir(exist_ok=True)
    y=np.array([r['label'] for r in rows])
    np.savez(folder/f'{name}.npz',score=z,y=y,ids=np.array([str(r['id']) for r in rows]))
    return z,metrics(z,y)


def run(seeds):
    setup()
    rows,arrays,indices,bank=refit.load()
    weights=class_weights(rows['fit'],'fit')
    rl=torch.tensor([r['label'] for r in rows['references']],device='cuda')
    for seed in seeds:
        checkpoint=FULL/'training'/f'spatial_s{seed}'/'model.pt'
        for arm in ARMS:
            folder=HERE/'training'/f'{arm}_s{seed}'
            if (folder/'training.json').exists():
                continue
            net=fresh(checkpoint,arm)
            # Same original endpoint for each arm; identity metric must preserve it.
            before,before_metrics=evaluate(net,arrays,indices['development'],rows['development'],
                                          rows['references'],bank,f'before_{arm}_s{seed}')
            with np.load(FULL/'predictions'/f'development_after_spatial_s{seed}.npz') as old:
                assert old['ids'].tolist()==[str(r['id']) for r in rows['development']]
                initial_error=float(abs(before-old['score']).max())
                assert initial_error<1e-4,initial_error
            groups=configure(net,arm)
            fixed={n:v.detach().cpu().clone() for n,v in net.state_dict().items() if not n.startswith('m3.')}
            params={n:p for n,p in net.named_parameters() if p.requires_grad}
            initial={n:p.detach().cpu().clone() for n,p in params.items()}
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
            rng=np.random.default_rng(seed)
            curve=[]
            tick=time.perf_counter()
            for epoch in range(1,4):
                terms_all=[]
                for ix in np.array_split(rng.permutation(len(rows['fit'])),range(24,len(rows['fit']),24)):
                    q=[rows['fit'][i] for i in ix]
                    optimizer.zero_grad(set_to_none=True)
                    z,d=refit.score(net,arrays,indices['fit'][ix],q,rows['references'],bank,True)
                    loss,terms=fitting_loss(z,d,torch.tensor([r['label'] for r in q],device='cuda'),
                        torch.tensor(weights[ix],device='cuda'),rl)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if epoch==1 and not terms_all:
                        gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in params.items()}
                        assert any(v is not None and v>0 for n,v in gradients.items() if 'selector' in n)
                        if arm=='metric':
                            assert all(gradients[n]>0 for n in gradients if 'metric_' in n)
                    optimizer.step()
                    terms_all.append({k:float(v.detach()) for k,v in terms.items()})
                entry=dict(epoch=epoch,**{k:float(np.mean([t[k] for t in terms_all])) for k in terms_all[0]})
                curve.append(entry)
                print('EPOCH',arm,seed,entry,round(time.perf_counter()-tick,1),flush=True)
            assert all(torch.equal(net.state_dict()[n].cpu(),v) for n,v in fixed.items())
            updates={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in params.items()}
            del fixed,initial,optimizer,z,d,loss
            net.requires_grad_(False)
            after,after_metrics=evaluate(net,arrays,indices['development'],rows['development'],
                                         rows['references'],bank,f'after_{arm}_s{seed}')
            save(net,folder/'model.pt',arm)
            # Native chips detect accidental stale cache or wrong checkpoint export.
            native_rows=rows['development'][:4]
            values=batch([prepare_record(r) for r in native_rows],'cuda')
            native_refs=batch([prepare_record(r) for r in rows['references']],'cuda')
            with torch.no_grad():
                native_refs['prefix']=net.prefix(native_refs['images'],stop=2)
                native=forward_batch(net,values,native_refs,pair_mask(native_rows,rows['references'],'cuda')).cpu().numpy()
            native_error=float(abs(native-after[:4]).max())
            assert native_error<1e-4,native_error
            del net,values,native_refs
            restored=load(folder/'model.pt')
            replay=refit.predict(restored,arrays,indices['development'][:4],native_rows,rows['references'],bank)
            replay_error=float(abs(replay-after[:4]).max())
            assert replay_error<1e-4,replay_error
            receipt=dict(arm=arm,seed=seed,checkpoint=str(checkpoint),steps=153,epochs=3,
                source_count=len(rows['fit']),reference_count=len(rl),held_count=len(rows['development']),
                comparison_lr=.001 if arm=='control' else .01,value_lr=.001,
                trainable=sum(p.numel() for p in params.values()),only_m3_changed=True,
                curve=curve,gradients=gradients,parameter_updates=updates,
                before=before_metrics,after=after_metrics,initial_error=initial_error,
                native_error=native_error,replay_error=replay_error,seconds=time.perf_counter()-tick,
                validation_selection=False,optimizer_restarted=True)
            write(folder/'training.json',receipt)
            print('COMPLETE',arm,seed,json.dumps(after_metrics),flush=True)
            del restored,params
            torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=parser.parse_args()
    run(args.seeds)
