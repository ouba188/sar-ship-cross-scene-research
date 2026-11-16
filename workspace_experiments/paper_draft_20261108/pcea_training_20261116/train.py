"""Bounded, paired OSS2 validation pilot. Never opens evaluation images."""
import argparse
import gc
import json
import math
import time
from model import ROOT, Model, full, torch, np

ARMS = ('baseline','independent','gate','pcea')
write, metrics = full.write, full.metrics


def optimizer(net, rate):
    params = {n:p for n,p in net.named_parameters() if p.requires_grad}
    buckets = [[],[],[]]
    for n,p in params.items():
        buckets[0 if n.startswith('net.base.backbone.') else 2 if n.startswith('exchange.') else 1].append(p)
    rates = (rate, 10*rate, 100*rate)
    return torch.optim.AdamW([dict(params=p,lr=r) for p,r in zip(buckets,rates) if p],weight_decay=.01), params


def smoke(inputs):
    ids = inputs.groups['validation'][:12]
    train = inputs.groups['small'][:12]
    report = {}
    expected = None
    for arm in ARMS:
        net = Model(17,arm)
        z = full.predict(net,inputs,ids,'validation')
        if expected is None:
            expected = z
        error = float(abs(z-expected).max())
        assert error<1e-5, (arm,error)
        if arm != 'baseline':
            assert float(net.exchange.trace['delta'].abs().max())==0
        opt, params = optimizer(net,1e-5)
        records=[]
        for step in range(2):
            net.train();opt.zero_grad(set_to_none=True)
            loss,_=full.loss_for(net,inputs,train,1)
            loss.backward()
            grads={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
            assert all(np.isfinite(list(grads.values())))
            assert grads['net.base.backbone.model.conv1.weight']>0
            if arm!='baseline':
                assert grads['exchange.output.0.weight']>0
                if step==1:
                    assert grads['exchange.value.0.weight']>0
                    if arm in ('gate','pcea'): assert grads['exchange.gate.0.weight']>0
            torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
            opt.step()
            records.append(dict(loss=float(loss.detach()),exchange_gradients={k:v for k,v in grads.items() if k.startswith('exchange.')}))
        report[arm]=dict(initial_logit_error=error,steps=records,
            trainable=sum(p.numel() for p in params.values()))
        print('SMOKE',arm,error,flush=True)
        del net,opt,params;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'smoke.json',report)


def fit(inputs,arm,seed,rate,epochs,branch_only=False):
    dest=ROOT/(f'{arm}_s{seed}_lr{rate:g}_e{epochs}'+('_branch' if branch_only else ''))
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists(): return json.loads((dest/'receipt.json').read_text())
    net=Model(seed,arm)
    if branch_only:
        assert arm != 'baseline'
        for name,param in net.named_parameters():
            if not name.startswith('exchange.'): param.requires_grad_(False)
    net.accelerate()
    opt,params=optimizer(net,rate)
    rates=[g['lr'] for g in opt.param_groups]
    ids,val=inputs.groups['small'],inputs.groups['validation']
    rng=np.random.default_rng(seed+20261116)
    aug=np.random.default_rng(seed+20261117)
    z=full.predict(net,inputs,val,'validation')
    previous=np.load(full.ROOT/f'selected_s{seed}/validation.npz')
    assert np.array_equal(val,previous['ids'])
    initial_error=float(abs(z-previous['score']).max())
    assert initial_error<1e-3,initial_error
    initial=metrics(inputs.y[val],z)
    best=dict(epoch=0,metrics=initial,key=[initial['ba'],initial['oa'],-initial['nll']])
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'validation.npz',ids=val,score=z,label=inputs.y[val])
    cfg=dict(arm=arm,seed=seed,backbone_lr=rate,old_modules_lr=10*rate,
        new_branch_lr=100*rate,epochs=epochs,batch=24,microbatch=12,source=719,validation=228,
        reference_records=32,initial_checkpoint=str(full.ROOT/f'selected_s{seed}/best.pt'),
        selection='validation BA, OA, -NLL; epoch0 included',
        scheduler='one epoch warmup then cosine to .1',
        loss='same class/vessel balanced CE + .02 M1 auxiliary CE',
        trainable_backbone='none; branch-only diagnostic' if branch_only else 'full; frozen BN running statistics',
        optimization_scope='branch' if branch_only else 'joint',
        intervention='checkpoint history retained, no new teacher selection',test_read=False)
    write(dest/'config.json',cfg)
    torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter();curve=[];steps=0
    for epoch in range(epochs):
        factor=.5 if epoch==0 else .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(epochs-2,1)))
        for group,lr in zip(opt.param_groups,rates): group['lr']=lr*factor
        net.train();order=rng.permutation(ids);total=0.;correct=0
        for first in range(0,len(order),24):
            batch=order[first:first+24];opt.zero_grad(set_to_none=True)
            code=int(aug.integers(4))+4*int(aug.random()<.5)
            for start in range(0,len(batch),12):
                part=batch[start:start+12]
                loss,score=full.loss_for(net,inputs,part,code)
                assert torch.isfinite(loss), (arm,epoch,first)
                (loss*len(part)/len(batch)).backward()
                total+=float(loss.detach())*len(part)
                correct+=int((score.argmax(1)==torch.as_tensor(inputs.y[part],device='cuda')).sum())
            torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
            opt.step();steps+=1
        z=full.predict(net,inputs,val,'validation');m=metrics(inputs.y[val],z)
        key=[m['ba'],m['oa'],-m['nll']]
        row=dict(epoch=epoch+1,train_loss=total/len(ids),train_oa=100*correct/len(ids),
            validation=m,seconds=time.perf_counter()-tick)
        curve.append(row)
        if key>best['key']:
            best=dict(epoch=epoch+1,metrics=m,key=key)
            torch.save(net.state_dict(),dest/'best.pt')
            np.savez_compressed(dest/'validation.npz',ids=val,score=z,label=inputs.y[val])
        write(dest/'curve.json',curve)
        write(ROOT/'status.json',dict(state='training',arm=arm,seed=seed,rate=rate,epoch=epoch+1,epochs=epochs,best=best))
        print('VAL',arm,seed,rate,epoch+1,'OA/BA',round(m['oa'],3),round(m['ba'],3),'best',best['epoch'],'sec',round(row['seconds'],1),flush=True)
    torch.save(net.state_dict(),dest/'last.pt')
    np.savez_compressed(dest/'last_validation.npz',ids=val,score=z,label=inputs.y[val])
    receipt=dict(**cfg,initial=initial,initial_logit_error=initial_error,best=best,last=curve[-1]['validation'],
        steps=steps,seconds=time.perf_counter()-tick,trainable=sum(p.numel() for p in params.values()),
        branch_parameters=sum(p.numel() for p in net.exchange.parameters() if p.requires_grad) if arm!='baseline' else 0,
        peak_gib=torch.cuda.max_memory_allocated()/1024**3)
    write(dest/'receipt.json',receipt)
    del net,opt,params;gc.collect();torch.cuda.empty_cache()
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    parser.add_argument('--rates',type=float,nargs='+',default=[1e-5,3e-5])
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--branch-only',action='store_true')
    args=parser.parse_args()
    inputs=full.Inputs()
    for split in ('small','validation'): inputs.prepare_geometry(split)
    if args.smoke: return smoke(inputs)
    assert (ROOT/'smoke.json').exists()
    for seed in args.seeds:
        for rate in args.rates:
            for arm in args.arms: fit(inputs,arm,seed,rate,args.epochs,args.branch_only)
    write(ROOT/'status.json',dict(state='requested_runs_complete',seeds=args.seeds,rates=args.rates,
                                  arms=args.arms,epochs=args.epochs,test_read=False))


if __name__=='__main__':main()
