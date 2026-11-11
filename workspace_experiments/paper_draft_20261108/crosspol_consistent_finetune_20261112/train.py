"""Matched Full fine-tuning with the same recovered-input rule in every split."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import math
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'full_finetune_20261111'
RECOVERY=ROOT.parent/'crosspol_box_recovery_20261112'
sys.path.insert(0,str(OLD))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


ft=load('matched_full_finetune',OLD/'train.py')
recovery=load('crosspol_recovery_policy',RECOVERY/'pilot.py')
np,torch=ft.np,ft.torch
write,metrics=ft.write,ft.metrics
LR=1e-4
SEEDS=ft.SEEDS


class Inputs(ft.Inputs):
    def __init__(self):
        super().__init__()
        self.restore={}
        active=np.concatenate([self.groups[s] for s in ('small','validation','evaluation')])
        for i in active:
            row,audit=recovery.recover(self.rows[i])
            if audit:
                self.rows[i]=row
                self.restore[int(i)]=audit
        self.reference_rows=[self.rows[i] for i in self.groups['references']]
        self.cache=ROOT/'geometry_cache'
        self.cache.mkdir(exist_ok=True)
        for split in ('small','validation'):
            self.restore_resident(split)
        write(ROOT/'input_counts.json',dict(
            recovered={s:sum(int(i) in self.restore for i in self.groups[s])
                       for s in ('small','validation','evaluation','references')},
            rule='same original pixel grid; donor box; no label in selection',
            native_scale=1.0,source_audit=str(RECOVERY/'AUDIT_ZH.md')))

    def restore_resident(self,split):
        lookup,pixels,footprint,available=self.resident[split]
        for i in self.groups[split]:
            if int(i) not in self.restore: continue
            p=0 if self.restore[int(i)]['restored']=='vv' else 1
            view=ft.staged.task.data.prepare(self.rows[i])[p]
            assert view['valid_target']
            j=lookup[int(i)]
            pixels[j,p]=torch.as_tensor(view['pixels'].copy(),device=ft.DEVICE)
            footprint[j,p]=torch.as_tensor(view['footprint'].copy(),device=ft.DEVICE)
            available[j,p]=1
        assert bool((available==1).all()),split

    def load_split(self,split):
        if split in self.loaded: return
        super().load_split(split)
        if hasattr(self,'restore'): self.restore_resident(split)


def smoke(inputs):
    ids=np.array([i for i in inputs.groups['small'] if int(i) in inputs.restore][:ft.MICRO])
    net=ft.LiveFull(17,'full').train()
    loss,z=ft.loss_for(net,inputs,ids,1)
    loss.backward()
    grads={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}
    for prefix in ('net.m1.','net.base.internal_m2.','net.matcher.','net.guide.output.'):
        assert any(n.startswith(prefix) and v>0 for n,v in grads.items()),prefix
    assert all(np.isfinite(v) for v in grads.values())
    assert grads['net.base.backbone.model.conv1.weight']>0
    original={n:p.grad.detach().clone() for n,p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True);net.accelerate()
    loss2,z2=ft.loss_for(net,inputs,ids,1);loss2.backward()
    err=float((z-z2).abs().max())
    ge=max(float((p.grad-original[n]).abs().max()) for n,p in net.named_parameters() if n in original)
    assert err<1e-5 and ge<1e-4,(err,ge)
    write(ROOT/'smoke.json',dict(ids=ids.tolist(),loss=float(loss.detach()),
        score_error=err,max_gradient_error=ge,gradients=grads,
        restored_reference_count=5,all_training_views_available=True))
    print('SMOKE',err,ge,flush=True)
    del net,original,loss,loss2,z,z2
    gc.collect();torch.cuda.empty_cache()


def fit(inputs,seed):
    dest=ROOT/f'selected_s{seed}'
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists():return
    cfg=ft.configuration('full',LR,seed)
    cfg.update(input_policy='cross-polarization recovery in training, references and queries',
               matched_rng=True,hyperparameter_search=False,
               baseline=str(OLD/f'selected_s{seed}'))
    write(dest/'config.json',cfg)
    net=ft.LiveFull(seed,'full')
    net.accelerate()
    params={n:p for n,p in net.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([
        dict(params=[p for n,p in params.items() if n.startswith('net.base.backbone.')],lr=LR),
        dict(params=[p for n,p in params.items() if not n.startswith('net.base.backbone.')],lr=ft.MODULE_LR)],weight_decay=.01)
    train,val=inputs.groups['small'],inputs.groups['validation']
    rng=np.random.default_rng(seed+20261111);aug=np.random.default_rng(seed+20261112)
    score=ft.predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
    # Restoring images and references intentionally changes the initial predictions.
    # The actual carrier loading and baseline live path were verified previously.
    write(dest/'initial_restored.json',dict(validation=m,epoch=0,
        checkpoint=f'staged_ablation_20261109/full_s{seed}_lr1/readout/final.pt'))
    best=dict(epoch=0,key=[m['ba'],m['oa'],-m['nll']],metrics=m)
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
    curve=[];steps=0;elapsed=0.;start=0
    if (dest/'resume.pt').exists():
        ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best,curve,start,steps,elapsed=(ck[k] for k in ('best','curve','epoch','steps','seconds'))
        rng.bit_generator.state=ck['rng'];aug.bit_generator.state=ck['aug'];del ck
        shutil.copy2(dest/'best_saved.pt',dest/'best.pt')
        shutil.copy2(dest/'validation_saved.npz',dest/'validation.npz')
    tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,ft.EPOCHS):
        factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(ft.EPOCHS-3)))
        for group,base in zip(opt.param_groups,(LR,ft.MODULE_LR)):group['lr']=base*factor
        net.train();order=rng.permutation(train);total=0.;correct=0
        for first in range(0,len(order),ft.BATCH):
            batch=order[first:first+ft.BATCH];opt.zero_grad(set_to_none=True)
            code=int(aug.integers(4))+4*int(aug.random()<.5)
            for part in range(0,len(batch),ft.MICRO):
                ids=batch[part:part+ft.MICRO]
                loss,z=ft.loss_for(net,inputs,ids,code)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {seed}/{epoch}')
                (loss*len(ids)/len(batch)).backward()
                total+=float(loss.detach())*len(ids)
                correct+=int((z.argmax(1)==torch.as_tensor(inputs.y[ids],device=ft.DEVICE)).sum())
            torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
            opt.step();steps+=1
        score=ft.predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
        key=(m['ba'],m['oa'],-m['nll'])
        row=dict(epoch=epoch+1,train_loss=total/len(train),train_oa=100*correct/len(train),
            validation=m,seconds=elapsed+time.perf_counter()-tick,backbone_lr=LR*factor)
        curve.append(row)
        if key>tuple(best['key']):
            best=dict(epoch=epoch+1,key=list(key),metrics=m)
            torch.save(net.state_dict(),dest/'best.pt')
            np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
        stop=epoch+1>=ft.MIN_EPOCHS and epoch+1-best['epoch']>=ft.PATIENCE
        print('VAL',seed,epoch+1,'OA/BA',round(m['oa'],3),round(m['ba'],3),'best',best['epoch'],'sec',round(row['seconds'],1),flush=True)
        write(ROOT/'status.json',dict(state='training',seed=seed,epoch=epoch+1,best=best,test_read=False,complete=False))
        write(dest/'curve.json',curve)
        if (epoch+1)%5==0 or stop or epoch+1==ft.EPOCHS:
            shutil.copy2(dest/'best.pt',dest/'best_saved.pt')
            shutil.copy2(dest/'validation.npz',dest/'validation_saved.npz')
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,curve=curve,
                epoch=epoch+1,steps=steps,seconds=row['seconds'],rng=rng.bit_generator.state,
                aug=aug.bit_generator.state),dest/'resume.pt')
        if stop:break
    write(dest/'receipt.json',dict(**cfg,epochs=epoch+1,steps=steps,best=best,
        seconds=elapsed+time.perf_counter()-tick,stop_reason='validation_patience' if stop else 'max_epochs',
        trainable=sum(p.numel() for p in params.values()),peak_gib=torch.cuda.max_memory_allocated()/1024**3,test_read=False))
    del net,opt,params;gc.collect();torch.cuda.empty_cache()


def evaluate(inputs):
    inputs.load_split('evaluation');inputs.prepare_geometry('evaluation')
    checks=[]
    for seed in SEEDS:
        for arm in ('selected','frozen_restored'):
            dest=ROOT/f'{arm}_s{seed}';dest.mkdir(exist_ok=True)
            net=ft.LiveFull(seed,'full')
            path=dest/'best.pt' if arm=='selected' else OLD/f'selected_s{seed}/best.pt'
            net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            for split in ('validation','evaluation'):
                ids=inputs.groups[split]
                z=ft.predict(net,inputs,ids,split)
                if arm=='selected' and split=='validation':
                    expected=np.load(dest/'validation.npz')['score']
                    err=float(abs(z-expected).max())
                    assert err<1e-3 and np.array_equal(z.argmax(1),expected.argmax(1)),err
                    checks.append(dict(seed=seed,validation_replay_error=err))
                np.savez_compressed(dest/f'{split}.npz',ids=ids,score=z,label=inputs.y[ids])
                m=metrics(inputs.y[ids],z);write(dest/f'{split}.json',m)
                print('FINAL',arm,seed,split,json.dumps(m),flush=True)
            del net;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'checkpoint_replays.json',checks)
    write(ROOT/'status.json',dict(state='training_test_complete',complete=True,analysis_pending=True,
        formal_model_changed=False,manuscript_changed=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    inputs=Inputs()
    for split in ('small','validation'):inputs.prepare_geometry(split)
    if args.prepare:return
    if args.smoke:smoke(inputs);return
    if args.evaluate:evaluate(inputs);return
    write(ROOT/'protocol.json',dict(seeds=SEEDS,scope='full',lr=LR,config=ft.configuration('full',LR,17),
        task='paired consistent cross-polarization input recovery',search=False,
        training_records=719,validation_records=228,test_records=378,reference_records=32,
        restored_training=48,restored_validation=22,restored_test=30,restored_references=5,
        arms=['historical original fine-tuning','historical query-only recovery',
              'frozen old fine-tuning with recovered query+reference','matched recovered-input fine-tuning']))
    if not (ROOT/'smoke.json').exists():smoke(inputs)
    for seed in SEEDS:fit(inputs,seed)
    evaluate(inputs)


if __name__=='__main__':main()
