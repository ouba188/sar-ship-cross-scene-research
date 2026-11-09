"""Reproducible staged training of surviving current operators and matched controls."""
from pathlib import Path
import sys
import json
import argparse
import time
import gc
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'experiment_completion_20261109'
sys.path.insert(0, str(PREVIOUS))
import common_ablation as old
from model_init import BaseNetwork

torch, nn, F, task, prev = old.torch, old.nn, old.F, old.task, old.prev
from torchvision.ops import SqueezeExcitation


class Network(BaseNetwork):
    def __init__(self, arm, seed, weight_path, classes=4):
        super().__init__('plain' if arm == 'se' else arm, seed, weight_path, classes)
        self.arm = arm
        self.phase = 0
        self.fixed_delta = None
        self.readout_batch = None
        if arm == 'se':
            torch.manual_seed(seed+10000)
            self.se = nn.ModuleList(SqueezeExcitation(c, c//16) for c in (512,1024)).to(task.DEVICE)
            bb = self.base.backbone.model
            bb.layer2[3].bn3.register_forward_hook(lambda m,args,out: self.se[0](out))
            bb.layer3[5].bn3.register_forward_hook(lambda m,args,out: self.se[1](out))
        self.eval()

    def forward(self, a, c, rows, ids, rid, m3_on=True):
        valid = task.tensor(a['valid'], ids)
        av = task.tensor(a['available'], ids)
        slots = task.tensor(a['slots'], ids)
        local = task.tensor(c['local_slots'], ids)
        x = task.tensor(c['early_x'], ids)
        aux, responses = [], []
        if self.use1 and self.phase >= 2:
            values = []
            for p in range(2):
                occ = old.grid_occupancy(valid[:,p], (20,20))*av[:,p,None,None,None]
                result, trace = self.m1[p](x[:,p], occ, details=True)
                values.append(result[0]); aux.append(result[1]); responses.append(trace['response'])
            x = torch.stack(values, 1)
        else:
            responses = [x.new_zeros((len(ids),self.classes,20,20)) for _ in range(2)]
        bb = self.base.backbone.model
        early = bb.layer2[3]
        z = early.relu(early.bn3(early.conv3(x.flatten(0,1)))+task.tensor(c['early_identity'],ids).flatten(0,1))
        live2 = z.reshape(len(ids),2,512,20,20)
        low, masks = [], []
        if self.use2:
            for p in range(2):
                reg,en = self.base.region_integrals(local[:,p],local.new_zeros(len(ids),20,3,20,20))
                low.append(self.base.m2[p](live2[:,p],reg,valid[:,p],en)['pooled'])
                masks.append(torch.minimum(reg,old.grid_occupancy(valid[:,p],(20,20))))
        for layer in list(bb.layer3.children())[:5]: z=layer(z)
        identity=z; block=bb.layer3[5]
        x=block.relu(block.bn1(block.conv1(z))).reshape(len(ids),2,256,10,10)
        if self.use2:
            values=[]
            for p in range(2):
                reg,en=self.base.region_integrals(slots[:,p],slots.new_zeros(len(ids),20,3,10,10))
                values.append(self.base.internal_m2[p](x[:,p],reg,valid[:,p],en)[0])
            x=torch.stack(values,1)
        x=block.relu(block.bn2(block.conv2(x.flatten(0,1))))
        x=block.relu(block.bn3(block.conv3(x))+identity).reshape(len(ids),2,1024,10,10)
        if self.use3 and self.phase>=1 and m3_on:
            if self.fixed_delta is not None:
                delta=task.tensor(self.fixed_delta,ids)
            else:
                pair=task.legal(rows,ids,rid)[...,None]&av[:,None].bool()&task.tensor(a['available'],rid)[None].bool()
                delta=self.matcher(task.tensor(a['desc'],ids),task.tensor(a['desc'],rid),
                    task.tensor(a['mass'],ids),task.tensor(a['mass'],rid),task.tensor(a['write'],ids),pair,
                    torch.tensor([rows[i]['label'] for i in rid],device=task.DEVICE),arm='aggregate')[0]
            x=x+delta
        deep=bb.layer4(x.flatten(0,1))
        glob=old.regional_mean(deep,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(len(ids),2,-1)
        if self.use2:
            pooled=[]
            for p in range(2):
                reg,en=self.base.region_integrals(slots[:,p],slots.new_zeros(len(ids),20,3,10,10))
                extra=self.base.deep_pool[p](x[:,p],reg,valid[:,p],en)['pooled']
                pooled.append((low[p]+self.base.deep_output[p](extra)).flatten(1))
            feat=torch.cat((glob,torch.stack(pooled,1)),-1)*av[...,None]
            b=dict(features=torch.cat((feat.flatten(1),av),1),x=live2,
                mask=torch.stack(masks,1),response=torch.stack(responses,1),available=av,
                valid=old.grid_occupancy(valid.flatten(0,1),(20,20)).reshape(len(ids),2,1,20,20))
            self.readout_batch=b
            score=self.guide(b) if self.phase>=3 else self.guide.head(b['features'])
        else:
            features=torch.cat(((glob*av[...,None]).flatten(1),av),1)
            self.readout_batch=dict(features=features)
            score=self.head(features)
        return score,torch.stack(aux,1) if aux else None

    def classifier(self):
        return self.guide.head if self.use2 else self.head


def configure(net, phase):
    net.phase=phase
    net.zero_grad(set_to_none=True)
    net.requires_grad_(False)
    net.classifier().requires_grad_(True)
    if phase==-1:
        net.classifier().requires_grad_(False)
        for i in (1,2,3):getattr(net.base.backbone.model.layer3[5],f'conv{i}').requires_grad_(True)
    elif phase==0:
        block=net.base.backbone.model.layer3[5]
        for i in (1,2,3):getattr(block,f'conv{i}').requires_grad_(True)
        if net.use2:
            net.base.internal_m2.requires_grad_(True)
            net.base.deep_output.requires_grad_(True)
        if net.arm=='se':net.se.requires_grad_(True)
    elif phase==1:
        if net.use3:net.matcher.requires_grad_(True)
    elif phase==2:
        net.base.backbone.model.layer2[3].conv3.requires_grad_(True)
        if net.use1:net.m1.requires_grad_(True)
        if net.arm=='se':net.se[0].requires_grad_(True)
    elif phase==3 and net.use2:
        net.guide.value.requires_grad_(True)
        net.guide.output.requires_grad_(True)
    net.eval()


def weights(rows, train, baseline=None):
    fr=prev.pd.DataFrame([rows[i] for i in train])
    w=np.zeros(len(rows),np.float32)
    w[train]=1/(fr.groupby('label').mmsi.transform('nunique').to_numpy()*fr.groupby('mmsi').mmsi.transform('size').to_numpy())
    if baseline is not None:
        y=np.array([rows[i]['label'] for i in train])
        margins=baseline[np.arange(len(train)),y]-np.where(np.arange(baseline.shape[1])[None]==y[:,None],-np.inf,baseline).max(1)
        w[train]*=np.where(margins<1,2.,1.)
    w[train]/=w[train].mean()
    return w


@torch.no_grad()
def predict(net,a,c,rows,ids,rid):
    return np.concatenate([net(a,c,rows,ids[i:i+12],rid)[0].cpu().numpy() for i in range(0,len(ids),12)])


@torch.no_grad()
def freeze_delta(net,a,rows,rid):
    if not net.use3:return
    result=torch.empty((len(rows),2,1024,10,10),device=task.DEVICE)
    rd=task.tensor(a['desc'],rid);rm=task.tensor(a['mass'],rid);ra=task.tensor(a['available'],rid).bool()
    label=torch.tensor([rows[i]['label'] for i in rid],device=task.DEVICE)
    for start in range(0,len(rows),12):
        ids=np.arange(start,min(start+12,len(rows)))
        pair=task.legal(rows,ids,rid)[...,None]&task.tensor(a['available'],ids)[:,None].bool()&ra[None]
        result[ids]=net.matcher(task.tensor(a['desc'],ids),rd,task.tensor(a['mass'],ids),rm,
                               task.tensor(a['write'],ids),pair,label,arm='aggregate')[0]
    net.fixed_delta=result


def resident(a,c):
    original=task.tensor
    def tensor(value,ids):
        if torch.is_tensor(value):return value[torch.as_tensor(ids,device=value.device)]
        return original(value,ids)
    task.tensor=tensor
    for table,keys in ((a,('valid','available','slots','desc','mass','write')),
                       (c,('early_x','early_identity','local_slots'))):
        for key in keys:
            value=table[key]
            out=torch.empty(value.shape,device=task.DEVICE,dtype=torch.float32)
            for first in range(0,len(value),16):out[first:first+16]=torch.from_numpy(np.array(value[first:first+16],copy=True)).to(task.DEVICE)
            table[key]=out
    print('RESIDENT_GIB',torch.cuda.memory_allocated()/1024**3,flush=True)


def orders(ids, seed, epochs=None, steps=None, batch=24, continuous=False):
    rng=np.random.default_rng(seed)
    if continuous:
        stream=[]
        while len(stream)<steps*batch:stream.extend(rng.permutation(ids).tolist())
        return list(np.array(stream[:steps*batch]).reshape(steps,batch))
    batches=[]
    while len(batches)<(steps or 10**9):
        order=rng.permutation(ids)
        batches.extend(order[i:i+batch] for i in range(0,len(order),batch))
        if epochs is not None:
            epochs-=1
            if not epochs:break
    return batches[:steps] if steps else batches


@torch.no_grad()
def readout_cache(net,a,c,rows,ids,rid):
    stores={}
    for start in range(0,len(ids),12):
        chosen=ids[start:start+12]
        net(a,c,rows,chosen,rid)
        for key,value in net.readout_batch.items():
            if key not in stores:
                stores[key]=torch.empty((len(ids),)+value.shape[1:],device=task.DEVICE)
            stores[key][start:start+len(chosen)]=value.detach()
    net.readout_batch=None
    return stores


def train_phase(net,cfg,seed,folder,rows,groups,a,c,scale=1.):
    phase=cfg['phase'];configure(net,phase)
    ids=groups[cfg['cohort']];rid=groups['references']
    dest=folder/cfg['name'];dest.mkdir(parents=True,exist_ok=True)
    y=np.array([r['label'] for r in rows])
    initial=predict(net,a,c,rows,ids,rid)
    weight=weights(rows,ids,initial if cfg.get('hard') else None)
    selected=None
    if phase==-1:
        from intervention_inputs import select
        selected=select(net,a,c,rows,ids,rid)
        task.write(dest/'teacher.json',selected)
        original=np.array([r['original'] for r in selected])
        assert [r['index'] for r in selected]==ids.tolist()
        assert float(abs(original-initial).max())<1e-4
        labels=y[ids];counts=np.bincount(labels,minlength=net.classes)
        weight[ids]=len(ids)/(net.classes*counts[labels])
    params={n:p for n,p in net.named_parameters() if p.requires_grad}
    buckets={}
    for name,p in params.items():
        if name.startswith('matcher.'):
            lr=cfg.get('output_lr',1e-4) if name.startswith('matcher.output.') else cfg.get('module_lr',1e-3)
        elif phase==2 and name.startswith('m1.'):
            lr=1e-3 if '.output.' in name or '.relation_output.' in name else 3e-4
        elif phase==3 and name.startswith('guide.') and not name.startswith('guide.head.'):
            lr=1e-3 if name.startswith('guide.output.') else 3e-4
        elif name.startswith('base.backbone.'):
            lr=cfg.get('backbone_lr',1e-5)
        elif name.startswith(('guide.head.','head.')):
            lr=cfg.get('head_lr',1e-5)
        else:lr=cfg.get('module_lr',3e-4)
        buckets.setdefault(lr*scale,[]).append(p)
    opt=torch.optim.AdamW([dict(params=v,lr=k) for k,v in buckets.items()],weight_decay=.01)
    batches=orders(ids,seed,epochs=cfg.get('epochs'),steps=cfg.get('steps'),
                   batch=cfg.get('batch',24),continuous=cfg.get('continuous',False))
    cache=readout_cache(net,a,c,rows,ids,rid) if phase==3 else None
    lookup={int(idx):j for j,idx in enumerate(ids)}
    task.write(dest/'orders.json',[x.tolist() for x in batches])
    tick=time.perf_counter();curve=[];grads=[]
    for step,ix in enumerate(batches,1):
        opt.zero_grad(set_to_none=True)
        if cache is None:z,aux=net(a,c,rows,ix,rid)
        else:
            j=torch.tensor([lookup[int(k)] for k in ix],device=task.DEVICE)
            b={key:value[j] for key,value in cache.items()}
            z=net.guide(b) if net.use2 else net.head(b['features']);aux=None
        labels=torch.tensor(y[ix],device=task.DEVICE);w=torch.tensor(weight[ix],device=task.DEVICE)
        ce=(F.cross_entropy(z,labels,reduction='none')*w).mean();ace=ce.new_zeros(())
        if aux is not None and phase==2:
            av=task.tensor(a['available'],ix)
            terms=F.cross_entropy(aux.flatten(0,1),labels[:,None].expand(-1,2).flatten(),reduction='none').reshape(len(ix),2)
            ace=((terms*av).sum(1)/av.sum(1).clamp_min(1)*w).mean()
        if phase==-1:
            from sar_m123.current.training import adaptation_loss
            targets=[selected[lookup[int(k)]] for k in ix]
            teacher=torch.tensor([r['original'] for r in targets],device=task.DEVICE)
            useful=torch.tensor([r['useful'] for r in targets],device=task.DEVICE)
            loss,terms=adaptation_loss(z,labels,teacher,w,useful,.3683066666126251)
        else:loss=ce+.02*ace
        assert torch.isfinite(loss)
        loss.backward()
        if step in (1,2):
            norm={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
            assert all(np.isfinite(list(norm.values())))
            grads.append(dict(step=step,norms=norm))
        torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
        opt.step();entry=dict(step=step,ce=float(ce.detach()),aux=float(ace.detach()))
        if phase==-1:entry.update({k:float(v.detach()) for k,v in terms.items()})
        curve.append(entry)
        if step%60==0 or step==len(batches):
            print('STEP',folder.name,cfg['name'],step,len(batches),round(float(ce.detach()),4),flush=True)
    net.readout_batch=None;del cache,opt
    gc.collect();torch.cuda.empty_cache()
    torch.save(net.state_dict(),dest/'final.pt')
    stats={}
    for split in ('small','validation'):
        score=predict(net,a,c,rows,groups[split],rid)
        np.savez_compressed(dest/f'{split}.npz',ids=groups[split],score=score)
        stats[split]=task.metrics(y[groups[split]],score)
    prev.pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False)
    task.write(dest/'receipt.json',dict(config=cfg,seed=seed,scale=scale,steps=len(batches),
        examples=sum(map(len,batches)),trainable=sum(p.numel() for p in params.values()),
        metrics=stats,seconds=time.perf_counter()-tick,gradients=grads,
        selected_count=None if selected is None else sum(r['useful'] for r in selected)))
    print('DONE',folder.name,cfg['name'],stats,flush=True)


def fit(arm,seed,scale,protocol,rows,groups,a,c):
    folder=ROOT/f'{arm}_s{seed}_lr{scale:g}'
    if (folder/'receipt.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    net=Network(arm,seed,protocol['pretrain'],protocol.get('classes',4))
    for cfg in protocol['phases']:
        dest=folder/cfg['name']
        if (dest/'receipt.json').exists():
            net.load_state_dict(torch.load(dest/'final.pt',map_location='cpu',weights_only=True))
            configure(net,cfg['phase'])
        else:train_phase(net,cfg,seed,folder,rows,groups,a,c,scale)
        if cfg['phase']==1:freeze_delta(net,a,rows,groups['references'])
    final=folder/protocol['phases'][-1]['name']
    task.write(folder/'receipt.json',dict(arm=arm,seed=seed,scale=scale,final=str(final/'final.pt'),
        phase=3,metrics=json.loads((final/'receipt.json').read_text())['metrics'],evaluation_read=False))
    del net;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arms',nargs='+',default=['full','plain','no_m1','no_m2','no_m3','se'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();task.current.setup(task.DEVICE)
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    rows,groups=prev.records();a,c=prev.arrays()
    a['write']=np.load(PREVIOUS/'common_init/write.npy',mmap_mode='r')
    resident(a,c)
    if args.smoke:
        result={};ids=groups['small'][:3]
        for arm in args.arms:
            net=Network(arm,17,protocol['pretrain'])
            for phase in range(4):
                configure(net,phase)
                z,aux=net(a,c,rows,ids,groups['references'])
                loss=F.cross_entropy(z,torch.tensor([rows[i]['label'] for i in ids],device=task.DEVICE))
                if aux is not None:loss=loss+.02*aux.square().mean()
                loss.backward()
                grads={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
                assert torch.isfinite(z).all() and all(np.isfinite(list(grads.values())))
                result[f'{arm}_{phase}']=dict(shape=list(z.shape),gradients=grads)
                net.zero_grad(set_to_none=True)
            del net;gc.collect();torch.cuda.empty_cache()
        task.write(ROOT/'smoke.json',result);print('SMOKE_OK',flush=True);return
    assert protocol['phases'], 'Finalize the staged protocol before training.'
    for seed in args.seeds:
        for arm in args.arms:fit(arm,seed,args.scale,protocol,rows,groups,a,c)


if __name__=='__main__':main()
