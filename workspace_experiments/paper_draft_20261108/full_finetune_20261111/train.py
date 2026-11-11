"""Live-chip joint fine-tuning of the actual staged Full checkpoint."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
WORK = PAPER.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r50 = load('r50_finetune_baseline', PAPER/'r50_tuned_20261111/train.py')
cost = load('full_live_path', PAPER/'computational_cost_20261110/benchmark.py')
sys.path.insert(0, str(PAPER/'staged_ablation_20261109'))
import staged
from geometry import _cell_area
torch, np, nn, F = r50.torch, r50.np, r50.nn, r50.F
staged.task.tensor = cost.resident_tensor
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False
DEVICE = 'cuda'
SEEDS = (17, 29, 43)
SCOPES = ('local', 'late', 'full')
LRS = (1e-5, 1e-4)
EPOCHS, MIN_EPOCHS, PATIENCE = 30, 12, 8
MODULE_LR = 1e-4
BATCH, MICRO = 24, 12
write, metrics = r50.write, r50.metrics


def transform(x, code):
    y = np.rot90(x, code % 4, axes=(-2, -1))
    return y[..., ::-1] if code // 4 else y


class Inputs(r50.Inputs):
    def __init__(self):
        super().__init__()
        self.geometry = {}
        self.cache = ROOT/'geometry_cache'
        self.cache.mkdir(exist_ok=True)
        assert set(self.groups['references']) <= set(self.groups['small'])
        self.reference_rows = [self.rows[i] for i in self.groups['references']]

    def prepare_geometry(self, split):
        if split in self.geometry:
            return
        ids = self.groups[split]
        folder = self.cache/split
        folder.mkdir(exist_ok=True)
        codes = 8 if split == 'small' else 1
        shapes = dict(slots=(20,10,10), local_slots=(20,20,20), read=(20,20,20),
                      write=(20,10,10), mass=(20,))
        if not (folder/'complete.json').exists():
            stores = {k: np.lib.format.open_memmap(folder/(k+'.npy'), mode='w+', dtype=np.float32,
                shape=(len(ids), codes, 2)+shape) for k,shape in shapes.items()}
            for j, idx in enumerate(ids):
                for p, view in enumerate(staged.task.data.prepare(self.rows[idx])):
                    if view is None or not view['valid_target']:
                        for value in stores.values(): value[j,:,p] = 0
                        continue
                    native = cost.raw.tokens.slots(view['geometry'], view['footprint'])
                    for code in range(codes):
                        mask = np.ascontiguousarray(transform(native,code))
                        read = cost.raw.tokens.cell_mass(mask,8)/_cell_area(20)
                        writer = cost.raw.tokens.cell_mass(mask,16)/_cell_area(10)
                        writer /= np.maximum(writer.sum(0,keepdims=True),1)
                        stores['read'][j,code,p] = read
                        stores['mass'][j,code,p] = read.sum((-2,-1))
                        stores['write'][j,code,p] = writer
                        t = torch.from_numpy(mask).float()[None]
                        for key,size in (('slots',10),('local_slots',20)):
                            stores[key][j,code,p] = cost.raw.grid_occupancy(t,(size,size))[0].numpy()
                if (j+1)%100 == 0:
                    print('GEOMETRY',split,j+1,len(ids),flush=True)
            for value in stores.values(): value.flush()
            del stores
            write(folder/'complete.json',dict(n=len(ids),codes=codes,ids=ids.tolist(),
                policy='transform native slot masks then integrate; no feature caching'))
        self.geometry[split] = {k: np.load(folder/(k+'.npy'),mmap_mode='r') for k in shapes}

    def data(self, ids, split, code=0):
        lookup, pixels, footprint, available = self.resident[split]
        ix = np.array([lookup[int(i)] for i in ids])
        it = torch.as_tensor(ix,device=DEVICE)
        images = pixels[it].float().unsqueeze(2)/255
        valid = footprint[it].float().unsqueeze(2)
        images = torch.rot90(images,code%4,(-2,-1))
        valid = torch.rot90(valid,code%4,(-2,-1))
        if code//4:
            images,valid=images.flip(-1),valid.flip(-1)
        data = dict(images=images,valid=valid,available=available[it])
        data.update({k: torch.as_tensor(np.array(v[ix,code],copy=True),device=DEVICE)
                     for k,v in self.geometry[split].items()})
        return data


class LiveFull(cost.ChipPath):
    def __init__(self, seed, scope):
        # Copy ImageNet descriptor projection BEFORE restoring trained Full weights.
        net = staged.Network('full',seed,r50.WEIGHTS)
        super().__init__(net)
        checkpoint = PAPER/'staged_ablation_20261109'/f'full_s{seed}_lr1/readout/final.pt'
        net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        net.phase,net.fixed_delta = 3,None
        self.requires_grad_(False)
        bb = net.base.backbone.model
        if scope == 'local':
            bb.layer2[3].conv3.requires_grad_(True)
            for i in (1,2,3): getattr(bb.layer3[5],f'conv{i}').requires_grad_(True)
        elif scope == 'late':
            bb.layer3.requires_grad_(True);bb.layer4.requires_grad_(True)
        elif scope == 'full':
            bb.requires_grad_(True)
        else:
            raise ValueError(scope)
        bb.fc.requires_grad_(False)
        for module in (net.m1,net.base.internal_m2,net.base.deep_output,net.matcher,
                       net.guide.value,net.guide.output,net.guide.head):
            module.requires_grad_(True)
        self.to(DEVICE).eval()
        self.transport_graph = None

    def accelerate(self):
        from transport_graph import install
        self.transport_graph = install(self.net.matcher)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        return self

    def finish(self,a,c,query_rows,reference,reference_rows,m3_on=True):
        n=len(query_rows);table=dict(a)
        for key in ('desc','mass','available'):
            table[key]=torch.cat((a[key],reference[key]),0)
        z,aux=self.net(table,c,query_rows+reference_rows,np.arange(n),
                       np.arange(n,n+len(reference_rows)),m3_on=m3_on)
        self.net.readout_batch=None
        return z,aux


@torch.no_grad()
def predict(net,inputs,ids,split):
    net.eval()
    reference=net.reference(inputs.data(inputs.groups['references'],'small'))
    scores=[]
    for first in range(0,len(ids),MICRO):
        part=ids[first:first+MICRO]
        z,_=net(inputs.data(part,split),[inputs.rows[i] for i in part],reference,inputs.reference_rows)
        scores.append(z.cpu().numpy())
    return np.concatenate(scores)


def loss_for(net,inputs,ids,code):
    # Reference descriptors are live and differentiable; no stale bank after updates.
    ref=net.reference(inputs.data(inputs.groups['references'],'small'))
    data=inputs.data(ids,'small',code)
    z,aux=net(data,[inputs.rows[i] for i in ids],ref,inputs.reference_rows)
    y=torch.as_tensor(inputs.y[ids],device=DEVICE)
    w=torch.as_tensor(inputs.weights[ids],device=DEVICE)
    ce=(F.cross_entropy(z.float(),y,reduction='none')*w).mean()
    ac=F.cross_entropy(aux.float().flatten(0,1),y[:,None].expand(-1,2).flatten(),
                       reduction='none').reshape(len(ids),2)
    av=data['available']
    ace=(((ac*av).sum(1)/av.sum(1).clamp_min(1))*w).mean()
    return ce+.02*ace,z


def configuration(scope,lr,seed):
    return dict(scope=scope,lr=lr,seed=seed,module_lr=MODULE_LR,epochs_max=EPOCHS,
        epochs_min=MIN_EPOCHS,patience=PATIENCE,batch=BATCH,microbatch=MICRO,
        initialization=f'staged Full seed{seed}, five stages already trained',
        optimizer='AdamW wd=.01, warmup2 cosine .05, clip5',
        loss='class/vessel balanced CE + .02 M1 auxiliary CE',
        augmentation='paired rot90/flip; native region masks transformed and re-integrated',
        references='32 fixed legal source records; current shared-prefix features each batch',
        bn='running statistics fixed; affine trainable in scope',
        intervention='retained checkpoint training history; no new teacher selection',
        selection='validation BA, OA, -NLL; epoch0 is eligible',
        inference='all M1 M2 M3 active; no spectral module; no feature/delta cache')


def smoke(inputs):
    ids=inputs.groups['validation'][:MICRO]
    report={}
    for scope in SCOPES:
        net=LiveFull(17,scope)
        z=predict(net,inputs,ids,'validation')
        old=np.load(PAPER/'staged_ablation_20261109/full_s17_lr1/validation.npz')
        lookup={int(i):j for j,i in enumerate(old['ids'])}
        expected=old['score'][[lookup[int(i)] for i in ids]]
        err=float(abs(z-expected).max())
        assert err<1e-3 and np.array_equal(z.argmax(1),expected.argmax(1)),err
        params={n:p for n,p in net.named_parameters() if p.requires_grad}
        opt=torch.optim.AdamW(params.values(),lr=1e-5)
        train=inputs.groups['small'][:MICRO]
        net.train()
        loss,_=loss_for(net,inputs,train,1)
        loss.backward()
        grads={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
        for prefix in ('net.m1.','net.base.internal_m2.','net.matcher.','net.guide.output.'):
            assert any(n.startswith(prefix) and v>0 for n,v in grads.items()),prefix
        assert all(np.isfinite(v) for v in grads.values())
        assert ('net.base.backbone.model.conv1.weight' in grads)==(scope=='full')
        assert ('net.base.backbone.model.layer4.2.conv3.weight' in grads)==(scope!='local')
        assert not any(p.grad is not None for p in net.anchor_conv.parameters())
        ref0=net.reference(inputs.data(inputs.groups['references'],'small'))['desc'].detach().clone()
        opt.step()
        with torch.no_grad():
            ref1=net.reference(inputs.data(inputs.groups['references'],'small'))['desc']
        refchange=float((ref1-ref0).abs().max())
        assert (refchange>0)==(scope=='full'),refchange
        report[scope]=dict(replay_error=err,loss=float(loss.detach()),gradients=grads,
            reference_change=refchange,trainable=sum(p.numel() for p in params.values()),
            peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        print('SMOKE',scope,err,refchange,flush=True)
        del net,opt,params,ref0,ref1;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'smoke.json',report)


def graph_smoke(inputs):
    ids=inputs.groups['small'][:MICRO]
    net=LiveFull(17,'full').train()
    loss,z=loss_for(net,inputs,ids,2);loss.backward()
    original={n:p.grad.detach().clone() for n,p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True);net.accelerate()
    other,z2=loss_for(net,inputs,ids,2);other.backward()
    score_error=float((z-z2).abs().max())
    gradient_errors={n:float((p.grad-original[n]).abs().max()) for n,p in net.named_parameters() if n in original}
    assert score_error<1e-5 and max(gradient_errors.values())<1e-4
    write(ROOT/'graph_full_check.json',dict(score_error=score_error,gradient_errors=gradient_errors,
        peak_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('GRAPH_FULL_CHECK',score_error,max(gradient_errors.values()),flush=True)


def fit(inputs,scope,lr,seed,dest):
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists(): return json.loads((dest/'receipt.json').read_text())
    cfg=configuration(scope,lr,seed);write(dest/'config.json',cfg)
    net=LiveFull(seed,scope)
    net.accelerate()
    params={n:p for n,p in net.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([
        dict(params=[p for n,p in params.items() if n.startswith('net.base.backbone.')],lr=lr),
        dict(params=[p for n,p in params.items() if not n.startswith('net.base.backbone.')],lr=MODULE_LR)],
        weight_decay=.01)
    train,val=inputs.groups['small'],inputs.groups['validation']
    rng=np.random.default_rng(seed+20261111);aug=np.random.default_rng(seed+20261112)
    score=predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
    prior=np.load(PAPER/'staged_ablation_20261109'/f'full_s{seed}_lr1/validation.npz')
    lookup={int(i):j for j,i in enumerate(prior['ids'])}
    expected=prior['score'][[lookup[int(i)] for i in val]]
    replay_error=float(abs(score-expected).max())
    assert replay_error<1e-3 and np.array_equal(score.argmax(1),expected.argmax(1)),replay_error
    write(dest/'initial_replay.json',dict(max_logit_error=replay_error,validation=m,
        checkpoint=f'full_s{seed}_lr1/readout/final.pt',full_validation_replay=True))
    best=dict(epoch=0,key=[m['ba'],m['oa'],-m['nll']],metrics=m)
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
    curve=[];steps=0;elapsed=0.;start=0
    if (dest/'resume.pt').exists():
        ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best,curve,start,steps,elapsed=(ck[k] for k in ('best','curve','epoch','steps','seconds'))
        rng.bit_generator.state=ck['rng'];aug.bit_generator.state=ck['aug'];del ck
        # best.pt/validation.npz are restored by the resume checkpoint, not epoch0.
        shutil.copy2(dest/'best_saved.pt',dest/'best.pt')
        shutil.copy2(dest/'validation_saved.npz',dest/'validation.npz')
    tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,EPOCHS):
        factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(EPOCHS-3)))
        for group,base in zip(opt.param_groups,(lr,MODULE_LR)): group['lr']=base*factor
        net.train();order=rng.permutation(train);total=0.;correct=0
        for first in range(0,len(order),BATCH):
            batch=order[first:first+BATCH];opt.zero_grad(set_to_none=True)
            code=int(aug.integers(4))+4*int(aug.random()<.5)
            for part in range(0,len(batch),MICRO):
                ids=batch[part:part+MICRO]
                loss,z=loss_for(net,inputs,ids,code)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {dest.name}/{epoch}')
                (loss*len(ids)/len(batch)).backward()
                total+=float(loss.detach())*len(ids)
                correct+=int((z.argmax(1)==torch.as_tensor(inputs.y[ids],device=DEVICE)).sum())
            torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
            opt.step();steps+=1
        score=predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
        key=(m['ba'],m['oa'],-m['nll'])
        row=dict(epoch=epoch+1,train_loss=total/len(train),train_oa=100*correct/len(train),
            validation=m,seconds=elapsed+time.perf_counter()-tick,backbone_lr=lr*factor)
        curve.append(row)
        if key>tuple(best['key']):
            best=dict(epoch=epoch+1,key=list(key),metrics=m)
            torch.save(net.state_dict(),dest/'best.pt')
            np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
        stop=epoch+1>=MIN_EPOCHS and epoch+1-best['epoch']>=PATIENCE
        print('VAL',scope,lr,seed,epoch+1,'OA/BA',round(m['oa'],3),round(m['ba'],3),
            'best',best['epoch'],'sec',round(row['seconds'],1),flush=True)
        write(ROOT/'status.json',dict(state='training',scope=scope,lr=lr,seed=seed,
            epoch=epoch+1,best=best,test_read=False,complete=False))
        write(dest/'curve.json',curve)
        if (epoch+1)%5==0 or stop or epoch+1==EPOCHS:
            shutil.copy2(dest/'best.pt',dest/'best_saved.pt')
            shutil.copy2(dest/'validation.npz',dest/'validation_saved.npz')
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,curve=curve,
                epoch=epoch+1,steps=steps,seconds=row['seconds'],rng=rng.bit_generator.state,
                aug=aug.bit_generator.state),dest/'resume.pt')
        if stop: break
    receipt=dict(**cfg,epochs=epoch+1,steps=steps,best=best,seconds=elapsed+time.perf_counter()-tick,
        stop_reason='validation_patience' if stop else 'max_epochs',
        trainable=sum(p.numel() for p in params.values()),peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        test_read=False)
    write(dest/'receipt.json',receipt)
    del net,opt,params;gc.collect();torch.cuda.empty_cache()
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--graph-smoke',action='store_true')
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    inputs=Inputs()
    for split in ('small','validation'): inputs.prepare_geometry(split)
    if args.prepare: return
    if args.smoke: return smoke(inputs)
    if args.graph_smoke: return graph_smoke(inputs)
    assert (ROOT/'smoke.json').exists()
    write(ROOT/'protocol.json',dict(scopes=SCOPES,lrs=LRS,seeds=SEEDS,
        template=configuration('search',0,17),selection='seed17 validation only; test selected3 afterfit'))
    candidates=[]
    for scope in SCOPES:
        for lr in LRS:
            dest=ROOT/f'scout_{scope}_lr{lr:g}_s17'
            r=fit(inputs,scope,lr,17,dest)
            candidates.append(dict(path=dest.name,scope=scope,lr=lr,best=r['best'],epochs=r['epochs']))
    selected=max(candidates,key=lambda r:tuple(r['best']['key']))
    write(ROOT/'selected.json',dict(**selected,candidates=candidates,seeds=SEEDS,test_read=False))
    print('SELECTED_VALIDATION',json.dumps(selected),flush=True)
    first=ROOT/'selected_s17';first.mkdir(exist_ok=True)
    for name in ('best.pt','config.json','receipt.json','curve.json','validation.npz'):
        shutil.copy2(ROOT/selected['path']/name,first/name)
    for seed in SEEDS[1:]: fit(inputs,selected['scope'],selected['lr'],seed,ROOT/f'selected_s{seed}')
    inputs.load_split('evaluation');inputs.prepare_geometry('evaluation')
    checks=[]
    for seed in SEEDS:
        dest=ROOT/f'selected_s{seed}';net=LiveFull(seed,selected['scope'])
        net.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True))
        ids=inputs.groups['validation'];z=predict(net,inputs,ids,'validation')
        old=np.load(dest/'validation.npz')['score'];err=float(abs(z-old).max())
        assert err<1e-3 and np.array_equal(z.argmax(1),old.argmax(1)),err
        ids=inputs.groups['evaluation'];z=predict(net,inputs,ids,'evaluation')
        np.savez_compressed(dest/'evaluation.npz',ids=ids,score=z,label=inputs.y[ids])
        m=metrics(inputs.y[ids],z);write(dest/'evaluation.json',m)
        checks.append(dict(seed=seed,validation_error=err))
        print('TEST_SELECTED',seed,json.dumps(m),flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'checkpoint_replays.json',checks)
    write(ROOT/'status.json',dict(state='training_and_test_complete',complete=True,analysis_pending=True))


if __name__=='__main__': main()
