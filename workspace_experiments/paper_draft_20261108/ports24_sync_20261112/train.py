"""Live-chip full-backbone fine-tuning on the frozen full-scale 24-port split."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
sys.path.insert(0, str(PAPER/'full_finetune_20261111'))
spec = importlib.util.spec_from_file_location('oss2_full_recipe', PAPER/'full_finetune_20261111/train.py')
ft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ft
spec.loader.exec_module(ft)
np, torch, nn, F = ft.np, ft.torch, ft.nn, ft.F
e, raw = ft.cost.e, ft.cost.raw
write = ft.write
SEEDS = (17, 29, 43)
LR = 1e-4
CACHE = ROOT/'native_cache'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(y, z):
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (y, z.argmax(1)), 1)
    count = cm.sum(1)
    prob = np.exp(z-z.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return dict(oa=float(100*np.mean(z.argmax(1)==y)),
        ba=float(100*np.mean(cm.diagonal()[count>0]/count[count>0])),
        macro_f1=float(100*np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+count, 1))),
        nll=float(-np.log(prob[np.arange(len(y)), y].clip(1e-12)).mean()),
        present_classes=int((count>0).sum()), confusion=cm.tolist())


def geometry(native):
    readout = raw.tokens.cell_mass(native, 8)/ft._cell_area(20)
    writer = raw.tokens.cell_mass(native, 16)/ft._cell_area(10)
    writer /= np.maximum(writer.sum(0, keepdims=True), 1)
    t = torch.from_numpy(native).float()[None]
    return dict(read=readout, write=writer, mass=readout.sum((-2, -1)),
        slots=raw.grid_occupancy(t, (10, 10))[0].numpy(),
        local_slots=raw.grid_occupancy(t, (20, 20))[0].numpy())


class Inputs:
    def __init__(self):
        self.rows = read(STORE/'manifest.json')
        self.groups = {k: np.asarray(v, np.int64) for k,v in read(STORE/'groups.json').items()}
        self.groups['small'] = np.concatenate((self.groups['source_fit'], self.groups['source_adapt']))
        self.y = np.array([r['label'] for r in self.rows])
        self.weights = e.weights(self.rows, self.groups['small'])
        self.reference_rows = [self.rows[i] for i in self.groups['references']]
        self.cache = CACHE
        self.arrays = {}
        assert [len(self.groups[k]) for k in ('small','validation','evaluation','references')] == [20653,160,3075,16]
        assert set(self.groups['references']) <= set(self.groups['small'])
        for a,b in (('small','validation'),('small','evaluation'),('validation','evaluation')):
            for field in ('id','mmsi'):
                assert not {self.rows[i][field] for i in self.groups[a]} & {self.rows[i][field] for i in self.groups[b]}
            assert not {'_'.join(self.rows[i]['pass_group'].split('_')[:2]) for i in self.groups[a]} & {
                '_'.join(self.rows[i]['pass_group'].split('_')[:2]) for i in self.groups[b]}

    def prepare(self):
        self.cache.mkdir(exist_ok=True)
        shapes = dict(pixels=(2,160,160), valid=(2,160,160), available=(2,),
            slots=(2,20,10,10), local_slots=(2,20,20,20), read=(2,20,20,20),
            write=(2,20,10,10), mass=(2,20))
        if not (self.cache/'complete.json').exists():
            progress = self.cache/'progress.json'
            first = read(progress)['next'] if progress.exists() else 0
            stores = {k: np.lib.format.open_memmap(self.cache/(k+'.npy'), mode='r+' if first else 'w+',
                dtype=np.uint8 if k in ('pixels','valid','available') else np.float32,
                shape=(len(self.rows),)+shape) for k,shape in shapes.items()}
            tick = time.perf_counter()
            for i in range(first, len(self.rows)):
                for value in stores.values(): value[i] = 0
                for p,v in enumerate(e.task.data.prepare(self.rows[i])):
                    if v is None or not v['valid_target']: continue
                    stores['pixels'][i,p] = v['pixels']
                    stores['valid'][i,p] = v['footprint']
                    stores['available'][i,p] = 1
                    native = raw.tokens.slots(v['geometry'], v['footprint'])
                    for k,value in geometry(native).items(): stores[k][i,p] = value
                if (i+1)%500 == 0 or i+1 == len(self.rows):
                    for v in stores.values(): v.flush()
                    write(progress, dict(next=i+1))
                    print('CACHE', i+1, len(self.rows), round(time.perf_counter()-tick,1), flush=True)
            write(self.cache/'complete.json', dict(n=len(self.rows), native160=True,
                note='CPU memory-mapped images and geometric masks; no cached neural features'))
            del stores
        self.arrays = {k:np.load(self.cache/(k+'.npy'), mmap_mode='r') for k in shapes}

    def data(self, ids, split=None, code=0):
        data = {k:torch.as_tensor(np.array(v[ids], copy=True), device=ft.DEVICE).float()
                for k,v in self.arrays.items()}
        data['images'] = data.pop('pixels').unsqueeze(2)/255
        data['valid'] = data['valid'].unsqueeze(2)
        if code:
            for k in ('images','valid'):
                data[k] = torch.rot90(data[k], code%4, (-2,-1))
                if code//4: data[k] = data[k].flip(-1)
            # Origin-zero feature cells are not symmetric at the boundary.
            # Re-integrate transformed native masks, not rotated coarse weights.
            masks = np.zeros((len(ids),2,20,160,160),np.float32)
            for j,idx in enumerate(ids):
                for p,v in enumerate(e.task.data.prepare(self.rows[idx])):
                    if v is None or not v['valid_target']: continue
                    native=raw.tokens.slots(v['geometry'],v['footprint'])
                    masks[j,p]=ft.transform(native,code)
            masks=torch.as_tensor(masks,device=ft.DEVICE).flatten(0,1)
            local=raw.grid_occupancy(masks,(20,20)).reshape(len(ids),2,20,20,20)
            slots=raw.grid_occupancy(masks,(10,10)).reshape(len(ids),2,20,10,10)
            data.update(slots=slots,local_slots=local,read=local,mass=local.sum((-2,-1)),
                        write=slots/slots.sum(2,keepdim=True).clamp_min(1))
        return data


class LiveFull(ft.LiveFull):
    def __init__(self, seed, scope='full'):
        net = e.Network('full', seed, ft.r50.WEIGHTS, 8)
        ft.cost.ChipPath.__init__(self, net)
        checkpoint = STORE/f'full_s{seed}_lr1/readout/final.pt'
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        net.phase, net.fixed_delta = 3, None
        self.requires_grad_(False)
        net.base.backbone.model.requires_grad_(True)
        net.base.backbone.model.fc.requires_grad_(False)
        for module in (net.m1,net.base.internal_m2,net.base.deep_output,net.matcher,
                       net.guide.value,net.guide.output,net.guide.head): module.requires_grad_(True)
        self.to(ft.DEVICE).eval()
        self.transport_graph = None


def smoke(inputs):
    errors = []
    shortcut_errors = []
    for i in inputs.groups['small'][:3]:
        for v in e.task.data.prepare(inputs.rows[i]):
            if v is None or not v['valid_target']: continue
            native = raw.tokens.slots(v['geometry'],v['footprint'])
            original = geometry(native)
            for code in range(8):
                actual = geometry(np.ascontiguousarray(ft.transform(native,code)))
                for k in original:
                    expected = original[k] if k=='mass' else ft.transform(original[k],code)
                    shortcut_errors.append(float(np.max(np.abs(expected-actual[k]))))
    # Check actual augmentation output against independent per-view preparation.
    for code in range(8):
        ids=inputs.groups['small'][:3]
        data=inputs.data(ids,'small',code)
        for j,i in enumerate(ids):
            for p,v in enumerate(e.task.data.prepare(inputs.rows[i])):
                if v is None or not v['valid_target']:continue
                target=geometry(np.ascontiguousarray(ft.transform(raw.tokens.slots(v['geometry'],v['footprint']),code)))
                errors.extend(float(np.max(np.abs(data[k][j,p].cpu().numpy()-value))) for k,value in target.items())
    assert max(errors)<1e-4,max(errors)
    net = LiveFull(17)
    ids = inputs.groups['validation'][:12]
    z = ft.predict(net,inputs,ids,'validation')
    old = np.load(STORE/'full_s17_lr1/readout/validation.npz')['score'][:12]
    err = float(np.max(np.abs(z-old)))
    assert err<1e-3 and np.array_equal(z.argmax(1),old.argmax(1)),err
    net.train()
    ids = inputs.groups['small'][:12]
    loss,z = ft.loss_for(net,inputs,ids,3)
    loss.backward()
    grads = {n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}
    assert all(np.isfinite(v) for v in grads.values())
    for prefix in ('net.m1.','net.base.internal_m2.','net.matcher.','net.guide.output.','net.base.backbone.model.conv1.'):
        assert any(n.startswith(prefix) and v>0 for n,v in grads.items()),prefix
    original = {n:p.grad.detach().clone() for n,p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True);net.accelerate()
    loss2,z2 = ft.loss_for(net,inputs,ids,3);loss2.backward()
    graph_error = float((z-z2).abs().max())
    grad_error = max(float((p.grad-original[n]).abs().max()) for n,p in net.named_parameters() if n in original)
    assert graph_error<1e-5 and grad_error<1e-4,(graph_error,grad_error)
    write(ROOT/'smoke.json',dict(replay_error=err,geometry_transform_error=max(errors),
        rejected_coarse_rotation_error=max(shortcut_errors),
        graph_error=graph_error,gradient_error=grad_error,gradients=grads))
    print('SMOKE_OK',err,max(errors),graph_error,grad_error,flush=True)
    del net,original;gc.collect();torch.cuda.empty_cache()


def fit(inputs,seed):
    dest = ROOT/f'selected_s{seed}'
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists(): return
    cfg = ft.configuration('full',LR,seed)
    cfg.update(dataset='24ports',references='16 original fixed source records, live shared-prefix features',
        initialization=str(STORE/f'full_s{seed}_lr1/readout/final.pt'),
        input_policy='original verified input availability; recovery tested separately',
        train_records=20653, validation_records=160, test_records=3075,
        hyperparameter_search=False,recipe='OSS2 selected full-backbone recipe transferred, not OSS2 weights')
    write(dest/'config.json',cfg)
    net = LiveFull(seed);net.accelerate()
    params = {n:p for n,p in net.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params.values(),lr=LR,weight_decay=.01)
    train,val = inputs.groups['small'],inputs.groups['validation']
    rng=np.random.default_rng(seed+20261111);aug=np.random.default_rng(seed+20261112)
    score=ft.predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
    expected=np.load(STORE/f'full_s{seed}_lr1/readout/validation.npz')['score']
    error=float(np.max(np.abs(expected-score)))
    assert error<1e-3 and np.array_equal(score.argmax(1),expected.argmax(1)),error
    best=dict(epoch=0,key=[m['ba'],m['oa'],-m['nll']],metrics=m)
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
    write(dest/'initial_replay.json',dict(error=error,validation=m))
    curve=[];steps=0;elapsed=0.;start=0
    if (dest/'resume.pt').exists():
        ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best,curve,start,steps,elapsed=(ck[k] for k in ('best','curve','epoch','steps','seconds'))
        rng.bit_generator.state=ck['rng'];aug.bit_generator.state=ck['aug']
        torch.save(ck['best_model'],dest/'best.pt')
        np.savez_compressed(dest/'validation.npz',ids=val,score=ck['best_score'],label=inputs.y[val])
        del ck
        print('RESUME',seed,'completed_epochs',start,'saved_steps',steps,'next_epoch',start+1,flush=True)
    tick=time.perf_counter()
    for epoch in range(start,ft.EPOCHS):
        factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(ft.EPOCHS-3)))
        for group in opt.param_groups:group['lr']=LR*factor
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
            if steps%200==0:
                write(ROOT/'status.json',dict(state='training',seed=seed,epoch=epoch+1,
                    seen_in_epoch=first+len(batch),steps=steps,complete=False))
                print('STEP',seed,epoch+1,first+len(batch),len(order),steps,round(time.perf_counter()-tick,1),flush=True)
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
        print('VAL',seed,epoch+1,'OA/BA',round(m['oa'],3),round(m['ba'],3),'best',best['epoch'],round(row['seconds'],1),flush=True)
        write(dest/'curve.json',curve)
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,curve=curve,
            epoch=epoch+1,steps=steps,seconds=row['seconds'],rng=rng.bit_generator.state,
            aug=aug.bit_generator.state,best_model=torch.load(dest/'best.pt',map_location='cpu',weights_only=True),
            best_score=np.load(dest/'validation.npz')['score']),dest/'resume.pt')
        if stop:break
    write(dest/'receipt.json',dict(**cfg,epochs=epoch+1,steps=steps,best=best,
        seconds=elapsed+time.perf_counter()-tick,stop_reason='validation_patience' if stop else 'max_epochs',
        trainable=sum(p.numel() for p in params.values()),test_read=False))
    del net,opt,params;gc.collect();torch.cuda.empty_cache()


def evaluate(inputs,seeds):
    for seed in seeds:
        dest=ROOT/f'selected_s{seed}'
        net=LiveFull(seed)
        net.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True))
        for split in ('validation','evaluation'):
            ids=inputs.groups[split];score=ft.predict(net,inputs,ids,split)
            if split=='validation':
                saved=np.load(dest/'validation.npz')['score']
                err=float(np.max(np.abs(saved-score)));assert err<1e-3,err
                write(dest/'replay.json',dict(validation_error=err))
            np.savez_compressed(dest/(split+'.npz'),ids=ids,score=score,label=inputs.y[ids])
            m=metrics(inputs.y[ids],score);write(dest/(split+'.json'),m)
            print('FINAL',seed,split,json.dumps(m),flush=True)
        del net;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','smoke','train','evaluate','all'])
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=parser.parse_args()
    inputs=Inputs();inputs.prepare()
    if args.action in ('smoke','all'):smoke(inputs)
    if args.action in ('train','all'):
        assert (ROOT/'smoke.json').exists()
        for seed in args.seeds:fit(inputs,seed)
    if args.action in ('evaluate','all'):
        evaluate(inputs,args.seeds)
        write(ROOT/'status.json',dict(state='training_and_test_complete',seeds=args.seeds,
            complete=True,analysis_pending=True,formal_model_changed=False))
