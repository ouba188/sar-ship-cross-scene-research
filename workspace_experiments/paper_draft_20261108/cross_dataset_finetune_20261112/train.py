"""Source-only live-chip fine-tuning for matched bidirectional dataset transfer."""
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
OLD = PAPER/'cross_dataset_retrained_20261111'
sys.path.insert(0, str(PAPER/'full_finetune_20261111'))
spec = importlib.util.spec_from_file_location('transfer_full_recipe', PAPER/'full_finetune_20261111/train.py')
ft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ft
spec.loader.exec_module(ft)
np, torch, nn, F = ft.np, ft.torch, ft.nn, ft.F
e = ft.cost.e
write, metrics = ft.write, ft.metrics
DIRECTIONS = ('oss2_to_ports24', 'ports24_to_oss2')
SEEDS, ARMS = (17, 29, 43), ('plain', 'full')
LR = 1e-4


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Inputs:
    prepare_geometry = ft.Inputs.prepare_geometry

    def __init__(self, direction):
        self.direction = direction
        source = OLD/direction
        self.rows = read(source/'manifest.json')
        self.groups = {k:np.asarray(v, np.int64) for k,v in read(source/'groups.json').items()}
        self.source_count = len(self.rows)
        targets = read(source/'target_manifest.json')
        self.rows += targets
        self.groups['evaluation'] = np.arange(self.source_count, len(self.rows))
        self.y = np.array([r['label'] for r in self.rows])
        self.weights = e.weights(self.rows, self.groups['small'])
        self.reference_rows = [self.rows[i] for i in self.groups['references']]
        self.cache = ROOT/direction/'geometry_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.geometry = {}
        self.lookup = {s:{int(i):j for j,i in enumerate(ids)} for s,ids in self.groups.items()}
        self.arrays = {}
        for dataset, folder, valid in (
            ('oss2', PAPER/'recent_baselines_20261111/native_cache', 'footprint'),
            ('ports24', PAPER/'ports24_sync_20261112/native_cache', 'valid')):
            self.arrays[dataset] = {k:np.load(folder/(name+'.npy'), mmap_mode='r')
                for k,name in (('pixels','pixels'), ('valid',valid), ('available','available'))}
        assert set(self.groups['references']) <= set(self.groups['small'])
        for a,b in (('small','validation'),('small','evaluation'),('validation','evaluation')):
            for field in ('mmsi','pass_group'):
                assert not {self.rows[i][field] for i in self.groups[a]} & {self.rows[i][field] for i in self.groups[b]}
        assert len(self.reference_rows) == 32

    def data(self, ids, split, code=0):
        rows = [self.rows[i] for i in ids]
        dataset = rows[0]['dataset']
        assert all(r['dataset'] == dataset for r in rows)
        original = np.asarray([r['original_index'] for r in rows])
        arrays = self.arrays[dataset]
        data = {k:torch.as_tensor(np.array(v[original],copy=True), device=ft.DEVICE).float()
                for k,v in arrays.items()}
        data['images'] = data.pop('pixels').unsqueeze(2)/255
        data['valid'] = data['valid'].unsqueeze(2)
        for k in ('images','valid'):
            data[k] = torch.rot90(data[k],code%4,(-2,-1))
            if code//4: data[k] = data[k].flip(-1)
        ix = np.array([self.lookup[split][int(i)] for i in ids])
        data.update({k:torch.as_tensor(np.array(v[ix,code],copy=True),device=ft.DEVICE)
                     for k,v in self.geometry[split].items()})
        return data


class LiveModel(ft.LiveFull):
    def __init__(self, direction, arm, seed):
        net = e.Network(arm, seed, ft.r50.WEIGHTS, 4)
        ft.cost.ChipPath.__init__(self, net)
        self.checkpoint = OLD/direction/f'{arm}_s{seed}_lr1/readout/final.pt'
        net.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True,mmap=True),strict=True)
        net.phase,net.fixed_delta = 3,None
        self.requires_grad_(False)
        net.base.backbone.model.requires_grad_(True)
        net.base.backbone.model.fc.requires_grad_(False)
        net.classifier().requires_grad_(True)
        if arm == 'full':
            for module in (net.m1,net.base.internal_m2,net.base.deep_output,net.matcher,
                           net.guide.value,net.guide.output): module.requires_grad_(True)
        self.to(ft.DEVICE).eval()
        self.transport_graph = None

    def accelerate(self):
        if self.net.use3: super().accelerate()

    def finish(self,a,c,query_rows,reference,reference_rows,m3_on=True):
        n = len(query_rows)
        table = dict(a)
        if self.net.use3 and m3_on:
            for key in ('desc','mass','available'):
                table[key] = torch.cat((a[key],reference[key]),0)
        z,aux = self.net(table,c,query_rows+reference_rows,np.arange(n),
                        np.arange(n,n+len(reference_rows)),m3_on=m3_on)
        self.net.readout_batch = None
        return z,aux


def reference(net,inputs):
    return net.reference(inputs.data(inputs.groups['references'],'small')) if net.net.use3 else {}


@torch.no_grad()
def predict(net,inputs,ids,split):
    net.eval()
    ref = reference(net,inputs)
    scores = []
    for first in range(0,len(ids),ft.MICRO):
        part = ids[first:first+ft.MICRO]
        z,_ = net(inputs.data(part,split),[inputs.rows[i] for i in part],ref,inputs.reference_rows)
        scores.append(z.cpu().numpy())
    return np.concatenate(scores)


def loss_for(net,inputs,ids,code):
    ref = reference(net,inputs)
    data = inputs.data(ids,'small',code)
    z,aux = net(data,[inputs.rows[i] for i in ids],ref,inputs.reference_rows)
    y = torch.as_tensor(inputs.y[ids],device=ft.DEVICE)
    w = torch.as_tensor(inputs.weights[ids],device=ft.DEVICE)
    loss = (F.cross_entropy(z.float(),y,reduction='none')*w).mean()
    if aux is not None:
        ac = F.cross_entropy(aux.float().flatten(0,1),y[:,None].expand(-1,2).flatten(),
                             reduction='none').reshape(len(ids),2)
        av = data['available']
        loss = loss+.02*((ac*av).sum(1)/av.sum(1).clamp_min(1)*w).mean()
    return loss,z


def smoke(inputs):
    dest = ROOT/inputs.direction/'smoke.json'
    if dest.exists(): return
    report = {}
    ids = inputs.groups['validation'][:12]
    for arm in ARMS:
        net = LiveModel(inputs.direction,arm,17)
        z = predict(net,inputs,ids,'validation')
        prior = np.load(OLD/inputs.direction/f'{arm}_s17_lr1/readout/validation.npz')
        lookup = {int(i):j for j,i in enumerate(prior['ids'])}
        expected = prior['score'][[lookup[int(i)] for i in ids]]
        error = float(np.max(abs(z-expected)))
        assert error<1e-3 and np.array_equal(z.argmax(1),expected.argmax(1)),(arm,error)
        net.train()
        loss,z = loss_for(net,inputs,inputs.groups['small'][:12],3)
        loss.backward()
        grads = {n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}
        assert all(np.isfinite(v) for v in grads.values())
        required = ['net.base.backbone.model.conv1.']
        required += ['net.m1.','net.base.internal_m2.','net.matcher.','net.guide.output.'] if arm=='full' else ['net.head.']
        for prefix in required:
            assert any(n.startswith(prefix) and v>0 for n,v in grads.items()),prefix
        graph = {}
        if arm=='full':
            original = {n:p.grad.detach().clone() for n,p in net.named_parameters() if p.grad is not None}
            net.zero_grad(set_to_none=True)
            net.accelerate()
            other,z2 = loss_for(net,inputs,inputs.groups['small'][:12],3)
            other.backward()
            graph = dict(logit_error=float((z-z2).abs().max()),
                gradient_error=max(float((p.grad-original[n]).abs().max()) for n,p in net.named_parameters() if n in original))
            assert graph['logit_error']<1e-5 and graph['gradient_error']<1e-4,graph
            del original,z2,other
        report[arm] = dict(replay_error=error,loss=float(loss.detach()),gradients=grads,graph=graph)
        print('SMOKE_OK',inputs.direction,arm,error,flush=True)
        del net,z,loss;gc.collect();torch.cuda.empty_cache()
    write(dest,report)


def fit(inputs,arm,seed):
    dest = ROOT/inputs.direction/f'{arm}_s{seed}'
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists(): return
    cfg = ft.configuration('full',LR,seed)
    cfg.update(direction=inputs.direction,arm=arm,
        initialization=str(OLD/inputs.direction/f'{arm}_s{seed}_lr1/readout/final.pt'),
        loss='class/vessel balanced CE'+(' + .02 M1 auxiliary CE' if arm=='full' else ''),
        target_training_images=0,target_training_labels=0,hyperparameter_search=False,
        train_records=len(inputs.groups['small']),validation_records=len(inputs.groups['validation']),
        target_records=len(inputs.groups['evaluation']),references_per_class=8 if arm=='full' else 0,
        input_policy='unchanged native160 and polarization availability; live backbone, no neural caches')
    write(dest/'config.json',cfg)
    net = LiveModel(inputs.direction,arm,seed)
    net.accelerate()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=LR,weight_decay=.01)
    train,val = inputs.groups['small'],inputs.groups['validation']
    rng,aug = np.random.default_rng(seed+20261111),np.random.default_rng(seed+20261112)
    if (dest/'resume.pt').exists():
        ck = torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best,curve,start,steps,elapsed = (ck[k] for k in ('best','curve','epoch','steps','seconds'))
        rng.bit_generator.state=ck['rng'];aug.bit_generator.state=ck['aug']
        del ck
    else:
        score = predict(net,inputs,val,'validation')
        prior = np.load(OLD/inputs.direction/f'{arm}_s{seed}_lr1/readout/validation.npz')
        lookup={int(i):j for j,i in enumerate(prior['ids'])}
        expected=prior['score'][[lookup[int(i)] for i in val]]
        error=float(abs(score-expected).max())
        assert error<1e-3 and np.array_equal(score.argmax(1),expected.argmax(1)),error
        m=metrics(inputs.y[val],score)
        write(dest/'initial_replay.json',dict(error=error,validation=m))
        best=dict(epoch=0,key=[m['ba'],m['oa'],-m['nll']],metrics=m)
        torch.save(net.state_dict(),dest/'best.pt')
        np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
        curve=[];start=steps=0;elapsed=0.
    tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    stop=False
    for epoch in range(start,ft.EPOCHS):
        factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(ft.EPOCHS-3)))
        opt.param_groups[0]['lr']=LR*factor
        net.train();order=rng.permutation(train);total=0.
        for first in range(0,len(order),ft.BATCH):
            batch=order[first:first+ft.BATCH];opt.zero_grad(set_to_none=True)
            code=int(aug.integers(4))+4*int(aug.random()<.5)
            for start_part in range(0,len(batch),ft.MICRO):
                ids=batch[start_part:start_part+ft.MICRO]
                loss,z=loss_for(net,inputs,ids,code)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {dest}/{epoch}')
                (loss*len(ids)/len(batch)).backward()
                total+=float(loss.detach())*len(ids)
            torch.nn.utils.clip_grad_norm_(params,5.)
            opt.step();steps+=1
        score=predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
        key=[m['ba'],m['oa'],-m['nll']]
        row=dict(epoch=epoch+1,train_loss=total/len(train),validation=m,
                 seconds=elapsed+time.perf_counter()-tick,lr=LR*factor)
        curve.append(row)
        if tuple(key)>tuple(best['key']):
            best=dict(epoch=epoch+1,key=key,metrics=m)
            torch.save(net.state_dict(),dest/'best.pt')
            np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
        stop=epoch+1>=ft.MIN_EPOCHS and epoch+1-best['epoch']>=ft.PATIENCE
        write(dest/'curve.json',curve)
        write(ROOT/'status.json',dict(complete=False,state='source_training',direction=inputs.direction,
            arm=arm,seed=seed,epoch=epoch+1,best=best,target_read=False))
        print('VAL',inputs.direction,arm,seed,epoch+1,round(m['oa'],3),round(m['ba'],3),
              'best',best['epoch'],'sec',round(row['seconds'],1),flush=True)
        # Save each epoch: small source queues make restart loss avoidable.
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,curve=curve,
            epoch=epoch+1,steps=steps,seconds=row['seconds'],rng=rng.bit_generator.state,
            aug=aug.bit_generator.state),dest/'resume.pt')
        if stop: break
    write(dest/'receipt.json',dict(**cfg,epochs=len(curve),steps=steps,best=best,
        seconds=elapsed+time.perf_counter()-tick,stop_reason='validation_patience' if stop else 'max_epochs',
        trainable=sum(p.numel() for p in params),peak_gib=torch.cuda.max_memory_allocated()/1024**3,test_read=False))
    del net,opt,params;gc.collect();torch.cuda.empty_cache()


def evaluate(inputs,arm,seed):
    dest=ROOT/inputs.direction/f'{arm}_s{seed}'
    if (dest/'evaluation.npz').exists():return
    assert (dest/'receipt.json').exists()
    net=LiveModel(inputs.direction,arm,seed)
    net.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True,mmap=True))
    net.accelerate()
    val=inputs.groups['validation']
    actual=predict(net,inputs,val,'validation')
    old=np.load(dest/'validation.npz')['score']
    error=float(abs(actual-old).max())
    assert error<1e-3 and np.array_equal(actual.argmax(1),old.argmax(1)),error
    ids=inputs.groups['evaluation'];score=predict(net,inputs,ids,'evaluation')
    saved_ids=np.arange(32,32+len(ids))
    np.savez_compressed(dest/'evaluation.npz',ids=saved_ids,score=score,label=inputs.y[ids])
    write(dest/'evaluation.json',dict(**metrics(inputs.y[ids],score),validation_replay_error=error))
    print('TARGET',inputs.direction,arm,seed,json.dumps(metrics(inputs.y[ids],score)),flush=True)
    del net;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','smoke','train','evaluate','all'))
    parser.add_argument('--directions',nargs='+',choices=DIRECTIONS,default=list(DIRECTIONS))
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    args=parser.parse_args()
    e.task.current.setup(ft.DEVICE)
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    write(ROOT/'protocol.json',dict(directions=DIRECTIONS,seeds=SEEDS,arms=ARMS,
        source_protocol=str(OLD),recipe=ft.configuration('full',LR,17),
        task='Four-class bidirectional source-only transfer; matched 644 source records, not full-source training',
        target_data_used_in_training=False,target_labels_for_selection=False,
        selection='source validation BA then OA then -NLL, epoch0 eligible',
        reference='32 source records; original legal-pair exclusions retained',
        reused='All completed source-only stages reused; no within-dataset trained weights'))
    for direction in args.directions:
        inputs=Inputs(direction)
        for split in ('small','validation'):inputs.prepare_geometry(split)
        if args.action=='prepare':continue
        if args.action in ('smoke','train','all'):smoke(inputs)
        if args.action in ('train','all'):
            for seed in args.seeds:
                for arm in ARMS:fit(inputs,arm,seed)
        del inputs;gc.collect()
    if args.action in ('evaluate','all'):
        # Freeze all requested source runs before any new target forward.
        for direction in args.directions:
            for seed in args.seeds:
                for arm in ARMS:assert (ROOT/direction/f'{arm}_s{seed}/receipt.json').exists()
        for direction in args.directions:
            inputs=Inputs(direction)
            for split in ('small','validation','evaluation'):inputs.prepare_geometry(split)
            for seed in args.seeds:
                for arm in ARMS:evaluate(inputs,arm,seed)
            del inputs;gc.collect()
        write(ROOT/'status.json',dict(complete=True,state='requested_training_and_evaluation_complete',
            directions=args.directions,seeds=args.seeds,analysis_pending=True))


if __name__=='__main__':main()
