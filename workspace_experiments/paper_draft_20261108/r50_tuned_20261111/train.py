"""Validation-selected ordinary dual-polarization R50, with live image forwards."""
from pathlib import Path
import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK/'sar_m123_standalone'))
from sar_m123.pool_replacement import regional_mean

WEIGHTS = ROOT.parent/'oss2_polarization_training_20261111/resnet50-11ad3fa6.pth'
CACHE = ROOT.parent/'recent_baselines_20261111/native_cache'
SCOPES = ('local', 'late', 'full')
LRS = (1e-5, 1e-4)
SEEDS = (17, 29, 43)
BATCH = 24
EPOCHS = 60
MIN_EPOCHS = 30
PATIENCE = 15
DEVICE = 'cuda'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(y, z):
    pred = z.argmax(1)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, pred), 1)
    prob = np.exp(z-z.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return dict(oa=float(100*np.mean(pred == y)),
        ba=float(100*np.mean(cm.diagonal()/np.maximum(cm.sum(1), 1))),
        macro_f1=float(100*np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1), 1))),
        nll=float(-np.log(prob[np.arange(len(y)), y].clip(1e-12)).mean()),
        confusion=cm.tolist())


class Inputs:
    def __init__(self):
        self.rows = json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
        self.groups = {k: np.array(v, dtype=np.int64) for k, v in
            json.loads((WORK/'m1_coverage_20261101/groups.json').read_text()).items()}
        self.y = np.array([r['label'] for r in self.rows])
        self.pixels = np.load(CACHE/'pixels.npy', mmap_mode='r')
        self.footprints = np.load(CACHE/'footprint.npy', mmap_mode='r')
        self.available = np.load(CACHE/'available.npy')
        self.loaded = set()
        self.resident = {}
        for split in ('small', 'validation'):
            self.load_split(split)
        train = self.groups['small']
        frame = pd.DataFrame([self.rows[i] for i in train])
        w = 1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*
               frame.groupby('mmsi').mmsi.transform('size').to_numpy())
        self.weights = np.zeros(len(self.rows), np.float32)
        self.weights[train] = w/w.mean()
        for left, right in (('small', 'validation'), ('small', 'evaluation'), ('validation', 'evaluation')):
            for field in ('id', 'mmsi', 'pass_group'):
                assert not {self.rows[i][field] for i in self.groups[left]} & {
                    self.rows[i][field] for i in self.groups[right]}
        assert [len(self.groups[k]) for k in ('small','validation','evaluation')] == [719,228,378]

    def load_split(self, split):
        if split in self.loaded:
            return
        ids = self.groups[split]
        self.resident[split] = (dict(zip(ids.tolist(), range(len(ids)))),
            torch.as_tensor(np.array(self.pixels[ids]), device=DEVICE),
            torch.as_tensor(np.array(self.footprints[ids]), device=DEVICE),
            torch.as_tensor(self.available[ids], device=DEVICE))
        self.loaded.add(split)

    def batch(self, ids, split, augment=None):
        lookup, pixels, footprint, available = self.resident[split]
        ix = torch.tensor([lookup[int(i)] for i in ids], device=DEVICE)
        x = pixels[ix].float().unsqueeze(2)/255
        mask = footprint[ix].float().unsqueeze(2)
        if augment is not None:
            k = int(augment.integers(4))
            x, mask = torch.rot90(x, k, (-2,-1)), torch.rot90(mask, k, (-2,-1))
            if augment.random() < .5:
                x, mask = x.flip(-1), mask.flip(-1)
        return x, mask, available[ix]


class PlainR50(nn.Module):
    def __init__(self, seed, scope):
        super().__init__()
        torch.manual_seed(seed)
        self.backbone = resnet50(weights=None)
        self.backbone.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(4098, 4)
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])
        self.backbone.requires_grad_(False)
        if scope == 'local':
            self.backbone.layer2[3].conv3.requires_grad_(True)
            for k in (1,2,3):
                getattr(self.backbone.layer3[5], f'conv{k}').requires_grad_(True)
        elif scope == 'late':
            self.backbone.layer3.requires_grad_(True)
            self.backbone.layer4.requires_grad_(True)
        elif scope == 'full':
            self.backbone.requires_grad_(True)
        else:
            raise ValueError(scope)
        self.to(DEVICE)

    def train(self, mode=True):
        super().train(mode)
        # Small paired SAR batches and unavailable views must not drift BN statistics.
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, valid, available):
        n = len(x)
        z = (x.flatten(0,1).expand(-1,3,-1,-1)-self.mean)/self.std
        b = self.backbone
        z = b.maxpool(b.relu(b.bn1(b.conv1(z))))
        z = b.layer4(b.layer3(b.layer2(b.layer1(z))))
        with torch.autocast('cuda', enabled=False):
            v = valid.flatten(0,1).float()
            pooled = regional_mean(z.float(), v, v)[0][:,0].reshape(n,2,2048)
            features = torch.cat(((pooled*available[...,None]).flatten(1), available), 1)
            return self.head(features)


@torch.no_grad()
def predict(net, inputs, ids, split):
    net.eval()
    return np.concatenate([net(*inputs.batch(ids[i:i+BATCH], split)).cpu().numpy()
                           for i in range(0,len(ids),BATCH)])


def configuration(scope, lr, seed):
    return dict(scope=scope, lr=lr, seed=seed, head_lr=.001, weight_decay=.01,
        batch=BATCH, epochs_max=EPOCHS, epochs_min=MIN_EPOCHS, patience=PATIENCE,
        optimizer='AdamW', schedule='2 epoch warmup then cosine to 0.05 at epoch60',
        selection='validation BA then OA then lower NLL',
        normalization='Full-compatible /255, gray replicated RGB, ImageNet mean/std',
        augmentation='paired 90-degree rotations and horizontal flip; native scale unchanged',
        loss='class and vessel balanced CE, same source-weight rule as Full',
        bn='all running statistics frozen; affine trainable only within late/full scopes',
        input='same native160; valid-area global pooling; availability-masked concatenation',
        backbone='shared ResNet50 ImageNet V2', modules='none', references='none',
        precision='bfloat16 convolution training; float32 pooling/head and evaluation')


def smoke(inputs):
    result = {}
    ids = inputs.groups['small'][:BATCH]
    batch = inputs.batch(ids, 'small')
    initial = None
    for scope in SCOPES:
        net = PlainR50(17, scope).eval()
        with torch.no_grad():
            z = net(*batch)
            if initial is None:
                initial = z.detach().clone()
            err = float((initial-z).abs().max())
            assert err == 0, err
            x, mask, av = [v.clone() for v in batch]
            av[:,1] = 0
            clean = net(x, mask, av)
            x[:,1] = torch.rand_like(x[:,1])
            missing = float((clean-net(x,mask,av)).abs().max())
            assert missing == 0, missing
        frozen = {n:p.detach().clone() for n,p in net.named_parameters()
                  if n in ('backbone.conv1.weight','backbone.layer4.2.conv3.weight')}
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(net(*batch), torch.tensor(inputs.y[ids], device=DEVICE))
        loss.backward()
        gradients = {n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None}
        assert gradients and all(np.isfinite(v) for v in gradients.values())
        assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
        opt.step()
        updates = {n:float((dict(net.named_parameters())[n]-v).abs().max()) for n,v in frozen.items()}
        assert (updates['backbone.conv1.weight'] > 0) == (scope == 'full')
        assert (updates['backbone.layer4.2.conv3.weight'] > 0) == (scope != 'local')
        result[scope] = dict(initial_error=err, missing_branch_error=missing,
            trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
            parameters=sum(p.numel() for p in net.parameters()), probe_updates=updates,
            gradient_layers=len(gradients), finite_gradients=True,
            peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        del net, opt, frozen, gradients
        gc.collect();torch.cuda.empty_cache()
    write(ROOT/'smoke.json', result)
    print('SMOKE', json.dumps(result), flush=True)


def fit(inputs, scope, lr, seed, dest):
    dest.mkdir(exist_ok=True)
    if (dest/'receipt.json').exists():
        return json.loads((dest/'receipt.json').read_text())
    cfg = configuration(scope,lr,seed)
    write(dest/'config.json',cfg)
    net = PlainR50(seed, scope)
    bb = [p for p in net.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=bb,lr=lr),dict(params=net.head.parameters(),lr=.001)],weight_decay=.01)
    train, val = inputs.groups['small'], inputs.groups['validation']
    rng = np.random.default_rng(seed+20261111)
    aug = np.random.default_rng(seed+20261112)
    best=None; curve=[]; steps=0; exposure=0; start=0; elapsed=0.
    if (dest/'resume.pt').exists():
        ck = torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best,curve,start,steps,exposure,elapsed = (ck[k] for k in ('best','curve','epoch','steps','exposure','seconds'))
        rng.bit_generator.state=ck['rng'];aug.bit_generator.state=ck['aug']
        del ck
    tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,EPOCHS):
        factor=(epoch+1)/2 if epoch<2 else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/(EPOCHS-3)))
        for group,base in zip(opt.param_groups,(lr,.001)):
            group['lr']=base*factor
        net.train();order=rng.permutation(train);loss_sum=0.;correct=0
        for first in range(0,len(order),BATCH):
            ids=order[first:first+BATCH]
            opt.zero_grad(set_to_none=True)
            x,mask,av=inputs.batch(ids,'small',aug)
            y=torch.tensor(inputs.y[ids],device=DEVICE)
            w=torch.tensor(inputs.weights[ids],device=DEVICE)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=net(x,mask,av)
                loss=(F.cross_entropy(z,y,reduction='none')*w).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss {scope}/{lr}/{seed}/{epoch}')
            loss.backward();opt.step()
            loss_sum+=float(loss.detach())*len(ids)
            correct+=int((z.argmax(1)==y).sum());steps+=1;exposure+=len(ids)
        score=predict(net,inputs,val,'validation');m=metrics(inputs.y[val],score)
        key=(m['ba'],m['oa'],-m['nll'])
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(train),train_oa=100*correct/len(train),
                 validation=m,seconds=elapsed+time.perf_counter()-tick,backbone_lr=lr*factor)
        curve.append(row)
        if best is None or key>tuple(best['key']):
            best=dict(epoch=epoch+1,key=list(key),metrics=m)
            torch.save(net.state_dict(),dest/'best.pt')
            np.savez_compressed(dest/'validation.npz',ids=val,score=score,label=inputs.y[val])
        stop=epoch+1>=MIN_EPOCHS and epoch+1-best['epoch']>=PATIENCE
        if epoch==0 or (epoch+1)%5==0 or stop:
            print('VAL',scope,lr,seed,epoch+1,'train',round(row['train_oa'],2),
                  'OA/BA',round(m['oa'],3),round(m['ba'],3),'best',best['epoch'],
                  'sec',round(row['seconds'],1),flush=True)
            write(ROOT/'status.json',dict(state='training',scope=scope,lr=lr,seed=seed,
                epoch=epoch+1,best=best,test_pixels_loaded=False,complete=False))
            write(dest/'curve.json',curve)
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),best=best,curve=curve,
                epoch=epoch+1,steps=steps,exposure=exposure,seconds=row['seconds'],
                rng=rng.bit_generator.state,aug=aug.bit_generator.state),dest/'resume.pt')
        if stop:
            break
    receipt=dict(**cfg,epochs=epoch+1,steps=steps,exposures=exposure,best=best,
        seconds=elapsed+time.perf_counter()-tick,stop_reason='validation_patience' if stop else 'max_epochs',
        parameters=sum(p.numel() for p in net.parameters()),
        trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
        peak_gib=torch.cuda.max_memory_allocated()/1024**3,test_read=False)
    write(dest/'curve.json',curve);write(dest/'receipt.json',receipt)
    del net,opt;gc.collect();torch.cuda.empty_cache()
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    inputs=Inputs()
    if args.smoke:
        return smoke(inputs)
    protocol=dict(scopes=SCOPES,backbone_lrs=LRS,seeds=SEEDS,training=configuration('search',0,17),
                  test_policy='only selected scope/LR three seeds, after all validation selection',
                  data=dict(train=719,validation=228,test=378),pretrain=str(WEIGHTS))
    write(ROOT/'protocol.json',protocol)
    candidates=[]
    for scope in SCOPES:
        for lr in LRS:
            path=ROOT/f'scout_{scope}_lr{lr:g}_s17'
            r=fit(inputs,scope,lr,17,path)
            candidates.append(dict(path=path.name,scope=scope,lr=lr,best=r['best'],epochs=r['epochs']))
    selected=max(candidates,key=lambda r:tuple(r['best']['key']))
    chosen=dict(selected_run=selected['path'],scope=selected['scope'],lr=selected['lr'],seeds=SEEDS,
                candidates=candidates,selected_paths=[f'selected_s{s}' for s in SEEDS],test_read=False)
    write(ROOT/'selected.json',chosen)
    print('SELECTED_VALIDATION',json.dumps(selected),flush=True)
    first=ROOT/'selected_s17';first.mkdir(exist_ok=True)
    for name in ('best.pt','config.json','receipt.json','curve.json','validation.npz'):
        shutil.copy2(ROOT/selected['path']/name,first/name)
    for seed in SEEDS[1:]:
        fit(inputs,selected['scope'],selected['lr'],seed,ROOT/f'selected_s{seed}')
    # Test pixels enter GPU memory only after architecture/LR selection and all fitting.
    inputs.load_split('evaluation')
    replays=[]
    for seed in SEEDS:
        dest=ROOT/f'selected_s{seed}'
        net=PlainR50(seed,selected['scope'])
        net.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True))
        old=np.load(dest/'validation.npz')
        val=predict(net,inputs,inputs.groups['validation'],'validation')
        err=float(abs(val-old['score']).max())
        same=bool(np.array_equal(val.argmax(1),old['score'].argmax(1)))
        assert same and err<1e-3,(err,same)
        ids=inputs.groups['evaluation'];score=predict(net,inputs,ids,'evaluation')
        np.savez_compressed(dest/'evaluation.npz',ids=ids,score=score,label=inputs.y[ids])
        m=metrics(inputs.y[ids],score)
        write(dest/'evaluation.json',m)
        replays.append(dict(seed=seed,validation_error=err,predictions_identical=same))
        print('TEST_SELECTED',seed,json.dumps(m),flush=True)
        del net;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'checkpoint_replays.json',replays)
    write(ROOT/'status.json',dict(state='training_and_test_complete',complete=True,
        analysis_pending=True,scope=selected['scope'],lr=selected['lr'],seeds=SEEDS))


if __name__=='__main__':
    main()
