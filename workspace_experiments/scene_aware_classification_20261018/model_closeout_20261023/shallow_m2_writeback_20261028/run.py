"""Matched early-M2 pilot; raw-image preprocessing, fixed anchors, live gradients."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
PACKAGE = WORK/'sar_m123_current_20261027'
RELEASE = WORK/'sar_m123_current_20261027'
sys.path.insert(0, str(PACKAGE))
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask
from sar_m123.current.model import load_model, setup, forward_batch, reference_bank
from sar_m123.current.training import adaptation_loss, class_weights
from sar_m123.shallow_writeback import ShallowM2
from sar_m123.pool_replacement import grid_occupancy
from sar_m123 import frequency, tokens

ARMS = dict(baseline=(), layer1=(1,), layer2=(2,), both=(1,2))
SEEDS = (17,29,43)
LIVE = tuple(f'base.backbone.model.layer3.5.conv{i}.weight' for i in (1,2,3))


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_early(record):
    item, views = prepare_record(record, True)
    item['regions1'] = np.zeros((2,3,40,40), np.float32)
    item['energy1'] = np.zeros((2,3,3,40,40), np.float32)
    for p, view in enumerate(views):
        if view is None or not item['available'][p]:
            continue
        slots = tokens.slots(view['geometry'], view['footprint'])
        region = np.stack([slots[r].sum(0) for r in (tokens.REGIONS[0],tokens.REGIONS[2],tokens.REGIONS[1])])
        field = frequency.extract(view['raw'], slots, view['placement'])['energy_field']
        item['regions1'][p] = grid_occupancy(torch.from_numpy(region)[None], (40,40))[0].numpy()
        weighted = (region[:,None]*field[None]).reshape(9,160,160)
        item['energy1'][p] = grid_occupancy(torch.from_numpy(weighted)[None],(40,40))[0].reshape(3,3,40,40).numpy()
    return item


@torch.no_grad()
def build_cache(model, records, name, device):
    dest = ROOT/'cache'/name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        return
    files = {}
    for first in range(0,len(records),4):
        items = [prepare_early(r) for r in records[first:first+4]]
        data = {k:torch.as_tensor(np.stack([x[k] for x in items]),device=device) for k in items[0]}
        stage1 = model.prefix(data['images'],stop=1)
        raw2 = model.backbone.model.layer2(stage1.flatten(0,1)).reshape(len(items),2,512,20,20)
        z = raw2.flatten(0,1)
        for block in list(model.backbone.model.layer3.children())[:-1]:
            z = block(z)
        raw3 = model.anchor_last(z).reshape(len(items),2,1024,10,10)
        data.update(stage1=stage1,raw2=raw2,anchor3=raw3)
        data.pop('images')
        for key,tensor in data.items():
            arr = tensor.cpu().numpy()
            if key not in files:
                files[key] = np.lib.format.open_memmap(dest/f'{key}.npy',mode='w+',dtype=np.float32,
                                                      shape=(len(records),*arr.shape[1:]))
            files[key][first:first+len(items)] = arr
        print(f'cache {name} {first+len(items)}/{len(records)}',flush=True)
    for arr in files.values():
        arr.flush()
    (dest/'complete.json').write_text(json.dumps(dict(n=len(records),ids=[str(r['id']) for r in records])),encoding='utf-8')


class Cache:
    def __init__(self,name):
        self.arrays = {p.stem:np.load(p,mmap_mode='r') for p in (ROOT/'cache'/name).glob('*.npy')}

    def get(self, indices, device='cuda'):
        return {k:torch.as_tensor(np.array(v[indices]),device=device) for k,v in self.arrays.items()}


@torch.no_grad()
def predict(model, cache, records, bank, refs, micro=4):
    out = []
    for first in range(0,len(records),micro):
        ix = np.arange(first,min(first+micro,len(records)))
        out.append(model(cache.get(ix),bank,pair_mask(records[first:first+len(ix)],refs,'cuda')).cpu().numpy())
    return np.concatenate(out)


def construct(state, stages, seed):
    # Per-stage initialization is identical between single-stage and both arms.
    torch.manual_seed(903)
    model = ShallowM2(state,stages)
    for stage in stages:
        for p in range(2):
            torch.manual_seed(903+stage*10+p)
            from sar_m123.bottleneck_evidence import RegionPoolWriteback
            model.early[f'{stage}_{p}'] = RegionPoolWriteback({1:256,2:512}[stage])
    return model.cuda().eval()


def diagnostics(state, source, refs, sc, bank, selection):
    q=sc.get(np.arange(4)); legal=pair_mask(source[:4],refs,'cuda')
    base=construct(state,(),17)
    with torch.no_grad():
        expected=base(q,bank,legal)
    old=base.base
    raw=batch([prepare_record(r) for r in source[:4]],'cuda')
    raw_bank=reference_bank(old,refs,'cuda')
    with torch.no_grad():
        raw_scores=forward_batch(old,raw,raw_bank,legal)
    diff=float((raw_scores-expected).abs().max())
    assert diff<1e-4, diff
    results=dict(raw_replay_max=diff,initial_zero_updates={},first_gradient={},later_gradient={})
    del base,old,raw_bank,raw
    gc.collect()
    for arm,stages in ARMS.items():
        m=construct(state,stages,17)
        with torch.no_grad():
            score=m(q,bank,legal)
        err=float((score-expected).abs().max())
        assert err<1e-4,(arm,err)
        results['initial_zero_updates'][arm]=err
        if arm=='both':
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4)
            labels=torch.tensor([r['label'] for r in source[:4]],device='cuda')
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(m(q,bank,legal),labels)
                loss.backward()
                for key,module in m.early.items():
                    output=sum(float(t[-1].weight.grad.abs().sum()) for t in module.transforms)
                    if step==0:
                        results['first_gradient'][key]=output
                        assert output>0,(key,'dead output')
                    if step==3:
                        inner=sum(float(t[0].weight.grad.abs().sum()) for t in module.transforms)
                        results['later_gradient'][key]=inner
                        assert inner>0,(key,'broken frozen downstream gradient')
                opt.step()
        del m
        gc.collect()
        torch.cuda.empty_cache()
    model=construct(state,(),17)
    scores=predict(model,sc,source,bank,refs)
    teacher=np.array([r['original'] for r in selection],np.float32)
    teacher_error=float(np.max(np.abs(scores-teacher)))
    assert teacher_error<1e-4,teacher_error
    results.update(selection_teacher_max_error=teacher_error,selected=sum(r['useful'] for r in selection),
                   cached_anchors='original frozen layer2 and last-block copy',
                   gradients='frozen downstream convolution operations remain differentiable')
    del model
    write('diagnostics.json',results)


def train_one(state, arm, seed, sc, dc, source, dev, bank, refs, selection, micro):
    model=construct(state,ARMS[arm],seed)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    opt=torch.optim.AdamW([dict(params=[p for n,p in trainable if n in LIVE],lr=3e-5),
                          dict(params=[p for n,p in trainable if n.startswith('early.')],lr=3e-4)],weight_decay=.01)
    labels=np.array([r['label'] for r in source])
    weights=class_weights(source,'adapt')
    teacher=np.array([r['original'] for r in selection],np.float32)
    useful=np.array([r['useful'] for r in selection],np.float32)
    rng=np.random.default_rng(seed)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); curve=[]; steps=0
    for epoch in range(1,7):
        totals=[]
        for indices in np.array_split(rng.permutation(len(source)),range(16,len(source),16)):
            opt.zero_grad(set_to_none=True)
            denominator=float(weights[indices].sum()); batch_terms={}
            for sub in np.array_split(indices,range(micro,len(indices),micro)):
                rows=[source[i] for i in sub]
                score=model(sc.get(sub),bank,pair_mask(rows,refs,'cuda'))
                w=torch.tensor(weights[sub],device='cuda')
                loss,terms=adaptation_loss(score,torch.tensor(labels[sub],device='cuda'),
                    torch.tensor(teacher[sub],device='cuda'),w,torch.tensor(useful[sub],device='cuda'),.3683066666126251)
                fraction=w.sum()/denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError((arm,seed,epoch))
                (loss*fraction).backward()
                for key,val in terms.items():
                    batch_terms[key]=batch_terms.get(key,0.)+float(val.detach()*fraction)
            opt.step();steps+=1;totals.append(batch_terms)
        row=dict(epoch=epoch,steps=steps,**{k:float(np.mean([t[k] for t in totals])) for k in totals[0]})
        curve.append(row)
        print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
    torch.cuda.synchronize(); seconds=time.perf_counter()-start
    score=predict(model,dc,dev,bank,refs,micro)
    for module in model.early.values():
        module.enabled=False
    bypass=predict(model,dc,dev,bank,refs,micro) if ARMS[arm] else score.copy()
    for module in model.early.values():
        module.enabled=True
    delta={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    checkpoint=ROOT/'checkpoints'/f'{arm}_{seed}.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(arm=arm,seed=seed,stages=ARMS[arm],state=delta,base_checkpoint=str(RELEASE/'weights/teacher.pt')),checkpoint)
    for name,tensor in model.base.state_dict().items():
        if f'base.{name}' not in LIVE and not torch.equal(tensor.cpu(),state[name]):
            raise AssertionError(f'Frozen state changed: {name}')
    receipt=dict(arm=arm,seed=seed,trainable=sum(p.numel() for _,p in trainable),seconds=seconds,steps=steps,
        curve=curve,peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2,endpoint='fixed_epoch6',
        selected=int(useful.sum()),checkpoint=str(checkpoint),frozen_state_unchanged=True,micro_batch=micro)
    del model,opt
    gc.collect();torch.cuda.empty_cache()
    replay=construct(state,ARMS[arm],seed)
    replay.load_state_dict(delta,strict=False)
    actual=predict(replay,dc,dev[:16],bank,refs,micro)
    error=float(np.max(abs(actual-score[:16])))
    assert error<1e-4,error
    receipt['saved_replay_error']=error
    del replay,delta
    gc.collect();torch.cuda.empty_cache()
    return score,bypass,receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('cache','check','train'),default='train')
    parser.add_argument('--micro',type=int,default=16)
    args=parser.parse_args();setup('cuda')
    records={k:read_manifest(RELEASE/'manifests'/f'{v}.json') for k,v in
             [('source','source_adapt'),('dev','development'),('refs','references')]}
    selection=json.loads((RELEASE/'manifests/selection_archived.json').read_text(encoding='utf-8'))
    lookup={str(r['id']):r for r in selection}
    selection=[lookup[str(r['id'])] for r in records['source']]
    state=torch.load(RELEASE/'weights/teacher.pt',map_location='cpu',weights_only=True)['state']
    if args.mode=='cache':
        model=load_model(RELEASE/'weights/teacher.pt','cuda')
        for name,rows in records.items():
            build_cache(model,rows,name,'cuda')
        return
    caches={name:Cache(name) for name in records}
    bank=caches['refs'].get(np.arange(len(records['refs'])))
    if args.mode=='check':
        diagnostics(state,records['source'],records['refs'],caches['source'],bank,selection)
        print('diagnostics passed',flush=True)
        return
    if not (ROOT/'diagnostics.json').exists():
        raise RuntimeError('Run --mode check before training.')
    n=len(records['dev']);scores=np.zeros((4,3,n,8),np.float32);bypasses=scores.copy();runs=[]
    initial_model=construct(state,(),17)
    initial=predict(initial_model,caches['dev'],records['dev'],bank,records['refs'],args.micro)
    del initial_model
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            saved=ROOT/f'completed_{arm}_{seed}.npz'
            receipt_path=ROOT/f'run_{arm}_{seed}.json'
            if saved.exists() and receipt_path.exists():
                data=np.load(saved)
                score,bypass=data['score'],data['bypass']
                receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            else:
                score,bypass,receipt=train_one(state,arm,seed,caches['source'],caches['dev'],records['source'],records['dev'],
                                               bank,records['refs'],selection,args.micro)
                np.savez(saved,score=score,bypass=bypass)
                write(receipt_path.name,receipt)
            scores[ai,si]=score;bypasses[ai,si]=bypass;runs.append(receipt)
            write('runs.json',runs)
    np.savez(ROOT/'predictions.npz',ids=np.array([str(r['id']) for r in records['dev']]),
        labels=np.array([r['label'] for r in records['dev']]),initial=initial,scores=scores,bypasses=bypasses,
        arms=np.array(list(ARMS)),seeds=np.array(SEEDS))
    print('COMPLETE: 12 fixed-endpoint matched fits',flush=True)


if __name__=='__main__':
    main()
