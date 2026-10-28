"""Frozen pre-add M2 expansion, reusing replayed baseline/stage predictions."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
RELEASE = WORK/'sar_m123_current_20261027'
PILOT = ROOT/'m2_residual_placement_20261028'
OLD = ROOT/'shallow_m2_expansion_20261028'
SHALLOW = ROOT/'shallow_m2_writeback_20261028'
sys.path.insert(0,str(RELEASE))
from sar_m123.block_m2_placement import Layer2BlockM2
from sar_m123.current.model import setup
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask

ARMS = ('baseline','stage','preadd')
SEEDS = (17,29,43)
BATCH = 16


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def records():
    rows = json.loads((OLD/'query_records.json').read_text(encoding='utf-8'))
    saved = np.load(OLD/'predictions.npz')
    assert len(rows)==3075
    assert saved['ids'].tolist()==[str(r['id']) for r in rows]
    assert saved['labels'].tolist()==[r['label'] for r in rows]
    for name in ('query_records.json','query_metadata.csv','coverage.json'):
        shutil.copy2(OLD/name,HERE/name)
    return rows


def model(state,arm,seed):
    net = Layer2BlockM2(state,'none' if arm=='baseline' else arm)
    checkpoint = (PILOT/'checkpoints'/f'preadd_{seed}.pt' if arm=='preadd' else
                  SHALLOW/'checkpoints'/f'{"layer2" if arm=="stage" else "baseline"}_{seed}.pt')
    saved = torch.load(checkpoint,weights_only=True,map_location='cpu')
    assert set(saved['state'])=={n for n,p in net.named_parameters() if p.requires_grad}
    assert saved['seed']==seed
    net.load_state_dict(saved['state'],strict=False)
    return net.cuda().eval().requires_grad_(False),str(checkpoint)


@torch.inference_mode()
def features(base,rows):
    prepared = [prepare_record(r,True) for r in rows]
    for row,(item,views) in zip(rows,prepared):
        assert item['available'].tolist()==[float(row.get(p) is not None) for p in ('vv','vh')]
        for view in views:
            if view is not None:
                assert view['placement']['retained_target_pixels']==view['placement']['native_target_pixels'],row['id']
    data = batch([item for item,_ in prepared],'cuda')
    stage1 = base.prefix(data['images'],stop=1)
    z = stage1.flatten(0,1)
    for block in list(base.backbone.model.layer2.children())[:-1]:
        z = block(z)
    prefix2 = z.reshape(len(rows),2,512,20,20)
    raw2 = base.backbone.model.layer2[-1](z).reshape_as(prefix2)
    x = raw2.flatten(0,1)
    for block in list(base.backbone.model.layer3.children())[:-1]:
        x = block(x)
    data.update(stage1=stage1,prefix2=prefix2,raw2=raw2,
                anchor3=base.anchor_last(x).reshape(len(rows),2,1024,10,10))
    data.pop('images')
    return data


@torch.inference_mode()
def replay(state,nets,bank,refs,rows):
    errors = {}
    old = np.load(OLD/'predictions.npz')
    take = np.linspace(0,len(rows)-1,16,dtype=int)
    subset = [rows[i] for i in take]
    data = features(nets[0].base,subset)
    legal = pair_mask(subset,refs,'cuda')
    for arm,oldarm in [('baseline','baseline'),('stage','layer2')]:
        for si,seed in enumerate(SEEDS):
            net,_ = model(state,arm,seed)
            score = net(data,bank,legal).cpu().numpy()
            expected = old['scores'][old['arms'].tolist().index(oldarm),si,take]
            error = float(abs(score-expected).max())
            assert error<1e-4,(arm,seed,error)
            errors[f'expanded_{arm}_{seed}']=error
            del net
            gc.collect();torch.cuda.empty_cache()
    dev = read_manifest(RELEASE/'manifests/development.json')
    take_dev = np.linspace(0,len(dev)-1,16,dtype=int)
    subset = [dev[i] for i in take_dev]
    data = features(nets[0].base,subset)
    legal = pair_mask(subset,refs,'cuda')
    pilot = np.load(PILOT/'predictions.npz')
    assert pilot['ids'].tolist()==[r['id'] for r in dev]
    for si,seed in enumerate(SEEDS):
        score = nets[si](data,bank,legal).cpu().numpy()
        expected = pilot['scores'][pilot['arms'].tolist().index('preadd'),si,take_dev]
        error = float(abs(score-expected).max())
        assert error<1e-4,('preadd',seed,error)
        errors[f'pilot_preadd_{seed}']=error
    write('replay.json',dict(errors=errors,expanded_ids=[rows[i]['id'] for i in take],
                            pilot_ids=[dev[i]['id'] for i in take_dev]))
    print('REPLAY',errors,flush=True)


@torch.inference_mode()
def main():
    setup('cuda')
    rows = records()
    refs = read_manifest(RELEASE/'manifests/references.json')
    state = torch.load(RELEASE/'weights/teacher.pt',weights_only=True,map_location='cpu')['state']
    nets,receipts = [],[]
    for seed in SEEDS:
        net,path = model(state,'preadd',seed)
        nets.append(net);receipts.append(dict(arm='preadd',seed=seed,checkpoint=path))
    bank = features(nets[0].base,refs)
    replay(state,nets,bank,refs,rows)
    del state
    write('models.json',dict(new_inference=receipts,reused=str(OLD/'models.json')))
    write('protocol.json',dict(arms=ARMS,seeds=SEEDS,queries=3075,references=16,training=False,
          tuning=False,default_changed=False,endpoint='previous fixed_epoch6',batch=BATCH,
          new_inference_models=3,reused_models=6,scope='historical expanded cohort, not a new blind dataset',
          reused_scores=str(OLD/'predictions.npz')))
    chunks = HERE/'chunks'; chunks.mkdir(exist_ok=True)
    start = time.perf_counter()
    for first in range(0,len(rows),BATCH):
        subset = rows[first:first+BATCH]; path = chunks/f'{first:05d}.npz'
        ids = np.array([r['id'] for r in subset])
        if path.exists():
            with np.load(path) as saved:
                assert np.array_equal(saved['ids'],ids)
            continue
        data = features(nets[0].base,subset)
        legal = pair_mask(subset,refs,'cuda')
        score = np.stack([net(data,bank,legal).cpu().numpy() for net in nets])
        assert np.isfinite(score).all()
        np.savez_compressed(path,ids=ids,scores=score)
        if first%(BATCH*8)==0 or first+BATCH>=len(rows):
            print('EVAL',min(first+BATCH,len(rows)),len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
    parts = [np.load(chunks/f'{i:05d}.npz') for i in range(0,len(rows),BATCH)]
    ids = np.concatenate([z['ids'] for z in parts])
    assert ids.tolist()==[r['id'] for r in rows]
    candidate = np.concatenate([z['scores'] for z in parts],axis=1)
    old = np.load(OLD/'predictions.npz')
    assert old['ids'].tolist()==ids.tolist() and old['seeds'].tolist()==list(SEEDS)
    scores = np.stack((old['scores'][old['arms'].tolist().index('baseline')],
                       old['scores'][old['arms'].tolist().index('layer2')],candidate))
    np.savez_compressed(HERE/'predictions.npz',ids=ids,labels=old['labels'],scores=scores,arms=ARMS,seeds=SEEDS)
    write('complete.json',dict(complete=True,queries=len(rows),new_inference_models=3,reused_models=6,
          training=False,tuning=False,seconds=time.perf_counter()-start,
          peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2))
    print('COMPLETE frozen pre-add expansion',flush=True)


if __name__=='__main__':main()
