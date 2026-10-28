"""Frozen 3075-query expansion: paired baseline versus early layer2 M2."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
RELEASE = WORK/'sar_m123_current_20261027'
PILOT = ROOT/'shallow_m2_writeback_20261028'
sys.path.insert(0, str(WORK/'sar_m123_current_20261027'))
from sar_m123.shallow_writeback import ShallowM2
from sar_m123.current.model import setup, CLASSES
from sar_m123.current.data import read_manifest, prepare_record, batch, pair_mask

ARMS = ('baseline', 'layer2')
SEEDS = (17, 29, 43)
BATCH = 16


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def records():
    table = pd.read_csv(ROOT/'intervention_selection_expansion_20261027/query_manifest.csv',
                        dtype={'mmsi':str})
    assert len(table) == 3075 and table.obs_id.is_unique
    truth = pd.read_csv(ROOT/'ports24_aligned_20261026/target_truth.csv', dtype={'mmsi':str})
    assert truth.obs_id.is_unique
    truth = truth.set_index('obs_id').loc[table.obs_id]
    for key in ('mmsi','pass_group','instance_id'):
        assert np.array_equal(truth[key].to_numpy(str),table[key].to_numpy(str)), key
    table['class_id'] = truth.class_id.to_numpy(int)
    snapshots = {str(x['obs_id']):x for x in json.loads((WORK/'same_vessel_gallery_20261019/local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    with np.load(ROOT/'local_evidence_transport/external_confirmation/tokens.npz') as z:
        assert np.array_equal(z['instance_ids'].astype(str), table.instance_id.to_numpy(str))
        available = z['valid'].astype(bool)
    result = []
    for i, row in table.iterrows():
        rec = dict(id=str(row.obs_id), mmsi=row.mmsi, port=row.port, pass_group=row.pass_group,
                   product=row['product'], label=int(row.class_id), split='external_evaluation')
        for p, pol in enumerate(('vv','vh')):
            info = snapshots[rec['id']]['images'].get(pol)
            if not available[i,p]:
                rec[pol] = None
                continue
            assert info is not None and Path(row['image_'+pol]).is_file(), (rec['id'],pol)
            priors = info['priors']
            rec[pol] = dict(image=row['image_'+pol], detections=priors['detections'],
                            target_index=priors['target_index'], ship_axis=priors.get('ship_axis'))
        result.append(rec)
    audit = dict(observations=len(result), vessels=table.mmsi.nunique(), passes=table.pass_group.nunique(),
                 ports=table.port.nunique(), classes=table.class_name.value_counts().to_dict(),
                 both_polarizations=int(available.all(1).sum()), overlaps={})
    for split in ('source_fit','source_adapt','development','references'):
        other = read_manifest(RELEASE/'manifests'/f'{split}.json')
        intersections = {k:sorted({str(r[k]) for r in result} & {str(r[k]) for r in other})
                         for k in ('id','mmsi','pass_group')}
        audit['overlaps'][split] = intersections
        assert not any(intersections.values()), (split,intersections)
    write('query_records.json',result)
    write('coverage.json',audit)
    table.to_csv(HERE/'query_metadata.csv',index=False)
    return result, available


@torch.inference_mode()
def features(base, rows):
    prepared = [prepare_record(r,True) for r in rows]
    for row, (item,views) in zip(rows,prepared):
        assert item['available'].tolist() == [float(row.get(p) is not None) for p in ('vv','vh')], row['id']
        for view in views:
            if view is not None:
                assert view['placement']['retained_target_pixels'] == view['placement']['native_target_pixels'], row['id']
    data = batch([x[0] for x in prepared], 'cuda')
    stage1 = base.prefix(data['images'],stop=1)
    raw2 = base.backbone.model.layer2(stage1.flatten(0,1)).reshape(len(rows),2,512,20,20)
    x = raw2.flatten(0,1)
    for block in list(base.backbone.model.layer3.children())[:-1]:
        x = block(x)
    data.update(stage1=stage1,raw2=raw2,anchor3=base.anchor_last(x).reshape(len(rows),2,1024,10,10))
    data.pop('images')
    return data


def models():
    teacher = torch.load(RELEASE/'weights/teacher.pt',map_location='cpu',weights_only=True)['state']
    nets = []
    receipts = []
    for arm in ARMS:
        for seed in SEEDS:
            net = ShallowM2(teacher,() if arm=='baseline' else (2,))
            checkpoint = PILOT/'checkpoints'/f'{arm}_{seed}.pt'
            saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
            expected = {n for n,p in net.named_parameters() if p.requires_grad}
            assert set(saved['state']) == expected, (arm,seed)
            assert saved['arm'] == arm and saved['seed'] == seed
            net.load_state_dict(saved['state'],strict=False)
            nets.append(net.cuda().eval().requires_grad_(False))
            receipts.append(dict(arm=arm,seed=seed,checkpoint=str(checkpoint)))
    del teacher
    gc.collect()
    write('models.json',receipts)
    return nets


@torch.inference_mode()
def replay(nets, bank, refs):
    saved = np.load(PILOT/'predictions.npz')
    dev = read_manifest(RELEASE/'manifests/development.json')
    take = np.linspace(0,len(dev)-1,16,dtype=int)
    assert np.array_equal(saved['ids'],np.array([r['id'] for r in dev]))
    rows = [dev[i] for i in take]
    data = features(nets[0].base,rows)
    errors = {}
    for ni, net in enumerate(nets):
        ai,si = divmod(ni,3)
        score = net(data,bank,pair_mask(rows,refs,'cuda')).cpu().numpy()
        old_ai = list(saved['arms']).index(ARMS[ai])
        expected = saved['scores'][old_ai,si,take]
        error = float(abs(score-expected).max())
        assert error < 1e-4, (ARMS[ai],SEEDS[si],error)
        errors[f'{ARMS[ai]}_{SEEDS[si]}'] = error
    write('replay.json',dict(n=16,ids=[r['id'] for r in rows],max_errors=errors))
    print('REPLAY',errors,flush=True)


@torch.inference_mode()
def main():
    setup('cuda')
    rows, available = records()
    refs = read_manifest(RELEASE/'manifests/references.json')
    nets = models()
    bank = features(nets[0].base,refs)
    replay(nets,bank,refs)
    write('protocol.json',dict(arms=ARMS,seeds=SEEDS,queries=3075,references=16,training=False,
          tuning=False,default_changed=False,endpoint='previous fixed_epoch6',batch=BATCH,
          scope='historical expanded cohort; not a new blind dataset'))
    chunks = HERE/'chunks'
    chunks.mkdir(exist_ok=True)
    start = time.perf_counter()
    for first in range(0,len(rows),BATCH):
        subset = rows[first:first+BATCH]
        path = chunks/f'{first:05d}.npz'
        if path.exists():
            continue
        data = features(nets[0].base,subset)
        assert np.array_equal(data['available'].cpu().numpy().astype(bool),available[first:first+len(subset)])
        legal = pair_mask(subset,refs,'cuda')
        score = np.stack([net(data,bank,legal).cpu().numpy() for net in nets]).reshape(2,3,len(subset),8)
        assert np.isfinite(score).all()
        np.savez_compressed(path,ids=np.array([r['id'] for r in subset]),scores=score)
        if first % (BATCH*8)==0 or first+BATCH>=len(rows):
            print('EVAL',min(first+BATCH,len(rows)),len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
    parts = [np.load(chunks/f'{i:05d}.npz') for i in range(0,len(rows),BATCH)]
    ids = np.concatenate([z['ids'] for z in parts])
    assert np.array_equal(ids,np.array([r['id'] for r in rows]))
    np.savez_compressed(HERE/'predictions.npz',ids=ids,labels=np.array([r['label'] for r in rows]),
                       scores=np.concatenate([z['scores'] for z in parts],axis=2),arms=ARMS,seeds=SEEDS)
    write('complete.json',dict(complete=True,queries=len(rows),models=6,training=False,
          seconds=time.perf_counter()-start,peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2))
    print('COMPLETE frozen expansion',flush=True)


if __name__=='__main__':
    main()
