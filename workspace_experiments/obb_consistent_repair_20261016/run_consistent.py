"""Consistent source/support/query regions with matched existing training recipes."""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'geometry_recovery_effect_20261016'))
import run_effect as effect

base=effect.base
geo=base.old
fit=effect.fit
PORT='Singapore'
ARMS=('wrong','fixed')
FEATURES={}
CONFIG=json.loads((HERE/'inventory.json').read_text())


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def features(arm):
    if arm not in FEATURES:
        with np.load(HERE/f'features_{arm}.npz') as z:
            values={k:z[k] for k in z.files}
        gallery=np.asarray(CONFIG['gallery_rows'],int)
        pos=np.searchsorted(values['rows'],gallery)
        np.testing.assert_array_equal(values['rows'][pos],gallery)
        pca=PCA(n_components=64,svd_solver='randomized',random_state=20261015)
        pca.fit(values['ship_peak'][pos])
        ship=pca.transform(values['ship_peak']).astype(np.float64)
        ship/=np.linalg.norm(ship,axis=1,keepdims=True).clip(1e-12)
        values['ship']=ship
        FEATURES[arm]=values
        out=HERE/arm; out.mkdir(exist_ok=True)
        np.savez_compressed(out/'source_pca.npz',gallery_rows=gallery,
                            mean=pca.mean_,components=pca.components_)
    return FEATURES[arm]


def load(arm,target=False):
    data=effect.load(PORT,target)
    if arm=='historical':
        return data
    values=features(arm)
    rows=data['meta'].global_row.to_numpy(int)
    pos=np.searchsorted(values['rows'],rows)
    np.testing.assert_array_equal(values['rows'][pos],rows)
    d=data['tensors']
    d['ship']=torch.as_tensor(values['ship'][pos],device=base.DEVICE,dtype=torch.float64)
    for k in ('tokens','positions','areas','img_tokens','ship_tokens','img_areas','ship_areas'):
        d[k]=torch.as_tensor(values[k][pos],device=base.DEVICE,dtype=torch.float32)
    for k in ('tokens','positions','areas','edge'):
        data['spatial'][k]=values[k][pos]
    data['region_arm']=arm
    return data


def prepare(data,arm,seed):
    if arm=='historical':
        effect.prepare(data,seed)
    else:
        # The old helper reads PREV/trained/<port>/B_seed.pt. Bind it only here.
        with patch.object(base,'PREV',HERE/arm):
            effect.prepare(data,seed)


def b_path(arm,seed):
    return HERE/arm/'trained'/PORT/f'B_seed{seed}.pt'


def train_all():
    started=time.perf_counter()
    assert (HERE/'FEATURES_DONE.json').exists()
    for arm in ARMS:
        data=load(arm)
        for seed in range(3):
            data['tensors']['geometry']=data['legacy_geometry']
            with patch.object(geo,'HERE',HERE/arm), patch.object(geo,'load_data',lambda port:data):
                geo.train(PORT,'B',seed,200)
            prepare(data,arm,seed)
            for family,(stage,reader_arm) in effect.FAMILIES.items():
                effect.set_input(data,family,'new')
                with patch.object(fit,'HERE',HERE/arm):
                    fit.train(data,stage,reader_arm,seed)
        del data
        torch.cuda.empty_cache()
    files=[b_path(a,s) for a in ARMS for s in range(3)] + [
        effect.checkpoint_path(HERE/a,PORT,f,s) for a in ARMS for f in effect.FAMILIES for s in range(3)]
    assert len(files)==18 and all(p.is_file() for p in files)
    # Both treatments must see identical source episodes and query positions.
    for family in effect.FAMILIES:
        for seed in range(3):
            traces=[effect.checkpoint_path(HERE/a,PORT,family,seed).with_name(
                effect.checkpoint_path(HERE/a,PORT,family,seed).stem+'_sampling.json') for a in ARMS]
            assert json.loads(traces[0].read_text())==json.loads(traces[1].read_text())
    dump(HERE/'SOURCE_FROZEN.json',dict(records=[str(p) for p in files],seeds=[0,1,2],
        steps=200,training_seconds=time.perf_counter()-started,
        matched_slot_sampling=True,target_model_selection=False))


def integration():
    # Historical feature pipeline check, no target predictions used.
    old=load('historical')
    old_ship=old['tensors']['ship'].cpu().numpy()
    values=features('historical')
    pos=np.searchsorted(values['rows'],old['meta'].global_row)
    error=float(np.abs(values['ship'][pos]-old_ship).max())
    # Raw peaks and the original 2264-row PCA batch replay exactly. Transforming
    # the 2713-row inventory changes float32 GEMM rounding by at most 2.26e-6.
    np.testing.assert_allclose(values['ship'][pos],old_ship,atol=3e-6,rtol=1e-6)
    prepare(old,'historical',0)
    frame=pd.read_csv(HERE/'inventory.csv',dtype={'mmsi':str}).set_index('global_row')
    roles=pd.read_csv(HERE/'roles.csv')
    groups={k:frame.loc[g.global_row.unique()] for k,g in roles.groupby('split')}
    overlaps=[]
    for i,a in enumerate(groups):
        for b in list(groups)[i+1:]:
            for field in ('mmsi','product'):
                overlap=set(groups[a][field])&set(groups[b][field])
                assert not overlap,(a,b,field,overlap)
                overlaps.append(dict(left=a,right=b,field=field,overlap=0))
    assert not frame.loc[CONFIG['gallery_rows']].port.eq(PORT).any()
    source_changes=[]
    for arm in ARMS:
        data=load(arm); d=data['tensors']
        torch.manual_seed(20261016)
        model=geo.GeometryReader('B').cuda().eval()
        task=data['tasks']['fit'][0]
        task={**task,'q':task['q'][:8]}
        ids=task['s']+task['q']
        context=geo.encode(model,d,ids)
        p=geo.prediction(d,task,context)[0]
        altered=dict(d); altered['labels']=d['labels'].clone()
        altered['labels'][task['q']]=(altered['labels'][task['q']]+1)%8
        q=geo.prediction(altered,task,context)[0]
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        geo.loss_fn(p,d['labels'][task['q']]).backward()
        assert sum(float(x.grad.square().sum()) for x in model.parameters() if x.grad is not None)>0
        changes={k:float((d[k]-old['tensors'][k]).abs().max()) for k in
                 ('ship','tokens','positions','areas','ship_tokens','img_tokens')}
        assert changes['ship']>0 and changes['tokens']>0 and changes['ship_tokens']>0
        source_changes.append(dict(arm=arm,changes=changes))
    dump(HERE/'INTEGRATION_CHECK.json',dict(source_historical_ship_max_error=error,
        historical_pca_rounding_note='Raw peaks and original batch replay exactly; inventory batch changes float32 GEMM rounding.',
        query_label_prediction_invariant=True,nonzero_B_gradients=True,
        region_input_changes=source_changes,role_overlap_checks=overlaps))
    print('INTEGRATION PASSED',error,flush=True)


@torch.no_grad()
def evaluate_all():
    assert (HERE/'SOURCE_FROZEN.json').is_file()
    replays=[]
    for arm in ('historical',*ARMS):
        data=load(arm,True)
        out=HERE/arm/'evaluation'; out.mkdir(parents=True,exist_ok=True)
        for seed in range(3):
            prepare(data,arm,seed)
            for family in ('B','strict','optional'):
                model=None
                if family=='B':
                    effect.set_input(data,'optional','new')
                else:
                    effect.set_input(data,family,'new')
                    root=effect.HERE if arm=='historical' else HERE/arm
                    model=effect.model_from(root,PORT,family,seed)
                metrics,pred=fit.evaluate(model,data,'target',True)
                for f in (metrics,pred):
                    f['arm']=arm; f['model']=family; f['seed']=seed
                metrics.to_csv(out/f'{family}_seed{seed}_metrics.csv',index=False)
                pred.to_csv(out/f'{family}_seed{seed}_predictions.csv',index=False)
                if arm=='historical':
                    name='B' if family=='B' else family+'_new'
                    path=effect.HERE/'evaluation'/PORT/f'{name}_seed{seed}_predictions.csv'
                    previous=pd.read_csv(path)
                    keys=['task','global_row']; prob=[f'p{i}' for i in range(8)]
                    paired=pred.merge(previous[keys+prob],on=keys,suffixes=('','_old'),validate='one_to_one')
                    assert len(paired)==len(pred)==len(previous)
                    error=float(np.abs(paired[prob].to_numpy()-paired[[c+'_old' for c in prob]].to_numpy()).max())
                    assert error<1e-10
                    replays.append(dict(model=family,seed=seed,max_error=error))
                print('EVALUATED',arm,family,seed,len(pred),flush=True)
        del data
        torch.cuda.empty_cache()
    dump(HERE/'EVALUATION_DONE.json',dict(historical_replays=replays,
         models=27,rows_per_model=len(pred),target_only_after_source_freeze=True))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['check','train','evaluate','all'])
    args=parser.parse_args(); torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase in ('check','all'):
            integration()
        if args.phase in ('train','all'):
            assert (HERE/'INTEGRATION_CHECK.json').is_file()
            train_all()
        if args.phase in ('evaluate','all'):
            evaluate_all()


if __name__=='__main__':
    main()
