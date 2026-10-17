"""Matched metadata recovery experiment; frozen B, readers and task identities."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'geometry_targeted_fix_20261016'
NEW=HERE.parent/'geometry_source_recovery_20261016'
sys.path.insert(0,str(OLD))
import run_fix as fit
import numpy as np
import pandas as pd
import torch

fit.HERE=HERE
base=fit.base
COLS=['ship_axis_rad','radar_axis_rad','incidence_rad','ship_valid','radar_valid','incidence_valid','ship_confidence']
FAMILIES={'strict':('bounded','geometry'),'optional':('recovered','conditional_rank')}
TABLES={v:pd.read_csv(p/'recovered_geometry.csv',low_memory=False).set_index('global_row')
        for v,p in [('old',OLD),('new',NEW)]}


def eligibility(geometry,family):
    return (geometry[:,3:6]>0).all(1) if family=='strict' else geometry[:,3]>0


def load(port,target=False):
    data=base.load(port,target)
    data['legacy_geometry']=data['tensors']['geometry'].clone()
    data['geometries']={}
    for version,table in TABLES.items():
        selected=table.loc[data['meta'].global_row]
        np.testing.assert_array_equal(selected.instance_id.to_numpy(),
                                      TABLES['old'].loc[data['meta'].global_row,'instance_id'].to_numpy())
        geometry=torch.as_tensor(np.nan_to_num(selected[COLS].to_numpy(np.float32)),device=base.DEVICE)
        data['geometries'][version]=geometry
        old=data['legacy_geometry']
        torch.testing.assert_close(old[:,3],geometry[:,3],rtol=0,atol=0)
        active=old[:,3]>0
        torch.testing.assert_close(old[active,0],geometry[active,0],rtol=0,atol=0)
    # Valid old fields must not change when restoring missing conditions.
    a,b=data['geometries']['old'],data['geometries']['new']
    for col in range(3):
        keep=a[:,col+3]>0
        torch.testing.assert_close(a[keep,col],b[keep,col],rtol=0,atol=0)
    data['metadata_changed']=(a!=b).any(1)
    return data


def prepare(data,seed):
    data['tensors']['geometry']=data['legacy_geometry']
    base.prepare_baseline(data,seed)


def set_input(data,family,version):
    d=data['tensors']
    d['geometry']=data['geometries'][version]
    d['valid']=eligibility(d['geometry'],family)


def checkpoint_path(root,port,family,seed):
    stage,arm=FAMILIES[family]
    return root/stage/'trained'/port.replace(' ','_')/f'{arm}_seed{seed}.pt'


def model_from(root,port,family,seed):
    saved=torch.load(checkpoint_path(root,port,family,seed),weights_only=False,map_location=base.DEVICE)
    if saved.get('unavailable'):
        return None
    model=fit.reader(*FAMILIES[family])
    model.load_state_dict(saved['state']); model.eval()
    return model


def train_all():
    for port in base.PORTS:
        data=load(port)
        for seed in range(3):
            prepare(data,seed)
            for family,(stage,arm) in FAMILIES.items():
                set_input(data,family,'new')
                fit.train(data,stage,arm,seed)
                if family=='optional':
                    trace=checkpoint_path(HERE,port,family,seed).with_name(f'{arm}_seed{seed}_sampling.json')
                    previous=checkpoint_path(OLD,port,family,seed).with_name(trace.name)
                    if trace.exists():
                        assert json.loads(trace.read_text())==json.loads(previous.read_text()),'Optional sampling changed'
        del data
        torch.cuda.empty_cache()
    records=[checkpoint_path(HERE,p,f,s) for p in base.PORTS for f in FAMILIES for s in range(3)]
    assert all(p.is_file() for p in records)
    fit.dump(HERE/'SOURCE_FROZEN.json',dict(records=len(records),seeds=[0,1,2],steps=200,
             metadata=str(NEW/'recovered_geometry.csv'),target_selection=False,
             optional_sampling_identical=True))


def add_coverage(pred,data,family,model):
    d=data['tensors']
    old=eligibility(data['geometries']['old'],family)
    new=eligibility(data['geometries']['new'],family)
    chunks=[]
    for ti,item in enumerate(data['cache']['target']):
        q,s=item['q'],item['s']
        f=pd.DataFrame(dict(task=ti,global_row=data['meta'].iloc[q.cpu().numpy()].global_row.to_numpy(),
            old_enabled=(old[q]&(old[s].sum()>=2)).cpu().numpy(),
            new_enabled=(new[q]&(new[s].sum()>=2)).cpu().numpy(),
            metadata_changed=data['metadata_changed'][q].cpu().numpy(),
            support_metadata_changes=int(data['metadata_changed'][s].sum())))
        chunks.append(f)
    coverage=pd.concat(chunks,ignore_index=True)
    result=pred.merge(coverage,on=['task','global_row'],validate='one_to_one')
    result['effective_enabled']=result.enabled & (model is not None)
    return result


def verify_replay(pred,port,arm,seed):
    if arm=='B':
        stage,oldarm='recovered','B'
    elif arm=='strict_old':
        stage,oldarm='bounded','geometry'
    elif arm=='optional_old':
        stage,oldarm='recovered','conditional_rank'
    else:
        return None
    historical=pd.read_csv(OLD/stage/'evaluation'/port.replace(' ','_')/f'{oldarm}_seed{seed}_predictions.csv')
    keys=['port','acquisition','origin','fold','draw','budget','sample_id']
    cols=[f'p{i}' for i in range(8)]
    joined=pred.merge(historical[keys+cols],on=keys,suffixes=('','_old'),validate='one_to_one')
    assert len(joined)==len(pred)==len(historical)
    error=float(np.abs(joined[cols].to_numpy()-joined[[c+'_old' for c in cols]].to_numpy()).max())
    assert error<1e-10,(port,arm,seed,error)
    return dict(port=port,arm=arm,seed=seed,rows=len(pred),max_error=error)


def target_all():
    assert (HERE/'SOURCE_FROZEN.json').is_file()
    replays=[]
    for port in base.PORTS:
        data=load(port,True)
        folder=HERE/'evaluation'/port.replace(' ','_')
        folder.mkdir(parents=True,exist_ok=True)
        for seed in range(3):
            prepare(data,seed)
            arms=[('B','optional','old',None)]
            for family in FAMILIES:
                oldmodel=model_from(OLD,port,family,seed)
                newmodel=model_from(HERE,port,family,seed)
                arms.extend([(family+'_old',family,'old',oldmodel),
                             (family+'_input',family,'new',oldmodel),
                             (family+'_new',family,'new',newmodel)])
            for arm,family,version,model in arms:
                set_input(data,family,version)
                metrics,pred=fit.evaluate(model,data,'target',True)
                pred=add_coverage(pred,data,family,model)
                for table in (metrics,pred):
                    table['arm'],table['seed']=arm,seed
                check=verify_replay(pred,port,arm,seed)
                if check:
                    replays.append(check)
                metrics.to_csv(folder/f'{arm}_seed{seed}_metrics.csv',index=False)
                pred.to_csv(folder/f'{arm}_seed{seed}_predictions.csv',index=False)
                print('TARGET',port,arm,seed,flush=True)
        del data
        torch.cuda.empty_cache()
    fit.dump(HERE/'REPLAY_CHECK.json',dict(checks=replays,max_error=max(c['max_error'] for c in replays)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','target','all'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.phase in ('train','all'):
        train_all()
    if args.phase in ('target','all'):
        target_all()


if __name__=='__main__':
    main()
