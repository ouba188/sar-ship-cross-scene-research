"""Paired source-only frozen-reader supplementation and extra-step control."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from frozen_reader import prior, EvidenceReader, read, predict, all_raw, freeze_axis, task_loss, fit

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'axis_access_margin_20261017'
ADDITIVE = HERE.parent/'axis_preserving_addition_20261017'
PKEYS = ['fold','seed','origin','draw','global_row']


@torch.no_grad()
def score_arrays(axis,common,data,features,tasks,arm,seed,folder):
    arrays = dict(global_rows=data['rows'],axis_raw=axis.numpy(),axis_valid=features['valid'].numpy())
    if common is not None:
        arrays.update(common_raw=common.numpy(),common_valid=features['common_valid'].numpy())
    np.savez_compressed(folder/'readouts.npz',**arrays)
    metrics,predictions = [],[]
    for task in tasks:
        ids,q,ns = task['ids'],task['q'],task['ns']
        cr = common[ids] if common is not None else None
        cv = features['common_valid'][ids] if common is not None else None
        tic = time.perf_counter()
        p,h,_ = predict(axis[ids],features['valid'][ids],task,cr,cv,axis_weight=1.)
        elapsed = time.perf_counter()-tic
        y,sy = data['labels'][q],data['labels'][task['s']]
        active = features['valid'][q] & (features['valid'][task['s']].sum()>=2)
        if common is not None:
            active = active | (features['common_valid'][q] & (features['common_valid'][task['s']].sum()>=2))
        kernel = task['qs']+h[ns:]@h[:ns].T
        kernel[~active] = task['qs'][~active]
        ap = prior.probe.map_at_r(kernel.numpy(),y,sy)
        record = dict(fold=data['fold'],arm=arm,seed=seed,origin=task['origin'],draw=task['draw'],
            query=len(q),**prior.old.scores(y,p.numpy()),macro_brier=float(prior.old.risk(p,torch.tensor(y))),
            support_map_r=float(np.nanmean(ap)),descriptor_energy=float(h[ns:].square().sum(1).mean()),
            cached_solve_seconds=elapsed)
        metrics.append(record)
        for j,r in enumerate(q):
            predictions.append(dict(fold=data['fold'],arm=arm,seed=seed,origin=task['origin'],draw=task['draw'],
                global_row=int(data['rows'][r]),truth=int(y[j]),prediction=int(p[j].argmax()),
                axis_valid=bool(features['valid'][r]),active=bool(active[j]),ap=ap[j],
                same_class_support=int((sy==y[j]).sum()),**{f'p{k}':float(p[j,k]) for k in range(8)}))
    return metrics,predictions


@torch.no_grad()
def cached_latency(axis_reader,common_reader,features,task):
    ids = task['ids']
    times = []
    for repeat in range(25):
        for name in (['margin','frozen_common'] if repeat%2==0 else ['frozen_common','margin']):
            tic = time.perf_counter()
            axis = read(axis_reader,features,ids)[0]
            common = read(common_reader,features,ids,True)[0] if name=='frozen_common' else None
            cv = features['common_valid'][ids] if common is not None else None
            predict(axis,features['valid'][ids],task,common,cv,axis_weight=1.)
            elapsed = time.perf_counter()-tic
            if repeat>=5:
                times.append(dict(arm=name,repeat=repeat,query=len(task['q']),support=task['ns'],seconds=elapsed))
    return times


def main():
    torch.set_num_threads(1)
    manifest = pd.read_csv(prior.old.MANIFEST,dtype={'mmsi':str})
    quality = dict(np.load(prior.HERE/'quality.npz'))
    old = pd.read_csv(PREVIOUS/'predictions.csv')
    old = old[old.arm.eq('margin')].set_index(PKEYS)
    shared = pd.read_csv(ADDITIVE/'predictions.csv')
    shared = shared[shared.arm.eq('additive')].set_index(PKEYS)
    metrics,predictions,timings,checks,latencies = [],[],[],[],[]
    start_all = time.perf_counter()
    with threadpool_limits(limits=1):
        for args in prior.old.folds():
            data,features,train_tasks,val_tasks,tasks,_,_ = prior.prepare_fold(args,manifest,quality)
            fold = data['fold']
            folder = HERE/fold
            folder.mkdir(exist_ok=True)
            task_folder = folder/'tasks'
            task_folder.mkdir(exist_ok=True)
            features['quality'] = torch.zeros_like(features['quality'])
            with np.load(PREVIOUS/fold/'universal_features.npz') as cache:
                np.testing.assert_array_equal(cache['sample_ids'].astype(str),manifest.sample_id.to_numpy(str))
                for key in ['body','tokens','positions','valid']:
                    features['common_'+key] = torch.tensor(cache[key][data['rows']])
            labels = torch.tensor(data['labels'],dtype=torch.long)
            for task in tasks:
                arrays = {key:task[key].numpy() if isinstance(task[key],torch.Tensor) else task[key]
                          for key in ['ids','s','q','ns','ss','qs','root','desired','prior_s','prior_q','base','old']}
                np.savez_compressed(task_folder/f"{task['origin']}_{task['draw']}.npz",**arrays)
            (folder/'data_protocol.json').write_text(json.dumps(dict(train=data['rows'][data['train']].tolist(),
                selection=data['rows'][data['val']].tolist(),query=data['rows'][data['query']].tolist(),
                old_reader_steps=200,new_steps=200,axis_weight=1.,common_weight=.5),indent=2),encoding='utf-8')
            print('FOLD',fold,'QUERY',len(data['query']),flush=True)
            for seed in range(3):
                axis_reader = EvidenceReader('static')
                axis_reader.load_state_dict(torch.load(PREVIOUS/fold/f'margin_{seed}'/'model.pt',weights_only=True))
                old_state = {key:value.clone() for key,value in axis_reader.state_dict().items()}
                fixed = freeze_axis(axis_reader,features)
                base_cache = np.load(ADDITIVE/fold/f'margin_{seed}'/'readouts.npz')
                np.testing.assert_array_equal(base_cache['global_rows'],data['rows'])
                np.testing.assert_array_equal(fixed.numpy(),base_cache['axis_raw'])
                for arm in ['margin','initial_common','frozen_common','axis_continue','shared_additive']:
                    out = folder/f'{arm}_{seed}'
                    out.mkdir(exist_ok=True)
                    tic = time.perf_counter()
                    axis,common = fixed,None
                    trained = arm in ['frozen_common','axis_continue']
                    grad_norm = np.nan
                    if arm in ['initial_common','frozen_common']:
                        torch.manual_seed(20261017+seed)
                        reader = EvidenceReader('static')
                        if trained:
                            loss = task_loss(reader,features,train_tasks[0],labels,fixed)[0]
                            grads = torch.autograd.grad(loss,[p for p in reader.parameters() if p.requires_grad])
                            grad_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
                            assert np.isfinite(grad_norm) and grad_norm>0
                            fit(reader,features,train_tasks,val_tasks,labels,seed,out,fixed)
                        common = all_raw(reader,features,True)
                    elif arm=='axis_continue':
                        reader = EvidenceReader('static')
                        reader.load_state_dict(old_state)
                        fit(reader,features,train_tasks,val_tasks,labels,seed,out)
                        axis = all_raw(reader,features)
                    elif arm=='shared_additive':
                        with np.load(ADDITIVE/fold/f'additive_{seed}'/'readouts.npz') as cache:
                            np.testing.assert_array_equal(cache['global_rows'],data['rows'])
                            axis,common = torch.tensor(cache['axis_raw']),torch.tensor(cache['common_raw'])
                    seconds = time.perf_counter()-tic
                    if trained:
                        expected = np.load(ADDITIVE/fold/f'additive_{seed}'/'schedule.npz')['episode']
                        np.testing.assert_array_equal(np.load(out/'schedule.npz')['episode'],expected)
                    for key,value in axis_reader.state_dict().items():
                        torch.testing.assert_close(value,old_state[key],atol=0,rtol=0)
                    assert all(p.grad is None and not p.requires_grad for p in axis_reader.parameters())
                    torch.testing.assert_close(all_raw(axis_reader,features),fixed,atol=0,rtol=0)
                    m,p = score_arrays(axis,common,data,features,tasks,arm,seed,out)
                    metrics.extend(m)
                    predictions.extend(p)
                    replay = np.nan
                    if arm in ['margin','shared_additive']:
                        current = pd.DataFrame(p).set_index(PKEYS)
                        columns = [f'p{k}' for k in range(8)]
                        reference = old if arm=='margin' else shared
                        replay = float(abs(current[columns].to_numpy()-reference.loc[current.index,columns].to_numpy()).max())
                        assert replay<1e-10
                    checks.append(dict(fold=fold,seed=seed,arm=arm,old_state_max_error=0.,old_readout_max_error=0.,
                        old_gradient_absent=True,common_initial_gradient_norm=grad_norm,probability_replay_max_error=replay))
                    timings.append(dict(fold=fold,seed=seed,arm=arm,trained=trained,seconds=seconds,
                        trainable_parameters=10973 if trained else 0,reader_parameters=22078 if common is not None and arm!='shared_additive' else 11039))
                    if arm=='frozen_common' and seed==0:
                        for row in cached_latency(axis_reader,reader,features,next(t for t in tasks if t['origin']=='cross')):
                            latencies.append(dict(fold=fold,**row))
                    print('DONE',fold,seed,arm,round(seconds,2),flush=True)
                for name,records in [('metrics',metrics),('predictions',predictions),('timings',timings),
                                     ('freeze_and_replay_checks',checks),('cached_latency',latencies)]:
                    pd.DataFrame(records).to_csv(HERE/(name+'.csv'),index=False)
    (HERE/'runtime.json').write_text(json.dumps(dict(main_seconds=time.perf_counter()-start_all),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).groupby(['origin','arm'])[['oa','ba','macro_brier','nll','support_map_r']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
