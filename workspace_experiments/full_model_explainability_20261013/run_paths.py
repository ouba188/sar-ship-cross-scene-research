"""Replay the frozen complete model and intervene on its exact score paths."""

import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
from backbone_ablation_core import AblationContext, r50_views, _conditions, EPISODES, _expert_inputs
from unified_head_run import base_kernel
from projection_residual_expert import discarded_coordinates

MODES=tuple(f'{p}{w}{b}' for p in (0,1) for w in (0,1) for b in (0,1))
PARTS=('base','r50_expert','prior_delta','old_relation_reweight','extra_relation')
BATCH=256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def arrays(path,keys=None):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in (z.files if keys is None else keys)}


def mapped(path,keys,directory):
    directory.mkdir(parents=True,exist_ok=True)
    answer={}
    with zipfile.ZipFile(path) as archive:
        for key in keys:
            dest=directory/(key+'.npy')
            if not dest.exists():
                with archive.open(key+'.npy') as source,dest.open('wb') as target:
                    shutil.copyfileobj(source,target,length=1024*1024)
            answer[key]=np.load(dest,mmap_mode='r',allow_pickle=False)
    return answer


class Geometry:
    def __init__(self,task,data,coeff,dest):
        self.ctx=AblationContext(Path(task['prepared']).parent)
        self.data,self.coeff=data,coeff
        c=self.ctx
        if c.dataset=='self24ports':
            c.model,c.stats=c.geometry['model'],c.geometry['stats']
            self.bank=np.load(c.parent/'target_bank.npy',mmap_mode='r')
            self.extra=np.load(EPISODES/c.task/'matched_joint/joint_extra.npy',mmap_mode='r')
            np.testing.assert_array_equal(np.load(c.parent/'target_rows.npy'),data['rows'])
            self.raw=c.raw
            self.conditions=_conditions()[data['rows']]
        else:
            target=Path(read(c.joint.parent/'protocol.json')['target'])
            if c.dataset=='oss2':
                self.raw=mapped(target/'features.npz',['paired_r50'],dest/'raw')['paired_r50']
            else:
                self.raw=np.load(target/'paired_r50.npy',mmap_mode='r')
            self.extra=np.load(c.joint/'target_extra.npy',mmap_mode='r')
            np.testing.assert_array_equal(data['rows'],np.arange(len(self.raw)))

    def encode(self,ids):
        c=self.ctx
        extra=np.asarray(self.extra[ids])
        if c.dataset=='self24ports':
            from run_protected_drift_activation import make_views
            bank=np.asarray(self.bank[ids])
            raw=self.raw[self.data['rows'][ids]]
            vv=make_views(bank,raw,extra,c.geometry)
            retained,comp=vv['common'][True]['retained'],vv['common'][True]['complement']
            rr,rd=r50_views(raw,c.model,c.stats,bank,c.geometry)
            aug=np.c_[bank,self.conditions[ids]]
            keep=np.r_[np.arange(128),np.arange(192,aug.shape[1])]
            pt=aug@self.coeff['dual_coef']+self.coeff['dual_intercept']
            pr=aug[:,keep]@self.coeff['r50_coef'][keep]+self.coeff['r50_intercept']
            stable,condition=vv['stable'],self.conditions[ids]
        else:
            raw=np.asarray(self.raw[ids])
            x=_expert_inputs(raw,extra,c.model)
            retained=(x/np.sqrt(192)).astype(float)
            residual=np.c_[discarded_coordinates(raw,c.model['pca']),
                           discarded_coordinates(extra,c.model['pca_ssl'])]
            comp=(residual-c.stats['discarded_center'])/c.stats['discarded_scale']/np.sqrt(2560)
            rr,rd=r50_views(raw,c.model,c.stats)
            pt=(x@self.coeff['dual_coef']+self.coeff['dual_intercept']).astype(np.float32).astype(float)
            # Use the original coordinate function; scaling back can perturb float32 prior rounding.
            from backbone_ablation_core import r50_coordinates
            x50=r50_coordinates(raw,c.model)
            pr=(x50@self.coeff['r50_coef']+self.coeff['r50_intercept']).astype(np.float32).astype(float)
            stable=condition=None
        np.testing.assert_allclose(rr,retained[:,:128],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(rd,comp[:,:2048],atol=1e-10,rtol=1e-10)
        return dict(rr=rr,rd=rd,br=retained[:,128:],bd=comp[:,2048:],
                    pr=pr,pt=pt,stable=stable,condition=condition)

    def kernels(self,q,s):
        kr=.5*(q['rr']@s['rr'].T+q['rd']@s['rd'].T)+1.
        kb=.5*(q['br']@s['br'].T+q['bd']@s['bd'].T)
        if q['stable'] is not None:
            kr+=.25*(q['stable']@s['stable'].T)+.125*(q['condition']@s['condition'].T)
        return kr,kb


def run(task):
    dest=OUT/'tasks'/task['task']
    if (dest/'FROZEN.json').exists():
        print('DONE',task['task'],flush=True)
        return
    dest.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    data=mapped(task['input_cache'],['rows','ids','support','support_labels','eval_masks','phi',
                'mean','conditions','prior'],dest/'input')
    emb=mapped(task['frequency_embedding'],['rows','embedding'],dest/'embedding')
    reference=arrays(task['predictions'])
    coeff=arrays(task['coefficients'])
    for key in ('rows','ids','support','eval_masks'):
        np.testing.assert_array_equal(data[key],reference[key])
    np.testing.assert_array_equal(data['rows'],emb['rows'])
    geo=Geometry(task,data,coeff,dest)
    union=np.unique(data['support'])
    refs=geo.encode(union)
    n=len(data['rows'])
    parts=np.lib.format.open_memmap(dest/'parts.npy',mode='w+',dtype='float64',shape=(5,5,n,8))
    pred=np.lib.format.open_memmap(dest/'predictions.npy',mode='w+',dtype='int8',shape=(5,8,n))
    expertpred=np.lib.format.open_memmap(dest/'expert_predictions.npy',mode='w+',dtype='int8',shape=(5,8,n))
    worst_identity=0.
    for lo in range(0,n,BATCH):
        ids=np.arange(lo,min(lo+BATCH,n))
        q=geo.encode(ids)
        kr,kb=geo.kernels(q,refs)
        for draw,support in enumerate(data['support']):
            cols=np.searchsorted(union,support)
            ra,ta=coeff[f'{draw}_r50_expert'],coeff[f'{draw}_dual_refit_expert']
            pdelta=q['pt']-q['pr']
            oldrel=kr[:,cols]@(ta-ra)
            extrarel=kb[:,cols]@ta
            er=q['pr']+kr[:,cols]@ra
            et=q['pt']+(kr[:,cols]+kb[:,cols])@ta
            np.testing.assert_allclose(er+pdelta+oldrel+extrarel,et,atol=1e-9,rtol=1e-10)
            worst_identity=max(worst_identity,float(abs(er+pdelta+oldrel+extrarel-et).max()))
            both=np.r_[support,ids]
            local={k:data[k][both] for k in ('phi','mean','conditions')}
            bk=base_kernel(local,emb['embedding'][both].astype(float),np.arange(64),geo.ctx.info['eta'])[64:]
            base=data['prior'][ids]+bk@coeff[f'{draw}_base']
            parts[draw,:,ids[0]:ids[-1]+1]=np.stack((base,er,pdelta,oldrel,extrarel))
            for m,mode in enumerate(MODES):
                z=er+int(mode[0])*pdelta+int(mode[1])*oldrel+int(mode[2])*extrarel
                expertpred[draw,m,ids]=z.argmax(1)
                pred[draw,m,ids]=(.5*softmax(base,axis=1)+.5*softmax(z,axis=1)).argmax(1)
            checks={'base':base.argmax(1),'r50_expert':expertpred[draw,0,ids],
                    'dual_refit_expert':expertpred[draw,7,ids],
                    'r50_late':pred[draw,0,ids],'dual_refit_late':pred[draw,7,ids]}
            for name,value in checks.items():
                expected=reference['predictions'][draw,list(reference['modes']).index(name),ids]
                if not np.array_equal(value,expected):
                    write(dest/'REPLAY_FAILED.json',dict(task=task['task'],draw=draw,mode=name,
                        rows=ids[value!=expected].tolist(),actual=value[value!=expected].tolist(),
                        expected=expected[value!=expected].tolist(),note='Stop; no relaxed tolerance.'))
                    raise AssertionError('Archive replay mismatch: '+task['task']+'/'+name)
        if lo%(BATCH*40)==0:
            print('REPLAY',task['task'],ids[-1]+1,n,round(time.perf_counter()-started,1),flush=True)
    for x in (parts,pred,expertpred):
        x.flush()
    write(dest/'FROZEN.json',dict(task=task['task'],rows=n,draws=5,modes=MODES,parts=PARTS,
        exact_argmax_replays=25,score_identity_max_error=worst_identity,
        seconds=time.perf_counter()-started,new_source_fits=0,new_support_fits=0,new_neural_fits=0))
    print('FROZEN',task['task'],round(time.perf_counter()-started,1),flush=True)
    del parts,pred,expertpred,geo,data,emb,reference
    gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',nargs='+')
    args=parser.parse_args()
    tasks=read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks']
    tasks=sorted(tasks,key=lambda t:(t['task'].startswith('f0921'),t['task']))
    if args.task:
        tasks=[t for t in tasks if t['task'] in args.task]
    with threadpool_limits(limits=1):
        for t in tasks:
            run(t)
