"""Small matched pre-pooling experiment on the frozen legal episode manifest."""
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
OLD=BRIDGE/'learned_geometry_crosspass_20261019'
sys.path.insert(0,str(OLD))
import runner as prior_run
from model import centered,residual_probability
from conv_model import ConvReader,scramble_cells


def matched_drop(weights,mass,available):
    high=np.zeros(weights.shape,bool);control=np.zeros(weights.shape,bool)
    for i in np.flatnonzero(available):
        for p in range(2):
            for region in range(2):
                m=mass[i,region]
                w=weights[i,p,region]
                candidates=np.flatnonzero(m>1e-8)
                for a in candidates[np.argsort(-w[candidates])]:
                    others=candidates[(abs(m[candidates]-m[a])<=.2*m[a])&(w[candidates]<w[a])]
                    if len(others):
                        b=others[np.argmin(w[others])]
                        high[i,p,region,a]=True
                        control[i,p,region,b]=True
                        break
    return high,control


class Study(prior_run.Study):
    def __init__(self):
        self.meta=pd.read_csv(OLD/'cohort.csv',dtype={'mmsi':str,'date':str})
        self.tasks=json.loads((OLD/'tasks.json').read_text())
        self.y=self.meta.label.to_numpy(int)
        self.features=dict(np.load(BRIDGE/'features.npz'))
        self.fit=np.asarray(self.tasks['fit'])
        geo=pd.read_csv(OLD/'recovery/expanded_geometry.csv')
        axes=pd.read_csv(BRIDGE/'orbit_geometry_readout_20261019/axes.csv')
        self.valid=(geo.orbit_valid&axes.half_width.gt(0)&axes.axis_reliability.gt(1e-8)).to_numpy()
        self.availability=torch.tensor(self.valid,device='cuda')
        self.labels=torch.tensor(self.y,device='cuda',dtype=torch.long)
        self.geometry_tensor=torch.zeros(len(self.meta),4,device='cuda')
        maps=torch.tensor(np.load(HERE/'maps.npy'),device='cuda')
        fitmaps=maps[self.fit]
        mean=fitmaps.mean((0,3,4),keepdim=True)
        std=fitmaps.std((0,3,4),unbiased=False,keepdim=True).clamp_min(.01)
        del fitmaps
        maps=(maps-mean)/std
        masks=torch.tensor(np.load(HERE/'masks32.npy'),device='cuda')
        positions=torch.tensor(np.load(OLD/'local_tokens.npz')['position'],device='cuda')
        self.inputs=[maps,masks,positions]
        self.slot_mass=F.avg_pool2d(masks,4).reshape(len(masks),2,64).cpu().numpy()
        self.intervention=None
        self.permutation=torch.tensor(np.random.default_rng(19021).permutation(16),device='cuda')
        self.drop_coverage=[]
        np.savez_compressed(HERE/'normalization.npz',mean=mean.cpu().numpy(),std=std.cpu().numpy(),fit=self.fit)

    def episodes(self):
        records=json.loads((OLD/'source_episodes.json').read_text())
        groups={'fit':[],'validation':[]}
        for r in records:
            groups[r['split']].append(self.context(r['source'],r['support'],r['query']))
        return groups['fit'],groups['validation']

    def predict(self,reader,ct,geometry=None):
        ids,ns=ct['ids'],ct['ns']
        maps,masks,position=[x[ids] for x in self.inputs]
        if self.intervention=='scramble':
            maps,masks=scramble_cells(maps,masks,self.permutation)
        raw,weights=reader(maps,masks,position,self.geometry_tensor[ids],self.availability[ids])
        if self.intervention in ('drop_high','drop_control'):
            high,control=matched_drop(weights.detach().cpu().numpy(),self.slot_mass[ids],self.valid[ids])
            drop=high if self.intervention=='drop_high' else control
            self.drop_coverage.append(dict(observations=len(ids),valid_observations=int(self.valid[ids].sum()),
                matched_region_polarizations=int(drop.sum()),possible_region_polarizations=int(self.valid[ids].sum())*4))
            raw,weights=reader(maps,masks,position,self.geometry_tensor[ids],self.availability[ids],torch.tensor(drop,device='cuda'))
        z=centered(raw,self.availability[ids],ns).double()
        k=ct['base']+.0625*z@z.T
        p,alpha,score=residual_probability(k,ct['prior'],ct['expert'],self.labels[ct['s']],ns)
        return p,k,alpha,score,weights


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Reuse the proven trainer, redirecting only its output directory and reader class.
    prior_run.HERE=HERE
    prior_run.Reader=ConvReader
    study=Study()
    fits,vals=study.episodes()
    contexts=[(t,d,study.context(t['source'],s,t['query'])) for t in study.tasks['tasks'] for d,s in enumerate(t['supports'])]
    training,metrics,predictions=[],[],[]
    invariance=[]
    for seed in (0,1,2):
        for mode in ('pool_first','point_first','conv_first'):
            study.intervention=None
            reader,timing=prior_run.fit(study,mode,seed,fits,vals)
            training.append(timing)
            for task,draw,ct in contexts:
                variants=[(mode,None)]
                if mode=='pool_first':variants.append(('base',None))
                if mode=='conv_first':
                    variants.extend([('conv_scramble','scramble'),('conv_drop_high','drop_high'),('conv_drop_control','drop_control')])
                for name,intervention in variants:
                    study.intervention=intervention
                    met,recs=prior_run.collect(study,reader,name,seed,task,draw,ct)
                    metrics.append(met);predictions.extend(recs)
                if mode in ('pool_first','point_first') and draw==0:
                    with torch.no_grad():
                        study.intervention=None;p0=study.predict(reader,ct)[0]
                        study.intervention='scramble';p1=study.predict(reader,ct)[0]
                        err=float((p0-p1).abs().max())
                        assert err<2e-7
                        invariance.append(dict(mode=mode,seed=seed,task=task['name'],probability_max_error=err))
                study.intervention=None
            pd.DataFrame(training).to_csv(HERE/'training.csv',index=False)
            pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
            pd.DataFrame(predictions).to_csv(HERE/'predictions.csv',index=False)
    met=pd.DataFrame(metrics)
    met.groupby(['line','budget','arm'])[['oa','ba','brier','nll','map_r']].mean().to_csv(HERE/'summary.csv')
    pd.DataFrame(invariance).to_csv(HERE/'scramble_invariance.csv',index=False)
    pd.DataFrame(study.drop_coverage).to_csv(HERE/'drop_coverage.csv',index=False)
    old=pd.read_csv(OLD/'predictions.csv')
    key=['task','seed','draw','row']
    p=pd.DataFrame(predictions)
    a=p[p.arm.eq('base')].set_index(key).sort_index()[[f'p{i}' for i in range(7)]]
    b=old[old.arm.eq('base')].set_index(key).sort_index()[[f'p{i}' for i in range(7)]]
    np.testing.assert_allclose(a,b,atol=2e-12)
    receipt=dict(finished=True,seconds=time.perf_counter()-started,training_runs=len(training),
        frozen_backbone=True,queries=48,source_validation_ships=6,independent_confirmation=False,
        baseline_replay_max_error=float(abs(a-b).to_numpy().max()),scramble_changes_only_fine_arrangement=True)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
