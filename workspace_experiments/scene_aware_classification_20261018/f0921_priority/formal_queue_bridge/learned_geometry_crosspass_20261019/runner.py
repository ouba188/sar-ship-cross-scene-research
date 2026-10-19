"""Learn only the supplemental readout, through the unchanged support solve."""
import json
import os
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from prepare import HERE, BRIDGE, load, unique_rows
from model import Reader, centered, residual_probability, substitute_geometry

roi = load('learned_roi',BRIDGE/'roi_controls_20261019/controls.py')
bridge = roi.bridge


class Study:
    def __init__(self):
        self.meta = pd.read_csv(HERE/'cohort.csv',dtype={'mmsi':str,'date':str})
        self.tasks = json.loads((HERE/'tasks.json').read_text())
        self.y = self.meta.label.to_numpy(int)
        self.features = dict(np.load(BRIDGE/'features.npz'))
        arrays = dict(np.load(HERE/'local_tokens.npz'))
        np.testing.assert_array_equal(arrays['clean_rows'],self.meta.clean_row)
        geo = pd.read_csv(HERE/'recovery/expanded_geometry.csv')
        axes = pd.read_csv(BRIDGE/'orbit_geometry_readout_20261019/axes.csv')
        self.valid = (geo.orbit_valid & axes.half_width.gt(0) & axes.axis_reliability.gt(1e-8)).to_numpy()
        relative = geo.los_angle_image_rad.to_numpy()-axes.ship_axis_rad.to_numpy()
        incidence = np.radians(geo.orbit_incidence_deg.to_numpy())
        self.geometry = np.nan_to_num(np.stack([np.cos(relative),np.sin(relative),np.sin(incidence),np.cos(incidence)],1)).astype(np.float32)
        self.geometry[~self.valid] = 0
        self.fit = np.array(self.tasks['fit'])
        self.donors = self.fit[self.valid[self.fit]]
        assert len(self.donors)>1
        mass = arrays['mass']
        content = arrays['content'].astype(np.float32)
        w = mass[self.fit,:,None]
        mean = (content[self.fit]*w[:,None]).sum((0,2))/w.sum()
        variance = ((content[self.fit]-mean[None,:,None])**2*w[:,None]).sum((0,2))/w.sum()
        std = np.sqrt(variance).clip(.01)
        content = np.clip((content-mean[None,:,None])/std[None,:,None],-8,8)
        self.inputs = [torch.tensor(a,device='cuda') for a in (content,arrays['position'],mass)]
        self.availability = torch.tensor(self.valid,device='cuda')
        self.labels = torch.tensor(self.y,device='cuda')
        self.geometry_tensor = torch.tensor(self.geometry,device='cuda')
        np.savez_compressed(HERE/'normalization.npz',mean=mean,std=std)
        report = []
        for name,ids in [('source',self.tasks['source']),('fit',self.fit),('validation',self.tasks['validation'])]:
            report.append(dict(split=name,ships=len(ids),geometry=int(geo.iloc[ids].orbit_valid.sum()),
                               readout=int(self.valid[ids].sum()),classes=sorted(np.unique(self.y[ids]).tolist())))
        (HERE/'learning_coverage.json').write_text(json.dumps(report,indent=2))

    def context(self, source, support, query):
        source,support,query = [np.asarray(a,dtype=int) for a in (source,support,query)]
        assert len(source)>0 and len(support)>0 and len(query)>0
        for a,b in [(source,query),(source,support),(support,query)]:
            assert not set(self.meta.iloc[a].mmsi)&set(self.meta.iloc[b].mmsi)
        assert not set(self.meta.iloc[source].date)&set(self.meta.iloc[query].date)
        g = bridge.global_descriptor(self.features['global'],source)
        b = bridge.global_descriptor(self.features['r18'],source)
        kg,kb = g@g.T,b@b.T
        pa,pb = bridge.prior(kg,source,self.y,7),bridge.prior(kb,source,self.y,7)
        merged,valid = roi.merged_regions(self.features['prepool'],self.features['mass12'])
        z,_ = bridge.token_descriptor(merged,valid,support)
        grid = bridge.cosine(self.features['grid2'].reshape(len(g),2,-1),support)
        base = .5*kg+.25*grid+.25*bridge.kernel(z)
        expert,_ = bridge.adapted(kb,pb,support,self.y)
        ids = np.r_[support,query]
        ns = len(support)
        ct = dict(source=source,s=support,q=query,ids=ids,ns=ns)
        ct.update({k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in
                   [('base',base[np.ix_(ids,ids)]),('prior',pa[ids]),('expert',expert[ids])]})
        ct['base_prob'],ct['base_alpha'],ct['base_score'] = residual_probability(ct['base'],ct['prior'],ct['expert'],self.labels[support],ns)
        score,_ = bridge.adapted(base,pa,support,self.y)
        np.testing.assert_allclose(ct['base_prob'].cpu(),bridge.late(score[query],expert[query]),atol=2e-12)
        return ct

    def source_context(self, support, query):
        held_dates = set(self.meta.iloc[query].date)
        excluded_ships = set(self.meta.iloc[np.r_[support,query]].mmsi)
        source = self.fit[~self.meta.iloc[self.fit].date.isin(held_dates)
                          & ~self.meta.iloc[self.fit].mmsi.isin(excluded_ships)]
        return self.context(source,support,query)

    def episodes(self):
        rng = np.random.default_rng(14191)
        fitmeta = self.meta.iloc[self.fit]
        dates = sorted(fitmeta.date.unique())
        same_groups = [g.row.to_numpy() for _,g in fitmeta[fitmeta.pass_known].groupby('pass_key') if len(g)>=24]
        fit_tasks, records = [], []
        for i in range(32):
            if i%2==0:
                bank = same_groups[(i//2)%len(same_groups)]
                order = rng.permutation(bank)
                s,q = order[:16],order[16:24]
                line = 'same_pass'
            else:
                date = dates[(i//2)%len(dates)]
                qpool = fitmeta[fitmeta.date.eq(date)].row.to_numpy()
                spool = fitmeta[~fitmeta.date.eq(date)].row.to_numpy()
                q,s = rng.choice(qpool,min(8,len(qpool)),replace=False),rng.choice(spool,16,replace=False)
                line = 'cross_date'
            context = self.source_context(s,q)
            fit_tasks.append(context)
            records.append(dict(split='fit',index=i,line=line,source=context['source'].tolist(),support=s.tolist(),query=q.tolist()))
        val_tasks = []
        q = np.asarray(self.tasks['validation'])
        for i in range(5):
            s = rng.choice(self.fit,16,replace=False)
            context = self.source_context(s,q)
            val_tasks.append(context)
            records.append(dict(split='validation',index=i,line='cross_date',source=context['source'].tolist(),support=s.tolist(),query=q.tolist()))
        (HERE/'source_episodes.json').write_text(json.dumps(records,indent=2))
        return fit_tasks,val_tasks

    def predict(self, reader, ct, geometry=None):
        ids,ns = ct['ids'],ct['ns']
        geometry = self.geometry_tensor if geometry is None else geometry
        raw,weights = reader(*(x[ids] for x in self.inputs),geometry[ids],self.availability[ids])
        z = centered(raw,self.availability[ids],ns).double()
        k = ct['base']+.0625*(z@z.T)
        p,alpha,score = residual_probability(k,ct['prior'],ct['expert'],self.labels[ct['s']],ns)
        return p,k,alpha,score,weights

    def wrong_geometry(self, seed):
        g,receivers,donors = substitute_geometry(self.geometry,self.valid,self.donors,np.random.default_rng(seed))
        return torch.tensor(g,device='cuda'),receivers,donors


def fit(study,arm,seed,fit_tasks,val_tasks):
    torch.manual_seed(seed)
    reader = Reader(arm).cuda()
    optimizer = torch.optim.AdamW(reader.parameters(),lr=.001,weight_decay=.0001)
    schedule = np.random.default_rng(22000+seed).integers(0,len(fit_tasks),(200,2))
    history = []
    best_loss,best_step,best_state = np.inf,None,None
    start = time.perf_counter()
    for step in range(201):
        if step:
            reader.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for j,index in enumerate(schedule[step-1]):
                ct = fit_tasks[index]
                geo = study.wrong_geometry(30000+seed*10000+step*2+j)[0] if arm=='D' else None
                p,*_ = study.predict(reader,ct,geo)
                losses.append(-p[torch.arange(len(ct['q']),device='cuda'),study.labels[ct['q']]].clamp_min(1e-12).log().mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(),5,error_if_nonfinite=True)
            optimizer.step()
        if step in (0,50,100,200):
            reader.eval()
            with torch.no_grad():
                risk = []
                for j,ct in enumerate(val_tasks):
                    geo = study.wrong_geometry(60000+seed*100+j)[0] if arm=='D' else None
                    p,*_ = study.predict(reader,ct,geo)
                    risk.append(float(-p[torch.arange(len(ct['q']),device='cuda'),study.labels[ct['q']]].clamp_min(1e-12).log().mean()))
                val = np.mean(risk)
                history.append(dict(arm=arm,seed=seed,step=step,validation_nll=val,
                    last_training_nll=None if not step else float(loss.detach()),gradient_norm=None if not step else float(norm)))
                if val<best_loss:
                    best_loss,best_step = val,step
                    best_state = {k:v.detach().clone() for k,v in reader.state_dict().items()}
    reader.load_state_dict(best_state)
    torch.save(best_state,HERE/f'{arm}_seed{seed}.pt')
    pd.DataFrame(history).to_csv(HERE/f'learning_{arm}_{seed}.csv',index=False)
    result = dict(arm=arm,seed=seed,steps_run=200,selected_step=best_step,
                  validation_nll=best_loss,seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in reader.parameters()))
    print('FIT',json.dumps(result),flush=True)
    return reader,result


def collect(study,reader,arm,seed,task,draw,ct,geometry=None):
    start = time.perf_counter()
    with torch.no_grad():
        if arm=='base':
            prob,k,alpha,score = ct['base_prob'],ct['base'],ct['base_alpha'],ct['base_score']
        else:
            prob,k,alpha,score,_ = study.predict(reader,ct,geometry)
        prob,k,alpha,score = [x.cpu().numpy() for x in (prob,k,alpha,score)]
    q,s,ns = ct['q'],ct['s'],ct['ns']
    sim = k[ns:,:ns]
    ap = np.array([bridge.ap_at_r(v,study.y[s]==c) for v,c in zip(sim,study.y[q])])
    common = dict(task=task['name'],line=task['line'],target_pass=task['target_pass'],budget=task['budget'],draw=draw,seed=seed,arm=arm)
    metric = dict(**common,queries=len(q),seconds=time.perf_counter()-start,**bridge.metrics(prob,study.y[q],ap))
    records = []
    for j,i in enumerate(q):
        truth = study.y[i]
        order = np.argsort(-sim[j],kind='stable')
        same = [int(s[t]) for t in order if study.y[s[t]]==truth]
        wrong = [int(s[t]) for t in order if study.y[s[t]]!=truth]
        rec = dict(**common,row=int(i),truth=int(truth),prediction=int(prob[j].argmax()),
            correct=bool(prob[j].argmax()==truth),margin=float(prob[j,truth]-np.delete(prob[j],truth).max()),
            brier=float(((prob[j]-np.eye(7)[truth])**2).sum()),nll=float(-np.log(max(prob[j,truth],1e-12))),
            ap=float(ap[j]),readout_valid=bool(study.valid[i]),same_support=len(same),
            top_same=same[0] if same else -1,top_wrong=wrong[0] if wrong else -1)
        rec.update({f'p{c}':float(prob[j,c]) for c in range(7)})
        records.append(rec)
    return metric,records


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    study = Study()
    fit_tasks,val_tasks = study.episodes()
    contexts = [(t,d,study.context(t['source'],s,t['query'])) for t in study.tasks['tasks'] for d,s in enumerate(t['supports'])]
    timings,metrics,predictions,donations = [],[],[],[]
    for seed in (0,1,2):
        for arm in ('A','B','C','D'):
            reader,timing = fit(study,arm,seed,fit_tasks,val_tasks)
            timings.append(timing)
            for task,draw,ct in contexts:
                key = 70000+seed*1000+int(task['name'][0])*100+draw
                wrong,receivers,donors = study.wrong_geometry(key)
                metric,recs = collect(study,reader,arm,seed,task,draw,ct,wrong if arm=='D' else None)
                metrics.append(metric);predictions.extend(recs)
                if arm=='C':
                    metric,recs = collect(study,reader,'C_test_mismatch',seed,task,draw,ct,wrong)
                    metrics.append(metric);predictions.extend(recs)
                if arm=='A':
                    metric,recs = collect(study,reader,'base',seed,task,draw,ct)
                    metrics.append(metric);predictions.extend(recs)
                    for r,d in zip(receivers,donors):
                        if r in ct['ids']:
                            angle = np.arccos(np.clip(study.geometry[r,:2]@study.geometry[d,:2],-1,1))*180/np.pi
                            donations.append(dict(seed=seed,task=task['name'],draw=draw,receiver=int(r),donor=int(d),angle_gap_deg=float(angle)))
            pd.DataFrame(timings).to_csv(HERE/'training.csv',index=False)
            pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
            pd.DataFrame(predictions).to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(donations).to_csv(HERE/'mismatch_donors.csv',index=False)
    met = pd.DataFrame(metrics)
    cols = ['oa','ba','brier','nll','map_r']
    met.groupby(['line','budget','arm'])[cols].mean().to_csv(HERE/'summary.csv')
    met.groupby(['line','budget','seed','arm'])[cols].mean().to_csv(HERE/'seed_summary.csv')
    receipt = dict(finished=True,seconds=time.perf_counter()-start,training_runs=len(timings),
        seeds=3,support_draws=5,target_tasks=len(study.tasks['tasks']),independent_confirmation=False,
        source_ships=len(study.tasks['source']),fit_ships=len(study.fit),validation_ships=len(study.tasks['validation']),
        geometry_valid=int(study.valid.sum()),learned_backbone=False)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
