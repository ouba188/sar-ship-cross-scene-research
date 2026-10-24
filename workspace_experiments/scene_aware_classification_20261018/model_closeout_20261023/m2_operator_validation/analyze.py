"""Paired effects and exact head/representation case attribution."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import importlib.util
import json
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train
from operators import Candidate, regional_raw


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def margin(score, y):
    rival = np.where(np.arange(8)[None] == y[:, None], -np.inf, score).max(1)
    return score[np.arange(len(y)), y]-rival


@torch.inference_mode()
def attribution(data, model, j, c, rival, reference='old'):
    before = torch.as_tensor(data.original_summary[j:j+1], dtype=torch.float32, device='cuda')
    old = data.baseline[j, c]-data.baseline[j, rival]
    if reference=='conv_zero':
        saved = model.conv.expand.weight.clone()
        model.conv.expand.weight.zero_()
        before_score,before = model(data.batch([j],model.arm),True)
        old = float(before_score[0,c]-before_score[0,rival])
        model.conv.expand.weight.copy_(saved)
    after_score, after = model(data.batch([j], model.arm), True)
    b = data.batch([j], 'grid_refit')
    values = []
    for subset in range(64):
        desc = before.clone()
        for k in range(6):
            if subset & (1 << k):
                desc[:, k//3, k%3] = after[:, k//3, k%3]
        raw = regional_raw(b['s0'], desc, b['quality'])
        s = b['rest']+model.head((raw-model.mu)/model.scale)
        values.append(float(s[0, c]-s[0, rival]))
    phi = []
    for k in range(6):
        total = 0.
        for subset in range(64):
            if subset & (1 << k):
                continue
            n = subset.bit_count()
            total += math.factorial(n)*math.factorial(5-n)/720*(values[subset | (1 << k)]-values[subset])
        phi.append(total)
    assert abs(sum(phi)-(values[-1]-values[0])) < 2e-5
    np.testing.assert_allclose(values[-1], float(after_score[0, c]-after_score[0, rival]), atol=2e-5)
    intervention = {}
    for name, parameter in [('conv_zero', model.conv.expand.weight if model.conv is not None else None),
                            ('pool_static', model.pool.score[-1].weight if model.pool is not None else None)]:
        if parameter is None:
            continue
        saved = parameter.clone()
        parameter.zero_()
        s = model(data.batch([j],model.arm))[0]
        intervention[name] = dict(prediction=int(s.argmax()),fixed_margin=float(s[c]-s[rival]))
        parameter.copy_(saved)
    return dict(old_fixed_margin=float(old), new_fixed_margin=values[-1],
                head_change=values[0]-float(old), representation_change=values[-1]-values[0],
                region_shapley=phi, region_order=['VV_body','VV_end','VV_side','VH_body','VH_end','VH_side'],operator_interventions=intervention)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    complete = json.loads((HERE/'complete.json').read_text())
    arms, seeds = complete['arms'], complete['seeds']
    f = pd.read_csv(HERE/'table.csv', dtype={'mmsi': str})
    te = np.flatnonzero(f.role.eq('evaluation'))
    y = f.class_id.to_numpy(int)
    scores, rows, effects, candidates = {}, [], [], []
    for seed in seeds:
        data = train.Data(seed)
        scores[seed, 'old_both'] = data.baseline
        for arm in arms:
            scores[seed, arm] = np.load(HERE/'results'/f'{arm}_{seed}.npz')['score']
        del data
        for arm in ['old_both', *arms]:
            s = scores[seed, arm]
            z = f.iloc[te][['up_index','obs_id','class_id','class_name','mmsi','pass_group']].copy()
            z['seed'], z['arm'] = seed, arm
            z['prediction'] = s[te].argmax(1)
            z['correct'] = z.prediction.eq(z.class_id)
            z['margin'] = margin(s[te], y[te])
            rows.append(z)
        comparisons = [(arm, 'old_both') for arm in arms]
        comparisons += [('conv_grid','grid_refit'), ('learned_pool','static_pool'),
                        ('conv_static_pool','static_pool'), ('conv_learned_pool','conv_static_pool'),
                        ('conv_learned_pool','learned_pool')]
        for arm, comp in comparisons:
            if arm not in arms or comp not in ['old_both', *arms]:
                continue
            a = scores[seed, arm][te].argmax(1) == y[te]
            b = scores[seed, comp][te].argmax(1) == y[te]
            d = a.astype(float)-b
            effects.append(dict(seed=seed, arm=arm, comparator=comp, rescued=int((a&~b).sum()),
                                harmed=int((b&~a).sum()), net=int(d.sum()), OA_pp=100*d.mean(),
                                BA_pp=100*np.mean([d[y[te]==c].mean() for c in range(8)])))
        # Main joint candidate is fixed in the protocol, not chosen on evaluation accuracy.
        for arm in ('conv_learned_pool', 'conv_grid', 'learned_pool'):
            if arm not in arms:
                continue
            a, b = scores[seed, arm][te], scores[seed, 'old_both'][te]
            change = margin(a,y[te])-margin(b,y[te])
            for k,j in enumerate(te):
                ac,bc = a[k].argmax()==y[j], b[k].argmax()==y[j]
                status = 'CC' if ac and bc else ('WW' if not ac and not bc else ('rescue' if ac else 'harm'))
                candidates.append(dict(seed=seed,arm=arm,index=int(j),case=status,change=float(change[k])))
    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(HERE/'predictions.csv.gz', index=False, compression='gzip')
    paired = pd.DataFrame(effects)
    paired.to_csv(HERE/'paired.csv', index=False)
    aggregate = pd.read_csv(HERE/'metrics.csv').query("role=='evaluation'").groupby('arm')[['OA','BA']].agg(['mean','std'])
    aggregate.to_csv(HERE/'summary.csv')
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    order = ['old_both',*arms]
    for ax,key in zip(axes,('OA','BA')):
        means = aggregate.loc[order,(key,'mean')].to_numpy()*100
        std = aggregate.loc[order,(key,'std')].to_numpy()*100
        ax.bar(np.arange(len(order)),means,yerr=std,color=['#555555','#78949c','#78949c','#138f86','#138f86','#bd6b32','#7f519b'],capsize=3)
        ax.set_xticks(np.arange(len(order)),order,rotation=30,ha='right',fontsize=8)
        ax.set_ylabel(f'{key} (%)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Complete-model evaluation: matched 3 configurations; 485 repeated queries')
    fig.tight_layout()
    fig.savefig(HERE/'comparison.png',dpi=150)
    plt.close(fig)
    detail.groupby(['seed','arm','class_id','class_name']).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).to_csv(HERE/'classes.csv')
    detail.groupby(['seed','arm','pass_group']).agg(n=('correct','size'),accuracy=('correct','mean')).to_csv(HERE/'passes.csv')
    # Acquisition-cluster resampling preserves within-scene dependence; seeds averaged first.
    pg = f.iloc[te].pass_group.to_numpy()
    groups = np.unique(pg)
    rng = np.random.default_rng(1024)
    draws = rng.integers(len(groups), size=(2000,len(groups)))
    ct0 = np.array([[((pg==g)&(y[te]==c)).sum() for c in range(8)] for g in groups])
    ct = ct0[draws].sum(1)
    allclasses = (ct>0).all(1)
    intervals = []
    for arm, comp in paired[['arm','comparator']].drop_duplicates().itertuples(index=False):
        d = np.mean([(scores[s,arm][te].argmax(1)==y[te]).astype(float)-(scores[s,comp][te].argmax(1)==y[te]) for s in seeds],0)
        sm = np.array([[d[(pg==g)&(y[te]==c)].sum() for c in range(8)] for g in groups])[draws].sum(1)
        oa = 100*sm.sum(1)/ct.sum(1)
        ba = 100*(sm[allclasses]/ct[allclasses]).mean(1)
        intervals.append(dict(arm=arm,comparator=comp,OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),BA_valid_draws=int(allclasses.sum())))
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv',index=False)
    cf = pd.DataFrame(candidates)
    chosen = []
    for status in ('rescue','harm','CC','WW'):
        pool = cf[cf.arm.eq('conv_learned_pool') & cf.case.eq(status)]
        if len(pool):
            pool = pool[pool.seed.eq(pool.seed.min())]
            chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    # Inspect changed component-only cases even when the joint source selector rejects it.
    for arm in ('conv_grid','learned_pool'):
        for status in ('rescue','harm'):
            pool = cf[cf.arm.eq(arm)&cf.case.eq(status)]
            if len(pool):
                pool = pool[pool.seed.eq(pool.seed.min())]
                chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    # Explicit later diagnostic: inspect the active seed43 joint operator as well.
    for status in ('rescue','harm'):
        pool = cf[cf.arm.eq('conv_learned_pool')&cf.seed.eq(43)&cf.case.eq(status)]
        if len(pool):
            chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    geo = load('operator_geo', HERE.parent.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshots = {int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    uptable = pd.read_csv(train.UP/'table.csv',dtype={'mmsi':str})
    refs = np.load(train.LOCAL/'scores_layer1.npz')['refs']
    valid = np.load(train.UP/'summaries.npz')['valid']
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    records = []
    for case in chosen:
        seed, j, arm = int(case['seed']),int(case['index']),case['arm']
        i,c,obs = int(f.iloc[j].up_index),int(y[j]),int(f.iloc[j].obs_id)
        before,after = scores[seed,'old_both'][j],scores[seed,arm][j]
        # A harmed sample may switch to a different rival than the old runner-up.
        rival = int(after.argmax()) if case['case']=='harm' else int(np.where(np.arange(8)==c,-np.inf,before).argmax())
        rr = refs[i,[c,rival]].ravel()
        fig,ax = plt.subplots(2,5,figsize=(16,7))
        provenance = []
        for p in range(2):
            for k,ri in enumerate([i,*rr]):
                pix,fp,g,_ = geo.native(uptable,snapshots,int(ri),p)
                ax[p,k].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if g is not None:
                    masks = geo.soft_regions(g)*fp
                    for r,col in enumerate(('#20a780','#d77329','#3287d0')):
                        if masks[r].max()>.5:
                            ax[p,k].contour(masks[r],levels=[.5],colors=[col],linewidths=.55)
                label = 'QUERY' if k==0 else ('TRUE' if k<3 else 'RIVAL')
                ax[p,k].set_title(f'{("VV","VH")[p]} {label} {int(uptable.iloc[ri].obs_id)} valid={int(valid[ri,p])}\n{uptable.iloc[ri].class_name}',fontsize=8)
                ax[p,k].set(xticks=[],yticks=[])
                provenance.append(dict(up_index=int(ri),polarization=('VV','VH')[p],path=uptable.iloc[ri]['image_'+('vv','vh')[p]]))
        fig.suptitle(f"{arm} {case['case']} seed{seed}: old {before.argmax()} -> new {after.argmax()} | true {c}\nMargin {margin(before[None],y[j:j+1])[0]:+.4f} -> {margin(after[None],y[j:j+1])[0]:+.4f}")
        fig.text(.02,.02,'Native160 crop/pad; no resize. Green=body, orange=end, blue=side. References belong to unchanged M3, not new M2.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.15)
        path = out/f'{arm}_{case["case"]}_{obs}_{seed}.png'
        fig.savefig(path,dpi=120)
        plt.close(fig)
        data = train.Data(seed)
        ck = torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
        if ck['receipt']['selected_step']<0:
            att = dict(old_fixed_margin=float(before[c]-before[rival]),new_fixed_margin=float(after[c]-after[rival]),head_change=0.,representation_change=0.,region_shapley=[0.]*6,fallback=True)
        else:
            model = Candidate(arm,data.ck,data.wproject,data.tau).cuda()
            model.load_state_dict(ck['state'])
            att = attribution(data,model,j,c,rival)
            np.testing.assert_allclose(att['new_fixed_margin'],after[c]-after[rival],atol=2e-5)
            del model
        records.append(dict(**case,obs_id=obs,true_class=c,rival=rival,rival_rule='new wrong prediction for harm; old strongest wrong class otherwise',old_pred=int(before.argmax()),new_pred=int(after.argmax()),selected_step=ck['receipt']['selected_step'],attribution=att,reference_obs=uptable.iloc[rr].obs_id.astype(int).tolist(),provenance=provenance,path=str(path)))
        del data
        torch.cuda.empty_cache()
    (out/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    receipt = dict(queries=len(te),ships=f.iloc[te].mmsi.nunique(),passes=len(groups),cases=len(records),bootstrap='2000 acquisition clusters; average seeds first; BA conditional on all classes',case_selection='joint four statuses at earliest seed; component-only rescue/harm; active seed43 joint rescue/harm added after operator activity diagnosis; largest absolute margin change; diagnostic not representative')
    (HERE/'analysis_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(paired.to_string(index=False),flush=True)
    print(pd.DataFrame(records)[['arm','case','seed','obs_id','attribution']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
