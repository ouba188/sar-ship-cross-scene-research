"""Paired frozen full-model versus regional controls, with native image panels."""
import os
from pathlib import Path
import importlib.util
import json
os.environ['OMP_NUM_THREADS'] = '2'
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    prior = HERE.parent / 'final_m3_coverage_20261026'
    run = load('evidence_24port_runtime', prior / 'evaluate.py')
    table = pd.read_csv(prior / 'table.csv', dtype={'mmsi': str})
    y = table.class_id.to_numpy(int)
    full = {s: dict(np.load(prior / 'results' / f'seed{s}.npz')) for s in run.SEEDS}
    controls = {r: dict(np.load(HERE / 'regional_24port' / r / 'expanded_predictions.npz'))
                for r in ('pooled', 'fit_only')}
    for z in controls.values():
        np.testing.assert_array_equal(z['obs_id'], table.obs_id)
    out = HERE / 'regional_24port' / 'analysis'
    out.mkdir(exist_ok=True)
    rows, paired, byclass, bygroup, intervals = [], [], [], [], []
    groups = list(table.groupby('pass_group').indices.values())
    counts = np.array([np.bincount(y[g], minlength=8) for g in groups])
    draws = np.random.default_rng(1026).integers(len(groups), size=(2000, len(groups)))
    ct = counts[draws].sum(1)
    complete = (ct > 0).all(1)
    for role, z in controls.items():
        differences = []
        for seed in run.SEEDS:
            a, b = full[seed]['frequency'], z[f'regional_mlp_seed{seed}']
            for method, score in (('full_frequency', a), ('regional_mlp', b)):
                rows.append(dict(role=role, seed=seed, method=method, **run.metrics(y, score)))
            ac, bc = a.argmax(1) == y, b.argmax(1) == y
            d = ac.astype(float) - bc
            differences.append(d)
            paired.append(dict(role=role, seed=seed, full_only=int((ac & ~bc).sum()),
                simple_only=int((bc & ~ac).sum()), OA_pp=100*d.mean(),
                BA_pp=100*np.mean([d[y == c].mean() for c in range(8)])))
            for c in range(8):
                byclass.append(dict(role=role, seed=seed, class_id=c, n=int((y == c).sum()),
                                    recall_pp=100*d[y == c].mean()))
            for group, ids in table.groupby('pass_group').indices.items():
                bygroup.append(dict(role=role, seed=seed, pass_group=group, n=len(ids), OA_pp=100*d[ids].mean()))
        d = np.mean(differences, 0)
        sums = np.array([np.bincount(y[g], weights=d[g], minlength=8) for g in groups])
        sm = sums[draws].sum(1)
        oa = 100*sm.sum(1)/ct.sum(1)
        ba = 100*(sm[complete]/ct[complete]).mean(1)
        intervals.append(dict(role=role, OA_low=np.quantile(oa,.025), OA_high=np.quantile(oa,.975),
            BA_low=np.quantile(ba,.025), BA_high=np.quantile(ba,.975), complete_draws=int(complete.sum())))
    for name, values in (('metrics',rows),('paired',paired),('by_class',byclass),
                          ('by_pass',bygroup),('uncertainty',intervals)):
        pd.DataFrame(values).to_csv(out / f'{name}.csv',index=False)
    pd.DataFrame(rows).groupby(['role','method'])[['OA','BA']].agg(['mean','std']).to_csv(out / 'summary.csv')
    ex = run.load('evidence_native_24port', run.COV / 'extract.py')
    snapshots = {int(v['obs_id']):v for v in json.loads((ex.joint.geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    source = pd.read_csv(run.UP / 'table.csv',dtype={'mmsi':str})
    bank = np.load(run.PACKAGE / 'assets/m3/bank.npz')
    st, tt = np.load(run.LOCAL / 'tokens.npz'), np.load(run.EXT / 'tokens.npz')
    a, b = full[17]['frequency'], controls['pooled']['regional_mlp_seed17']
    ac, bc = a.argmax(1) == y, b.argmax(1) == y
    def margin(score):
        return score[np.arange(len(y)),y] - np.where(np.arange(8)[None] == y[:,None], -np.inf,score).max(1)
    delta = margin(a)-margin(b)
    cases = []
    for status, mask in dict(full_only=ac & ~bc,simple_only=bc & ~ac,CC=ac & bc,WW=~ac & ~bc).items():
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        j = int(ids[np.abs(delta[ids]).argmax()])
        true = int(y[j])
        rival = int(np.where(np.arange(8) == true,-np.inf,a[j]).argmax())
        refs = bank['source_index'][full[17]['refs'][j,[true,rival]]].ravel()
        fig, ax = plt.subplots(2,5,figsize=(16,7))
        provenance = []
        for p, pol in enumerate(('VV','VH')):
            for k, i in enumerate([j,*refs]):
                f,tok = (table,tt) if k == 0 else (source,st)
                pix,fp,g,placement = ex.joint.geo.native(f,snapshots,int(i),p)
                ax[p,k].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                valid = bool(tok['valid'][i,p])
                if valid and g is not None:
                    mask_region = ex.joint.geo.soft_regions(g)[0]*fp
                    if mask_region.max() > .5:
                        ax[p,k].contour(mask_region,levels=[.5],colors=['#20a780'],linewidths=.7)
                label = 'QUERY' if k == 0 else ('TRUE REF' if k < 3 else 'RIVAL REF')
                ax[p,k].set_title(f'{pol} {label} {int(f.iloc[i].obs_id)} valid={int(valid)}\n{f.iloc[i].class_name}',fontsize=8)
                ax[p,k].set(xticks=[],yticks=[])
                provenance.append(dict(obs_id=int(f.iloc[i].obs_id),polarization=pol,
                    path=str(f.iloc[i]['image_'+pol.lower()]),valid=valid))
        name = f'{status}_{int(table.iloc[j].obs_id)}.png'
        fig.suptitle(f'{status}, seed17; true={true}; full={a[j].argmax()}, regional MLP={b[j].argmax()}\nNative crop/pad; reference chips are used by FULL model only.')
        fig.subplots_adjust(top=.86,bottom=.05,wspace=.15,hspace=.2)
        fig.savefig(out/name,dpi=120)
        plt.close(fig)
        cases.append(dict(status=status,index=j,obs_id=int(table.iloc[j].obs_id),true=true,
            full_pred=int(a[j].argmax()),simple_pred=int(b[j].argmax()),rival=rival,
            references=source.iloc[refs].obs_id.astype(int).tolist(),image=name,provenance=provenance))
    run.write_json(out / 'cases.json',cases)
    run.write_json(out / 'receipt.json',dict(seed=17,classes=8,n=len(y),passes=len(groups),
        case_selection='largest absolute uncalibrated margin difference within each status; explanatory only',
        bootstrap='paired pass-cluster, average across seeds first; BA conditional on all classes present',
        raw_visual_review_pending=True))
    print(pd.DataFrame(rows).groupby(['role','method'])[['OA','BA']].mean().to_string())
    print(pd.DataFrame(intervals).to_string(index=False))
    print(pd.DataFrame(cases)[['status','obs_id','image']].to_string(index=False))


if __name__ == '__main__':
    main()
