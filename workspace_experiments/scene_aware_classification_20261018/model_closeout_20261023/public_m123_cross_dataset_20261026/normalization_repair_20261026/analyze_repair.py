"""Paired before/after repair evaluation and actual-chip decision evidence."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import transfer as tr
import analyze_transfer as analysis

SEEDS = (17, 29, 43)
STAGES = ('axis', 'A', 'U', 'frontend_base', 'm1', 's12', *tr.ARMS)


def analyze(direction):
    root, original = HERE / direction, HERE.parent / direction
    if not tr.read_json(root / 'complete.json')['evaluated']:
        raise ValueError('Incomplete prediction/evaluation run.')
    target = pd.read_csv(root / 'target_table.csv', dtype={'mmsi': str})
    y = target.class_id.to_numpy(int)
    pairs = {seed: (tr.load_npz(original / f'seed{seed}/predictions.npz'),
                    tr.load_npz(root / f'seed{seed}/predictions.npz')) for seed in SEEDS}
    rows, changes, grouped, intervals, limits = [], [], [], [], []
    for seed, (old, new) in pairs.items():
        for stage in STAGES:
            for mode, z in (('original', old), ('repaired', new)):
                rows.append(dict(direction=direction, seed=seed, arm=stage, mode=mode, **tr.m3.metrics(y, z[stage])))
                limits.append(dict(seed=seed, stage=stage, mode=mode, max_abs=float(np.abs(z[stage]).max()),
                    p99_abs=float(np.quantile(np.abs(z[stage]), .99)),
                    rows_max_gt100=int((np.abs(z[stage]).max(1)>100).sum())))
            a, b = new[stage].argmax(1)==y, old[stage].argmax(1)==y
            d = a.astype(float)-b
            changes.append(dict(direction=direction, seed=seed, arm=stage, rescued=int((a&~b).sum()),
                harmed=int((b&~a).sum()), prediction_changes=int((new[stage].argmax(1)!=old[stage].argmax(1)).sum()),
                OA_pp=d.mean()*100, BA_pp=np.mean([d[y==c].mean() for c in np.unique(y)])*100,
                max_score_change=float(np.abs(new[stage]-old[stage]).max())))
            for key in ('class_id', 'physical_pass_group'):
                for g, ix in target.groupby(key).groups.items():
                    ix=np.array(ix)
                    grouped.append(dict(seed=seed, arm=stage, group_type=key, group=g, n=len(ix),
                        rescued=int((a[ix]&~b[ix]).sum()), harmed=int((b[ix]&~a[ix]).sum()),
                        OA_pp=d[ix].mean()*100,
                        BA_pp=np.mean([d[ix[y[ix]==c]].mean() for c in np.unique(y[ix])])*100))
    groups=[np.array(ix) for ix in target.groupby('physical_pass_group').groups.values()]
    for arm in tr.ARMS:
        d=np.mean([(new[arm].argmax(1)==y).astype(float)-(old[arm].argmax(1)==y)
                   for old,new in pairs.values()],0)
        rng=np.random.default_rng(1026)
        sample=[]
        for _ in range(2000):
            ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            sample.append([d[ix].mean()*100, np.mean([d[ix[y[ix]==c]].mean() for c in np.unique(y[ix])])*100])
        lo,hi=np.quantile(sample,[.025,.975],axis=0)
        intervals.append(dict(arm=arm, OA_low=lo[0], OA_high=hi[0], BA_low=lo[1], BA_high=hi[1], passes=len(groups)))
    pd.DataFrame(rows).to_csv(root/'repair_metrics.csv',index=False)
    pd.DataFrame(changes).to_csv(root/'repair_paired.csv',index=False)
    pd.DataFrame(grouped).to_csv(root/'repair_groups.csv',index=False)
    pd.DataFrame(intervals).to_csv(root/'repair_bootstrap.csv',index=False)
    pd.DataFrame(limits).to_csv(root/'score_ranges.csv',index=False)
    print(direction, pd.DataFrame(changes).to_string(index=False), flush=True)
    if direction == 'oss1-to-oss2':
        gallery(root, target, pairs)
        analysis.main(root)
    else:
        tr.write_json(root/'gallery_reuse.json',dict(reason='Unchanged reverse predictions and inputs; review original panels.',
            source=str(original/'analysis'), pending_visual_review=True))


def gallery(root, target, pairs):
    table=pd.read_csv(root/'table.csv',dtype={'mmsi':str})
    records=analysis.input_records(table,tr.read_json(root/'protocol.json'))
    y=target.class_id.to_numpy(int)
    out=root/'repair_gallery';out.mkdir(exist_ok=True)
    cases=[]
    picks=[]
    arm='B_fixed'
    for status in ('rescue','harm','CC','WW'):
        options=[]
        for seed,(old,new) in pairs.items():
            a,b=new[arm].argmax(1)==y,old[arm].argmax(1)==y
            mask=dict(rescue=a&~b,harm=b&~a,CC=a&b,WW=~a&~b)[status]
            delta=analysis.margin(new[arm],y)-analysis.margin(old[arm],y)
            options.extend((abs(delta[i]),seed,int(i)) for i in np.flatnonzero(mask & (abs(delta)>1e-6)))
        if options:
            _,seed,i=max(options);picks.append((status,seed,i))
    for obs in (2688,106):
        ix=np.flatnonzero(target.obs_id.to_numpy()==obs)
        if len(ix) and not any(seed==17 and i==ix[0] for _,seed,i in picks):
            picks.append(('previous_extreme',17,int(ix[0])))
    for status,seed,i in picks:
        old,new=pairs[seed];true=int(y[i])
        rival=int(np.where(np.arange(4)==true,-np.inf,old[arm][i]).argmax())
        def pair(z,key):return float(z[key][i,true]-z[key][i,rival])
        before,after=pair(old,arm),pair(new,arm)
        stage=np.array([pair(new,k)-pair(old,k) for k in ('A','U','frontend_base','m1','s12',arm)])
        effects=np.diff(stage)
        np.testing.assert_allclose(effects.sum(),after-before,atol=2e-5)
        qi=int(new['query_rows'][i]);refs=new['refs'][i,[true,rival]].ravel()
        fig,axes=plt.subplots(2,5,figsize=(16,7));provenance=[]
        for col,j in enumerate([qi,*refs]):
            for p,v in enumerate(tr.extract.prepare(records[j])):
                ax=axes[p,col]
                if v is not None:
                    ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    if v['masks'][0].any():ax.contour(v['masks'][0],levels=[.5],colors=['#19a580'],linewidths=.7)
                    provenance.append(dict(sample_id=table.iloc[j].sample_id,pol=('VV','VH')[p],
                        path=records[j][('vv','vh')[p]]['image'],placement=v['placement']))
                label='QUERY' if col==0 else ('TRUE REF' if col<3 else 'RIVAL REF')
                ax.set_title(f"{('VV','VH')[p]} {label} {table.iloc[j].original_obs_id}\n{analysis.CLASSES[int(table.iloc[j].class_id)]}",fontsize=8)
                ax.set(xticks=[],yticks=[])
        fig.suptitle(f"Source-constant mask: {arm}, {status}, seed{seed}\n"
            f"{analysis.CLASSES[old[arm][i].argmax()]} -> {analysis.CLASSES[new[arm][i].argmax()]}; fixed margin {before:+.4f} -> {after:+.4f}")
        fig.text(.02,.025,f'Change through D/T/M1/M2/M3: {np.round(effects,4)}. Same pixels and references; not physical attribution.',fontsize=9)
        fig.subplots_adjust(top=.82,bottom=.1,hspace=.2,wspace=.15)
        name=f'{status}_{int(target.iloc[i].obs_id)}_{seed}.png'
        fig.savefig(out/name,dpi=120);plt.close(fig)
        cases.append(dict(status=status,arm=arm,seed=seed,obs_id=int(target.iloc[i].obs_id),
            query_row=i,true=true,rival=rival,before_pred=int(old[arm][i].argmax()),after_pred=int(new[arm][i].argmax()),
            before_margin=before,after_margin=after,effects_D_T_M1_M2_M3=effects.tolist(),
            refs=refs.tolist(),image=name,provenance=provenance))
    tr.write_json(out/'cases.json',cases)
    tr.write_json(out/'receipt.json',dict(panels=len(cases),raw_chip_visual_review_pending=True))


if __name__=='__main__':
    for direction in ('oss1-to-oss2','oss2-to-oss1'):
        analyze(direction)
