"""Frozen transfer: paired class/acquisition results and native query/reference panels."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transfer import extract, m3, ARMS, read_json, write_json

CLASSES = ('bulk', 'general', 'chemical', 'container')


def margin(score, y):
    return score[np.arange(len(y)),y]-np.where(np.arange(4)[None]==y[:,None],-np.inf,score).max(1)


def input_records(table, protocol):
    records = [None]*len(table)
    for side in ('source','target'):
        root = Path(protocol[f'{side}_root'])
        part = table[table.dataset_side.eq(side)].copy()
        part['obs_id'] = part.original_obs_id.astype(int)
        extract.VIEW_TABLE = root/'view_inventory.csv'
        extract.RADAR_TABLE = (Path(protocol['geometry']) if side=='target' else
            root/'geometry.csv' if (root/'geometry.csv').exists() else
            Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv'))
        for ix, record in zip(part.index, extract.prepare_inventory(part)):
            records[ix] = record
    return records


def main(root):
    protocol = read_json(root/'protocol.json')
    if protocol['mode'] != 'frozen_cross_dataset':
        raise ValueError('Source replay is not external validation.')
    table = pd.read_csv(root/'table.csv', dtype={'mmsi':str})
    target = pd.read_csv(root/'target_table.csv', dtype={'mmsi':str})
    y = target.class_id.to_numpy(int)
    data = {int(p.name[4:]): dict(np.load(p/'predictions.npz')) for p in sorted(root.glob('seed*'))}
    paired, bygroup, predictions = [],[],[]
    pairs = [('B_fixed','old_both'),('frequency','old_both'),('spatial','B_fixed'),
             ('frequency','B_fixed'),('frequency','spatial')]
    for seed,z in data.items():
        for arm in ARMS:
            row = target[['obs_id','sample_id','mmsi','pass_group','class_id']].copy()
            row['seed'],row['arm'],row['prediction'] = seed,arm,z[arm].argmax(1)
            row['margin'] = margin(z[arm],y)
            predictions.append(row)
        for arm,base in pairs:
            a,b = z[arm].argmax(1)==y,z[base].argmax(1)==y
            d=a.astype(float)-b
            paired.append(dict(seed=seed,arm=arm,baseline=base,n=len(y),rescued=int((a&~b).sum()),
                harmed=int((b&~a).sum()),OA_pp=d.mean()*100,
                BA_pp=np.mean([d[y==c].mean() for c in np.unique(y)])*100))
        for key in ('class_id','pass_group'):
            for group,ix in target.groupby(key).groups.items():
                ix=np.array(ix)
                for arm in (*ARMS,'frontend_base','m1','s12'):
                    bygroup.append(dict(seed=seed,arm=arm,group_type=key,group=group,**m3.metrics(y[ix],z[arm][ix])))
    pd.DataFrame(paired).to_csv(root/'paired.csv',index=False)
    pd.DataFrame(bygroup).to_csv(root/'group_metrics.csv',index=False)
    pd.concat(predictions).to_csv(root/'predictions.csv.gz',index=False,compression='gzip')
    groups=[np.array(ix) for ix in target.groupby('pass_group').groups.values()]
    rng=np.random.default_rng(1026)
    intervals=[]
    for arm,base in pairs:
        d=np.mean([(z[arm].argmax(1)==y).astype(float)-(z[base].argmax(1)==y) for z in data.values()],0)
        samples=[]
        for _ in range(2000):
            ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            samples.append([100*d[ix].mean(),100*np.mean([d[ix[y[ix]==c]].mean() for c in np.unique(y[ix])])])
        low,high=np.quantile(samples,[.025,.975],axis=0)
        intervals.append(dict(arm=arm,baseline=base,OA_low=low[0],OA_high=high[0],BA_low=low[1],BA_high=high[1],
            passes=len(groups),scope='target acquisition bootstrap, seeds averaged within query; few groups limit precision'))
    pd.DataFrame(intervals).to_csv(root/'acquisition_bootstrap.csv',index=False)
    records=input_records(table,protocol)
    cases=[]
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    for arm,base,statuses in [('B_fixed','old_both',('rescue','harm')),
        ('frequency','B_fixed',('rescue','harm','CC','WW'))]:
        for status in statuses:
            candidates=[]
            for seed,z in data.items():
                a,b=z[arm].argmax(1)==y,z[base].argmax(1)==y
                mask=dict(rescue=a&~b,harm=b&~a,CC=a&b,WW=~a&~b)[status]
                delta=margin(z[arm],y)-margin(z[base],y)
                candidates.extend((abs(delta[i]),seed,int(i)) for i in np.flatnonzero(mask))
            if not candidates:
                continue
            _,seed,i=max(candidates)
            z=data[seed]
            true=int(y[i]); before=z[base][i]; after=z[arm][i]
            rival=int(np.where(np.arange(4)==true,-np.inf,before).argmax())
            qi=int(z['query_rows'][i]); refs=z['refs'][i,[true,rival]].ravel()
            wa=z[f'weight_{arm}'][i]
            wb=None if base=='old_both' else z[f'weight_{base}'][i]
            dp=np.array([z['delta_vv'][i,true]-z['delta_vv'][i,rival],z['delta_vh'][i,true]-z['delta_vh'][i,rival]])
            mb,ma=float(before[true]-before[rival]),float(after[true]-after[rival])
            change=(wa-wb)*dp if wb is not None else np.r_[wa*dp,-(mb-(z['base'][i,true]-z['base'][i,rival]))]
            np.testing.assert_allclose(change.sum(),ma-mb,atol=8e-6)
            fig,axes=plt.subplots(2,5,figsize=(16,7)); provenance=[]
            for col,j in enumerate([qi,*refs]):
                views=extract.prepare(records[j])
                for p,v in enumerate(views):
                    ax=axes[p,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():
                            ax.contour(v['masks'][0],levels=[.5],colors=['#19a580'],linewidths=.7)
                        provenance.append(dict(sample_id=table.iloc[j].sample_id,pol=('VV','VH')[p],
                            path=records[j][('vv','vh')[p]]['image'],placement=v['placement']))
                    label='QUERY' if col==0 else ('TRUE REF' if col<3 else 'RIVAL REF')
                    ax.set_title(f"{('VV','VH')[p]} {label} {table.iloc[j].original_obs_id}\n{CLASSES[int(table.iloc[j].class_id)]}",fontsize=8)
                    ax.set(xticks=[],yticks=[])
            fig.suptitle(f'{arm} vs {base}: {status}, seed{seed}; {CLASSES[before.argmax()]} -> {CLASSES[after.argmax()]}\n'
                         f'Fixed pair margin {mb:+.4f} -> {ma:+.4f}; weights {wb} -> {wa.round(3)}')
            fig.text(.02,.02,f'Native crop/pad, no resizing. Signed contribution change: {np.round(change,4)}; not physical attribution.',fontsize=9)
            fig.subplots_adjust(top=.82,bottom=.09,hspace=.2,wspace=.15)
            name=f'{arm}_{base}_{status}_{int(target.iloc[i].obs_id)}_{seed}.png'
            fig.savefig(out/name,dpi=120);plt.close(fig)
            cases.append(dict(arm=arm,baseline=base,status=status,seed=seed,query_row=i,
                sample_id=target.iloc[i].sample_id,obs_id=int(target.iloc[i].obs_id),true=true,rival=rival,
                before_pred=int(before.argmax()),after_pred=int(after.argmax()),margin_before=mb,margin_after=ma,
                weights_before=None if wb is None else wb.tolist(),weights_after=wa.tolist(),
                contribution_change=change.tolist(),refs=refs.tolist(),image=name,provenance=provenance))
    write_json(out/'cases.json',cases)
    write_json(root/'analysis_receipt.json',dict(panels=len(cases),query_n=len(y),passes=len(groups),raw_chip_visual_review_pending=True))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__=='__main__':
    main(Path(sys.argv[1]))
