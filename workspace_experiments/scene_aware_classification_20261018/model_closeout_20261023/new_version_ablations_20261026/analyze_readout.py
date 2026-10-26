"""Pass-paired effects, exact comparisons, and native VV/VH inspection material."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import run_readout as run
from analyze_transfer import input_records, CLASSES
tr = run.tr

PAIRS = (('mean_meta','full'), ('mean_only','attention_only'),
         ('attention_only','full'), ('mean_only','mean_meta'),
         ('raw_only','full'), ('merged_raw','raw_only'), ('body_raw','raw_only'))


def interval(table, effects):
    y = table.class_id.to_numpy(int)
    d = np.mean(effects, axis=0)
    groups = list(table.groupby('pass_group').indices.values())
    counts = np.array([np.bincount(y[ix], minlength=4) for ix in groups])
    sums = np.array([np.bincount(y[ix], weights=d[ix], minlength=4) for ix in groups])
    draws = np.random.default_rng(1026).integers(len(groups), size=(2000,len(groups)))
    ct, sm = counts[draws].sum(1), sums[draws].sum(1)
    complete = (ct>0).all(1)
    oa = 100*sm.sum(1)/ct.sum(1)
    ba = 100*(sm[complete]/ct[complete]).mean(1)
    return dict(OA_pp=100*d.mean(), BA_pp=100*np.mean([d[y==c].mean() for c in range(4)]),
                OA_low=np.quantile(oa,.025), OA_high=np.quantile(oa,.975),
                BA_low=np.quantile(ba,.025), BA_high=np.quantile(ba,.975),
                passes=len(groups), BA_complete_draws=int(complete.sum()))


def display(ax, record, pol):
    item = tr.extract.prepare(record)[pol]
    if item is None:
        ax.text(.5,.5,'Unavailable',ha='center')
        return dict(pol=('VV','VH')[pol], missing=True)
    ax.imshow(item['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if item['masks'][0].any():
        ax.contour(item['masks'][0], levels=[.5], colors=['#20a780'], linewidths=.6)
    return dict(pol=('VV','VH')[pol], path=record[('vv','vh')[pol]]['image'], placement=item['placement'])


def main():
    assert (run.DEST/'evaluation_complete.json').exists()
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    intervals, classes, cases, passrows = [], [], [], []
    for source, target in (('oss1','oss2'), ('oss2','oss1')):
        direction = f'{source}-to-{target}'
        cross = tr.HERE/direction
        table = tr.read_table(cross/'target_table.csv', labels=True)
        y = table.class_id.to_numpy(int)
        z = {arm:{s:tr.load_npz(run.DEST/source/arm/f'seed{s}/cross_predictions.npz') for s in run.SEEDS} for arm in run.ARMS}
        for before, after in PAIRS:
            for endpoint in ('bare','frequency'):
                effects = []
                for seed in run.SEEDS:
                    a, b = z[after][seed][endpoint], z[before][seed][endpoint]
                    d = (a.argmax(1)==y).astype(float)-(b.argmax(1)==y)
                    effects.append(d)
                    for c in range(4):
                        mask = y==c
                        classes.append(dict(source=source,before=before,after=after,endpoint=endpoint,seed=seed,
                                            class_name=CLASSES[c], n=int(mask.sum()), recall_pp=100*d[mask].mean(),
                                            rescue=int((d[mask]>0).sum()),harm=int((d[mask]<0).sum())))
                    for name, ids in table.groupby('pass_group').indices.items():
                        passrows.append(dict(source=source,before=before,after=after,endpoint=endpoint,seed=seed,
                                             pass_group=name,n=len(ids),OA_pp=100*d[ids].mean()))
                intervals.append(dict(source=source,before=before,after=after,endpoint=endpoint,**interval(table,effects)))
        records = input_records(pd.read_csv(cross/'table.csv',dtype={'mmsi':str}), tr.read_json(cross/'protocol.json'))
        mapping = tr.load_npz(run.aligned.REPAIR/direction/'seed17/predictions.npz')
        actual = tr.load_npz(run.aligned.HERE/direction/'seed17/cross_predictions.npz')
        for before, after, endpoint in (('mean_meta','full','bare'), ('merged_raw','raw_only','bare'), ('raw_only','full','frequency')):
            a, b = z[after][17][endpoint], z[before][17][endpoint]
            ac, bc = a.argmax(1)==y, b.argmax(1)==y
            masks = dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc)
            picks = []
            for state, mask in masks.items():
                ids=np.flatnonzero(mask)
                if len(ids): picks.append((state,int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])))
            fig, axes = plt.subplots(2,len(picks),figsize=(4*len(picks),7),squeeze=False)
            name=f'{source}_{before}_to_{after}_{endpoint}.png'
            for col,(state,j) in enumerate(picks):
                true=int(y[j]); rival=int(b[j].argmax() if ac[j] and not bc[j] else a[j].argmax())
                if rival==true:rival=int(np.where(np.arange(4)==true,-np.inf,b[j]).argmax())
                rec=records[int(mapping['query_rows'][j])]
                prov=[]
                for p in range(2):
                    prov.append(display(axes[p,col],rec,p))
                    axes[p,col].set_title(f'{state} obs{int(table.iloc[j].obs_id)} {CLASSES[true]} {("VV","VH")[p]}\n{CLASSES[b[j].argmax()]} -> {CLASSES[a[j].argmax()]}',fontsize=9)
                    axes[p,col].set(xticks=[],yticks=[])
                entry=dict(source=source,before=before,after=after,endpoint=endpoint,seed=17,status=state,
                           obs_id=int(table.iloc[j].obs_id),true=true,rival=rival,before_pred=int(b[j].argmax()),after_pred=int(a[j].argmax()),
                           before_margin=float(b[j,true]-b[j,rival]),after_margin=float(a[j,true]-a[j,rival]),
                           image=name,provenance=prov)
                if endpoint=='frequency':
                    entry['old_weights']=z[before][17]['weight_frequency'][j].tolist()
                    entry['new_weights']=z[after][17]['weight_frequency'][j].tolist()
                    if state in ('rescue','harm'):
                        refs=actual['refs'][j,[true,rival]].astype(int).ravel()
                        rfig,rax=plt.subplots(2,5,figsize=(15,6.5))
                        rprov=[]
                        for c,row in enumerate([int(mapping['query_rows'][j]),*refs.tolist()]):
                            for p in range(2):
                                rprov.append(display(rax[p,c],records[row],p))
                                rax[p,c].set_title(f'{("QUERY","TRUE1","TRUE2","RIVAL1","RIVAL2")[c]} {("VV","VH")[p]}',fontsize=9)
                                rax[p,c].set(xticks=[],yticks=[])
                        rname=f'{source}_{state}_{entry["obs_id"]}_actual_refs.png'
                        rfig.suptitle(f'{direction} seed17 {state}; same actual reference bank for both arms')
                        rfig.subplots_adjust(top=.88,bottom=.04,hspace=.25,wspace=.08)
                        rfig.savefig(out/rname,dpi=110);plt.close(rfig)
                        entry.update(reference_image=rname,reference_rows=refs.tolist(),reference_provenance=rprov)
                cases.append(entry)
            fig.suptitle(f'{direction} seed17 {before} -> {after}, {endpoint}')
            fig.text(.02,.02,'Native160 crop/pad; unchanged grayscale; OBB boundary only. Minimum obsID per correctness state.',fontsize=8)
            fig.subplots_adjust(top=.88,bottom=.08,hspace=.2,wspace=.1)
            fig.savefig(out/name,dpi=115);plt.close(fig)
    for name,rows in (('uncertainty',intervals),('class_effects',classes),('pass_effects',passrows)):
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    tr.write_json(out/'cases.json',cases)
    tr.write_json(out/'receipt.json',dict(native_review_pending=True,case_positions=len(cases),case_seed=17,
                  policy='lowest obsID per state; no manual success picking',actual_references=True))
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


if __name__=='__main__':main()
