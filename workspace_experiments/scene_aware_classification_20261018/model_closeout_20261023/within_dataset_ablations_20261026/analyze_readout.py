"""Evaluate frozen within-dataset held acquisitions; no fitting or model selection."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'new_version_ablations_20261026'))
import run_readout as run
previous = run.module('cross_readout_analysis', ROOT/'new_version_ablations_20261026/analyze_readout.py')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

OUT = HERE/'readout'
CLASSES = ('bulk', 'general', 'chemical', 'container')


def interval(table, effects):
    y = table.class_id.to_numpy(int)
    d = np.mean(effects, axis=0)
    if table.pass_group.nunique() > 1:
        result = previous.interval(table, effects)
        result['scope'] = 'paired acquisition bootstrap; fixed trained seeds and observed acquisitions'
        return result
    assert table.mmsi.is_unique
    rng = np.random.default_rng(1026)
    samples = [rng.choice(np.flatnonzero(y == c), size=(2000, sum(y == c))) for c in range(4)]
    values = [d[s] for s in samples]
    oa = np.concatenate(values, axis=1).mean(1)*100
    ba = np.stack([v.mean(1) for v in values]).mean(0)*100
    return dict(OA_pp=d.mean()*100, BA_pp=np.mean([d[y == c].mean() for c in range(4)])*100,
                OA_low=np.quantile(oa,.025), OA_high=np.quantile(oa,.975),
                BA_low=np.quantile(ba,.025), BA_high=np.quantile(ba,.975), passes=1,
                BA_complete_draws=2000, scope='class-stratified distinct ships in ONE fixed acquisition; NOT cross-acquisition uncertainty')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pairrows, classes, passrows, cis, coverage, checks, cases = [], [], [], [], [], [], [], []
    for source in ('oss1', 'oss2'):
        root = ROOT/f'{source}_m123_replication_20261026'
        table = pd.read_csv(root/'stack_table.csv', dtype={'mmsi': str})
        ids = np.flatnonzero(table.role.eq('evaluation'))
        target = table.iloc[ids].reset_index(drop=True)
        y = target.class_id.to_numpy(int)
        assert len(ids) == {'oss1':63, 'oss2':378}[source]
        assert set(y) == set(range(4))
        for field in ('obs_id', 'mmsi', 'pass_group'):
            assert not set(target[field]) & set(table.loc[table.role.ne('evaluation'),field])
        coverage.append(dict(dataset=source, observations=len(ids), vessels=target.mmsi.nunique(),
            passes=target.pass_group.nunique(), counts=np.bincount(y,minlength=4).tolist(),
            roles=table.role.value_counts().to_dict()))
        target.to_csv(OUT/f'{source}_evaluation.csv',index=False)
        z = {arm:{s:run.tr.load_npz(run.DEST/source/arm/f'seed{s}/source_predictions.npz')
                  for s in run.SEEDS} for arm in run.ARMS}
        for arm in run.ARMS:
            for seed in run.SEEDS:
                np.testing.assert_array_equal(z[arm][seed]['obs_id'],table.obs_id)
                for endpoint in ('bare','B_fixed','frequency'):
                    score = z[arm][seed][endpoint][ids]
                    rows.append(dict(dataset=source,arm=arm,seed=seed,endpoint=endpoint,
                                     **run.base.classification_metrics(y,score)))
        # Compare new frozen source evaluation outputs, not cross-dataset outputs.
        for before,after in previous.PAIRS:
            for endpoint in ('bare','frequency'):
                effects=[]
                for seed in run.SEEDS:
                    a=z[after][seed][endpoint][ids].argmax(1)==y
                    b=z[before][seed][endpoint][ids].argmax(1)==y
                    d=a.astype(float)-b
                    effects.append(d)
                    meta=dict(dataset=source,before=before,after=after,endpoint=endpoint,seed=seed)
                    pairrows.append(dict(**meta,rescue=int((a&~b).sum()),harm=int((b&~a).sum()),
                                         OA_pp=d.mean()*100,BA_pp=np.mean([d[y==c].mean() for c in range(4)])*100))
                    for c in range(4):
                        mask=y==c
                        classes.append(dict(**meta,class_name=CLASSES[c],n=int(mask.sum()),recall_pp=d[mask].mean()*100,
                                            rescue=int((d[mask]>0).sum()),harm=int((d[mask]<0).sum())))
                    for name, ix in target.groupby('pass_group').indices.items():
                        passrows.append(dict(**meta,pass_group=name,n=len(ix),OA_pp=d[ix].mean()*100))
                cis.append(dict(dataset=source,before=before,after=after,endpoint=endpoint,**interval(target,effects)))
        extract=run.tr.extract
        extract.VIEW_TABLE=root/'view_inventory.csv'
        extract.RADAR_TABLE=(root/'geometry.csv' if source=='oss2' else ROOT.parents[1]/'oss1_joint_geometry_20261017/geometry.csv')
        records=extract.prepare_inventory(table)
        direction=f'{source}-to-'+('oss2' if source=='oss1' else 'oss1')
        actual=run.tr.load_npz(run.aligned.HERE/direction/'seed17/source_predictions.npz')
        np.testing.assert_array_equal(actual['obs_id'],table.obs_id)
        np.testing.assert_array_equal(actual['frequency'],z['full'][17]['frequency'])
        assert table.iloc[np.unique(actual['refs'][ids])].role.eq('fit_train').all()
        checks.append(dict(dataset=source,identity_and_holdout_disjoint=True,full_exact_replay=True,refs_fit_only=True))
        for before,after,endpoint in (('mean_meta','full','bare'),('merged_raw','raw_only','bare'),('raw_only','full','frequency')):
            a=z[after][17][endpoint][ids];b=z[before][17][endpoint][ids]
            ac=a.argmax(1)==y;bc=b.argmax(1)==y
            picks=[]
            for state, mask in dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc).items():
                available=np.flatnonzero(mask)
                if len(available):
                    j=int(available[np.argmin(target.iloc[available].obs_id.to_numpy())])
                    picks.append((state,j))
            fig,axes=plt.subplots(2,len(picks),figsize=(4*len(picks),7),squeeze=False)
            name=f'{source}_{before}_to_{after}_{endpoint}.png'
            for col,(state,j) in enumerate(picks):
                i=int(ids[j]);true=int(y[j]);rival=int(b[j].argmax() if state=='rescue' else a[j].argmax())
                if rival==true:rival=int(np.where(np.arange(4)==true,-np.inf,b[j]).argmax())
                prov=[]
                for p in range(2):
                    prov.append(previous.display(axes[p,col],records[i],p))
                    axes[p,col].set_title(f'{state} obs{int(table.iloc[i].obs_id)} {CLASSES[true]} {("VV","VH")[p]}\n{CLASSES[b[j].argmax()]} -> {CLASSES[a[j].argmax()]}',fontsize=9)
                    axes[p,col].set(xticks=[],yticks=[])
                entry=dict(dataset=source,seed=17,before=before,after=after,endpoint=endpoint,status=state,
                    obs_id=int(table.iloc[i].obs_id),row=i,true=true,rival=rival,before_pred=int(b[j].argmax()),after_pred=int(a[j].argmax()),
                    before_margin=float(b[j,true]-b[j,rival]),after_margin=float(a[j,true]-a[j,rival]),image=name,provenance=prov)
                if endpoint=='frequency':
                    entry.update(old_weights=z[before][17]['weight_frequency'][i].tolist(),
                                 new_weights=z[after][17]['weight_frequency'][i].tolist())
                    for arm,label in ((before,'before'),(after,'after')):
                        zz=z[arm][17]
                        contrib=[float(zz['bare'][i,true]-zz['bare'][i,rival])]
                        for p,key in enumerate(('delta_vv','delta_vh')):
                            contrib.append(float(zz['weight_frequency'][i,p]*(zz[key][i,true]-zz[key][i,rival])))
                        assert abs(sum(contrib)-entry[f'{label}_margin'])<1e-4
                        entry[f'{label}_base_vv_vh']=contrib
                    if state in ('rescue','harm'):
                        refs=actual['refs'][i,[true,rival]].astype(int).ravel()
                        rf,ra=plt.subplots(2,5,figsize=(15,6.5))
                        rprov=[]
                        for c,row in enumerate([i,*refs]):
                            for p in range(2):
                                rprov.append(previous.display(ra[p,c],records[row],p))
                                ra[p,c].set_title(f'{("QUERY","TRUE1","TRUE2","RIVAL1","RIVAL2")[c]} obs{int(table.iloc[row].obs_id)} {("VV","VH")[p]}',fontsize=8)
                                ra[p,c].set(xticks=[],yticks=[])
                        rname=f'{source}_{state}_{entry["obs_id"]}_actual_refs.png'
                        rf.suptitle(f'{source} held acquisitions seed17: actual same bank / retrieval; {state}')
                        rf.subplots_adjust(top=.88,bottom=.04,hspace=.25,wspace=.08)
                        rf.savefig(OUT/rname,dpi=110);plt.close(rf)
                        entry.update(reference_image=rname,reference_rows=refs.tolist(),reference_provenance=rprov)
                cases.append(entry)
            fig.suptitle(f'{source} WITHIN-DATASET held acquisitions seed17: {before} -> {after}, {endpoint}')
            fig.text(.02,.02,'Native160 crop/pad; unchanged grayscale; OBB boundary. Lowest obsID per correctness state.',fontsize=8)
            fig.subplots_adjust(top=.88,bottom=.08,hspace=.2,wspace=.1)
            fig.savefig(OUT/name,dpi=115);plt.close(fig)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv',index=False)
    frame.groupby(['dataset','arm','endpoint'])[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(OUT/'summary.csv')
    for name,data in (('paired',pairrows),('class_effects',classes),('pass_effects',passrows),('uncertainty',cis)):
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    for name,data in (('coverage',coverage),('checks',checks),('cases',cases)):
        run.tr.write_json(OUT/f'{name}.json',data)
    run.tr.write_json(OUT/'receipt.json',dict(configurations_evaluated=42,new_training_runs=0,
        scope='source-role evaluation only; no selection metrics used as efficacy',native_review_pending=True,case_positions=len(cases)))
    print(frame.groupby(['dataset','arm','endpoint'])[['OA','BA']].mean().to_string())
    print(pd.DataFrame(cis).to_string(index=False))


if __name__=='__main__':main()
