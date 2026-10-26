"""Frozen main tables, paired uncertainty, and native query/reference review panels."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_classical import ROOT, TRANSFER, SEEDS, tr, write_json
from analyze_transfer import input_records, margin

REPAIR = TRANSFER/'normalization_repair_20261026'
CLASSES = ('bulk','general','chemical','container')


def cohorts(name):
    root = ROOT/f'{name}_m123_replication_20261026'
    source = pd.read_csv(root/'stack_table.csv',dtype={'mmsi':str})
    other = 'oss2' if name == 'oss1' else 'oss1'
    direction = f'{name}-to-{other}'
    external = tr.read_table(TRANSFER/direction/'target_table.csv',labels=True)
    return source, external, direction


def load_comparison(name, cohort):
    source, external, direction = cohorts(name)
    table = source[source.role.eq('evaluation')].reset_index(drop=True) if cohort=='within' else external
    ix = np.flatnonzero(source.role.eq('evaluation')) if cohort=='within' else np.arange(len(external))
    filename = 'source_predictions.npz' if cohort=='within' else 'cross_predictions.npz'
    result = {}
    for family in ('classical','classical_fit_only','local'):
        z = tr.load_npz(HERE/family/name/filename)
        expected = source.obs_id if cohort=='within' else external.obs_id
        np.testing.assert_array_equal(z.pop('obs_id'), expected)
        for k,v in z.items():
            if family=='local' and k.startswith('s12_'):
                continue
            match = re.fullmatch(r'(.+)_seed(\d+)', k)
            method, seed = (match.group(1),int(match.group(2))) if match else (k,0)
            result.setdefault(f'{family}/{method}',{})[seed] = v[ix]
    repair_dir = REPAIR/(f'source_{name}' if cohort=='within' else direction)
    for seed in SEEDS:
        z = tr.load_npz(repair_dir/f'seed{seed}/predictions.npz')
        for arm in ('frontend_base','m1','s12','old_both','B_fixed','frequency'):
            result.setdefault('ours/'+arm,{})[seed] = z[arm][ix]
    return table,result


def make_panel(name,cohort,table,scores,status,out,records,full_table,query_rows,refs):
    seed=17
    before=scores['classical/regional_mlp'][seed]
    after=scores['ours/frequency'][seed]
    y=table.class_id.to_numpy(int)
    a,b=after.argmax(1)==y,before.argmax(1)==y
    mask={'ours_only':a&~b,'simple_only':b&~a,'CC':a&b,'WW':~a&~b}[status]
    pool=np.flatnonzero(mask)
    if not len(pool):
        return None
    # Fixed rule: largest absolute margin difference within each outcome group.
    delta=margin(after,y)-margin(before,y)
    i=int(pool[np.argmax(np.abs(delta[pool]))])
    true=int(y[i])
    rival=int(np.where(np.arange(4)==true,-np.inf,after[i]).argmax())
    ref=refs[i,[true,rival]].ravel()
    fig,axes=plt.subplots(2,5,figsize=(15,6.7))
    provenance=[]
    for col,j in enumerate([int(query_rows[i]),*ref]):
        views=tr.extract.prepare(records[j])
        for p,v in enumerate(views):
            ax=axes[p,col]
            if v is not None:
                ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if v['masks'][0].any():
                    ax.contour(v['masks'][0],levels=[.5],colors=['#11a579'],linewidths=.7)
                provenance.append(dict(pol=('VV','VH')[p],path=records[j][('vv','vh')[p]]['image'],
                                       placement=v['placement']))
            r=full_table.iloc[j]
            role='QUERY' if col==0 else ('TRUE REF' if col<3 else 'RIVAL REF')
            obs=r.get('original_obs_id',r.obs_id)
            ax.set_title(f'{("VV","VH")[p]} {role} {obs}\n{CLASSES[int(r.class_id)]}',fontsize=8)
            ax.set(xticks=[],yticks=[])
    bm=float(before[i,true]-before[i,rival]);am=float(after[i,true]-after[i,rival])
    fig.suptitle(f'{name} {cohort} {status} | seed17 | true={CLASSES[true]}\n'
                 f'Regional MLP: {CLASSES[before[i].argmax()]} ({bm:+.3f}) | '
                 f'M1-M3: {CLASSES[after[i].argmax()]} ({am:+.3f})',fontsize=11)
    fig.text(.02,.02,'Native160 crop/pad, VV/VH separate. References belong to M3, NOT the MLP. Different score scales; margins are not calibrated.',fontsize=8)
    fig.subplots_adjust(top=.82,bottom=.08,wspace=.08,hspace=.2)
    path=out/f'{name}_{cohort}_{status}_{int(table.iloc[i].obs_id)}.png'
    fig.savefig(path,dpi=110);plt.close(fig)
    return dict(source=name,cohort=cohort,status=status,seed=seed,obs_id=int(table.iloc[i].obs_id),
        query_position=i,true=true,rival=rival,mlp_prediction=int(before[i].argmax()),
        ours_prediction=int(after[i].argmax()),mlp_margin=bm,ours_margin=am,
        refs=ref.tolist(),image=path.name,provenance=provenance)


def main():
    out=HERE/'stage1_analysis'
    out.mkdir(exist_ok=True)
    rows,paired,groups,intervals,cases=[],[],[],[],[]
    for name in ('oss1','oss2'):
        source,external,direction=cohorts(name)
        root=ROOT/f'{name}_m123_replication_20261026'
        for cohort in ('within','cross'):
            table,scores=load_comparison(name,cohort)
            y=table.class_id.to_numpy(int)
            for method,seed_scores in scores.items():
                for seed,s in seed_scores.items():
                    rows.append(dict(source=name,cohort=cohort,method=method,seed=seed,**tr.m3.metrics(y,s)))
            for baseline in ('classical/regional_mlp','classical_fit_only/regional_mlp',
                             'classical/regional_ridge','local/dn4_residual','local/context_only_residual'):
                for seed in SEEDS:
                    s=scores['ours/frequency'][seed]
                    b=scores[baseline].get(seed,scores[baseline].get(0))
                    d=(s.argmax(1)==y).astype(float)-(b.argmax(1)==y)
                    paired.append(dict(source=name,cohort=cohort,baseline=baseline,seed=seed,
                        rescued=int((d==1).sum()),harmed=int((d==-1).sum()),OA_pp=100*d.mean(),
                        BA_pp=100*np.mean([d[y==c].mean() for c in range(4)])))
                    for key in ('class_id','pass_group'):
                        for group,ids in table.groupby(key).indices.items():
                            groups.append(dict(source=name,cohort=cohort,baseline=baseline,seed=seed,
                                group_type=key,group=str(group),n=len(ids),net=int(d[ids].sum()),
                                OA_pp=float(100*d[ids].mean())))
                mean_d=np.mean([(scores['ours/frequency'][seed].argmax(1)==y).astype(float)-
                    (scores[baseline].get(seed,scores[baseline].get(0)).argmax(1)==y) for seed in SEEDS],0)
                rng=np.random.default_rng(1026)
                units=list(table.groupby('pass_group').indices.values())
                values=[]
                for _ in range(2000):
                    if len(units)>1:
                        idx=np.concatenate([units[j] for j in rng.integers(len(units),size=len(units))])
                    else:
                        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),size=(y==c).sum()) for c in range(4)])
                    # BA is undefined for the four-class estimand if a class is absent.
                    if len(np.unique(y[idx]))!=4:
                        continue
                    values.append([100*mean_d[idx].mean(),100*np.mean([mean_d[idx[y[idx]==c]].mean() for c in range(4)])])
                low,high=np.quantile(values,[.025,.975],axis=0)
                intervals.append(dict(source=name,cohort=cohort,baseline=baseline,OA_low=low[0],OA_high=high[0],
                    BA_low=low[1],BA_high=high[1],valid_bootstraps=len(values),passes=len(units),
                    resampling='acquisition, conditional all classes' if len(units)>1 else 'class-stratified within one acquisition'))
            if cohort=='cross':
                full_table=pd.read_csv(REPAIR/direction/'table.csv',dtype={'mmsi':str})
                protocol=tr.read_json(REPAIR/direction/'protocol.json')
                records=input_records(full_table,protocol)
                z=tr.load_npz(REPAIR/direction/'seed17/predictions.npz')
                query_rows,refs=z['query_rows'],z['refs']
            else:
                tr.extract.VIEW_TABLE=root/'view_inventory.csv'
                tr.extract.RADAR_TABLE=root/'geometry.csv' if (root/'geometry.csv').exists() else Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv')
                full_table=source
                records=tr.extract.prepare_inventory(source)
                z=tr.load_npz(root/('m3_results_grid01' if name=='oss1' else 'm3_results')/'seed17/predictions.npz')
                query_rows=np.flatnonzero(source.role.eq('evaluation'))
                refs=z['refs'][query_rows]
            for status in ('ours_only','simple_only','CC','WW'):
                case=make_panel(name,cohort,table,scores,status,out,records,full_table,query_rows,refs)
                if case:
                    cases.append(case)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'all_metrics.csv',index=False)
    frame.groupby(['source','cohort','method'])[['OA','BA','NLL','Brier']].agg(['mean','std','count']).to_csv(out/'main_table.csv')
    pd.DataFrame(paired).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'group_differences.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'uncertainty.csv',index=False)
    write_json(out/'cases.json',cases)
    write_json(out/'receipt.json',dict(panels=len(cases),raw_visual_review_pending=True,
                  statistical_unit='source training seed paired; acquisitions cluster queries, not independent seeds'))
    print(frame[frame.method.isin(['ours/frequency','classical/regional_mlp','classical_fit_only/regional_mlp',
          'classical/regional_ridge','local/dn4_residual'])].groupby(['source','cohort','method'])[['OA','BA']].mean().to_string())


if __name__=='__main__':
    main()
