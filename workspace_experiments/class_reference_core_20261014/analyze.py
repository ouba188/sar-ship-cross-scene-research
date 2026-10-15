"""Paired summaries of the frozen source-reference core experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
KEY=['dataset','study','cohort','budget','cell','mode']
VALUES=['oa','ba','brier','macro_brier','nll','macro_nll']


def add_scopes(frame):
    ports=frame[frame.dataset=='ports'].copy()
    ports['cohort']='all48'
    return pd.concat([frame,ports],ignore_index=True)


def aggregate(frame, keys, values):
    per_port=frame.groupby(keys+['draw','port'],dropna=False)[values].mean().reset_index()
    return per_port.groupby(keys+['draw'],dropna=False)[values].mean().reset_index()


def main():
    data=pd.read_csv(OUT/'metrics.csv')
    x=add_scopes(data)
    draw=aggregate(x,KEY+['arm'],VALUES)
    draw.to_csv(OUT/'draw_metrics.csv',index=False)
    absolute=draw.groupby(KEY+['arm'])[VALUES].agg(['mean','std']).reset_index()
    absolute.columns=['_'.join(k).rstrip('_') if isinstance(k,tuple) else k for k in absolute.columns]
    absolute.to_csv(OUT/'absolute_summary.csv',index=False)
    rows=[]
    for keys,g in draw.groupby(KEY):
        p=g.pivot(index='draw',columns='arm',values=VALUES)
        for candidate in ['class_mean','cluster_0','cluster_1','cluster_2']:
            for baseline in ['original','shrink','cluster_average']:
                if candidate!='class_mean' and baseline!='original':
                    continue
                for metric in VALUES:
                    baseline_values=p[metric][['cluster_0','cluster_1','cluster_2']].mean(axis=1) if baseline=='cluster_average' else p[metric][baseline]
                    delta=p[metric][candidate]-baseline_values
                    rows.append(dict(zip(KEY,keys),candidate=candidate,baseline=baseline,metric=metric,
                        mean=float(delta.mean()),std=float(delta.std(ddof=1)),
                        positive=int((delta>1e-12).sum()),negative=int((delta< -1e-12).sum()),
                        minimum=float(delta.min()),maximum=float(delta.max()),draws=len(delta)))
    effects=pd.DataFrame(rows);effects.to_csv(OUT/'effects_summary.csv',index=False)

    cases=pd.read_csv(OUT/'case_paths.csv',dtype={'ship':str})
    cases['rescued']=(~cases.old_correct)&cases.new_correct
    cases['harmed']=cases.old_correct&(~cases.new_correct)
    cases['net_hit']=cases.new_correct.astype(int)-cases.old_correct.astype(int)
    cases['map_delta']=cases.map_new-cases.map_old
    cases['margin_positive']=cases.margin_delta>0
    cases['support_present']=cases.correct_support>0
    for status,mask in [('rescued',cases.rescued),('harmed',cases.harmed)]:
        subset=cases[(cases.arm=='class_mean')&mask]
        subset.groupby(['dataset','study','budget','cell','class_id']).agg(
            exposures=('row','size'),ships=('ship','nunique'),observations=('row','nunique'),
            margin_delta=('margin_delta','mean'),relation_margin=('relation_margin','mean'),
            refit_margin=('refit_margin','mean')).reset_index().to_csv(OUT/f'{status}_groups.csv',index=False)
    diagnostics=[]
    for keys,g in cases.groupby(['dataset','study','budget','cell','arm']):
        good=g[g.correct_support>0]
        diagnostics.append(dict(zip(['dataset','study','budget','cell','arm'],keys),
            exposures=len(g),query_ships=g.ship.nunique(),rescues=int(g.rescued.sum()),harms=int(g.harmed.sum()),
            rescued_ships=g.loc[g.rescued,'ship'].nunique(),harmed_ships=g.loc[g.harmed,'ship'].nunique(),
            margin_delta=g.margin_delta.mean(),relation_margin=g.relation_margin.mean(),
            refit_margin=g.refit_margin.mean(),map_old=good.map_old.mean(),map_new=good.map_new.mean(),
            missing_support_exposures=int((g.correct_support==0).sum())))
    pd.DataFrame(diagnostics).to_csv(OUT/'path_diagnostics_pooled.csv',index=False)

    # Preserve task/class macro weights rather than confusing pooled events with BA.
    cg=cases.groupby(['dataset','study','cohort','budget','cell','arm','draw','port','episode','class_id'])[
        ['net_hit','margin_delta','relation_margin','refit_margin','map_delta']].mean().reset_index()
    cg['oa_delta_pp']=100*cg.net_hit
    cg.to_csv(OUT/'class_path_effects.csv',index=False)
    cscene=cg.groupby(['dataset','study','cohort','budget','cell','arm','draw','port','episode'])[
        ['oa_delta_pp','margin_delta','relation_margin','refit_margin','map_delta']].mean().reset_index()
    cscene=cscene.rename(columns={'oa_delta_pp':'ba_delta_pp'})
    cdraw=aggregate(add_scopes(cscene),['dataset','study','cohort','budget','cell','arm'],
        ['ba_delta_pp','margin_delta','relation_margin','refit_margin','map_delta'])
    cdraw.to_csv(OUT/'macro_path_draws.csv',index=False)
    cdraw.groupby(['dataset','study','cohort','budget','cell','arm'])[
        ['ba_delta_pp','margin_delta','relation_margin','refit_margin','map_delta']].mean().reset_index().to_csv(
        OUT/'macro_path_summary.csv',index=False)
    checks=[]
    for _,r in cdraw.iterrows():
        a=draw[(draw.dataset==r.dataset)&(draw.study==r.study)&(draw.cohort==r.cohort)&
            (draw.budget==r.budget)&(draw.cell==r.cell)&(draw['mode']=='full')&(draw.draw==r.draw)]
        expected=float(a[a.arm==r.arm].ba.iloc[0]-a[a.arm=='original'].ba.iloc[0])
        np.testing.assert_allclose(r.ba_delta_pp,expected,atol=1e-10)
        checks.append(abs(r.ba_delta_pp-expected))

    timings=pd.read_csv(OUT/'timings.csv')
    timings['total_ms']=1000*(timings.fit_seconds+timings.predict_seconds)
    timings.groupby(['dataset','budget','arm'])[['fit_seconds','predict_seconds','total_ms']].median().reset_index().to_csv(
        OUT/'timing_summary.csv',index=False)
    tk=['dataset','episode','budget','draw','cell']
    baseline=timings[timings.arm=='original'][tk+['total_ms']].rename(columns={'total_ms':'baseline_ms'})
    paired_time=timings.merge(baseline,on=tk,validate='many_to_one')
    paired_time['extra_ms']=paired_time.total_ms-paired_time.baseline_ms
    paired_time.groupby(['dataset','budget','cell','arm'])[['total_ms','baseline_ms','extra_ms']].median().reset_index().to_csv(
        OUT/'paired_timing_summary.csv',index=False)
    composition=[]
    for domain in ['ports','external']:
        cc=pd.read_csv(OUT/f'{domain}_cluster_composition.csv')
        for seed,g in cc.groupby('seed'):
            table=g.pivot(index='cluster',columns='class_id',values='n')
            composition.append(dict(domain=domain,seed=seed,
                purity=float(table.max(axis=1).sum()/table.to_numpy().sum()),
                majority_classes=int(table.idxmax(axis=1).nunique()),
                class_maximum_coverage=json.dumps(table.idxmax(axis=1).tolist())))
    pd.DataFrame(composition).to_csv(OUT/'cluster_structure.csv',index=False)

    pick=effects[(effects.budget==64)&(effects['mode']=='full')&(effects.metric=='ba')&
        (effects.baseline=='original')&(effects.cell=='other_acquisition')]
    specs=[('ports','all48','Ports / all48'),('ports','new_acquisition','Ports / new42'),
           ('f0921','external','F0921'),('oss2','external','OpenSARShip2')]
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,(dataset,cohort,label) in enumerate(specs):
        sub=pick[(pick.dataset==dataset)&(pick.cohort==cohort)]
        if sub.empty:
            continue
        a=sub[sub.candidate=='class_mean'].iloc[0]
        clusters=sub[sub.candidate.str.startswith('cluster_')]['mean']
        ax.bar(i-.16,a['mean'],width=.3,color='#26787a',label='Class reference' if i==0 else None)
        ax.bar(i+.16,clusters.mean(),width=.3,color='#b46c70',label='Unlabelled anchors (3 fits)' if i==0 else None)
        ax.scatter(np.repeat(i+.16,len(clusters)),clusters,color='#502c30',s=16,zorder=4)
    ax.axhline(0,color='#777777',linewidth=.8)
    ax.set_xticks(range(len(specs)),[s[2] for s in specs])
    ax.set_ylabel('BA change from original (percentage points)')
    ax.set_title('Cross-acquisition, B=64, fixed 0.5 fusion')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'core_comparison.png',dpi=170);plt.close(fig)
    (OUT/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(
        macro_reconstruction_checks=len(checks),max_error=max(checks),
        note='Cluster average is average performance over 3 reference fits, NOT a prediction ensemble'),indent=2),encoding='utf-8')
    print(effects[(effects.budget==64)&(effects['mode']=='full')&(effects.metric=='ba')&
        (effects.candidate=='class_mean')].to_string(index=False))


if __name__=='__main__':
    main()
