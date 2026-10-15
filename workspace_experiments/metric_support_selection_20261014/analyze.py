"""Paired system effects, support-CV resolution, and data-level accounting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
VALUES=['oa','ba','macro_nll','macro_brier']
KEYS=['stage','port','episode','draw','cell','seed']
POLICIES=['fixed_0','fixed_0.25','fixed_0.5','support_ba','support_nll']


def main():
    assert (OUT/'EVALUATION_COMPLETE.json').exists()
    frame=pd.read_csv(OUT/'metrics.csv')
    fixed=frame[frame.policy.eq('fixed_0')][KEYS+VALUES]
    paired=frame.merge(fixed,on=KEYS,suffixes=('','_fixed'),validate='many_to_one')
    for v in VALUES:
        paired[v+'_delta']=paired[v]-paired[v+'_fixed']
    values=VALUES+[v+'_delta' for v in VALUES]
    group=['stage','cell','policy']
    byportseed=paired.groupby(group+['port','seed'])[values].mean().reset_index()
    seeds=byportseed.groupby(group+['seed'])[values].mean().reset_index()
    ports=byportseed.groupby(group+['port'])[values].mean().reset_index()
    summary=seeds.groupby(group)[values].mean().reset_index()
    seed_sd=seeds.groupby(group).ba_delta.std().rename('ba_seed_sd').reset_index()
    draws=paired.groupby(group+['port','draw'])[values].mean().groupby(group+['draw']).mean().reset_index()
    draw_sd=draws.groupby(group).ba_delta.std().rename('ba_draw_sd').reset_index()
    positive=draws.groupby(group).ba_delta.apply(lambda x:int((x>1e-12).sum())).rename('positive_draws').reset_index()
    summary=summary.merge(seed_sd,on=group).merge(draw_sd,on=group).merge(positive,on=group)
    a=summary.set_index(group);b=draws.groupby(group)[values].mean()
    np.testing.assert_allclose(a.loc[b.index,values],b,atol=1e-11)
    for name,data in [('paired_metrics',paired),('summary',summary),('by_port',ports),('by_seed',seeds),('by_support_draw',draws)]:
        data.to_csv(OUT/(name+'.csv'),index=False)

    choices=pd.read_csv(OUT/'choices.csv')
    frequencies=choices.groupby(group+['alpha']).size().rename('count').reset_index()
    frequencies['fraction']=frequencies['count']/frequencies.groupby(group)['count'].transform('sum')
    frequencies.to_csv(OUT/'selection_frequencies.csv',index=False)
    quality=choices.merge(paired[KEYS+['policy']+values],on=KEYS+['policy'],validate='one_to_one')
    quality.to_csv(OUT/'selection_vs_query.csv',index=False)
    qsum=[]
    for key,g in quality.groupby(group):
        nonzero=g[g.alpha.gt(0)]
        qsum.append(dict(zip(group,key))|dict(cells=len(g),nonzero=int(len(nonzero)),
            ba_tie_fraction=float(g.ba_tied_candidates.gt(1).mean()),all_ba_tie_fraction=float(g.ba_tied_candidates.eq(3).mean()),
            oof_no_change_fraction=float(g.changed_oof_predictions.eq(0).mean()),
            selected_cv_ba_gain=float(g.cv_ba_gain.mean()),selection_ms_median=float(g.selection_ms.median()),
            nonzero_query_ba_positive=int(nonzero.ba_delta.gt(1e-12).sum()),
            nonzero_query_ba_negative=int(nonzero.ba_delta.lt(-1e-12).sum()),
            nonzero_query_ba_unchanged=int(nonzero.ba_delta.abs().le(1e-12).sum())))
    pd.DataFrame(qsum).to_csv(OUT/'selection_quality.csv',index=False)

    # Diagnostic only: exact expected score under random allocation of the same
    # alpha frequencies within each port/seed/cell, not averaged probabilities.
    allocation_keys=['stage','cell','port','seed','policy']
    counts=choices.groupby(allocation_keys+['alpha']).size().rename('n').reset_index()
    counts['probability']=counts.n/counts.groupby(allocation_keys).n.transform('sum')
    fixed_mean=paired[paired.policy.str.startswith('fixed_')].groupby(['stage','cell','port','seed','alpha'])[values].mean().reset_index()
    random=counts.merge(fixed_mean,on=['stage','cell','port','seed','alpha'],validate='many_to_one')
    for v in values:random[v]=random[v]*random.probability
    random=random.groupby(allocation_keys)[values].sum().reset_index()
    actual=byportseed[byportseed.policy.str.startswith('support_')]
    allocation=random.merge(actual,on=allocation_keys,suffixes=('_random','_actual'),validate='one_to_one')
    for v in ['oa_delta','ba_delta','macro_nll_delta']:
        allocation[v+'_selection_advantage']=allocation[v+'_actual']-allocation[v+'_random']
    allocation.to_csv(OUT/'matched_frequency_control_by_port_seed.csv',index=False)
    numeric=[c for c in allocation.columns if c not in allocation_keys]
    allocation.groupby(group)[numeric].mean().reset_index().to_csv(OUT/'matched_frequency_control.csv',index=False)

    cv=pd.read_csv(OUT/'cv_metrics.csv')
    cv0=cv[cv.alpha.eq(0)][KEYS+VALUES]
    cv=cv.merge(cv0,on=KEYS,suffixes=('','_zero'),validate='many_to_one')
    for v in VALUES:cv['cv_'+v+'_delta']=cv[v]-cv[v+'_zero']
    cv=cv.merge(paired[paired.policy.str.startswith('fixed_')][KEYS+['alpha']+[v+'_delta' for v in VALUES]],
                on=KEYS+['alpha'],validate='one_to_one')
    cv.to_csv(OUT/'cv_vs_query_candidates.csv',index=False)

    events=pd.read_csv(OUT/'class_events.csv')
    event_keys=KEYS+['policy']
    reconstructed=events.groupby(event_keys).recall.mean().reset_index().merge(frame[event_keys+['ba']],on=event_keys,validate='one_to_one')
    np.testing.assert_allclose(reconstructed.recall,reconstructed.ba,atol=1e-10)
    events['nclasses']=events.groupby(event_keys).class_id.transform('size')
    events['nscenes']=events.groupby(['stage','cell','policy','port']).episode.transform('nunique')
    events['ba_contribution']=100*(events.rescues-events.harms)/events.query_n/events.nclasses/events.nscenes/3/5/3
    contributions=events.groupby(group+['port','episode','class_id','support_n'])[['ba_contribution','rescues','harms']].sum().reset_index()
    contributions.to_csv(OUT/'scene_class_contributions.csv',index=False)
    reconstructed_delta=events.groupby(group).ba_contribution.sum()
    np.testing.assert_allclose(reconstructed_delta,a.loc[reconstructed_delta.index,'ba_delta'],atol=1e-10)
    events['support_covered']=events.support_n.gt(0)
    events.groupby(group+['support_covered']).agg(ba_contribution=('ba_contribution','sum'),
        rescues=('rescues','sum'),harms=('harms','sum')).reset_index().to_csv(OUT/'support_coverage_contributions.csv',index=False)
    changes=pd.read_csv(OUT/'changed_cases.csv')
    changes.groupby(group+['event']).agg(exposures=('row','size'),unique_rows=('row','nunique'),unique_ships=('ship','nunique')).reset_index().to_csv(OUT/'changed_case_counts.csv',index=False)

    # Decompose the actual support-CV choice into covered and missing-class evidence.
    enrollment=[e for e in json.loads((OUT.parent/'class_reference_metric_pilot_20261014/ENROLLMENT.json').read_text(encoding='utf-8')) if e['role']=='held']
    keyed=choices.set_index(KEYS+['policy'])
    number={};oof_parts=[];coverage=[];max_error=0.
    for e in enrollment:
        held=e['held'];ti=number.get(held,0);number[held]=ti+1
        key={k:e[k] for k in ['port','episode','draw','cell']}
        with np.load(OUT/'oof'/f"{held.replace(' ','_')}_{ti:03d}.npz") as archive:
            y,fold=archive['sy'],archive['fold'];groups=archive['acquisition']
            full=archive['full'];alphas=archive['alphas'];stages=archive['stages'];seeds_=archive['seeds']
        missing=np.zeros(len(y),dtype=bool)
        for k in np.unique(fold):missing[fold==k]=~np.isin(y[fold==k],y[fold!=k])
        classes,counts=np.unique(y,return_counts=True)
        weights=np.zeros(len(y))
        for c,n in zip(classes,counts):
            weights[y==c]=100/(len(classes)*n)
            coverage.append(dict(**key,class_id=int(c),support_n=int(n),acquisition_n=len(np.unique(groups[y==c])),
                fold_missing_n=int(missing[y==c].sum())))
        for gi,stage in enumerate(stages):
            for si,seed in enumerate(seeds_):
                before=full[gi,si,0].argmax(1)==y
                for policy in ['support_ba','support_nll']:
                    ch=keyed.loc[(stage,held,e['episode'],e['draw'],e['cell'],seed,policy)]
                    ai=int(np.flatnonzero(alphas==ch.alpha)[0])
                    after=full[gi,si,ai].argmax(1)==y
                    delta=after.astype(int)-before.astype(int)
                    parts=[float((delta[~missing]*weights[~missing]).sum()),float((delta[missing]*weights[missing]).sum())]
                    error=abs(sum(parts)-ch.cv_ba_gain);max_error=max(max_error,error)
                    assert error < 1e-10
                    oof_parts.append(dict(stage=stage,**key,seed=int(seed),policy=policy,alpha=ch.alpha,
                        covered_ba_contribution=parts[0],missing_ba_contribution=parts[1],
                        missing_oof_n=int(missing.sum()),cv_ba_gain=ch.cv_ba_gain))
    pd.DataFrame(oof_parts).to_csv(OUT/'cv_coverage_contributions.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'support_class_acquisition_coverage.csv',index=False)
    checks=dict(full_ba_reconstructions=len(reconstructed),max_ba_error=float(abs(reconstructed.recall-reconstructed.ba).max()),
        cv_coverage_reconstructions=len(oof_parts),max_cv_error=max_error,draw_seed_mean_agreement=True)
    (OUT/'SUMMARY_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    plot(summary,draws)
    print('CROSS SUMMARY')
    print(summary[summary.cell.eq('other_acquisition')][['stage','policy','oa','ba','oa_delta','ba_delta','macro_nll_delta','ba_seed_sd','ba_draw_sd','positive_draws']].to_string(index=False))
    print('CROSS SELECTION QUALITY')
    print(pd.DataFrame(qsum).query("cell=='other_acquisition'").to_string(index=False))
    print('CROSS CHOICES')
    print(frequencies.query("cell=='other_acquisition'").to_string(index=False))


def plot(summary,draws):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    labels=['Fixed 0','Fixed .25','Fixed .5','Support BA','Support NLL']
    for ax,stage in zip(axs,['initial','matched']):
        g=summary[(summary.stage==stage)&(summary.cell=='other_acquisition')].set_index('policy').loc[POLICIES]
        ax.bar(np.arange(5),g.ba_delta,color=['#999999','#718f9d','#397991','#28927b','#b36b4e'],width=.6)
        for draw,h in draws[(draws.stage==stage)&(draws.cell=='other_acquisition')].groupby('draw'):
            h=h.set_index('policy').loc[POLICIES]
            ax.scatter(np.arange(5)+(draw-2)*.05,h.ba_delta,s=16,color='#444444',alpha=.7)
        ax.axhline(0,color='#555555',linewidth=.8)
        ax.set_xticks(np.arange(5),[f'{label}\n{v:+.3f} pp' for label,v in zip(labels,g.ba_delta)],fontsize=9)
        ax.set_title('Initial training' if stage=='initial' else 'Quota-matched training')
        ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('BA change (percentage points)')
    fig.suptitle('Support-only metric selection: cross-acquisition, 48 scenes / 3 ports')
    fig.text(.5,.02,'Bars: equal-port mean. Dots: five support draws, each averaged over three training seeds. Reused development data.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(OUT/'support_selection.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
