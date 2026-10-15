"""Paired full-model OA/BA, same-case rescue/harm, and class contributions."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def main():
    plan = json.loads((OUT/'SUPPORT_PLAN.json').read_text(encoding='utf-8'))
    lookup = {(r['task'],r['draw'],r['arm']):r for r in plan}
    rows, classes, events = [], [], []
    checks = 0
    for path in sorted((OUT/'tasks').rglob('predictions.npz')):
        task = path.parent.relative_to(OUT/'tasks').as_posix()
        z = np.load(path); y = z['labels']; present = np.unique(y)
        group = '23ports' if task.startswith('self24ports/') else task.split('/')[0]
        modes = list(z['modes']); arms = list(z['arms'])
        frame = pd.read_csv(path.parent/'manifest.csv', dtype={'identity':str}).set_index('local_row')
        assert len(set(z['identities'])) == len(y)
        np.testing.assert_array_equal(frame.loc[z['query'],'class_id'], y)
        for draw in range(5):
            for ai, arm in enumerate(arms):
                if (task,draw,arm) not in lookup:
                    continue
                record = lookup[(task,draw,arm)]
                oldcounts = np.array(record['old_counts']); counts = np.array(record['counts'])
                supports = z['supports'][draw,ai]
                np.testing.assert_array_equal(frame.loc[supports,'class_id'], z['support_labels'][draw,ai])
                assert len(set(frame.loc[supports,'identity'])) == 64
                assert not set(frame.loc[supports,'identity']) & set(z['identities'])
                for mi, mode in enumerate(modes):
                    prob = z['probability'][draw,ai,mi]
                    prob0 = z['probability'][draw,0,mi]
                    pred = prob.argmax(1); pred0 = prob0.argmax(1)
                    if record['changed'] == 0:
                        np.testing.assert_array_equal(prob, prob0)
                    correct = pred == y; oldcorrect = pred0 == y
                    rescues = ~oldcorrect & correct; harms = oldcorrect & ~correct
                    oa = 100*correct.mean()
                    ba = 100*np.mean([correct[y==c].mean() for c in present])
                    delta_oa = 100*(correct.astype(int)-oldcorrect).mean()
                    delta_ba = 100*np.mean([(correct[y==c].astype(int)-oldcorrect[y==c]).mean() for c in present])
                    truep = prob[np.arange(len(y)),y]
                    truep0 = prob0[np.arange(len(y)),y]
                    competitor = prob.copy(); competitor[np.arange(len(y)),y] = -np.inf
                    competitor0 = prob0.copy(); competitor0[np.arange(len(y)),y] = -np.inf
                    margin = truep-competitor.max(1); margin0 = truep0-competitor0.max(1)
                    rows.append(dict(task=task, dataset=group, draw=draw, arm=arm, mode=mode,
                        n=len(y), classes=len(present), changed=record['changed'],
                        missing_before=int(np.sum(oldcounts[present]==0)), missing_after=int(np.sum(counts[present]==0)),
                        oa=oa, ba=ba, delta_oa=delta_oa, delta_ba=delta_ba,
                        rescue_n=int(rescues.sum()), harm_n=int(harms.sum()),
                        mean_margin=float(margin.mean()), delta_margin=float((margin-margin0).mean())))
                    contributions = []
                    for c in present:
                        mask = y == c
                        role = ('repaired_class' if oldcounts[c]==0 and counts[c]>0 else
                                'still_missing' if oldcounts[c]==0 else 'donor' if counts[c]<oldcounts[c] else 'other_supported')
                        drecall = 100*(correct[mask].astype(int)-oldcorrect[mask]).mean()
                        oa_contribution = 100*(int(rescues[mask].sum())-int(harms[mask].sum()))/len(y)
                        ba_contribution = drecall/len(present)
                        contributions.append((oa_contribution,ba_contribution))
                        classes.append(dict(task=task, dataset=group, draw=draw, arm=arm, mode=mode,
                            class_id=int(c),role=role,n=int(mask.sum()),old_support=int(oldcounts[c]),support=int(counts[c]),
                            recall_before=100*oldcorrect[mask].mean(),recall_after=100*correct[mask].mean(),
                            delta_recall=drecall,oa_contribution=oa_contribution,ba_contribution=ba_contribution,
                            rescue_n=int(rescues[mask].sum()),harm_n=int(harms[mask].sum()),
                            delta_margin=float((margin[mask]-margin0[mask]).mean())))
                    np.testing.assert_allclose(np.sum(contributions,axis=0),[delta_oa,delta_ba],atol=1e-10)
                    np.testing.assert_allclose(100*(rescues.sum()-harms.sum())/len(y),delta_oa,atol=1e-10)
                    checks += 3
                    if arm != 'random' and mode == 'full':
                        for i in np.flatnonzero(rescues|harms):
                            events.append(dict(task=task,draw=draw,arm=arm,query_local=int(z['query'][i]),
                                identity=str(z['identities'][i]),class_id=int(y[i]),before=int(pred0[i]),after=int(pred[i]),
                                event='rescue' if rescues[i] else 'harm',margin_before=float(margin0[i]),margin_after=float(margin[i])))
    m = pd.DataFrame(rows); c = pd.DataFrame(classes)
    m.to_csv(OUT/'metrics.csv',index=False)
    c.to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
    means = m.groupby(['dataset','task','arm','mode']).mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'task_summary.csv',index=False)
    summary = means.groupby(['dataset','arm','mode']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    draws = m.groupby(['dataset','arm','mode','draw'])[['oa','ba','delta_oa','delta_ba']].mean().reset_index()
    draws.to_csv(OUT/'draws.csv',index=False)
    keys = ['task','draw','mode']
    sham = m[m.arm=='sham1'][keys+['oa','ba']].rename(columns={'oa':'sham_oa','ba':'sham_ba'})
    versus = m[m.arm=='cover1'].merge(sham,on=keys,validate='one_to_one')
    versus['cover_minus_sham_oa'] = versus.oa-versus.sham_oa
    versus['cover_minus_sham_ba'] = versus.ba-versus.sham_ba
    versus.to_csv(OUT/'coverage_vs_sham.csv',index=False)
    # Fill absent roles with zero contribution before averaging over complete units.
    roles = ('repaired_class','still_missing','donor','other_supported')
    keys = ['dataset','task','draw','arm','mode']
    contribution = c.groupby(keys+['role'])[['oa_contribution','ba_contribution']].sum()
    index = pd.MultiIndex.from_tuples([tuple(r)+ (role,) for r in m[keys].itertuples(index=False,name=None) for role in roles],names=keys+['role'])
    contribution = contribution.reindex(index,fill_value=0).reset_index()
    contribution.to_csv(OUT/'role_contributions.csv',index=False)
    grouped = contribution.groupby(['dataset','arm','mode','role'])[['oa_contribution','ba_contribution']].mean().reset_index()
    grouped.to_csv(OUT/'role_summary.csv',index=False)
    print('FULL MODEL',summary[summary['mode']=='full'][['dataset','arm','oa','ba','delta_oa','delta_ba','changed','rescue_n','harm_n']].to_string(index=False))
    print('HEADS',summary[summary.arm=='cover1'][['dataset','mode','delta_oa','delta_ba']].to_string(index=False))
    print('DRAW EFFECT',draws[(draws.arm=='cover1')&(draws['mode']=='full')].to_string(index=False))
    print('BY TASK',means[(means.arm=='cover1')&(means['mode']=='full')][['task','changed','delta_oa','delta_ba']].to_string(index=False))
    print('ROLES',grouped[(grouped.arm=='cover1')&(grouped['mode']=='full')].to_string(index=False))
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    names=['23ports','f0921','oss2']
    for ax,metric,title in zip(axes,['delta_oa','delta_ba'],['Overall accuracy change','Balanced accuracy change']):
        for j,(arm,color) in enumerate([('cover1','#328a9e'),('sham1','#979797')]):
            table=summary[(summary.arm==arm)&(summary['mode']=='full')].set_index('dataset')
            ax.bar(np.arange(3)+(j-.5)*.3,table.loc[names,metric],.3,label=arm,color=color)
        ax.set_xticks(range(3),['23 ports (equal task)','F_0921','OSS2'])
        ax.axhline(0,color='#666666',lw=.8);ax.set_title(title);ax.set_ylabel('Percentage points vs random')
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen complete model, 64 supports, fixed vessel queries',fontsize=11)
    fig.savefig(OUT/'FULL_MODEL_COVERAGE.png',dpi=180);fig.savefig(OUT/'FULL_MODEL_COVERAGE.pdf');plt.close(fig)
    (OUT/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(metric_rows=len(m),class_rows=len(c),
        contribution_checks=checks,main_model_changed=False,unknown_detection_evaluated=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
