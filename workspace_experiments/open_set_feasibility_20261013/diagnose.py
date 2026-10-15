"""Data-level coverage diagnosis and bounded independent metric verification."""

from study import OUT, write, METHODS
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    m=pd.read_csv(OUT/'metrics.csv');p=pd.read_csv(OUT/'partial_metrics.csv')
    source=pd.read_csv(OUT/'source_manifest.csv',dtype={'identity':str})
    enrollment=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    groups=[];confusions=[];checks=0
    for path in sorted((OUT/'tasks').rglob('unknown_*.npz')):
        task=path.parent.relative_to(OUT/'tasks').as_posix();unknown=int(path.stem.split('_')[1])
        z=np.load(path);y=z['labels'];known=y!=unknown
        frame=pd.read_csv(path.parent/'manifest.csv',dtype={'identity':str}).set_index('local_row')
        qids=frame.loc[z['query'],'identity']
        assert not set(qids)&set(source.identity)
        with (OUT/'fits'/f'without_{unknown}.pkl').open('rb') as h:model=pickle.load(h)
        assert unknown not in model['fit_class_counts']
        assert model['pca'].n_samples_==int((source.class_id!=unknown).sum())
        assert model['ssl'].n_samples_==model['pca'].n_samples_
        for draw in range(5):
            sids=frame.loc[z['supports'][draw],'identity']
            assert len(set(sids))==64 and not set(sids)&set(qids)
            assert not set(sids)&set(source.identity)
            np.testing.assert_array_equal(frame.loc[z['supports'][draw],'class_id'],z['support_labels'][draw])
            np.testing.assert_array_equal(frame.loc[z['query'],'class_id'],y)
            counts=np.bincount(z['support_labels'][draw],minlength=8)
            pred=z['classes'][z['probability'][draw].argmax(1)]
            closed=np.mean([np.mean(pred[y==c]==c) for c in np.unique(y[known])])*100
            for mi,method in enumerate(METHODS):
                score=z['score'][draw,:,mi];reject=score>z['thresholds'][draw,1,mi]
                for c in range(8):
                    use=y==c
                    if not use.any():continue
                    groups.append(dict(task=task,unknown=unknown,draw=draw,method=method,true_class=c,
                        stratum='unknown' if c==unknown else 'known_no_support' if counts[c]==0 else 'known_supported',
                        n=int(use.sum()),class_support=int(counts[c]),
                        reject_rate=100*np.mean(reject[use]),mean_score=float(score[use].mean()),
                        correct_lost=100*np.mean((pred[use]==c)&reject[use])))
                # Direct class-by-class evaluation independent of the analysis weight-vector helper.
                known_classes=np.unique(y[known])
                after=100*np.mean([np.mean((pred[y==c]==c)&~reject[y==c]) for c in known_classes])
                frr=100*np.mean([np.mean(reject[y==c]) for c in known_classes])
                tpr=100*np.mean(reject[~known]);utility=.2*tpr-.8*(closed-after)
                actual=m[(m.task==task)&(m.unknown==unknown)&(m.draw==draw)&(m.method==method)&(m['quantile']==.95)].iloc[0]
                np.testing.assert_allclose([actual.known_ba_before,actual.known_ba_after,actual.known_false_reject,
                    actual.unknown_tpr,actual.utility_20],[closed,after,frr,tpr,utility],atol=1e-9,rtol=0)
                # With one unknown class, macro AUROC equals the mean of its pairwise AUROC against each known class.
                auc=100*np.mean([roc_auc_score(y[(y==c)|(~known)]==unknown,score[(y==c)|(~known)]) for c in known_classes])
                np.testing.assert_allclose(actual.auroc,auc,atol=1e-9,rtol=0);checks+=6
            for c in range(8):
                confusions.append(dict(task=task,unknown=unknown,draw=draw,predicted_class=c,
                    unknown_n=int((~known).sum()),prediction_percent=100*np.mean(pred[~known]==c)))
    strata=pd.DataFrame(groups);strata.to_csv(OUT/'class_coverage_diagnostics.csv',index=False)
    ss=strata.groupby(['task','method','stratum'])[['reject_rate','correct_lost','n']].mean().reset_index()
    ss.to_csv(OUT/'coverage_diagnostic_summary.csv',index=False)
    pd.DataFrame(confusions).to_csv(OUT/'unknown_confusions.csv',index=False)
    paired=pd.read_csv(OUT/'paired_effects.csv');paired=paired[paired['quantile']==.95]
    draws=paired.groupby(['task','method','draw'])[['delta_utility_20','delta_auroc']].mean().reset_index()
    draws.to_csv(OUT/'draw_effects.csv',index=False)
    partial=p.groupby(['task','draw','mode']).delta_ba.mean().reset_index()
    partial.to_csv(OUT/'partial_draw_effects.csv',index=False)
    write(OUT/'VERIFIED.json',dict(contexts=200,metric_checks=checks,source_target_identity_disjoint=True,
        support_query_identity_disjoint=True,known_only_pca_and_heads=True,
        independent_primal_checks=40,all_passed=True,
        remaining_boundary='Historical frozen encoder exposure and compact pilot, not full-model open-set validation'))
    print(ss[ss.method=='support_knn'].to_string(index=False))
    print('FISHING AS UNKNOWN')
    print(m[(m.unknown==1)&(m['quantile']==.95)].groupby(['task','method'])[['unknown_n','auroc','unknown_tpr','known_false_reject','utility_20','utility_05']].mean().to_string())
    print('UTILITY SENSITIVITY')
    print(m[m['quantile']==.95].groupby(['task','method'])[['utility_05','utility_20','utility_50']].mean().to_string())


if __name__=='__main__':main()
