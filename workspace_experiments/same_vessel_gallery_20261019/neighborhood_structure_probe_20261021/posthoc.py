"""Bounded diagnostics; no sample removal from primary results or new fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent


def metrics(y,p):
    return dict(oa=accuracy_score(y,p),ba=balanced_accuracy_score(y,p),
                macro_f1=f1_score(y,p,average='macro',zero_division=0))


def main():
    for body in (False,True):
        root=ROOT/'body_only' if body else ROOT
        pred=pd.read_csv(root/'predictions.csv')
        results=[]
        for (mode,arm,seed),g in pred.groupby(['mode','arm','seed']):
            if mode=='fixed':continue
            for scope,rows in (('primary_all',g),('posthoc_without_10461',g[g.obs_id.ne(10461)])):
                results.append(dict(scope=scope,mode=mode,arm=arm,seed=seed,n=len(rows),
                    **metrics(rows.class_id,rows.prediction)))
        pd.DataFrame(results).to_csv(root/'outlier_sensitivity.csv',index=False,lineterminator='\n')
        constants=[]
        for seed in range(3):
            lookup=pred[pred['mode'].eq('reference')&pred.arm.eq('baseline')&pred.seed.eq(seed)].reset_index(drop=True)
            for variant in ('source_constant','scores'):
                files=[np.load((root/f'scores_f{k}_original_matched_s{seed}.npz') if body else
                    ROOT.parent/'neighborhood_level_split_20261021'/f'scores_f{k}_separated_s{seed}.npz') for k in range(3)]
                for k,z in enumerate(files):
                    ref=np.load(root/f'scores_f{k}_reference_s{seed}.npz')
                    np.testing.assert_array_equal(z['rows'],ref['rows'])
                score=np.concatenate([z[variant] for z in files])
                constants.append(dict(seed=seed,arm=variant,
                    **metrics(lookup.class_id,score.argmax(1))))
        pd.DataFrame(constants).to_csv(root/'reference_constant.csv',index=False,lineterminator='\n')
        cases=pd.read_csv(root/'case_index.csv')
        parts=pd.read_csv(ROOT.parent/'neighborhood_level_split_20261021'/'partitions.csv')
        cases=cases.merge(parts[parts.partition.eq('test')][['obs_id','split_fold']],on='obs_id',validate='many_to_one')
        fit=pd.read_csv(root/'metrics.csv')
        fit=fit[fit.partition.eq('test')].rename(columns={'fold':'split_fold'})
        cases=cases.merge(fit[['split_fold','seed','mode','arm','step']],on=['split_fold','seed','mode','arm'],validate='many_to_one')
        cases.to_csv(root/'case_selected_steps.csv',index=False,lineterminator='\n')
        print('body_only' if body else 'whole',pd.DataFrame(constants).groupby('arm')[['oa','ba']].mean().to_string())
        print(pd.DataFrame(results).query("arm=='flat'").groupby('scope')[['oa','ba']].mean().to_string())


if __name__=='__main__':main()
