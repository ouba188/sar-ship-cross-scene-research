"""Verify the bounded decomposition and render expansion results."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from expand import OUT,metadata,write


def main():
    d=metadata();plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'class_shared_diagnosis/metrics.csv');checked=0
    # One newly enrolled scene per port checks the new component readout and labels.
    for ei in [1,15,35]:
        e=plans[ei];y=d.iloc[e['query']].class_id.to_numpy(int)
        for b in [16,64]:
            p=np.load(OUT/'class_shared_diagnosis'/f'episode_{ei:02d}_b{b}.npz')
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    for ai,arm in enumerate(p['arms']):
                        for mi,mode in enumerate(['expert','compact_full']):
                            pred=p['probabilities'][draw,ci,ai,mi].argmax(1)
                            a=metrics[(metrics.episode==ei)&(metrics.budget==b)&(metrics.draw==draw)&(metrics.cell==cell)&
                                (metrics.arm==arm)&(metrics['mode']==mode)&(metrics.subset=='all_queries')].iloc[0]
                            np.testing.assert_allclose([a.oa,a.ba],[100*accuracy_score(y,pred),
                                100*recall_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)],atol=1e-10)
                            checked+=1
    all_metrics=pd.read_csv(OUT/'metrics.csv')
    all48=all_metrics[(all_metrics.subset=='all_queries')&(all_metrics['mode']=='compact_full')]
    all48.groupby(['port','budget','cell','arm'])[['delta_oa','delta_ba','delta_macro_brier','delta_macro_nll']].mean().groupby(['budget','cell','arm']).mean().to_csv(OUT/'all48_effects.csv')
    ef=pd.read_csv(OUT/'effects_summary.csv')
    df=ef[(ef.cohort=='new_acquisition')&(ef.subset=='all_queries')&(ef['mode']=='compact_full')&(ef.cell=='other')]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    arms=['class_mean','class_cluster','scene_memory'];labels=['Class mean','Class clusters','Scene memory']
    for ax,budget in zip(axes,[16,64]):
        a=df[df.budget==budget].set_index('arm');x=np.arange(3)
        ax.bar(x-.17,a.loc[arms].delta_oa,.34,color='#267f79',label='OA')
        ax.bar(x+.17,a.loc[arms].delta_ba,.34,color='#b55269',label='BA')
        ax.set_xticks(x,labels);ax.axhline(0,color='#333333',linewidth=.8)
        ax.set_ylabel('Change vs original (percentage points)');ax.set_title(f'42 added query acquisitions, B={budget}')
        ax.spines[['top','right']].set_visible(False);ax.legend()
    fig.savefig(OUT/'expanded_results.png',dpi=160);fig.savefig(OUT/'expanded_results.svg');plt.close(fig)
    audit=json.loads((OUT/'VERIFIED.json').read_text(encoding='utf-8'))
    extra=json.loads((OUT/'class_shared_diagnosis/COMPLETE.json').read_text(encoding='utf-8'))
    write('COMPLETE.json',dict(date='2026-10-14',status='completed',acquisitions=48,additional_query_acquisitions=42,
        primary_expert_solves=3840,diagnostic_expert_solves=extra['expert_solves'],neural_training=0,
        source_geometry_fits=0,memory_fits=0,primary_metric_checks=audit['independent_metric_rows'],
        diagnostic_metric_checks=checked,independent_kernel_ridge_cases=48+extra['independent_solver_cases'],
        pilot_probability_replays=12,external_datasets_run=False,main_model_changed=False,
        conclusion='Scene memory weak pilot gain did not reproduce; class-mean control has scoped added-query gains, preserved pilot absent-class counterexample',
        correction='One source-scaled class-shared memory repair did not restore cross-acquisition utility; retained, not promoted',
        next_experiment_started=False))
    print('COMPLETE',checked,flush=True)


if __name__=='__main__':main()
