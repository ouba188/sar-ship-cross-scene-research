"""Final independent metric replay, small cached timing comparison and summary plot."""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, recall_score
from threadpoolctl import threadpool_limits
from core import relation, kernel
from run_pilot import OUT, metadata, operator, write


def main():
    d=metadata(); plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    rep=OUT/'relative_repair'; m=pd.read_csv(rep/'metrics.csv'); checked=0
    for e in plans:
        y=d.iloc[e['query']].class_id.to_numpy(int)
        for b in [16,64]:
            p=np.load(rep/f"episode_{e['episode']:02d}_b{b}.npz")
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    for ai,arm in enumerate(p['arms']):
                        for mi,mode in enumerate(['expert','compact_full']):
                            pp=p['probabilities'][draw,ci,ai,mi]
                            a=m[(m.episode==e['episode'])&(m.budget==b)&(m.draw==draw)&(m.cell==cell)&(m.arm==arm)&(m['mode']==mode)].iloc[0]
                            np.testing.assert_allclose([a.oa,a.ba],[100*accuracy_score(y,pp.argmax(1)),100*recall_score(y,pp.argmax(1),labels=np.unique(y),average='macro',zero_division=0)],atol=1e-10)
                            checked+=1
    f=np.load(OUT/'encoded.npz'); rows=f['rows']; z=np.c_[f['r'],f['t']]
    with (OUT/'memories.pkl').open('rb') as h:memories=pickle.load(h)
    spec=json.loads((rep/'PROTOCOL.json').read_text(encoding='utf-8'))['calibration']['scene_memory']
    ev=operator(); times=[]
    def mapping(x):
        r=relation(x,memories['scene_memory'])
        return (r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-15)-np.array(spec['center']))*spec['scale']
    rr=np.load(rep/'relations.npz')['scene_memory']
    np.testing.assert_allclose(mapping(z[:100]),rr[:100],atol=1e-12)
    for e in plans:
        qi=np.searchsorted(rows,e['query'])
        for b in [16,64]:
            s=np.array(e['draws'][0][str(b)][1]); si=np.searchsorted(rows,s); sy=d.iloc[s].class_id.to_numpy(int)
            for kind in ['original','relative_scene']:
                repetitions=[]
                for repeat in range(11):
                    tick=time.perf_counter()
                    if kind=='original':
                        ks=kernel(z[si],z[si]); kq=kernel(z[qi],z[si])
                    else:
                        rs=mapping(z[si]);rq=mapping(z[qi]);ks=kernel(z[si],z[si],rs,rs);kq=kernel(z[qi],z[si],rq,rs)
                    alpha=ev.coefficients(ks,f['pd'][si],sy,8.)
                    softmax(f['pd'][qi]+kq@alpha,axis=1)
                    elapsed=time.perf_counter()-tick
                    if repeat:repetitions.append(elapsed)
                times.append(dict(episode=e['episode'],budget=b,queries=len(qi),arm=kind,median_seconds=float(np.median(repetitions))))
    pd.DataFrame(times).to_csv(OUT/'matched_cached_timing.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=pd.read_csv(OUT/'effect_summary.csv'); new=pd.read_csv(rep/'effect_summary.csv')
    fig, axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    names=['class_mean','class_cluster','scene_memory']; labels=['Class mean','Class clusters','Scene memory']
    for ax,b in zip(axes,[16,64]):
        before=old[(old['mode']=='compact_full')&(old.cell=='other')&(old.budget==b)].set_index('arm')
        after=new[(new['mode']=='compact_full')&(new.cell=='other')&(new.budget==b)].set_index('arm')
        x=np.arange(3)
        ax.bar(x-.18,before.loc[names].delta_ba,.36,label='Raw reference response',color='#89929c')
        ax.bar(x+.18,after.loc[names].delta_ba,.36,label='Relative-profile repair',color='#267f79')
        ax.axhline(0,color='#333333',linewidth=.8);ax.set_xticks(x,labels);ax.set_title(f'Cross-acquisition support, B={b}')
        ax.set_ylabel('BA change vs original (percentage points)');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'pilot_results.png',dpi=160);fig.savefig(OUT/'pilot_results.svg');plt.close(fig)
    scene=new[(new['mode']=='compact_full')&(new.arm=='scene_memory')]
    t=pd.read_csv(rep/'transitions.csv')
    pooled=t[(t['mode']=='compact_full')&(t.arm=='scene_memory')].groupby(['budget','cell'])[['rescued','harmed']].sum()
    pooled.to_csv(OUT/'repair_event_counts.csv')
    write('COMPLETE.json',dict(status='completed',primary_expert_solves=480,diagnostic_expert_solves=360,repair_expert_solves=360,
        independent_primary_metrics=960,independent_repair_metrics=checked,independent_solver_cases=48+36+36,
        new_source_geometry=1,new_source_heads=3,neural_training=0,external_datasets_run=False,main_model_changed=False,
        scope='3source-held ports6acquisitions300observations288ships,5support draws,2budgets',
        conclusion='Raw memory ineffective; common response dominated. One relative-profile repair gives small scoped classification gains, not a mature universal module',
        next_experiment_started=False))
    print('REPAIR METRICS VERIFIED',checked,flush=True)
    print('CACHED TIMING\n',pd.DataFrame(times).groupby(['budget','arm']).median(numeric_only=True).to_string(),flush=True)
    print('REPEATED EXPOSURE COUNTS\n',pooled.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
