"""Paired accounting of prespecified region controls; never selects target models."""
import sys
from pathlib import Path
import importlib.util
import argparse
import json
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_accounting', HERE.parent / 'analysis/analyze_results.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
PAIRS = [('dynamic','shuffled'), ('dynamic','body'), ('body_end','body'), ('body_side','body'),
         ('dynamic','body_end'), ('dynamic','body_side')]


def analyze(stage):
    role = 'validation' if stage == 'source' else 'test'
    f = pd.read_csv(HERE / f'{stage}_predictions.csv', dtype={'mmsi':str}, float_precision='round_trip')
    f = f[f.role.eq(role)].copy()
    if stage == 'source':
        full = pd.read_csv(HERE.parent / 'source_predictions.csv', dtype={'mmsi':str}, float_precision='round_trip')
        f = pd.concat([f, full[full.role.eq(role) & full.arm.eq('dynamic')]], ignore_index=True)
    assert not f.duplicated(['arm','seed','fold','instance_id']).any()
    metrics, contrasts, groups, active, intervals, checks = [], [], [], [], [], []
    for seed, g in f.groupby('seed'):
        arms = {name:part.sort_values(['fold','instance_id']).reset_index(drop=True) for name,part in g.groupby('arm')}
        ref = arms['dynamic']
        y = ref.class_id.to_numpy()
        base = ref[a.BASE].to_numpy()
        scores = {'baseline':base}
        for name,part in arms.items():
            pd.testing.assert_frame_equal(part[['fold','instance_id','class_id']], ref[['fold','instance_id','class_id']])
            np.testing.assert_array_equal(part[a.BASE].to_numpy(), base)
            scores[name] = part[a.SCORE].to_numpy()
        if stage == 'test':
            assert len(ref) == 4744 and ref.instance_id.is_unique
        checks.append(dict(seed=seed,n=len(y),paired_ids_exact=True,paired_base_exact=True))
        for name,score in scores.items():
            m = a.assess(y,score)
            metrics.append(dict(seed=seed,arm=name,J=(m['OA']+m['BA'])/2,**m))
        for left,right in PAIRS + [(name,'baseline') for name in arms]:
            ls,rs = scores[left],scores[right]
            meta = dict(seed=seed,left=left,right=right)
            v = a.paired(y,ls,rs)
            contrasts.append(dict(**meta,delta_J=(v['delta_OA']+v['delta_BA'])/2,**v))
            for field in ['class_name','port']:
                for level,idx in ref.groupby(field).groups.items():
                    idx = np.asarray(idx)
                    groups.append(dict(**meta,field=field,group=level,**a.paired(y[idx],ls[idx],rs[idx])))
            if right != 'baseline':
                le,re = arms[left].epoch.to_numpy(),arms[right].epoch.to_numpy()
                for fold in range(3):
                    take = ref.fold.eq(fold).to_numpy()
                    if not take.any():
                        continue
                    active.append(dict(**meta,fold=fold,left_epoch=int(le[take][0]),right_epoch=int(re[take][0]),
                        both_active=bool(le[take][0]>0 and re[take][0]>0),**a.paired(y[take],ls[take],rs[take])))
            if stage == 'test' and (left,right) in PAIRS:
                delta = (ls.argmax(1)==y).astype(int)-(rs.argmax(1)==y).astype(int)
                for field in ['mmsi','pass_group']:
                    intervals.append(dict(**meta,cluster=field,**a.clustered(y,delta,ref[field],20261023+int(seed))))
    for name,rows in [('metrics',metrics),('contrasts',contrasts),('groups',groups),('active',active),('intervals',intervals),('pairing_checks',checks)]:
        pd.DataFrame(rows).to_csv(HERE / f'{stage}_{name}.csv',index=False)
    m = pd.DataFrame(metrics)
    summary = m.groupby('arm')[['OA','BA','J','NLL','Brier']].agg(['mean','std'])
    summary.to_csv(HERE / f'{stage}_summary.csv')
    print(summary.to_string())
    print(pd.DataFrame(contrasts).to_string(index=False))
    if stage == 'test':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        c = pd.DataFrame(contrasts)
        fig,axes = plt.subplots(1,2,figsize=(13,5))
        labels = [f'{l} - {r}' for l,r in PAIRS]
        for ax,key in zip(axes,['delta_OA','delta_BA']):
            for j,seed in enumerate([17,29,43]):
                values = [float(c.loc[(c.seed==seed)&(c.left==l)&(c.right==r),key].iloc[0]) for l,r in PAIRS]
                ax.bar(np.arange(len(PAIRS))+(j-1)*.24,values,width=.24,label=f'seed {seed}')
            ax.axhline(0,color='black',linewidth=.8)
            ax.set_xticks(np.arange(len(PAIRS)),labels,rotation=30,ha='right')
            ax.set_ylabel(key.replace('delta_','')+' difference (percentage points)')
        axes[0].legend()
        fig.suptitle('Frozen full readout versus matched region controls: all three seeds')
        fig.tight_layout()
        fig.savefig(HERE/'paired_controls.png',dpi=150)
        plt.close(fig)
    (HERE/f'{stage}_analysis_receipt.json').write_text(json.dumps(dict(stage=stage,new_fits=0,
        seed_count=3,rows=len(f),full_reused=True,cluster_note='Separate vessel/pass resampling within each seed; not independent seed-expanded ships.'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['source','test'])
    analyze(p.parse_args().stage)
