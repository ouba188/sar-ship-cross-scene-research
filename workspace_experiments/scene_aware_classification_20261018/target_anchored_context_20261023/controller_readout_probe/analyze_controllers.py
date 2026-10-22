"""Analyze completed arms without refitting or touching parent artifacts."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from run_controllers import HERE, PARENT, EXP, ARMS
sys.path.insert(0, str(EXP))
from analyze_expanded import clustered_ci


def margins(q, prefix):
    scores = q[[prefix+str(c) for c in range(3)]].to_numpy()
    y = q.class_id.to_numpy()
    other = scores.copy()
    other[np.arange(len(q)), y] = -np.inf
    return scores[np.arange(len(q)), y]-other.max(1)


def main():
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi':str})
    parent = pd.read_csv(PARENT/'control_predictions.csv')
    parent = parent[parent.arm.eq('own_regions_full')].rename(columns={
        'prediction':'parent_prediction', 'base_prediction':'parent_base_prediction',
        **{f's{c}':f'parent_s{c}' for c in range(3)}})
    completed = [arm for arm in ARMS if (HERE/arm/'summary.json').exists()]
    for filename in ('metrics','predictions','pooled','curves','verification','gates','env_norms'):
        tables = [pd.read_csv(HERE/arm/(filename+'.csv')) for arm in completed]
        pd.concat(tables, ignore_index=True).to_csv(HERE/(filename+'.csv'), index=False)
    predictions, metrics_rows, classes, acquisitions, intervals, cases = [], [], [], [], [], []
    existing = {}
    for pattern in ('case_pair_*.png', 'control_case_pair_*.png'):
        for path in sorted(PARENT.glob(pattern)):
            existing['pair_'+path.stem.split('pair_',1)[1]] = path
    for arm in completed:
        p = pd.read_csv(HERE/arm/'predictions.csv')
        p = p.merge(parent[['fold','seed','instance_id','class_id','parent_prediction','parent_base_prediction',
                           'parent_s0','parent_s1','parent_s2']],
                    on=['fold','seed','instance_id','class_id'], validate='one_to_one')
        assert len(p) == len(t)*3 and p.base_prediction.eq(p.parent_base_prediction).all()
        p = p.merge(t[['instance_id','mmsi','pass_group','port']], on='instance_id', validate='many_to_one')
        p['correct'] = p.prediction.eq(p.class_id)
        p['base_correct'] = p.base_prediction.eq(p.class_id)
        p['parent_correct'] = p.parent_prediction.eq(p.class_id)
        p['net_vs_base'] = p.correct.astype(int)-p.base_correct.astype(int)
        p['net_vs_parent'] = p.correct.astype(int)-p.parent_correct.astype(int)
        p['kind'] = np.select([p.net_vs_base.eq(1), p.net_vs_base.eq(-1), p.correct],
                              ['rescued','harmed','CC'], default='WW')
        for prefix, label in [('s','margin'),('b','base_margin'),('parent_s','parent_margin')]:
            p[label] = margins(p, prefix)
        assert np.isfinite(p[['s0','s1','s2','margin','parent_margin','base_margin']]).all().all()
        predictions.append(p)
        for seed, rows in p.groupby('seed'):
            for scope, q in [('all',rows),('external',rows[rows.has_external]),('no_external',rows[~rows.has_external])]:
                for ref in ('base','parent'):
                    diff = q.correct.astype(int)-q[ref+'_correct'].astype(int)
                    metrics_rows.append(dict(arm=arm, seed=seed, scope=scope, reference=ref, n=len(q),
                        delta_OA=float(diff.mean()*100), delta_BA=float(diff.groupby(q.class_id).mean().mean()*100),
                        rescued=int((q.correct & ~q[ref+'_correct']).sum()),
                        harmed=int((~q.correct & q[ref+'_correct']).sum()), net=int(diff.sum())))
        for (seed,c), q in p.groupby(['seed','class_id']):
            classes.append(dict(arm=arm, seed=seed, class_id=c, n=len(q),
                recall=float(q.correct.mean()*100), parent_recall=float(q.parent_correct.mean()*100),
                net_vs_base=int(q.net_vs_base.sum()), net_vs_parent=int(q.net_vs_parent.sum())))
        for (seed,acq), q in p.groupby(['seed','pass_group']):
            acquisitions.append(dict(arm=arm, seed=seed, pass_group=acq, n=len(q),
                net_vs_base=int(q.net_vs_base.sum()), net_vs_parent=int(q.net_vs_parent.sum())))
        for ref in ('base','parent'):
            diff = p.groupby('instance_id')['net_vs_'+ref].mean().reindex(t.instance_id).to_numpy()
            for block in ('pass_group','mmsi'):
                intervals.append(dict(arm=arm, reference=ref, block=block, delta_OA=float(diff.mean()*100),
                    delta_BA=float(pd.Series(diff).groupby(t.class_id).mean().mean()*100), **clustered_ci(t,diff,block)))
        for kind in ('rescued','harmed','CC','WW'):
            options = p[p.seed.eq(29)&p.kind.eq(kind)].copy()
            options['existing'] = options.instance_id.isin(existing)
            row = options.sort_values(['existing','instance_id'], ascending=[False,True]).iloc[0]
            path = existing.get(row.instance_id)
            reuse = path is not None
            if path is None:
                path = HERE/f'{arm}_{row.instance_id}.png'
                panel = Image.new('RGB',(820,350),'white')
                draw = ImageDraw.Draw(panel)
                font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
                draw.text((10,10), f'{row.instance_id} {arm} {kind} seed29 true={row.class_id}', font=font, fill='black')
                draw.text((10,35),f'A={row.base_prediction} parent_own={row.parent_prediction} current={row.prediction}',font=font,fill='black')
                for k,pol in enumerate(('vv','vh')):
                    with Image.open(EXP/'readout/native_targets'/f'{row.instance_id}_{pol}.png') as im:
                        im = im.convert('RGB')
                        draw.text((10+400*k,80), f'{pol.upper()} native {im.width}x{im.height}', font=font, fill='black')
                        panel.paste(im,(10+400*k,110))
                draw.text((10,315),'Native target pixels; parent panel captions elsewhere are not current predictions.',font=font,fill='black')
                panel.save(path)
            record = row.drop(labels=['existing']).to_dict()
            record.update(panel=str(path), reused_parent_panel=reuse,
                selection='seed29 outcome_vs_A, prefer existing parent panel, then ascending ID',
                native_vv=str(EXP/'readout/native_targets'/f'{row.instance_id}_vv.png'),
                native_vh=str(EXP/'readout/native_targets'/f'{row.instance_id}_vh.png'))
            cases.append(record)
    pd.concat(predictions).to_csv(HERE/'paired_predictions.csv',index=False)
    for name, values in [('paired_metrics',metrics_rows),('paired_intervals',intervals),
                         ('class_effects',classes),('acquisition_effects',acquisitions),('review_cases',cases)]:
        pd.DataFrame(values).to_csv(HERE/(name+'.csv'),index=False)
    if 'whole64' in completed and 'constant_own_full' in completed:
        pp = pd.concat(predictions)
        a = pp[pp.arm.eq('whole64')].set_index(['seed','instance_id'])
        b = pp[pp.arm.eq('constant_own_full')].set_index(['seed','instance_id']).reindex(a.index)
        difference = a.correct.astype(int)-b.correct.astype(int)
        comparison = []
        for seed in sorted(a.index.get_level_values('seed').unique()):
            ix = a.index.get_level_values('seed') == seed
            diff = difference.loc[ix]
            comparison.append(dict(arm='whole64',reference='constant_own_full',seed=int(seed),n=int(ix.sum()),
                delta_OA=float(diff.mean()*100),delta_BA=float(diff.groupby(a.loc[ix,'class_id']).mean().mean()*100),
                rescued=int((a.loc[ix,'correct'] & ~b.loc[ix,'correct']).sum()),
                harmed=int((~a.loc[ix,'correct'] & b.loc[ix,'correct']).sum()),net=int(diff.sum())))
        pd.DataFrame(comparison).to_csv(HERE/'whole_vs_constant.csv',index=False)
        avg = difference.groupby('instance_id').mean().reindex(t.instance_id).to_numpy()
        extra_ci = [dict(arm='whole64',reference='constant_own_full',block=block,
            delta_OA=float(avg.mean()*100),delta_BA=float(pd.Series(avg).groupby(t.class_id).mean().mean()*100),
            **clustered_ci(t,avg,block)) for block in ('pass_group','mmsi')]
        pd.DataFrame(extra_ci).to_csv(HERE/'whole_vs_constant_intervals.csv',index=False)
    summary = dict(completed_arms=completed, completed_heads=18*len(completed), planned_heads=72,
        pending_arms=[arm for arm in ARMS if arm not in completed],
        unique_case_panels=len(set(c['panel'] for c in cases)),
        reused_parent_case_rows=sum(c['reused_parent_panel'] for c in cases),
        ci='2000 seed20261022 bootstrap; seed-averaged target differences; separate pass and MMSI blocks, not joint.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics_rows).loc[lambda q:q.scope.eq('all')].to_string(index=False),flush=True)
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)
    print(pd.DataFrame(cases)[['arm','kind','instance_id','base_margin','parent_margin','margin','panel']].to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
